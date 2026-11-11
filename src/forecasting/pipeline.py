"""Rolling-origin backtest: train on history, forecast the next 28 days, repeat.

    PYTHONPATH=src python -m forecasting.pipeline --data data/raw/sales.csv --out reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecasting.evaluate import bias, inventory_impact, wape  # noqa: E402
from forecasting.features import FEATURES, HORIZON, build_features  # noqa: E402
from forecasting.models import all_models  # noqa: E402

N_FOLDS = 3
UNIT_COST_RATIO = 0.6  # cost of goods as a share of shelf price


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    last = df["date"].max()
    preds = []
    for k in range(N_FOLDS, 0, -1):
        origin = last - pd.Timedelta(days=HORIZON * k - 1)
        end = origin + pd.Timedelta(days=HORIZON - 1)
        train = df[df["date"] < origin]
        test = df[(df["date"] >= origin) & (df["date"] <= end)].copy()
        test["fold"] = N_FOLDS - k + 1
        print(f"Fold {test['fold'].iloc[0]}: train through {origin.date() - pd.Timedelta(days=1)}, "
              f"forecast {origin.date()} to {end.date()}")
        for model in all_models():
            model.fit(train)
            test[f"pred_{model.name}"] = model.predict(test)
            print(f"  {model.name:<18} WAPE {wape(test.units_sold.values, test[f'pred_{model.name}'].values):.3f}")
        preds.append(test)
    return pd.concat(preds, ignore_index=True)


def summarize(bt: pd.DataFrame, names: list[str]) -> dict[str, pd.DataFrame]:
    y = bt["units_sold"].to_numpy()
    overall = pd.DataFrame([
        {"model": m, "wape": wape(y, bt[f"pred_{m}"].values), "bias": bias(y, bt[f"pred_{m}"].values)}
        for m in names
    ])
    by_fold = (bt.groupby("fold")
               .apply(lambda g: pd.Series({m: wape(g.units_sold.values, g[f"pred_{m}"].values) for m in names}),
                      include_groups=False)
               .reset_index())
    by_category = (bt.groupby("category", observed=True)
                   .apply(lambda g: pd.Series({m: wape(g.units_sold.values, g[f"pred_{m}"].values) for m in names}),
                          include_groups=False)
                   .reset_index())
    promo = (bt.groupby("on_promo")
             .apply(lambda g: pd.Series({m: wape(g.units_sold.values, g[f"pred_{m}"].values) for m in names}),
                    include_groups=False)
             .reset_index().replace({"on_promo": {0: "regular", 1: "promotion"}}))
    return {"overall": overall, "by_fold": by_fold, "by_category": by_category, "by_promo": promo}


def figures(bt: pd.DataFrame, tables: dict, inv: pd.DataFrame, names: list[str], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    colors = {"seasonal_naive": "#9aa5b1", "weekday_average": "#6c8ebf", "gradient_boosting": "#1f4e8c"}
    labels = {m: m.replace("_", " ") for m in names}

    # Accuracy by segment
    fig, ax = plt.subplots(figsize=(8, 4))
    segs = pd.concat([
        tables["overall"].set_index("model")["wape"].rename("All").to_frame().T,
        tables["by_promo"].set_index("on_promo")[names],
    ])
    x = np.arange(len(segs))
    w = 0.26
    for i, m in enumerate(names):
        ax.bar(x + (i - 1) * w, segs[m], w, label=labels[m], color=colors[m])
    ax.set_xticks(x, ["All days", "Regular days", "Promotion days"])
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("Forecast error, 3 × 28-day backtest")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "wape_by_segment.png", dpi=150)
    plt.close(fig)

    # Example series
    sample = bt[(bt.store_id == "S01")].groupby("item_id", observed=True)["units_sold"].sum().sort_values()
    item = sample.index[int(len(sample) * 0.7)]
    s = bt[(bt.store_id == "S01") & (bt.item_id == item)].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(s["date"], s["units_sold"], color="black", lw=1.6, label="actual")
    for m in ("seasonal_naive", "gradient_boosting"):
        ax.plot(s["date"], s[f"pred_{m}"], lw=1.4, label=labels[m], color=colors[m])
    promo_days = s[s.on_promo == 1]["date"]
    for d in promo_days:
        ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="#f4c27a", alpha=0.35, lw=0)
    ax.set_title(f"Store S01, item {item}: 12 weeks of forecasts (shaded = promotion)")
    ax.set_ylabel("Units")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "example_series.png", dpi=150)
    plt.close(fig)

    # Inventory
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh([labels[m] for m in inv["model"]], inv["excess_value_at_cost"], color=[colors[m] for m in inv["model"]])
    ax.set_xlabel("Excess inventory at cost, 84 days ($)")
    ax.set_title("Inventory needed to hit a 95% fill rate")
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    fig.tight_layout()
    fig.savefig(fig_dir / "inventory_impact.png", dpi=150)
    plt.close(fig)


def feature_importance(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Permutation importance of the GBM on the final fold (sampled for speed)."""
    from sklearn.inspection import permutation_importance

    from forecasting.models import GlobalGBM

    last = df["date"].max()
    origin = last - pd.Timedelta(days=HORIZON - 1)
    model = GlobalGBM().fit(df[df["date"] < origin])
    test = df[df["date"] >= origin].sample(8000, random_state=0)
    pi = permutation_importance(model.model, test[FEATURES], test["units_sold"], n_repeats=5,
                                random_state=0, scoring="neg_mean_absolute_error")
    imp = (pd.DataFrame({"feature": FEATURES, "mae_increase": pi.importances_mean})
           .sort_values("mae_increase", ascending=False))
    imp.to_csv(out, index=False)
    return imp


def run(data: Path, out: Path) -> dict:
    raw = pd.read_csv(data, parse_dates=["date"])
    df = build_features(raw)
    bt = backtest(df)
    names = [m.name for m in all_models()]
    tables = summarize(bt, names)

    bt["unit_cost"] = bt["price"] / (1 - bt["discount_pct"]) * UNIT_COST_RATIO
    bt["sigma"] = bt["roll_std_28"]
    inv = inventory_impact(bt, names)

    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    for k, t in tables.items():
        t.to_csv(out / "tables" / f"wape_{k}.csv", index=False)
    inv.to_csv(out / "tables" / "inventory_impact.csv", index=False)
    imp = feature_importance(df, out / "tables" / "feature_importance.csv")
    figures(bt, tables, inv, names, out / "figures")

    o = tables["overall"].set_index("model")
    inv_i = inv.set_index("model")
    best_base = o.loc[["seasonal_naive", "weekday_average"], "wape"].idxmin()
    metrics = {
        "backtest": {"folds": N_FOLDS, "horizon_days": HORIZON, "series": int(raw.groupby(["store_id", "item_id"]).ngroups)},
        "wape": o["wape"].round(4).to_dict(),
        "bias": o["bias"].round(4).to_dict(),
        "best_baseline": best_base,
        "wape_reduction_vs_best_baseline": float(1 - o.loc["gradient_boosting", "wape"] / o.loc[best_base, "wape"]),
        "promo_day_wape": tables["by_promo"].set_index("on_promo").loc["promotion", names].round(4).to_dict(),
        "excess_inventory_reduction_vs_best_baseline": float(
            1 - inv_i.loc["gradient_boosting", "excess_value_at_cost"] / inv_i.loc[best_base, "excess_value_at_cost"]),
        "excess_inventory_value": inv_i["excess_value_at_cost"].round(0).to_dict(),
        "top_features": imp["feature"].head(5).tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/raw/sales.csv"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    a = p.parse_args()
    run(a.data, a.out)


if __name__ == "__main__":
    main()
