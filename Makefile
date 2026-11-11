.PHONY: setup data backtest test all

setup:
	python -m pip install -r requirements.txt

data:
	python data/generate_data.py --stores 10 --items 50 --days 730 --out data/raw/sales.csv

backtest:
	PYTHONPATH=src python -m forecasting.pipeline --data data/raw/sales.csv --out reports

test:
	python -m pytest -q

all: data backtest test
