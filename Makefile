PYTHON ?= python

.PHONY: setup data features targets backtest

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.cli data

features: data
	$(PYTHON) -m src.cli features

targets: features
	$(PYTHON) -m src.cli targets

backtest: targets
	$(PYTHON) -m src.cli backtest
