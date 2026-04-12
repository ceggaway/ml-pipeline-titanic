.PHONY: install train predict generate test lint freeze help

## Install dependencies
install:
	pip install -r requirements.txt

## Train the model
train:
	python -m src.pipeline.pipeline --config config/config.yaml --data data/raw/train.csv

## Generate synthetic daily input data
generate:
	python scripts/generate_sample_data.py --rows 100

## Run batch prediction on daily input
predict: generate
	python -m src.pipeline.pipeline --predict \
		--input data/raw/daily_input.csv \
		--output models/batch_outputs/predictions_$$(date +%Y%m%d_%H%M%S).csv

## Run all tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Run unit tests only
test-unit:
	pytest tests/test_pipeline.py -v

## Run integration tests only
test-integration:
	pytest tests/test_integration.py -v

## Reproduce the final model exactly (train + evaluate)
reproduce:
	python -m src.pipeline.pipeline --config config/config.yaml --data data/raw/train.csv

## Lint with flake8
lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

## Pin current package versions to requirements-lock.txt
freeze:
	pip freeze > requirements-lock.txt

help:
	@grep -E '^##' Makefile | sed 's/## //'
