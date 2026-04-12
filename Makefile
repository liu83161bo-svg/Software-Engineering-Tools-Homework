# Makefile for AI System Course Project

# Data generation
generate-data:
	python data/generate_data.py

create-splits:
	python data/create_splits.py

# Data validation
data-check:
	python tests/test_data_checks.py

# DVC commands
dvc-pull:
	dvc pull

dvc-push:
	dvc push

dvc-repro:
	dvc repro

# Full pipeline
data-pipeline: generate-data create-splits data-check

# Test commands
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src/ --cov-report=html

# Clean up
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -name "*.pyc" -delete

clean-data:
	rm -rf data/raw/*.csv data/processed/*.jsonl data/splits/*.txt

.PHONY: generate-data create-splits data-check test test-coverage clean clean-data