# Minimal Makefile for HW1-HW4 requirements

.PHONY: test data-check eval-gate

test:
	python -m pytest tests/ -v

data-check:
	python tests/test_data_checks.py

eval-gate:
	python pipelines/eval_gate.py