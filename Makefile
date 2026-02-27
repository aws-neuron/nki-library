.PHONY: all test lint lint-fix format clean install wheel unit_test venv install_wheelhouse

# Default target
all: lint unit_test

wheel:
	mkdir -p dist
	rm -rf dist/*
	pip --require-virtualenv --isolated wheel . --wheel-dir dist --no-deps

test:
	pytest test $(ARGS)

unit_test:
	pytest test/unit

lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	ruff format .

clean:
	rm -rf dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	pip freeze --require-virtualenv --exclude-editable | cut -d "@" -f1 | xargs pip --require-virtualenv uninstall -y

venv:
	python3 -m venv .venv

install:
	pip install --require-virtualenv --editable .[test]

install_wheelhouse:
	pip install --require-virtualenv wheelhouse/*.whl

# Include internal targets if available 
-include Makefile.internal.mk
