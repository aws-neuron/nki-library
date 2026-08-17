.PHONY: all test lint lint-fix format clean install wheel unit_test venv install_wheelhouse

# Default target
all: lint unit_test

wheel:
	mkdir -p dist
	rm -rf dist/*
	pip --require-virtualenv --isolated wheel . --wheel-dir dist --no-deps

test:
	PYTHONPATH=$(CURDIR):$$PYTHONPATH pytest -c test/pytest.ini -p test.utils.pytest_plugin $(ARGS)

unit_test:
	PYTHONPATH=$(CURDIR):$$PYTHONPATH pytest -c test/pytest.ini -p test.utils.pytest_plugin $(ARGS) test/unit

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

install_neuron:
	pip install --require-virtualenv --extra-index-url https://pip.repos.neuron.amazonaws.com .

install_wheelhouse:
	pip install --require-virtualenv wheelhouse/*.whl

# Include internal targets if available
-include Makefile.internal.mk
