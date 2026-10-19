.PHONY: all test lint lint-fix format clean install wheel unit_test ty compile_test build venv install_wheelhouse

# Worker count for the compile-only gate; override in CI to match the host.
MAXPROCS ?= $(shell nproc)
# Interpreter for `ty`; set when the checker must target an environment other
# than the one on PATH.
TY_PYTHON ?=

# Default target
all: lint unit_test

# The full gate: type check, unit tests, then the compile-only set. Runs the same
# way locally, in CI, and in a container.
build: ty unit_test compile_test

ty:
	ty check $(if $(TY_PYTHON),--python $(TY_PYTHON),) src/nkilib_src test

wheel:
	mkdir -p dist
	rm -rf dist/*
	pip --require-virtualenv --isolated wheel . --wheel-dir dist --no-deps

test:
	PYTHONPATH=$(CURDIR):$$PYTHONPATH pytest -c test/pytest.ini $(ARGS)

unit_test:
	PYTHONPATH=$(CURDIR):$$PYTHONPATH pytest -c test/pytest.ini $(ARGS) test/unit

compile_test:
	PYTHONPATH=$(CURDIR):$$PYTHONPATH pytest -c test/pytest.ini test/integration/nkilib \
	  -m fast --test-mode compile-only --dist worksteal -n auto --maxprocesses $(MAXPROCS) \
	  --timeout 600 --cpu-timeout 240 --monitor-durations --memory-limit 3000 --monitor-memory \
	  --platform-target trn2,trn3_a0 --force-local-cleanup --skip-model-tests $(ARGS)

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
