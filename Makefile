.PHONY: all test lint lint-fix format clean install install_neuron wheel unit_test ty compile_test build venv install_wheelhouse

# Worker count for the compile-only gate; override in CI to match the host.
MAXPROCS ?= $(shell nproc)
UV ?= uv
UV_ENVIRONMENT = $(or $(UV_PROJECT_ENVIRONMENT),.venv)
export UV

CHECK_ENV = test -x "$(UV_ENVIRONMENT)/bin/python" || { echo 'Environment missing: run make install or make internal_setup first.' >&2; exit 1; };
# Pin the prepared interpreter so uv cannot select/download a different Python.
UV_RUN = $(CHECK_ENV) $(UV_NATIVE_ENV) "$(UV)" run --no-sync --python "$(UV_ENVIRONMENT)/bin/python"
TY_PYTHON ?= $(UV_ENVIRONMENT)/bin/python

# Default target
all: lint unit_test

# The full gate: type check, unit tests, then the compile-only set. Runs the same
# way locally, in CI, and in a container.
build: ty unit_test compile_test

ty:
	$(UV_RUN) python -m ty check --python "$(TY_PYTHON)" src/nkilib_src test

wheel:
	$(CHECK_ENV)
	mkdir -p dist
	rm -rf dist/*
	"$(UV)" build --wheel --python "$(UV_ENVIRONMENT)/bin/python" --out-dir dist

test:
	$(UV_RUN) python -m pytest -c test/pytest.ini $(ARGS)

unit_test:
	$(UV_RUN) python -m pytest -c test/pytest.ini $(ARGS) test/unit

compile_test:
	$(UV_RUN) python -m pytest -c test/pytest.ini test/integration/nkilib \
	  -m fast --test-mode compile-only --dist worksteal -n auto --maxprocesses $(MAXPROCS) \
	  --timeout 600 --cpu-timeout 240 --monitor-durations 20 --memory-limit 3000 --monitor-memory 20 \
	  --platform-target trn2,trn3_a0 --force-local-cleanup --skip-model-tests $(ARGS)

lint:
	$(UV_RUN) python -m ruff check .

lint-fix:
	$(UV_RUN) python -m ruff check . --fix

format:
	$(UV_RUN) python -m ruff format .

clean:
	$(CHECK_ENV)
	$(INSTALL_PREP)
	rm -rf dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@req=$$(mktemp) || exit 1; \
	"$(UV)" pip freeze --python "$(UV_ENVIRONMENT)/bin/python" --exclude-editable > "$$req" && \
	{ [ ! -s "$$req" ] || "$(UV)" pip uninstall --python "$(UV_ENVIRONMENT)/bin/python" -r "$$req"; }; \
	status=$$?; rm -f "$$req"; exit $$status

venv:
	"$(UV)" venv "$(UV_ENVIRONMENT)"

install:
	$(INSTALL_PREP)
	"$(UV)" sync --frozen --extra test

# Index-based installation, unlike install's locked uv sync. No implicit upgrade.
install_neuron:
	$(CHECK_ENV)
	$(INSTALL_PREP)
	"$(UV)" pip install --python "$(UV_ENVIRONMENT)/bin/python" --extra-index-url https://pip.repos.neuron.amazonaws.com/ .

install_wheelhouse:
	$(CHECK_ENV)
	$(INSTALL_PREP)
	"$(UV)" pip install --python "$(UV_ENVIRONMENT)/bin/python" wheelhouse/*.whl

# Include internal targets if available
-include Makefile.internal.mk
