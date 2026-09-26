# =============================================================================
# agentic-workflow — developer task runner
# `make help` lists every target.
# =============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

PY        := .venv/bin/python
PIP       := .venv/bin/pip
UVICORN   := .venv/bin/uvicorn
PYTEST    := .venv/bin/pytest
RUFF      := .venv/bin/ruff
MYPY      := .venv/bin/mypy
COMPOSE   := docker compose
RUN_ID    ?= local-$(shell date +%s)

export PYTHONPATH := src

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------- #
# Environment
# ---------------------------------------------------------------------------- #
.PHONY: venv
venv: ## Create the local virtualenv
	python3 -m venv .venv
	$(PIP) install --upgrade pip wheel
	$(PIP) install -e ".[dev,api,postgres]"

.PHONY: install
install: ## Install the project with all extras (editable)
	$(PIP) install -e ".[all]"

.PHONY: lock
lock: ## Refresh the dependency lock file
	$(PIP) install pip-tools
	$(PY) -m piptools compile pyproject.toml --extra all -o requirements.lock

.PHONY: sync
sync: ## Sync the venv against requirements.lock
	$(PIP) install -r requirements.lock

# ---------------------------------------------------------------------------- #
# Quality gates
# ---------------------------------------------------------------------------- #
.PHONY: fmt
fmt: ## Auto-format and auto-fix
	$(RUFF) format .
	$(RUFF) check --fix .

.PHONY: fmt-check
fmt-check: ## Verify formatting without writing
	$(RUFF) format --check .
	$(RUFF) check .

.PHONY: lint
lint: ## Lint
	$(RUFF) check .

.PHONY: typecheck
typecheck: ## Static type check (strict)
	$(MYPY) src evals

.PHONY: check
check: fmt-check lint typecheck ## Run every static gate

.PHONY: test
test: ## Run the unit + integration test suite
	$(PYTEST) -m "not slow and not postgres" -q

.PHONY: test-all
test-all: ## Run everything, including the postgres-marked tests
	$(PYTEST) -q

.PHONY: test-cov
test-cov: ## Run tests with a coverage report
	$(PYTEST) --cov --cov-report=term-missing --cov-report=html -q

.PHONY: eval
eval: ## Score the golden dataset and print the report
	$(PY) -m agentic_workflow.cli eval

.PHONY: eval-json
eval-json: ## Score the golden dataset, emit JSON for a CI job to consume
	$(PY) -m agentic_workflow.cli eval --json

.PHONY: bench
bench: ## Measure the engine and print the tables quoted in the README
	$(PY) benchmarks/bench.py

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #
.PHONY: api
api: ## Start the REST/WebSocket control plane
	$(UVICORN) agentic_workflow.api.app:create_app --factory --reload --port $${AWF_API_PORT:-8000}

.PHONY: worker
worker: ## Start the API with the embedded in-process run executor
	AWF_API_EMBEDDED_WORKER=true $(UVICORN) agentic_workflow.api.app:create_app --factory --port $${AWF_API_PORT:-8000}

.PHONY: demo
demo: ## Execute one pipeline run end-to-end and print the trace
	$(PY) -m agentic_workflow.cli demo --run-id $(RUN_ID)

.PHONY: demo-interactive
demo-interactive: ## Same, but read every human gate verdict from the prompt
	$(PY) -m agentic_workflow.cli demo --run-id $(RUN_ID) --interactive

.PHONY: replay
replay: ## Time travel. Usage: make replay RUN=<run_id> INDEX=-1
	$(PY) -m agentic_workflow.cli replay $(RUN) $(if $(INDEX),--index $(INDEX),)

.PHONY: history
history: ## List a run's checkpoints. Usage: make history RUN=<run_id>
	$(PY) -m agentic_workflow.cli replay $(RUN) --limit 50

.PHONY: topology
topology: ## Print the graph's nodes, edges, loops and routing table
	$(PY) -m agentic_workflow.cli topology

.PHONY: janitor
janitor: ## Run one checkpoint-retention pass against the in-memory store
	$(PY) -m agentic_workflow.cli janitor --memory --dry-run

# ---------------------------------------------------------------------------- #
# Infrastructure
# ---------------------------------------------------------------------------- #
.PHONY: up
up: ## Start PostgreSQL + the API via docker compose
	$(COMPOSE) up -d --build
	@echo "API  -> http://localhost:8000/health"
	@echo "Docs -> http://localhost:8000/docs"

.PHONY: down
down: ## Stop the compose stack
	$(COMPOSE) down -v --remove-orphans

.PHONY: logs
logs: ## Tail compose logs
	$(COMPOSE) logs -f --tail=100

.PHONY: db-migrate
db-migrate: ## Apply Alembic migrations
	$(PY) -m alembic upgrade head

.PHONY: db-revision
db-revision: ## Autogenerate a migration: make db-revision MSG="add runs table"
	$(PY) -m alembic revision --autogenerate -m "$(or $(MSG),auto)"

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
