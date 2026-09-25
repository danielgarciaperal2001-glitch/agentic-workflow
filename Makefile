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
	$(MYPY) src

.PHONY: check
check: fmt-check lint typecheck ## Run every static gate

.PHONY: test
test: ## Run the unit + integration test suite
	$(PYTEST) -m "not slow and not eval" -q

.PHONY: test-all
test-all: ## Run everything, including evals
	$(PYTEST) -q

.PHONY: test-cov
test-cov: ## Run tests with a coverage report
	$(PYTEST) -m "not slow and not eval" --cov --cov-report=term-missing --cov-report=html -q

.PHONY: eval
eval: ## Run the automated LLM-quality evaluation suite
	$(PYTEST) -m eval -q --no-header -p no:randomly

.PHONY: eval-report
eval-report: ## Produce a persisted evaluation report
	$(PY) -m agentic_workflow.evals.report

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

.PHONY: replay
replay: ## Replay a persisted run from a given checkpoint. Usage: make replay THREAD_ID=... STEP=-1
	$(PY) -m agentic_workflow.cli replay --thread-id $(THREAD_ID) --step $(or $(STEP),-1)

.PHONY: graph
graph: ## Render the LangGraph state machine to a PNG
	$(PY) -m agentic_workflow.cli draw --out docs/diagrams/generated/graph.png

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
