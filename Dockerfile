# syntax=docker/dockerfile:1.7
# =============================================================================
# agentic-workflow — multi-stage production image
# =============================================================================
ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------------- #
# Builder: compile wheels so the runtime layer never needs a toolchain
# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libpq-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install --no-cache-dir ".[postgres,api,obs]"

# --------------------------------------------------------------------------- #
# Runtime: minimal, non-root, no build tooling
# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="agentic-workflow" \
      org.opencontainers.image.description="Multi-Agent AI Workflow Engine with Human-in-the-Loop" \
      org.opencontainers.image.source="https://github.com/danielgarciaperal2001-glitch/agentic-workflow" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    AWF_ENVIRONMENT=production

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 agentic \
    && useradd --system --uid 1001 --gid agentic --create-home --home-dir /home/agentic agentic

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=agentic:agentic src ./src
COPY --chown=agentic:agentic docs ./docs
COPY --chown=agentic:agentic evals ./evals

USER agentic
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["agentic_workflow.api.app:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers"]
