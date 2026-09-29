"""Application factory for the REST/WebSocket control plane.

Everything an operator or a CI job needs is here: submit a run, watch it, answer
its gates, read its history, replay a branch, audit who decided what.

The factory pattern is not a style choice. ``uvicorn --factory`` needs a callable
that builds a *fresh* app per worker, and the test suite needs two independent
apps in one process (one for the happy path, one that fails). A module-level
``app = FastAPI()`` gives neither.

Lifecycle
---------
::

    startup  → configure logging, build the hub, start the engine
    requests → routers resolve engine/approvals/hub from app.state
    shutdown → close the hub, stop the engine (cancelling in-flight runs)

A misconfigured durable store fails at *startup*, not on the first customer
request. A workflow engine that cannot reach its database is not degraded, it is
broken, and the orchestrator should never route traffic to it.

Example:
    --------
    >>> from agentic_workflow.api.app import create_app  # doctest: +SKIP
    >>> app = create_app()  # doctest: +SKIP
    >>> app.title  # doctest: +SKIP
    'agentic-workflow control plane'
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from agentic_workflow import __version__
from agentic_workflow.api.deps import RateLimiter, rate_limited, require_auth
from agentic_workflow.api.error_handlers import install_error_handlers
from agentic_workflow.api.events import EventHub
from agentic_workflow.api.routers import approvals, events, health, runs, threads
from agentic_workflow.config import Settings, load_settings
from agentic_workflow.domain.schemas import utcnow
from agentic_workflow.human.service import ApprovalService
from agentic_workflow.llm.base import Usage
from agentic_workflow.logging import bind_context, configure_logging, get_logger
from agentic_workflow.services.engine import WorkflowEngine

log = get_logger(__name__)

DESCRIPTION = """
Control plane for a **multi-agent AI workflow engine with Human-in-the-Loop**.

Submit a review request, watch the agents work, approve or reject what they
propose, and re-run any past step of any run. Every super-step is checkpointed,
so a parked run is durable, resumable and fully auditable.

* Runs that stop on a human gate answer **`202 Accepted`** — a parked run is a
  normal outcome, not a failure.
* Errors always carry a stable `error.code`; branch on that, never on the message.
* Subscribe to `ws://…/ws/runs/{run_id}` for push updates, or poll the REST
  resources.
"""

TAGS_METADATA: list[dict[str, Any]] = [
    {"name": "health", "description": "Liveness and readiness probes."},
    {"name": "runs", "description": "Start, inspect, resume and cancel runs."},
    {
        "name": "approvals",
        "description": "The human-in-the-loop inbox: list, answer, replay and audit gates.",
    },
    {
        "name": "threads",
        "description": "Checkpoint history, point-in-time state and time-travel replay.",
    },
    {"name": "events", "description": "WebSocket streams of run lifecycle events."},
]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, measure latency and bind both to the log context.

    A request id that is generated in the handler is useless: by then the access
    log and the first error have already been written. Binding it in middleware
    means every log line emitted while serving the request — including the ones
    from deep inside a graph node — carries the same correlation id.

    Attributes:
        app: The wrapped ASGI application.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap *app*.

        Args:
            app: The downstream application.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Attach a request id, call the app and stamp the response.

        Args:
            request: The inbound request.
            call_next: The downstream handler.

        Returns:
            The response, with ``X-Request-ID`` set.
        """
        incoming = request.headers.get("x-request-id")
        request_id = incoming or f"req_{uuid4().hex[:16]}"
        request.state.request_id = request_id
        started = perf_counter()
        with bind_context(request_id=request_id, path=request.url.path, method=request.method):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        elapsed_ms = (perf_counter() - started) * 1_000.0
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
        if elapsed_ms > 1_000.0:
            # Log only the slow requests: an access log for everything is noise
            # that trains people to ignore logs.
            log.warning("api.slow_request", ms=round(elapsed_ms, 2), status=response.status_code)
        return response


def create_app(
    settings: Settings | None = None,
    *,
    engine: WorkflowEngine | None = None,
    configure_logs: bool = True,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Configuration. Defaults to the cached process settings.
        engine: A pre-built engine. Tests inject one with a memory checkpointer;
            production leaves it ``None`` so the engine builds its own.
        configure_logs: Whether to install the structured logger. Tests pass
            ``False`` to keep pytest output readable.

    Returns:
        The configured :class:`~fastapi.FastAPI` application.

    Raises:
        ConfigurationError: Propagated from the settings validator, so an invalid
            environment fails at import rather than mid-flight.
    """
    resolved = settings or load_settings()
    if configure_logs:
        configure_logging(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start and stop every long-lived resource exactly once."""
        configure_logging(resolved)
        hub = EventHub(resolved)
        app.state.settings = resolved
        app.state.hub = hub
        app.state.rate_limiter = RateLimiter(resolved.api_rate_limit_per_minute)

        owned = engine is None
        active = engine or WorkflowEngine(resolved)
        app.state.engine = active
        app.state.approvals = ApprovalService(active)
        # Wired unconditionally, not only for an engine this app built. A hub is
        # per-app state; an injected engine carries whatever sink it was built
        # with, and an engine with no sink produces an app that accepts the
        # WebSocket handshake and then emits nothing at all. The failure is
        # silent, which is exactly what makes it worth guarding here.
        active.set_event_sink(hub.publish)

        try:
            await active.startup()
        except Exception:
            # Fail the boot rather than serve a half-initialised app: an
            # orchestrator can retry a failed startup, but it cannot detect a
            # process that answers /health/live while every run 500s.
            log.error("api.startup_failed", **resolved.safe_summary())
            raise
        log.info("api.ready", **resolved.safe_summary())
        # Said here, and not in the engine, because this is a statement about a
        # *deployment*: a control plane whose human decisions are unauthenticated
        # and unsigned has an audit trail nobody can rely on, and the operator
        # reading this log is the one who can fix it. The engine is a library and
        # is also driven by `awf demo`, where the human at the keyboard is the
        # trust boundary and the same warning would be noise.
        signing_warning = resolved.signing_warning()
        if signing_warning is not None:
            log.warning("hitl.signatures_unavailable", detail=signing_warning)
        try:
            yield
        finally:
            await hub.close()
            if owned:
                await active.shutdown()
            log.info("api.stopped")

    docs_enabled = resolved.api_docs_enabled and not resolved.is_production
    app = FastAPI(
        title="agentic-workflow control plane",
        description=DESCRIPTION,
        version=__version__,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        root_path=resolved.api_root_path,
    )

    # Middleware order is significant: the request-id middleware is added last so
    # it runs *first*, which is what lets CORS and rate-limit rejections carry a
    # correlation id too.
    app.add_middleware(GZipMiddleware, minimum_size=1_024)
    if resolved.api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.api_cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Response-Time-ms", "Retry-After"],
        )
    app.add_middleware(RequestContextMiddleware)

    install_error_handlers(app)

    # `dependencies=` is applied at the router level rather than per handler so
    # neither the throttle nor authentication can be forgotten on a new endpoint.
    # Auth was per handler for its whole life, which left the twenty-one
    # endpoints that carried it as the only thing between a new endpoint and an
    # unauthenticated control plane; the two properties belong in the same
    # place, and `tests/api/test_auth_surface.py` holds both to account.
    #
    # WebSockets are exempt from both: they authenticate in the handshake and
    # have no HTTP request to throttle. The health probes, the root listing and
    # `/metrics` are exempt from authentication only, and `docs/security.md`
    # gives the reason for each.
    secured = [Depends(rate_limited), Depends(require_auth)]
    app.include_router(health.router)
    app.include_router(runs.router, dependencies=secured)
    app.include_router(approvals.router, dependencies=secured)
    app.include_router(threads.router, dependencies=secured)
    app.include_router(events.router)
    app.include_router(_meta_router())

    log.info("api.created", version=__version__, environment=resolved.environment.value)
    return app


def _meta_router() -> APIRouter:
    """Return the root and metrics router.

    Kept as a function because it closes over nothing but constants, and building
    it lazily keeps :func:`create_app` readable as a single top-to-bottom script.
    """
    router = APIRouter(tags=["meta"])

    @router.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        """Point a first-time visitor at the useful endpoints."""
        return {
            "service": "agentic-workflow",
            "version": __version__,
            "docs": "/docs",
            "endpoints": {
                "start_run": "POST /v1/runs",
                "list_runs": "GET /v1/runs",
                "approvals": "GET /v1/approvals",
                "resolve": "POST /v1/approvals/{approval_id}/resolve",
                "history": "GET /v1/threads/{run_id}/history",
                "replay": "POST /v1/threads/{run_id}/replay",
                "events": "WS /ws/runs/{run_id}",
                "live": "GET /health/live",
                "ready": "GET /health/ready",
                "metrics": "GET /metrics",
            },
        }

    @router.get(
        "/metrics",
        include_in_schema=False,
        summary="Operational counters (Prometheus text format when available)",
    )
    async def metrics(request: Request) -> Response:
        """Return lightweight operational counters.

        Emits Prometheus text if ``prometheus_client`` is installed and the
        OpenMetrics exposition of the engine's own gauges is available;
        otherwise returns a JSON object with the same numbers. The endpoint is
        always present, because a metrics route that 404s takes down every
        dashboard the moment an optional extra is missing.
        """
        engine = getattr(request.app.state, "engine", None)
        hub = getattr(request.app.state, "hub", None)
        payload: dict[str, Any] = {
            "ts": utcnow().isoformat(),
            "version": __version__,
            "runs": len(engine.registry) if engine is not None else 0,
            "events": hub.stats() if hub is not None else {},
            # Process-wide LLM spend since boot: the engine's one client serves
            # every run, so this is the total, not a per-run slice.
            "usage": (engine.llm_usage.as_dict() if engine is not None else Usage().as_dict()),
        }
        return _render_metrics(payload)

    return router


# Usage keys on the payload map to their Prometheus counter names. Cached and
# completion tokens share a family with plain tokens: the process-wide totals
# since boot, exported with the ``_total`` suffix monotonic counters require.
_USAGE_COUNTERS: tuple[tuple[str, str], ...] = (
    ("calls", "awf_llm_calls_total"),
    ("prompt_tokens", "awf_llm_prompt_tokens_total"),
    ("completion_tokens", "awf_llm_completion_tokens_total"),
    ("cached_tokens", "awf_llm_cached_tokens_total"),
)


def _render_metrics(payload: dict[str, Any]) -> Response:
    """Render counters as Prometheus text, falling back to JSON.

    Event and run counters are gauges under a single ``awf_metric`` family;
    the process-wide LLM usage is exported as monotonic counters, which is the
    semantics a budget dashboard expects of token spend.

    Args:
        payload: The counters collected from the engine and the hub.

    Returns:
        A :class:`~fastapi.responses.Response` in the exposition format the
        metrics pipeline expects, or JSON when the counters cannot be expressed
        as flat gauges.
    """
    flat: dict[str, float] = {}
    counters: dict[str, Any] = payload.get("events") or {}
    for key in ("published", "delivered", "dropped", "subscribers", "runs_observed"):
        if key in counters:
            flat[f"awf_events_{key}"] = float(counters[key])
    flat["awf_runs_registered"] = float(payload.get("runs", 0))
    if not flat:
        return Response(
            content=json.dumps(payload, default=str),
            media_type="application/json",
        )
    usage: dict[str, Any] = payload.get("usage") or {}
    lines = [
        "# HELP awf_metric agentic-workflow operational counter.",
        "# TYPE awf_metric gauge",
        *(f'awf_metric{{name="{name}"}} {value:g}' for name, value in sorted(flat.items())),
    ]
    for key, name in _USAGE_COUNTERS:
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {int(usage.get(key, 0)):g}")
    lines.append("")
    return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8")


__all__ = ["DESCRIPTION", "RequestContextMiddleware", "create_app"]
