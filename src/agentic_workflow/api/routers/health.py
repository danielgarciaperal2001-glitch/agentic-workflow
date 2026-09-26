"""Liveness and readiness probes.

The two are not the same question and answering them with the same handler is a
classic production outage: a probe that checks its dependencies will restart a
process that is perfectly able to serve traffic, and a probe that does not will
route traffic to a process that cannot answer.

* ``/health/live`` — "am I running?" Only checks that the process is up. It must
  never fail because a *dependency* is down, or Kubernetes will kill healthy pods
  during a database blip and turn a degradation into an outage.
* ``/health/ready`` — "can I serve?" Checks the checkpointer and the approval
  inbox. A database outage makes this 503, which removes the instance from the
  load balancer without killing it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request, Response, status

from agentic_workflow import __version__
from agentic_workflow.api.schemas import HealthResponse, HealthStatus
from agentic_workflow.domain.schemas import utcnow
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

#: Bounded probe budget. An orchestrator treats a hanging probe as a dead
#: instance, and a bounded 503 is recoverable where a timeout is not.
PROBE_TIMEOUT_SECONDS: float = 5.0


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
async def live(request: Request) -> HealthResponse:
    """Report that the process is alive.

    Intentionally dependency-free: this probe answers "should the supervisor
    restart me?", and the answer is no as long as the interpreter is running.

    Args:
        request: The inbound request, used to read application state.

    Returns:
        A :class:`~agentic_workflow.api.schemas.HealthResponse` with status
        ``ok`` and no dependency checks.
    """
    settings = request.app.state.settings
    return HealthResponse(
        status=HealthStatus.OK,
        version=__version__,
        environment=settings.environment.value,
        checks={"event_loop": "ok"},
        detail=None,
        timestamp=utcnow(),
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def ready(request: Request, response: Response) -> HealthResponse:
    """Report whether the instance can serve traffic.

    Checks the durable store by asking the engine for its own state: if the
    checkpointer cannot answer a trivial read, no run can be started, resumed or
    inspected, so the instance must not receive traffic.

    The engine check is bounded by a short timeout. A probe that hangs is
    indistinguishable from a dead instance to most orchestrators, and hanging is
    worse than reporting "not ready" — a bounded, honest 503 is recoverable.

    Args:
        request: The inbound request, used to read application state.
        response: The outbound response, whose status code is set to 503 when
            the instance is not ready.

    Returns:
        A :class:`~agentic_workflow.api.schemas.HealthResponse` describing every
        check.
    """

    settings = request.app.state.settings
    checks: dict[str, Any] = {}
    healthy = True

    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        checks["engine"] = {"ok": False, "detail": "engine not initialised"}
        healthy = False
    else:
        checks["engine"] = {
            "ok": True,
            "runs": len(engine.registry),
            "durable": settings.use_durable_checkpointer,
        }
        if settings.use_durable_checkpointer:
            try:
                await asyncio.wait_for(_check_store(engine), timeout=PROBE_TIMEOUT_SECONDS)
            except TimeoutError:
                checks["checkpointer"] = {"ok": False, "detail": "read timed out after 5s"}
                healthy = False
            except Exception as exc:
                checks["checkpointer"] = {
                    "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                healthy = False
            else:
                checks["checkpointer"] = {"ok": True}

    hub = getattr(request.app.state, "hub", None)
    if hub is not None:
        checks["events"] = hub.stats()

    approvals = getattr(request.app.state, "approvals", None)
    if approvals is not None:
        try:
            checks["approvals"] = await asyncio.wait_for(
                approvals.stats(), timeout=PROBE_TIMEOUT_SECONDS
            )
        except Exception as exc:
            checks["approvals"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
            healthy = False

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("api.not_ready", checks=checks)
    return HealthResponse(
        status=HealthStatus.OK if healthy else HealthStatus.DOWN,
        version=__version__,
        environment=settings.environment.value,
        checks=checks,
        detail=None if healthy else "one or more readiness checks failed",
        timestamp=utcnow(),
    )


async def _check_store(engine: Any) -> None:
    """Touch the durable store so a broken connection surfaces here.

    Args:
        engine: The workflow engine.

    Raises:
        Exception: Whatever the checkpointer raises on a trivial read.
    """
    saver = engine.checkpointer
    if saver is None:
        return
    setup = getattr(saver, "setup", None)
    if setup is not None and not getattr(saver, "_awf_ready", False):
        # `setup()` is idempotent, so this doubles as a "does the schema exist?"
        # check. The flag avoids re-running DDL on every probe.
        await setup()
        object.__setattr__(saver, "_awf_ready", True)


__all__ = ["router"]
