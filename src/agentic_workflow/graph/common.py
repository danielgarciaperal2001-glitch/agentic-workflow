"""Cross-cutting helpers for graph nodes.

Every node needs the same four things: a logger bound to its identity, a
per-node timeout, a token/latency metric, and a transcript entry. Centralising
that in a decorator keeps the node bodies focused on *what the agent decides*
rather than *how to run safely*.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import time
from typing import Any, ParamSpec, TypeVar

from agentic_workflow.domain.schemas import NodeTiming, utcnow
from agentic_workflow.domain.state import WorkflowState
from agentic_workflow.errors import RunTimeoutError
from agentic_workflow.logging import bind_context, get_logger

P = ParamSpec("P")
R = TypeVar("R")

log = get_logger(__name__)


def transcript_entry(
    *,
    node: str,
    role: str,
    summary: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a transcript record.

    The transcript is trimmed by the state reducer and is used for observability
    and by the evaluation suite, so it stays structured rather than prose.
    """
    return {
        "node": node,
        "role": role,
        "summary": summary[:1_000],
        "payload": payload or {},
        "at": utcnow().isoformat(),
    }


def timing(node: str, started: float, *, ok: bool, error: str | None = None) -> NodeTiming:
    """Build a :class:`NodeTiming` from a monotonic start timestamp."""
    return NodeTiming(
        node=node,
        started_at=utcnow(),
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        ok=ok,
        error=error,
    )


def node(
    name: str,
    *,
    timeout_seconds: float | None = None,
) -> Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]]:
    """Decorate a graph node with logging, timing, timeout and state hygiene.

    The wrapper:

    1. binds ``run_id``/``node`` correlation context,
    2. enforces the node timeout (surfacing :class:`RunTimeoutError`),
    3. records a :class:`NodeTiming` appended to ``node_timings``,
    4. stamps ``updated_at`` on every returned update,
    5. converts unexpected exceptions into a domain error carrying the node name.

    Args:
        name: Node name used in logs, timings and error messages.
        timeout_seconds: Per-node budget. Falls back to the global default.

    Returns:
        A decorator producing the wrapped coroutine function.
    """

    def decorator(
        func: Callable[..., Awaitable[dict[str, Any]]],
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        async def wrapper(state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
            from agentic_workflow.domain.schemas import ReviewRequest
            from agentic_workflow.domain.state import as_model

            request = as_model(state, "request", ReviewRequest)
            run_id = getattr(request, "run_id", "unknown")
            node_log = get_logger(func.__module__).bind(node=name, run_id=run_id)
            started = time.perf_counter()
            node_log.info("node.start")

            with bind_context(run_id=run_id, node=name):
                try:
                    update = await _with_timeout(
                        func(state, runtime), name, run_id, timeout_seconds
                    )
                except BaseException as exc:
                    if _is_control_flow(exc):
                        # `GraphInterrupt` / `GraphBubbleUp` are LangGraph's
                        # control-flow signals, not failures. Wrapping them in a
                        # domain error would break resume and checkpointing.
                        node_log.info("node.interrupted")
                        raise
                    duration = (time.perf_counter() - started) * 1000.0
                    node_log.error("node.failed", error=str(exc), duration_ms=round(duration, 2))
                    raise _as_domain_error(exc, node=name, run_id=run_id) from exc

                duration = (time.perf_counter() - started) * 1000.0
                update = dict(update or {})
                update["node_timings"] = [timing(name, started, ok=True)]
                update["updated_at"] = utcnow().isoformat()
                node_log.info(
                    "node.done",
                    duration_ms=round(duration, 2),
                    keys=sorted(k for k in update if k != "node_timings"),
                )
                return update

        wrapper.__name__ = getattr(func, "__name__", name)
        wrapper.__doc__ = func.__doc__
        wrapper.__qualname__ = getattr(func, "__qualname__", name)
        wrapper.__annotations__ = dict(getattr(func, "__annotations__", {}))
        return wrapper

    return decorator


async def _with_timeout(
    awaitable: Awaitable[dict[str, Any]],
    name: str,
    run_id: str,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    """Await *awaitable* under a deadline.

    Uses :func:`asyncio.timeout` when available (3.11+) so the cancellation
    propagates cleanly into the awaiting task, letting the checkpointer roll the
    task back instead of leaving orphaned work behind.
    """
    import asyncio

    if not timeout_seconds or timeout_seconds <= 0:
        return await awaitable

    try:
        async with asyncio.timeout(timeout_seconds):
            return await awaitable
    except TimeoutError as exc:
        raise RunTimeoutError(
            f"node {name!r} exceeded its {timeout_seconds:.0f}s budget",
            run_id=run_id,
            node=name,
            timeout_seconds=timeout_seconds,
        ) from exc


def _is_control_flow(exc: BaseException) -> bool:
    """``True`` when *exc* is a LangGraph control-flow signal, not a failure.

    ``GraphInterrupt`` pauses the graph for human input and ``GraphBubbleUp``
    propagates a parent graph's interrupt. Both must reach LangGraph untouched
    or the run cannot be checkpointed and resumed.
    """
    try:
        from langgraph.errors import GraphBubbleUp, GraphInterrupt
    except ImportError:  # pragma: no cover - langgraph is a hard dependency
        return False
    return isinstance(exc, GraphInterrupt | GraphBubbleUp)


def _as_domain_error(exc: BaseException, *, node: str, run_id: str) -> Exception:
    """Attach node context to an error, leaving domain errors untouched."""
    from agentic_workflow.errors import AgentError, WorkflowError

    if isinstance(exc, WorkflowError):
        return exc.with_context(node=node)
    return AgentError(
        f"node {node!r} failed: {exc}",
        run_id=run_id,
        node=node,
        cause=type(exc).__name__,
    )


__all__ = ["node", "timing", "transcript_entry"]
