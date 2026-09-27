"""Runtime helpers for invoking a compiled graph.

Everything needed to run the graph correctly lives here: the thread-scoped
config, the recursion limit, and detection of a pending interrupt in the output
stream. The API, the CLI and the tests all go through this module so the
semantics of a run are identical everywhere.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from agentic_workflow.config import Settings, load_settings
from agentic_workflow.domain.schemas import ApprovalRequest
from agentic_workflow.errors import InvalidStateError
from agentic_workflow.human.gates import decode_interrupt
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

#: Config key identifying the checkpoint thread. One thread == one logical run.
THREAD_ID = "thread_id"

StreamMode = Literal["values", "updates", "debug", "tasks", "messages"]

#: Super-steps consumed before the first repair loop begins: triage, programmer,
#: reviewer, router, tester, reporter.
_BASE_SUPER_STEPS = 25

#: Super-steps one repair loop costs: programmer, reviewer, router, tester, router.
_SUPER_STEPS_PER_ITERATION = 10


def recursion_limit_for(settings: Settings | None = None) -> int:
    """Compute the Pregel recursion limit for *settings*.

    This is the graph's own backstop, and it has to clear the iteration budget
    the router enforces on business grounds. The two are not alternatives: the
    router decides a run needs a human, while this decides the process has a
    runaway graph on its hands. When only the second is available the run dies
    of infrastructure exhaustion instead of of an exhausted allowance, and the
    operator learns nothing about which budget was spent.

    The margin is generous because the cost per loop is an estimate, not a
    constant. One iteration re-enters the programmer, the reviewer, the router,
    the tester and the router again; an interrupt re-executes the gated node
    from the top, so the resume path is the more expensive one. Over-provisioning
    costs nothing — the limit only ever fires on a graph that is not converging.

    Args:
        settings: Application configuration. Defaults to the process settings.

    Returns:
        A safe upper bound on super-steps for a single ``ainvoke``.
    """
    resolved = settings or load_settings()
    return _BASE_SUPER_STEPS + _SUPER_STEPS_PER_ITERATION * resolved.max_iterations


def create_run_config(
    thread_id: str,
    *,
    run_id: str | None = None,
    settings: Settings | None = None,
    recursion_limit: int | None = None,
    extra: dict[str, Any] | None = None,
) -> RunnableConfig:
    """Build the ``RunnableConfig`` for one graph invocation.

    The thread id is the durable identity of a run: it keys the checkpoint
    history, so resuming, replaying and inspecting a run all work off the same
    value.

    Args:
        thread_id: Durable thread identifier, normally the run id.
        run_id: Optional logical run id recorded in metadata for tracing.
        settings: Configuration supplying the recursion limit.
        recursion_limit: Explicit override for the recursion limit.
        extra: Additional ``configurable`` entries.

    Returns:
        A ``RunnableConfig`` accepted by ``ainvoke``/``astream``.

    Raises:
        InvalidStateError: If ``thread_id`` is empty.

    Example:
        --------
        >>> cfg = create_run_config("run-1")
        >>> cfg["configurable"]["thread_id"]
        'run-1'
    """
    if not thread_id:
        raise InvalidStateError("thread_id is required to address a checkpoint")

    settings = settings or load_settings()
    configurable: dict[str, Any] = {THREAD_ID: thread_id}
    if run_id:
        configurable["awf_run_id"] = run_id
    if extra:
        configurable.update(extra)

    return RunnableConfig(
        {
            "configurable": configurable,
            "recursion_limit": recursion_limit
            if recursion_limit is not None
            else recursion_limit_for(settings),
            "max_concurrency": settings.max_parallel_runs,
            "tags": ["agentic-workflow", f"thread:{thread_id}"],
            "metadata": {"run_id": run_id or thread_id, "thread_id": thread_id},
        }
    )


def thread_config(thread_id: str, **metadata: Any) -> RunnableConfig:
    """Minimal config for read-only checkpoint operations (no recursion limit)."""
    return RunnableConfig({"configurable": {THREAD_ID: thread_id}, "metadata": dict(metadata)})


def extract_interrupts(chunk: Any) -> list[ApprovalRequest]:
    """Pull every pending approval out of a streamed graph chunk.

    LangGraph surfaces interrupts either as a top-level ``__interrupt__`` key in
    a ``values`` chunk or inside an ``updates`` payload. This helper normalises
    both shapes, because a caller should never have to care which one arrived.

    Args:
        chunk: One item yielded by ``graph.astream``.

    Returns:
        The decoded approval requests, in encounter order. Empty when the chunk
        carries no interrupt.

    Example:
        --------
        >>> extract_interrupts({"updates": {}})
        []
    """
    if not isinstance(chunk, dict):
        return []

    out: list[ApprovalRequest] = []
    candidates: list[Any] = []

    if "__interrupt__" in chunk:
        candidates.extend(_as_list(chunk["__interrupt__"]))
    updates = chunk.get("updates")
    if isinstance(updates, dict):
        for value in updates.values():
            if isinstance(value, dict) and "__interrupt__" in value:
                candidates.extend(_as_list(value["__interrupt__"]))
    if isinstance(updates, list):
        for item in updates:
            if isinstance(item, dict) and "__interrupt__" in item:
                candidates.extend(_as_list(item["__interrupt__"]))

    for candidate in candidates:
        try:
            request, _meta = decode_interrupt(_unwrap(candidate))
        except Exception as exc:
            log.warning("interrupt.undecodable", error=str(exc), type=type(candidate).__name__)
            continue
        out.append(request)
    return out


def _as_list(value: Any) -> list[Any]:
    """Normalise a value that may be a single item or a sequence."""
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _unwrap(candidate: Any) -> Any:
    """Unwrap LangGraph's ``Interrupt`` record to its raw value."""
    value = getattr(candidate, "value", None)
    if value is not None:
        return value
    if isinstance(candidate, dict) and "value" in candidate and "id" in candidate:
        return candidate["value"]
    return candidate


def is_interrupted(state: dict[str, Any] | None) -> bool:
    """``True`` when *state* shows a parked human gate.

    Args:
        state: A state mapping, possibly carrying LangGraph's ``__interrupt__``.

    Returns:
        Whether execution is currently waiting on a human.
    """
    if not state:
        return False
    if state.get("__interrupt__"):
        return True
    return bool(state.get("pending_approval"))


__all__ = [
    "THREAD_ID",
    "create_run_config",
    "extract_interrupts",
    "is_interrupted",
    "recursion_limit_for",
    "thread_config",
]
