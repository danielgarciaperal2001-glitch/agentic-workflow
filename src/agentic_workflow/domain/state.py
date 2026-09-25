"""The LangGraph state contract.

:class:`WorkflowState` is the single object threaded through every node. Two
design choices matter a lot in practice:

1. **Reducers, not overwrites.** Fields declared with ``Annotated[..., add]``
   are *merged* when several branches (e.g. parallel agents, or a node re-run
   after a time-travel resume) write to them. Without a reducer, LangGraph
   raises ``InvalidUpdateError`` on concurrent writes, and a replayed node
   would silently lose the previous iteration's data.
2. **Explicit reset markers.** Loops need to *clear* accumulated data. Nodes do
   that by returning the sentinel from :func:`reset` (see
   :func:`workflow_state_reducer`), which removes the key entirely — making the
   next write indistinguishable from the first.

The state is a ``TypedDict`` (not a dataclass) because LangGraph introspects the
type annotations to build the channel map and the input/output JSON schemas.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Final, TypedDict

from agentic_workflow.domain.schemas import (
    ApprovalRequest,
    Finding,
    FinalReport,
    NodeTiming,
    Patch,
    ReviewRequest,
    ReviewResult,
    TaskBrief,
    TestReport,
)

#: Sentinel returned by a node to clear a reduced channel. ``None`` cannot be
#: used because it is a legal list element; the marker object is unambiguous.
RESET: Final = "__reset__"

#: Upper bound on retained transcript entries. Keeps checkpoint payloads small
#: and predictable no matter how chatty a run becomes.
MAX_TRANSCRIPT_ENTRIES: Final = 200


def append_unique(existing: list[Any] | None, new: Any) -> list[Any]:
    """Reducer that appends while de-duplicating by ``id``.

    Used for findings and transcript entries: a node re-executed after a
    time-travel resume would otherwise duplicate every record.

    Args:
        existing: Current channel value, possibly ``None``.
        new: A single item, an iterable of items, or :data:`RESET`.

    Returns:
        A new list. The input lists are never mutated.
    """
    if new is None:
        return list(existing or [])
    if new == RESET:
        return []

    items = new if isinstance(new, list | tuple) else [new]
    merged = list(existing or [])
    seen = {getattr(item, "id", None) or _fingerprint(item) for item in merged}
    for item in items:
        key = getattr(item, "id", None) or _fingerprint(item)
        if key in seen:
            # Replace in place so updated records win over stale duplicates.
            for idx, existing_item in enumerate(merged):
                existing_key = getattr(existing_item, "id", None) or _fingerprint(existing_item)
                if existing_key == key:
                    merged[idx] = item
                    break
            continue
        seen.add(key)
        merged.append(item)
    return merged


def append_capped(existing: list[Any] | None, new: Any) -> list[Any]:
    """Reducer that appends and keeps only the newest :data:`MAX_TRANSCRIPT_ENTRIES`.

    Transcript growth is the main driver of checkpoint size. Capping it at the
    tail is safe because old messages are summaries, not evidence.
    """
    if new is None:
        return list(existing or [])
    if new == RESET:
        return []
    items = new if isinstance(new, list | tuple) else [new]
    merged = [*(existing or []), *items]
    return merged[-MAX_TRANSCRIPT_ENTRIES:]


def replace_or_reset(_existing: Any, new: Any) -> Any:
    """Reducer for "last writer wins" channels that can also be cleared.

    Returns :data:`RESET` for :data:`RESET` input; the graph layer translates
    that into a key deletion. Anything else overwrites the previous value.
    """
    if new == RESET:
        return RESET
    return new


def latest(_existing: Any, new: Any) -> Any:
    """Reducer keeping the newest non-``None`` value (ignore stale writes)."""
    return new if new is not None else _existing


def _fingerprint(item: Any) -> str:
    """Cheap identity for de-duplication when an item exposes no ``id``."""
    if isinstance(item, dict):
        return str(item.get("id") or item.get("role", "") + item.get("content", "")[:64])
    return f"{type(item).__name__}:{item!r:.128}"


def workflow_state_reducer(_existing: Any, new: Any) -> Any:
    """Top-level reducer understanding the :data:`RESET` sentinel.

    Applied to the whole state object, it lets any node clear a channel by
    returning ``{"findings": RESET}``.
    """
    if new == RESET:
        return {}
    return new


class WorkflowState(TypedDict, total=False):
    """Shared state for the multi-agent review workflow.

    The keys are grouped by the agent that owns them. ``Annotated`` reducers
    define the merge semantics; the plain keys are last-writer-wins.

    Attributes:
        request: Immutable input, injected at graph start.
        task_brief: Output of the ``triage`` node.
        patch: Output of the ``programmer`` node.
        review: Output of the ``reviewer`` node for the current iteration.
        test_report: Output of the ``tester`` node for the current iteration.
        findings: Every finding raised so far, de-duplicated by id.
        report: Output of the ``reporter`` node (terminal).
        iteration: Number of completed feedback loops.
        status: Coarse lifecycle status mirrored in the run registry.
        next_action: Router decision consumed by the ``router`` node.
        pending_approval: Approval request currently blocking the graph.
        human_decisions: Chronological log of resolved approvals.
        transcript: Trimmed conversation log, used for observability and evals.
        error: Terminal error message, if any.
        node_timings: Per-node wall-clock accounting.
        started_at / updated_at: ISO-8601 lifecycle markers.
    """

    # -- immutable input ------------------------------------------------- #
    request: ReviewRequest

    # -- agent outputs (last writer wins) -------------------------------- #
    task_brief: Annotated[TaskBrief | None, replace_or_reset]
    patch: Annotated[Patch | None, replace_or_reset]
    review: Annotated[ReviewResult | None, replace_or_reset]
    test_report: Annotated[TestReport | None, replace_or_reset]
    report: Annotated[FinalReport | None, replace_or_reset]

    # -- accumulated channels (reduced) ---------------------------------- #
    findings: Annotated[list[Finding], append_unique]
    transcript: Annotated[list[dict[str, Any]], append_capped]
    node_timings: Annotated[list[NodeTiming], operator.add]

    # -- control plane --------------------------------------------------- #
    iteration: int
    status: str
    next_action: str
    pending_approval: Annotated[ApprovalRequest | None, replace_or_reset]
    human_decisions: Annotated[list[dict[str, Any]], append_capped]
    error: Annotated[str | None, replace_or_reset]

    # -- bookkeeping ----------------------------------------------------- #
    started_at: str
    updated_at: str
    #: Free-form key/value bag for downstream consumers (e.g. eval harness).
    context: Annotated[dict[str, Any], latest]


#: Keys that must be present in the graph's *output* schema.
OUTPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "report",
        "review",
        "test_report",
        "findings",
        "iteration",
        "status",
        "pending_approval",
        "human_decisions",
        "error",
    }
)


def initial_state(request: ReviewRequest) -> WorkflowState:
    """Build the seed state for a new run.

    Args:
        request: The business request to process.

    Returns:
        A fully initialised state ready to be passed as the graph input.
    """
    from agentic_workflow.domain.schemas import utcnow

    now = utcnow().isoformat()
    return WorkflowState(
        request=request,
        task_brief=None,
        patch=None,
        review=None,
        test_report=None,
        report=None,
        findings=[],
        transcript=[],
        node_timings=[],
        iteration=0,
        status="pending",
        next_action="start",
        pending_approval=None,
        human_decisions=[],
        error=None,
        started_at=now,
        updated_at=now,
        context={},
    )


__all__ = [
    "MAX_TRANSCRIPT_ENTRIES",
    "OUTPUT_KEYS",
    "RESET",
    "WorkflowState",
    "append_capped",
    "append_unique",
    "initial_state",
    "latest",
    "replace_or_reset",
    "workflow_state_reducer",
]
