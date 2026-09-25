"""The LangGraph state contract.

:class:`WorkflowState` is the single object threaded through every node. Three
design choices matter in practice:

1. **Reducers, not overwrites.** Fields declared with a reducer are *merged*
   when several branches (parallel agents, or a node re-executed after a
   time-travel resume) write to them. Without a reducer, LangGraph raises
   ``InvalidUpdateError`` on concurrent writes and a replayed node silently
   loses the previous iteration's data.
2. **De-duplicating reducers.** A node that runs twice must not double-count, so
   the list reducers key on ``id`` and replace in place.
3. **``None`` clears an optional channel.** Loops must be able to invalidate a
   previous verdict (a new patch makes the old test report meaningless). Writing
   ``None`` is the idiomatic LangGraph way to do that; every read goes through
   :func:`as_model`, which maps ``None`` to "absent".

The state is a ``TypedDict`` (not a dataclass) because LangGraph introspects the
type annotations to build the channel map and the input/output JSON schemas.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Final, TypedDict, TypeVar, cast

from pydantic import BaseModel

from agentic_workflow.domain.schemas import (
    ApprovalRequest,
    FinalReport,
    Finding,
    NodeTiming,
    Patch,
    ReviewRequest,
    ReviewResult,
    TaskBrief,
    TestReport,
)

#: Upper bound on retained transcript entries. Keeps checkpoint payloads small
#: and predictable no matter how chatty a run becomes.
MAX_TRANSCRIPT_ENTRIES: Final = 200


# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #
def append_unique(existing: list[Any] | None, new: Any) -> list[Any]:
    """Reducer that appends while de-duplicating by ``id``.

    Used for findings: a node re-executed after a time-travel resume would
    otherwise duplicate every record, inflating the counts that gate the loop.

    Args:
        existing: Current channel value, possibly ``None``.
        new: A single item, an iterable of items, or ``None`` to clear.

    Returns:
        A new list. The input lists are never mutated.
    """
    if new is None:
        return []
    items = new if isinstance(new, list | tuple) else [new]

    merged = list(existing or [])
    index = {_identity(item): pos for pos, item in enumerate(merged)}
    for item in items:
        key = _identity(item)
        position = index.get(key)
        if position is None:
            index[key] = len(merged)
            merged.append(item)
        else:
            # Replace in place so an updated record supersedes a stale one.
            merged[position] = item
    return merged


def append_capped(existing: list[Any] | None, new: Any) -> list[Any]:
    """Reducer that appends and keeps only the newest entries.

    Transcript and audit growth are the main drivers of checkpoint size. Capping
    at the tail is safe: old entries are narration, not evidence.

    Args:
        existing: Current channel value, possibly ``None``.
        new: A single item, an iterable of items, or ``None`` to clear.

    Returns:
        A new, length-capped list.
    """
    if new is None:
        return []
    items = new if isinstance(new, list | tuple) else [new]
    merged = [*(existing or []), *items]
    return merged[-MAX_TRANSCRIPT_ENTRIES:]


def overwrite(_existing: Any, new: Any) -> Any:
    """Reducer for "last writer wins" channels, where ``None`` means *cleared*.

    Used for the optional agent outputs (``patch``, ``review``, …) so a node can
    invalidate a stale value simply by writing ``None``.
    """
    return new


def latest(_existing: Any, new: Any) -> Any:
    """Reducer keeping the newest non-``None`` value, ignoring stale writes."""
    return new if new is not None else _existing


def _identity(item: Any) -> str:
    """Cheap, stable identity for de-duplication when an item has no ``id``."""
    ident = getattr(item, "id", None)
    if ident:
        return str(ident)
    if isinstance(item, dict):
        return str(item.get("id") or f"{item.get('node')}:{item.get('summary')}"[:96])
    return f"{type(item).__name__}:{item!r:.96]}"


def add_timings(existing: list[NodeTiming] | None, new: Any) -> list[NodeTiming]:
    """Reducer accumulating node timings, tolerating a ``None`` clear.

    Args:
        existing: Current timings.
        new: One timing, a list of timings, or ``None`` to clear.

    Returns:
        The concatenated list.
    """
    if new is None:
        return []
    items = new if isinstance(new, list | tuple) else [new]
    return [*(existing or []), *items]


# --------------------------------------------------------------------------- #
# The state
# --------------------------------------------------------------------------- #
class WorkflowState(TypedDict, total=False):
    """Shared state for the multi-agent review workflow.

    Channels are grouped by ownership. ``Annotated`` reducers define the merge
    semantics; the exclusive channels are guarded at runtime by
    :meth:`~agentic_workflow.graph.nodes.base.AgentNode.assert_owns`.

    Attributes:
        request: Immutable input, injected at graph start.
        task_brief: Output of the ``triage`` node.
        patch: Output of the ``programmer`` node.
        review: Output of the ``reviewer`` node for the current iteration.
        test_report: Output of the ``tester`` node for the current iteration.
        report: Output of the ``reporter`` node (terminal).
        findings: Every finding raised so far, de-duplicated by id.
        transcript: Trimmed activity log used for observability and evals.
        node_timings: Per-node wall-clock accounting.
        iteration: Number of completed feedback loops.
        status: Coarse lifecycle status mirrored in the run registry.
        next_action: Router decision consumed by the conditional edge.
        pending_approval: Approval request currently blocking the graph.
        human_decisions: Chronological log of resolved approvals.
        error: Terminal error message, if any.
        started_at / updated_at: ISO-8601 lifecycle markers.
        context: Free-form key/value bag for downstream consumers.
    """

    # -- immutable input ------------------------------------------------- #
    request: ReviewRequest

    # -- exclusive agent outputs (last writer wins; None clears) --------- #
    task_brief: Annotated[TaskBrief | None, overwrite]
    patch: Annotated[Patch | None, overwrite]
    review: Annotated[ReviewResult | None, overwrite]
    test_report: Annotated[TestReport | None, overwrite]
    report: Annotated[FinalReport | None, overwrite]
    pending_approval: Annotated[ApprovalRequest | None, overwrite]
    error: Annotated[str | None, overwrite]

    # -- accumulated channels (reduced) ---------------------------------- #
    findings: Annotated[list[Finding], append_unique]
    transcript: Annotated[list[dict[str, Any]], append_capped]
    human_decisions: Annotated[list[dict[str, Any]], append_capped]
    node_timings: Annotated[list[NodeTiming], add_timings]

    # -- control plane --------------------------------------------------- #
    iteration: int
    status: str
    next_action: str

    # -- bookkeeping ----------------------------------------------------- #
    started_at: str
    updated_at: str
    context: Annotated[dict[str, Any], latest]


#: Keys guaranteed to be present in the graph's output schema.
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

M = TypeVar("M", bound=BaseModel)


def initial_state(request: ReviewRequest) -> WorkflowState:
    """Build the seed state for a new run.

    Args:
        request: The business request to process.

    Returns:
        A fully initialised state ready to be passed as the graph input.

    Example:
        --------
        >>> from agentic_workflow.domain import ReviewRequest
        >>> s = initial_state(ReviewRequest(run_id="r", request_id="p", title="t"))
        >>> s["status"], s["iteration"]
        ('pending', 0)
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
        pending_approval=None,
        error=None,
        findings=[],
        transcript=[],
        human_decisions=[],
        node_timings=[],
        iteration=0,
        status="pending",
        next_action="start",
        started_at=now,
        updated_at=now,
        context={},
    )


# --------------------------------------------------------------------------- #
# Tolerant accessors
# --------------------------------------------------------------------------- #
def as_model(state: WorkflowState, key: str, model: type[M]) -> M | None:
    """Read *key* from *state* as *model*, tolerating a dict-shaped value.

    A checkpoint round-trip normally returns the real model, but a checkpointer
    configured without our serializer degrades models to plain dicts. Doing the
    revalidation here — once, in one place — lets nodes keep using attribute
    access instead of defensive ``dict.get`` chains, and turns a misconfigured
    store into a clear :class:`SchemaValidationError` instead of an
    ``AttributeError`` three nodes deep.

    Args:
        state: The workflow state.
        key: Channel to read.
        model: The expected pydantic model type.

    Returns:
        A validated instance, or ``None`` when the channel is empty or cleared.

    Raises:
        SchemaValidationError: If the stored value cannot be validated.
    """
    from agentic_workflow.errors import SchemaValidationError

    value = cast("Any", state.get(key))
    if value is None:
        return None
    try:
        payload: Any = _as_payload(value)
        return model.model_validate(payload)
    except Exception as exc:
        raise SchemaValidationError(
            f"state channel {key!r} does not hold a valid {model.__name__}: {exc}",
            channel=key,
            model=model.__name__,
        ) from exc


def as_models(state: WorkflowState, key: str, model: type[M]) -> list[M]:
    """Read *key* as a list of *model*, skipping entries that cannot be validated.

    Accumulated channels are the most likely to be affected by a lossy store, and
    a single unreadable record must not take down an otherwise valid run. Skipped
    entries are logged at warning level.

    Args:
        state: The workflow state.
        key: Channel to read.
        model: The element model type.

    Returns:
        The validated entries, in stored order.
    """
    from agentic_workflow.logging import get_logger

    log = get_logger(__name__)
    raw = cast("list[Any]", state.get(key) or [])
    out: list[M] = []
    for index, item in enumerate(raw):
        try:
            out.append(model.model_validate(_as_payload(item)))
        except Exception as exc:
            log.warning("state.entry_skipped", channel=key, index=index, error=str(exc))
    return out


def _as_payload(value: Any) -> Any:
    """Return a validation payload for *value*, whatever shape the store left it in.

    A checkpoint round-trip normally yields the real model, but a store
    configured without our serializer degrades it to a plain dict, and a lossy
    round-trip can leave a ``dict`` where a model is expected (or vice versa).
    Normalising both directions here is what lets every caller write plain
    ``model.model_validate(...)`` code.

    Args:
        value: The raw channel value.

    Returns:
        A ``dict`` payload ready for validation.
    """
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        # `exclude_computed_fields` keeps derived keys out of the payload;
        # `revalidate_instances="always"` on StrictModel turns this into a deep
        # rebuild that repairs nested models the store degraded to dicts.
        return dump(exclude_computed_fields=True)
    return value


__all__ = [
    "MAX_TRANSCRIPT_ENTRIES",
    "OUTPUT_KEYS",
    "WorkflowState",
    "add_timings",
    "append_capped",
    "append_unique",
    "as_model",
    "as_models",
    "initial_state",
    "latest",
    "operator",
    "overwrite",
]
