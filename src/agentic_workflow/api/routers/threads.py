"""Time-travel endpoints over the checkpoint history.

Every super-step LangGraph takes is addressable, which turns "why did the
reviewer reject iteration two?" from a guess into a query. This router exposes
that:

* ``GET /v1/threads/{run_id}/history`` — every checkpoint, newest first.
* ``GET /v1/threads/{run_id}/checkpoints/{checkpoint_id}`` — the state *as of*
  one point, read-only.
* ``POST /v1/threads/{run_id}/replay`` — re-execute from a historical
  checkpoint as a **fork**: the new work is appended to the same thread as a
  sibling of the original history, so the branch that misbehaved stays intact and
  the two can be diffed.

The fork semantics are the reason replay is worth an endpoint at all. Re-running
in place would destroy the evidence.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Path, Query, Response, status

from agentic_workflow.api.deps import AuthDep, EngineDep
from agentic_workflow.api.routers.runs import project_detail
from agentic_workflow.api.schemas import (
    CheckpointResponse,
    HistoryResponse,
    ReplayRequest,
    RunDetail,
)
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1/threads", tags=["threads"])

#: Status returned when a replay parks on a human gate.
ACCEPTED = status.HTTP_202_ACCEPTED


@router.get(
    "/{run_id}/history",
    response_model=HistoryResponse,
    summary="List the checkpoint history of a run",
    responses={404: {"description": "The run has no checkpoints."}},
)
async def thread_history(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
    _: AuthDep,
    limit: Annotated[int, Query(ge=1, le=1_000)] = 100,
) -> HistoryResponse:
    """Return a run's checkpoints, newest first.

    Args:
        run_id: The run to inspect.
        engine: The workflow engine.
        _: Authentication dependency.
        limit: Maximum number of checkpoints to return.

    Returns:
        A :class:`~agentic_workflow.api.schemas.HistoryResponse`.

    Raises:
        RunNotFoundError: If the run has no checkpoint history.
    """
    entries = await engine.history(run_id, limit=limit)
    items = [
        CheckpointResponse(
            checkpoint_id=entry.checkpoint_id,
            step=entry.step,
            source=entry.source,
            next_nodes=list(entry.next_nodes),
            created_at=entry.created_at,
            pending_approval=entry.pending.approval_id if entry.pending else None,
        )
        for entry in entries
    ]
    return HistoryResponse(run_id=run_id, items=items, count=len(items))


@router.get(
    "/{run_id}/checkpoints/{checkpoint_id}",
    response_model=RunDetail,
    summary="Read the run state as of a historical checkpoint",
    responses={
        404: {"description": "Unknown checkpoint."},
        ACCEPTED: {"description": "The run was parked at that checkpoint."},
    },
)
async def checkpoint_state(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    checkpoint_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
    _: AuthDep,
    response: Response,
) -> RunDetail:
    """Project a run as it was at a given checkpoint. Read-only.

    Nothing is written and the head of the run is untouched, so a UI can let an
    operator scrub through a run without disturbing it.

    Args:
        run_id: The run to inspect.
        checkpoint_id: The checkpoint to read.
        engine: The workflow engine.
        _: Authentication dependency.
        response: The outbound response, downgraded to ``202`` when the run was
            parked at that checkpoint.

    Returns:
        A :class:`~agentic_workflow.api.schemas.RunDetail` as of that point.

    Raises:
        RunNotFoundError: If the checkpoint cannot be read.
    """
    outcome = await engine.state_at(run_id, checkpoint_id)
    if outcome.is_parked:
        response.status_code = ACCEPTED
    return project_detail(outcome)


@router.post(
    "/{run_id}/replay",
    response_model=RunDetail,
    summary="Re-execute a run from a historical checkpoint",
    responses={
        404: {"description": "Unknown checkpoint."},
        ACCEPTED: {"description": "The replay parked on a human gate."},
        504: {"description": "The replay exceeded the run budget."},
    },
)
async def replay_thread(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    body: Annotated[ReplayRequest, Body()],
    engine: EngineDep,
    _: AuthDep,
    response: Response,
) -> RunDetail:
    """Branch a run from a past checkpoint and drive the branch to a stopping point.

    Args:
        run_id: The run to replay.
        body: The checkpoint to branch from and an optional reason for the audit
            trail.
        engine: The workflow engine.
        _: Authentication dependency.
        response: The outbound response, downgraded to ``202`` when the branch
            parks.

    Returns:
        A :class:`~agentic_workflow.api.schemas.RunDetail` for the branch.

    Raises:
        RunNotFoundError: If the checkpoint cannot be read.
        RunTimeoutError: If the branch exceeds the run budget.
    """
    if body.reason:
        log.info(
            "api.replay",
            run_id=run_id,
            checkpoint_id=body.checkpoint_id,
            reason=body.reason,
        )
    outcome = await engine.replay_from(run_id, body.checkpoint_id)
    if outcome.is_parked:
        response.status_code = ACCEPTED
    return project_detail(outcome)


@router.get(
    "/{run_id}/checkpoints",
    summary="Compare two checkpoints of a run",
    responses={404: {"description": "Unknown checkpoint."}},
)
async def diff_checkpoints(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
    _: AuthDep,
    from_checkpoint: Annotated[str, Query(alias="from", min_length=1, max_length=128)],
    to_checkpoint: Annotated[str, Query(alias="to", min_length=1, max_length=128)],
) -> dict[str, Any]:
    """Return the state channels that differ between two checkpoints.

    A channel-level diff is what an operator actually wants: "what changed
    between the checkpoint before the reviewer and the one after it" answers in
    one list, whereas two full state dumps make the reader do the comparison.

    Args:
        run_id: The run to inspect.
        engine: The workflow engine.
        _: Authentication dependency.
        from_checkpoint: The earlier checkpoint.
        to_checkpoint: The later checkpoint.

    Returns:
        The changed channels with before/after values, plus the unchanged channel
        names.

    Raises:
        RunNotFoundError: If either checkpoint cannot be read.
    """
    before = (await engine.state_at(run_id, from_checkpoint)).state
    after = (await engine.state_at(run_id, to_checkpoint)).state
    keys = set(before) | set(after)
    changed = [
        {"channel": key, "before": before.get(key), "after": after.get(key)}
        for key in sorted(keys)
        if before.get(key) != after.get(key)
    ]
    changed_names = {entry["channel"] for entry in changed}
    return {
        "run_id": run_id,
        "from": from_checkpoint,
        "to": to_checkpoint,
        "changed": changed,
        "changed_count": len(changed),
        "unchanged": sorted(keys - changed_names),
    }


__all__ = ["router"]
