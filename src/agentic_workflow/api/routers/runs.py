"""Run lifecycle endpoints: start, inspect, resume, cancel.

Design decisions worth stating, because they are the difference between a control
plane clients can build on and one they fight:

* **Starting a run is a synchronous call that may park.** ``POST /v1/runs`` drives
  the graph until it finishes or hits a human gate, then answers ``202 Accepted``
  with the parked state. A caller that only wants a fire-and-forget submission
  passes ``auto_resolve=true`` and gets a finished run in one round trip; a
  caller that wants the gate gets it without polling.
* **Every read is served from the checkpointer**, not from the in-process
  registry, so a restart or a second replica answers identically. The registry is
  a rebuildable projection, never a source of truth.
* **Idempotency is opt-in via the content hash.** ``X-Content-Hash`` lets a
  retried submission return the original run instead of creating a duplicate
  thread; a conflicting run id is a ``409``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Header, Path, Query, Response, status

from agentic_workflow.api.deps import EngineDep
from agentic_workflow.api.schemas import (
    ResumeRunRequest,
    RunDetail,
    RunListResponse,
    StartRunRequest,
)
from agentic_workflow.domain.schemas import Decision
from agentic_workflow.errors import InvalidRequestError, RunNotFoundError
from agentic_workflow.logging import get_logger
from agentic_workflow.services.engine import RunOutcome

log = get_logger(__name__)

router = APIRouter(prefix="/v1/runs", tags=["runs"])

#: Status returned when a run stopped on a human gate rather than finishing.
ACCEPTED = status.HTTP_202_ACCEPTED


def project_detail(outcome: RunOutcome) -> RunDetail:
    """Project an engine outcome onto the wire model.

    Args:
        outcome: The engine's view of the run.

    Returns:
        A :class:`~agentic_workflow.api.schemas.RunDetail`.
    """
    return RunDetail(
        run_id=outcome.run_id,
        status=outcome.status,
        iteration=outcome.iteration,
        next_node=outcome.next_nodes[0] if outcome.next_nodes else None,
        checkpoint_id=outcome.checkpoint_id,
        pending_approval=outcome.pending,
        report=outcome.report,
        decisions=outcome.decisions,
        timings=outcome.timings,
        error=outcome.error,
        is_parked=outcome.is_parked,
        is_finished=outcome.is_finished,
        usage=outcome.usage.as_dict(),
    )


def _outcome_status(outcome: RunOutcome) -> int:
    """Map a run outcome to the HTTP status that describes it.

    Args:
        outcome: The engine's view of the run.

    Returns:
        ``202`` when the run is parked (it is waiting, not failing), ``200`` when
        it finished, and ``500`` only when it actually failed.
    """
    if outcome.is_parked:
        return ACCEPTED
    if outcome.status == "failed":
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    return status.HTTP_200_OK


@router.post(
    "",
    response_model=RunDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Start a review run",
    responses={
        ACCEPTED: {"description": "The run parked on a human gate."},
        409: {"description": "The run id is already in use."},
        429: {"description": "The concurrency or request budget is exhausted."},
    },
)
async def start_run(
    body: Annotated[StartRunRequest, Body()],
    engine: EngineDep,
    response: Response,
    content_hash: Annotated[str | None, Header(alias="X-Content-Hash")] = None,
) -> RunDetail:
    """Start a run and drive it to completion or to its first human gate.

    Args:
        body: The run request. Set ``auto_resolve`` to answer every gate
            automatically and get a finished run in a single call.
        engine: The workflow engine.
        response: The outbound response, whose status is downgraded to ``202``
            when the run parks.
        content_hash: Optional idempotency key. When it matches the hash of a
            run that already exists, that run is returned instead of starting a
            duplicate.

    Returns:
        A :class:`~agentic_workflow.api.schemas.RunDetail` describing the run as
        it stopped.

    Raises:
        RunAlreadyExistsError: If the run id is taken and the content hash does
            not match.
        InvalidRequestError: If ``X-Content-Hash`` disagrees with the body.
    """
    request = body.to_domain()
    if content_hash is not None and content_hash != request.content_hash:
        raise InvalidRequestError(
            "X-Content-Hash does not match the submitted content; compute it as "
            "sha256 over (file.path, file.content) pairs followed by description",
            run_id=request.run_id,
            expected=request.content_hash,
            received=content_hash,
        )

    if content_hash is not None:
        # Idempotent replay: a client retrying a submission it already sent gets
        # the original run back instead of a duplicate thread.
        try:
            existing = await engine.status(request.run_id)
        except RunNotFoundError:
            pass
        else:
            log.info("api.run_replayed", run_id=request.run_id, status=existing.status)
            response.status_code = _outcome_status(existing)
            return project_detail(existing)

    if body.auto_resolve:
        outcome = await engine.run_until_done(
            request,
            decide=_auto_decider(body.auto_decision),
            max_gates=body.max_gates,
        )
    else:
        outcome = await engine.start(request, metadata={"auto_resolve": False})

    response.status_code = _outcome_status(outcome)
    return project_detail(outcome)


@router.get("", response_model=RunListResponse, summary="List runs")
async def list_runs(
    engine: EngineDep,
    status_filter: Annotated[
        str | None,
        Query(alias="status", description="Filter by run status."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunListResponse:
    """List runs, most recently updated first.

    Args:
        engine: The workflow engine.
        status_filter: Optional status filter.
        limit: Maximum number of runs to return.
        offset: Pagination offset.

    Returns:
        A :class:`~agentic_workflow.api.schemas.RunListResponse`.
    """
    items = await engine.list_runs(status=status_filter, limit=limit, offset=offset)
    return RunListResponse(
        items=items, count=len(items), total=len(engine.registry), limit=limit, offset=offset
    )


@router.get(
    "/{run_id}",
    response_model=RunDetail,
    summary="Read a run",
    responses={ACCEPTED: {"description": "The run is parked on a human gate."}},
)
async def get_run(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
    response: Response,
) -> RunDetail:
    """Read the current state of a run from the checkpoint store.

    Args:
        run_id: The run to inspect.
        engine: The workflow engine.
        response: The outbound response, downgraded to ``202`` when parked.

    Returns:
        A :class:`~agentic_workflow.api.schemas.RunDetail`.

    Raises:
        RunNotFoundError: If no such run exists.
    """
    outcome = await engine.status(run_id)
    response.status_code = _outcome_status(outcome)
    return project_detail(outcome)


@router.get(
    "/{run_id}/report",
    summary="Fetch the final report of a completed run",
    responses={
        ACCEPTED: {"description": "The run has not produced a report yet."},
        404: {"description": "No such run."},
    },
)
async def get_report(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
    response: Response,
) -> Any:
    """Return the final report produced by a run.

    Args:
        run_id: The run to inspect.
        engine: The workflow engine.
        response: The outbound response.

    Returns:
        The serialised :class:`~agentic_workflow.domain.schemas.FinalReport`,
        or ``None`` with a ``202`` when the run has not produced one yet.

    Raises:
        RunNotFoundError: If no such run exists.
    """
    outcome = await engine.status(run_id)
    if outcome.report is None:
        response.status_code = ACCEPTED
        return None
    return outcome.report.model_dump(mode="json")


@router.post(
    "/{run_id}/resume",
    response_model=RunDetail,
    summary="Resume a parked run with a human decision",
    responses={
        ACCEPTED: {"description": "Resumed, and parked again on the next gate."},
        409: {"description": "The run is not parked, or the decision is stale."},
    },
)
async def resume_run(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    body: Annotated[ResumeRunRequest, Body()],
    engine: EngineDep,
    response: Response,
) -> RunDetail:
    """Answer the pending gate and let the run continue.

    The decision is passed through verbatim, including its ``approval_id``. The
    graph re-verifies that the id matches the gate it is actually parked on, so a
    client that answers a stale approval gets a ``409`` instead of silently
    approving the wrong thing.

    Args:
        run_id: The parked run.
        body: The human decision.
        engine: The workflow engine.
        response: The outbound response, downgraded to ``202`` when parked again.

    Returns:
        A :class:`~agentic_workflow.api.schemas.RunDetail`.

    Raises:
        RunNotFoundError: If no such run exists.
        InvalidStateError: If the run is not parked.
    """
    outcome = await engine.resume(run_id, body.to_domain())
    response.status_code = _outcome_status(outcome)
    return project_detail(outcome)


@router.post(
    "/{run_id}/cancel",
    response_model=RunDetail,
    summary="Cancel a run",
    responses={404: {"description": "No such run exists."}},
)
async def cancel_run(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
    reason: Annotated[str, Query(max_length=500)] = "cancelled via API",
) -> RunDetail:
    """Cancel a run, keeping every completed checkpoint.

    Args:
        run_id: The run to cancel.
        engine: The workflow engine.
        reason: Recorded on the run for the audit trail.

    Returns:
        A :class:`~agentic_workflow.api.schemas.RunDetail` reflecting the
        cancelled state.

    Raises:
        RunNotFoundError: If no such run exists.
    """
    await engine.cancel(run_id, reason=reason)
    return project_detail(await engine.status(run_id))


@router.get(
    "/{run_id}/decisions",
    summary="List the human decisions recorded for a run",
    responses={404: {"description": "No such run exists."}},
)
async def list_decisions(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
) -> dict[str, Any]:
    """Return the append-only decision log of a run.

    Args:
        run_id: The run to inspect.
        engine: The workflow engine.

    Returns:
        A mapping with the ordered decisions and a count.

    Raises:
        RunNotFoundError: If no such run exists.
    """
    outcome = await engine.status(run_id)
    return {"run_id": run_id, "items": outcome.decisions, "count": len(outcome.decisions)}


@router.get(
    "/{run_id}/timings",
    summary="Per-node wall-clock accounting for a run",
    responses={404: {"description": "No such run exists."}},
)
async def list_timings(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
) -> dict[str, Any]:
    """Return per-node timings recorded during a run.

    Args:
        run_id: The run to inspect.
        engine: The workflow engine.

    Returns:
        A mapping with the timings, their total and the slowest node.

    Raises:
        RunNotFoundError: If no such run exists.
    """
    outcome = await engine.status(run_id)
    total = sum(timing.duration_ms for timing in outcome.timings)
    slowest = max(outcome.timings, key=lambda t: t.duration_ms, default=None)
    return {
        "run_id": run_id,
        "items": [timing.model_dump(mode="json") for timing in outcome.timings],
        "count": len(outcome.timings),
        "total_ms": round(total, 3),
        "slowest_node": slowest.node if slowest else None,
        "slowest_ms": round(slowest.duration_ms, 3) if slowest else None,
    }


@router.get(
    "/{run_id}/usage",
    summary="LLM usage attributed to a run",
    responses={404: {"description": "No such run exists."}},
)
async def get_run_usage(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    engine: EngineDep,
) -> dict[str, Any]:
    """Return the LLM usage attributed to this run so far.

    The totals accumulate on the registry across every drive the engine has
    performed for the run — the parked start, each resolve — so this answers
    "what has it cost" without counting model calls in log lines. Unlike a
    checkpoint read, the attribution is process memory: after a restart it
    begins empty again even though the run's state survives.

    Args:
        run_id: The run to inspect.
        engine: The workflow engine.

    Returns:
        A mapping with the attributed usage counters.

    Raises:
        RunNotFoundError: If no such run exists.
    """
    record = engine.registry.find(run_id)
    if record is None:
        raise RunNotFoundError("no such run", run_id=run_id)
    return {"run_id": run_id, "usage": record.usage.as_dict()}


def _auto_decider(decision: Decision) -> Any:
    """Build the callable that answers gates when ``auto_resolve`` is set.

    Args:
        decision: The verdict to apply to every gate.

    Returns:
        A synchronous callable suitable for
        :meth:`~agentic_workflow.services.engine.WorkflowEngine.run_until_done`.
    """

    def decide(pending: Any) -> dict[str, Any]:
        """Return the canned decision for a pending gate.

        Args:
            pending: The approval request the graph raised.

        Returns:
            The resume payload.
        """
        payload: dict[str, Any] = {
            "approval_id": pending.approval_id,
            "decision": decision.value,
            "reviewer": "auto",
            "comment": f"auto-resolved as {decision.value}",
        }
        if decision is Decision.REJECT:
            # A rejection needs no payload, but the human-facing comment is what
            # the audit trail will show six months from now.
            payload["comment"] = "auto-resolved: rejected by policy"
        return payload

    return decide


__all__ = ["ACCEPTED", "project_detail", "router"]
