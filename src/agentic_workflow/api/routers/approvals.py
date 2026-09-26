"""Human-in-the-Loop endpoints: the approval inbox.

This router is the human side of the pause/resume cycle. It is the query and
command layer over parked runs:

* ``GET /v1/approvals`` — the inbox, oldest first, so a queue drains fairly.
* ``POST /v1/approvals/{id}/resolve`` — answer a gate and resume the run.
* ``POST /v1/approvals/{id}/replay`` — safely re-submit a decision that a
  timeout made you retry.
* ``GET /v1/runs/{run_id}/audit`` — re-verify the HMACs of every decision.

Idempotency is the design constraint that shapes this router. Operators
double-click, proxies retry, and browser tabs sit open for hours. Answering the
same gate twice is therefore *normal*, and each of those cases has a different
correct answer: an identical retry replays the recorded decision, a retry that
disagrees with the log is a ``409`` (the client's state is stale), and a fresh
answer to a live gate applies. The three are distinguished in
:meth:`~agentic_workflow.human.service.ApprovalService.resolve` and
:meth:`~agentic_workflow.human.service.ApprovalService.replay`.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Path, Query, Response, status

from agentic_workflow.api.deps import ApprovalsDep, AuthDep, EngineDep
from agentic_workflow.api.schemas import (
    ApprovalListResponse,
    ResolutionResponse,
    ResolveApprovalRequest,
)
from agentic_workflow.domain.schemas import ApprovalDecision, ApprovalRequest
from agentic_workflow.errors import InvalidStateError
from agentic_workflow.logging import get_logger
from agentic_workflow.services.engine import RunOutcome

log = get_logger(__name__)

router = APIRouter(prefix="/v1/approvals", tags=["approvals"])

#: Status returned when a resumed run parks on the *next* gate.
ACCEPTED = status.HTTP_202_ACCEPTED


@router.get(
    "",
    response_model=ApprovalListResponse,
    summary="List every approval currently blocking a run",
)
async def list_approvals(
    approvals: ApprovalsDep,
    _: AuthDep,
    run_id: Annotated[str | None, Query(description="Restrict the inbox to a single run.")] = None,
) -> ApprovalListResponse:
    """Return the approval inbox, oldest request first.

    Expired approvals are **included** and flagged rather than hidden. A reviewer
    looking for a run that vanished from the queue needs to see that it timed
    out and for how long; an empty list would look identical to "nothing was ever
    submitted".

    Args:
        approvals: The approval service.
        _: Authentication dependency.
        run_id: Optional run filter.

    Returns:
        An :class:`~agentic_workflow.api.schemas.ApprovalListResponse`.
    """
    views = await approvals.inbox(run_id=run_id)
    return ApprovalListResponse(items=[view.to_dict() for view in views], count=len(views))


@router.get(
    "/stats",
    summary="Inbox counters for dashboards and probes",
)
async def approval_stats(approvals: ApprovalsDep, _: AuthDep) -> dict[str, Any]:
    """Return pending / expired / resolved counters, grouped by stage.

    Args:
        approvals: The approval service.
        _: Authentication dependency.

    Returns:
        A mapping of counters.
    """
    return await approvals.stats()


@router.get(
    "/{approval_id}",
    summary="Fetch a single pending approval",
    responses={404: {"description": "No pending approval with that id."}},
)
async def get_approval(
    approval_id: Annotated[str, Path(min_length=1, max_length=128)],
    approvals: ApprovalsDep,
    _: AuthDep,
) -> dict[str, Any]:
    """Return one approval enriched with its run context.

    Args:
        approval_id: The approval to fetch.
        approvals: The approval service.
        _: Authentication dependency.

    Returns:
        The serialised approval view.

    Raises:
        ApprovalNotFoundError: If no run is parked on that id.
    """
    return (await approvals.get(approval_id)).to_dict()


@router.post(
    "/{approval_id}/resolve",
    response_model=ResolutionResponse,
    summary="Answer an approval and resume the run",
    responses={
        ACCEPTED: {"description": "Resumed, and parked again on the next gate."},
        200: {"description": "Resumed, or the decision was a rejection."},
        404: {"description": "No pending approval with that id."},
        409: {"description": "Expired, or already resolved."},
    },
)
async def resolve_approval(
    approval_id: Annotated[str, Path(min_length=1, max_length=128)],
    body: Annotated[ResolveApprovalRequest, Body()],
    approvals: ApprovalsDep,
    _: AuthDep,
    response: Response,
) -> ResolutionResponse:
    """Answer a gate and let the run continue.

    Args:
        approval_id: The approval being answered.
        body: The verdict, the reviewer and any replacement content.
        approvals: The approval service.
        _: Authentication dependency.
        response: The outbound response, downgraded to ``202`` when the run parks
            on the next gate.

    Returns:
        A :class:`~agentic_workflow.api.schemas.ResolutionResponse`.

    Raises:
        ApprovalNotFoundError: If the approval is not pending.
        ApprovalExpiredError: If the decision window has closed.
        ApprovalAlreadyResolvedError: If the gate was already answered.
        InvalidStateError: If an ``edit`` arrives without replacement content.
    """
    result = await approvals.resolve(
        approval_id,
        decision=body.decision,
        reviewer=body.reviewer,
        comment=body.comment,
        payload=body.payload or None,
    )
    outcome = result.outcome
    response.status_code = _resume_status(outcome)
    return _resolution(approval_id, result.decision, result.run_id, outcome, result.replayed)


@router.post(
    "/{approval_id}/replay",
    response_model=ResolutionResponse,
    summary="Re-submit an already-applied decision idempotently",
    responses={
        409: {"description": "The retry disagrees with the recorded decision."},
        404: {"description": "No such approval was ever resolved."},
    },
)
async def replay_approval(
    approval_id: Annotated[str, Path(min_length=1, max_length=128)],
    body: Annotated[ResolveApprovalRequest, Body()],
    approvals: ApprovalsDep,
    _: AuthDep,
) -> ResolutionResponse:
    """Re-submit a decision that was already applied, without re-running the graph.

    Clients retry after timeouts. Without this endpoint every retry is a ``409``,
    and operators learn to fear the button. With it, an identical retry returns
    the recorded outcome, while a retry that *disagrees* with the log is a
    ``409`` — because that combination means the client's state is stale, and
    silently returning the old decision would hide it.

    Args:
        approval_id: The approval being re-submitted.
        body: The decision the client believes it sent.
        approvals: The approval service.
        _: Authentication dependency.

    Returns:
        A :class:`~agentic_workflow.api.schemas.ResolutionResponse` with
        ``replayed=true``.

    Raises:
        ApprovalNotFoundError: If the approval was never resolved.
        ApprovalAlreadyResolvedError: If the retry disagrees with the log.
    """
    result = await approvals.replay(approval_id, reviewer=body.reviewer, decision=body.decision)
    return _resolution(approval_id, result.decision, result.run_id, result.outcome, result.replayed)


@router.post(
    "/sweep",
    summary="Report approvals whose decision window has closed",
)
async def sweep_expired(approvals: ApprovalsDep, _: AuthDep) -> dict[str, Any]:
    """Return the ids of approvals nobody answered in time.

    This does **not** cancel or auto-decline anything. Expiry is a signal for an
    operator, not a decision: the run stays parked so a human can still answer it
    deliberately. Silently declining on a timer is how a review pipeline starts
    approving things nobody read.

    Args:
        approvals: The approval service.
        _: Authentication dependency.

    Returns:
        The expired approval ids and how long they have been waiting.
    """
    expired = await approvals.expire_stale()
    return {"expired": expired, "count": len(expired), "auto_resolved": False}


@router.get(
    "/{approval_id}/diff",
    summary="Rendered diff for a patch-apply gate",
    responses={404: {"description": "The approval is not a patch gate."}},
)
async def approval_diff(
    approval_id: Annotated[str, Path(min_length=1, max_length=128)],
    approvals: ApprovalsDep,
    _: AuthDep,
) -> dict[str, Any]:
    """Return the unified diff a reviewer is being asked to approve.

    Args:
        approval_id: The approval to inspect.
        approvals: The approval service.
        _: Authentication dependency.

    Returns:
        The diff text and the files it touches.

    Raises:
        ApprovalNotFoundError: If no run is parked on that id.
        InvalidStateError: If the gate carries no diff.
    """
    view = await approvals.get(approval_id)
    diff = view.request.diff_preview
    if not diff:
        raise InvalidStateError(
            "this approval does not carry a diff preview",
            approval_id=approval_id,
            stage=view.request.stage,
        )
    return {
        "approval_id": approval_id,
        "stage": view.request.stage,
        "diff": diff,
        "files": list(view.request.payload.get("files_changed", [])),
        "iteration": view.iteration,
    }


@router.get(
    "/by-run/{run_id}/audit",
    summary="Re-verify the HMAC of every decision recorded for a run",
)
async def audit_run(
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    approvals: ApprovalsDep,
    engine: EngineDep,
    _: AuthDep,
    response: Response,
) -> dict[str, Any]:
    """Re-derive every decision signature for a run.

    The decision log lives inside the graph state, so it inherits the trust level
    of whatever can write to the checkpoint store. Re-deriving the HMAC is what
    turns "we recorded that alice approved this" into a checkable claim.

    Args:
        run_id: The run to audit.
        approvals: The approval service.
        engine: The workflow engine, used to confirm the run exists.
        _: Authentication dependency.
        response: The outbound response.

    Returns:
        One entry per decision with a ``verified`` flag. ``null`` means no signing
        secret is configured, so the decision could not be verified either way.

    Raises:
        RunNotFoundError: If no such run exists.
    """
    outcome: RunOutcome = await engine.status(run_id)
    findings = await approvals.verify_log(run_id)
    unverified = [entry for entry in findings if entry.get("verified") is False]
    if unverified:
        # A signature that does not verify is a security finding, not a soft
        # error: surface it loudly with a 409 so monitoring can alert on it.
        response.status_code = status.HTTP_409_CONFLICT
        log.error("api.audit_failed", run_id=run_id, findings=len(unverified))
    return {
        "run_id": run_id,
        "status": outcome.status,
        "decisions": findings,
        "count": len(findings),
        "unverified": len(unverified),
        "signing_configured": any(entry.get("verified") is not None for entry in findings),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resume_status(outcome: RunOutcome | None) -> int:
    """Return the status describing a resumed run.

    Args:
        outcome: The run's state after the resume, when the resume succeeded.

    Returns:
        ``202`` when the run parked again, ``200`` otherwise. A rejection ends the
        run and is a completed review, not a failure.
    """
    if outcome is None:
        return status.HTTP_200_OK
    return ACCEPTED if outcome.is_parked else status.HTTP_200_OK


def _resolution(
    approval_id: str,
    decision: ApprovalDecision,
    run_id: str,
    outcome: RunOutcome | None,
    replayed: bool,
) -> ResolutionResponse:
    """Project a resolution onto the wire model.

    Args:
        approval_id: The approval that was answered.
        decision: The applied decision.
        run_id: The resumed run.
        outcome: The run's state afterwards, if the resume ran.
        replayed: Whether the decision was a no-op replay.

    Returns:
        A :class:`~agentic_workflow.api.schemas.ResolutionResponse`.
    """
    pending: ApprovalRequest | None = outcome.pending if outcome else None
    return ResolutionResponse(
        run_id=run_id,
        status=outcome.status if outcome else "unknown",
        replayed=replayed,
        approval_id=approval_id,
        decision=decision.decision.value,
        pending_approval=pending,
    )


__all__ = ["router"]
