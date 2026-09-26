"""LangGraph interrupt plumbing for Human-in-the-Loop gates.

How a gate works
----------------
A node calls :func:`request_decision`. On the **first** pass LangGraph raises
``GraphInterrupt`` and rolls the task back; the interrupt value is surfaced to
the caller (the run manager), persisted, and pushed to any WebSocket client.
The graph is now durably parked — the process can die and come back.

When a human responds, the client calls ``graph.ainvoke(Command(resume=...))``.
LangGraph re-executes the node from the top, and the **second** call to
:func:`request_decision` returns the resume value instead of raising.

Two consequences drive the implementation below:

1. **Node re-execution is real.** Everything above the interrupt call runs
   again. Nodes must therefore be written so that re-execution is idempotent —
   :func:`request_decision` enforces the cheap half of that by *verifying* the
   resume value belongs to the gate being answered, which catches the classic
   "stale UI answered an old gate" bug.
2. **The resume value must be self-describing.** A bare ``True`` is
   indistinguishable from "someone clicked approve on a different screen", so
   decisions always travel as a full :class:`ApprovalDecision`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
import hashlib
import hmac
import json
from typing import Any, Final

from agentic_workflow.config import Settings
from agentic_workflow.domain.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    Decision,
    FinalReport,
    Patch,
    ReviewResult,
    TestReport,
    utcnow,
)
from agentic_workflow.errors import (
    ApprovalAlreadyResolvedError,
    ApprovalExpiredError,
    ApprovalRejectedError,
    InvalidStateError,
    SchemaValidationError,
)
from agentic_workflow.human.policy import (
    EscalationPolicy,
    EscalationReason,
    GateDecision,
    Stage,
    allowed_decisions,
)
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

#: Key under which the gate bookkeeping is stored in the interrupt payload. The
#: whole interrupt value is a single serialisable object; nesting metadata under
#: a reserved key keeps the user-facing fields at the top level.
_METADATA_KEY = "__awf__"

#: Prefix every derived gate identifier carries. See :func:`run_id_from_approval_id`.
APPROVAL_ID_PREFIX = "apr_"


# --------------------------------------------------------------------------- #
# Signing
# --------------------------------------------------------------------------- #
def sign_decision(decision: ApprovalDecision, secret: str) -> str:
    """Return an HMAC-SHA256 signature over the canonical decision.

    The signature lets the audit log prove a decision was not tampered with
    after the fact, and lets a verifier reject a payload replayed from another
    run (the run id is part of the signed material).

    Args:
        decision: The decision being signed.
        secret: Shared secret. Must be non-empty.

    Returns:
        A hex-encoded digest prefixed with the algorithm.
    """
    if not secret:
        raise ValueError("a non-empty secret is required to sign decisions")
    canonical = json.dumps(
        {
            "approval_id": decision.approval_id,
            "decision": decision.decision.value,
            "reviewer": decision.reviewer,
            "comment": decision.comment,
            "payload": decision.payload,
            "decided_at": decision.decided_at.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_decision(decision: ApprovalDecision, secret: str) -> bool:
    """Check the signature produced by :func:`sign_decision`.

    Uses :func:`hmac.compare_digest` to avoid leaking timing information.
    """
    if not decision.signature:
        return False
    expected = sign_decision(decision, secret)
    return hmac.compare_digest(expected, decision.signature)


# --------------------------------------------------------------------------- #
# Payload construction
# --------------------------------------------------------------------------- #
def approval_id_for(run_id: str, stage: Stage, iteration: int) -> str:
    """Derive the *stable* identifier of a gate.

    Stability is not cosmetic — it is load-bearing. LangGraph re-executes a node
    from the top after an interrupt, so :func:`build_approval_request` is called
    twice for a single human decision. If the id were random, the second pass
    would not match the ``approval_id`` the client was shown, and the stale-answer
    guard in :func:`request_decision` would reject every legitimate decision.

    So the id is a pure function of the coordinates that identify the gate —
    ``(run_id, stage, iteration)``. A genuinely *different* gate (the run moved
    on to another stage or another feedback iteration) produces a different id and
    is correctly rejected, which is exactly the "stale browser tab" case we care
    about.

    The trailing digest keeps ids unique across runs that share a run-id prefix
    and makes them visually distinct from user data.

    Args:
        run_id: Owning run.
        stage: The gated stage.
        iteration: Feedback-loop iteration the gate belongs to.

    Returns:
        An identifier of the form ``apr_<run>_<stage>_<iteration>_<digest8>``.
    """
    material = f"{run_id}|{stage.value}|{iteration}".encode()
    digest = hashlib.sha256(material).hexdigest()[:8]
    return f"apr_{run_id}_{stage.value}_{iteration:02d}_{digest}"


def run_id_from_approval_id(approval_id: str) -> str | None:
    """Recover the run id an approval id was derived from.

    The inverse of :func:`approval_id_for`, and it exists for a performance
    reason: locating a gate's run by scanning every run's decision log is
    O(runs) on the hot path of every human decision, and the answer is already
    encoded in the id. Parsing is done from the right because stage names contain
    underscores while run ids do not — the wire layer enforces that alphabet, so
    the three trailing components are unambiguous.

    Args:
        approval_id: The gate identifier to invert.

    Returns:
        The run id, or ``None`` if the value is not a gate id this function can
        parse. ``None`` means "ask the store", not "no such run".
    """
    if not approval_id.startswith(APPROVAL_ID_PREFIX):
        return None
    body = approval_id[len(APPROVAL_ID_PREFIX) :]
    head, _, _digest = body.rpartition("_")
    candidate, _, iteration = head.rpartition("_")
    if not iteration.isdigit() or not candidate:
        return None
    for stage in Stage:
        marker = f"_{stage.value}"
        if candidate.endswith(marker):
            return candidate[: -len(marker)] or None
    return None


def build_approval_request(
    *,
    run_id: str,
    stage: Stage,
    title: str,
    gate: GateDecision,
    policy: EscalationPolicy,
    settings: Settings | None = None,
    payload: dict[str, Any] | None = None,
    diff_preview: str = "",
    confidence: float | None = None,
    iteration: int = 0,
) -> ApprovalRequest:
    """Assemble the :class:`ApprovalRequest` surfaced at an interrupt.

    The identifier is :func:`approval_id_for` — deterministic, so re-executing
    the node after the interrupt yields the *same* id and a stale answer to a
    superseded gate is still caught.

    Args:
        run_id: Owning run.
        stage: The stage requesting approval.
        title: One-line summary.
        gate: The policy decision that triggered the gate.
        policy: Active escalation policy (controls offered actions + timeout).
        settings: Application settings; overrides the policy timeout when given.
        payload: Node-specific inspection data.
        diff_preview: Truncated diff shown to the human.
        confidence: Agent confidence driving the escalation.
        iteration: Feedback-loop iteration, part of the gate's identity.

    Returns:
        A fully populated approval request.
    """
    timeout = policy.default_timeout_seconds
    if settings is not None:
        timeout = settings.hitl_default_timeout_seconds

    return ApprovalRequest(
        approval_id=approval_id_for(run_id, stage, iteration),
        run_id=run_id,
        node=stage.value,
        stage=stage.value,
        title=title[:256],
        rationale=gate.rationale,
        payload=payload or {},
        options=allowed_decisions(policy),
        confidence=confidence if confidence is not None else gate.confidence,
        created_at=utcnow(),
        expires_at=utcnow() + timedelta(seconds=timeout),
        diff_preview=diff_preview,
        requested_by="workflow-engine",
    )


def encode_interrupt(
    request: ApprovalRequest,
    *,
    reason: EscalationReason = EscalationReason.AGENT_REQUESTED,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialise an approval request into the object sent through ``interrupt()``.

    Returning a plain dict (rather than the pydantic model) means the value
    survives JSON serialisation in the checkpointer without custom encoders, and
    can be returned verbatim by the API.

    Args:
        request: The approval to surface.
        reason: Why the gate was raised, recorded in the metadata block.
        metadata: Extra structured context from the policy decision.

    Returns:
        A JSON-serialisable interrupt value.
    """
    return {
        "type": "approval_request",
        "request": request.model_dump(mode="json"),
        _METADATA_KEY: {
            "reason": reason.value,
            "metadata": metadata or {},
            "issued_at": utcnow().isoformat(),
        },
    }


def decode_interrupt(payload: Any) -> tuple[ApprovalRequest, dict[str, Any]]:
    """Recover the :class:`ApprovalRequest` from a decoded interrupt payload.

    Args:
        payload: The interrupt value, either raw or wrapped in a LangGraph
            ``__interrupt__`` record.

    Returns:
        The approval request and the accompanying metadata.

    Raises:
        SchemaValidationError: If the payload is not a recognisable approval.
    """
    if isinstance(payload, ApprovalRequest):
        return payload, {}
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"unexpected interrupt payload type: {type(payload).__name__}")
    if payload.get("type") != "approval_request":
        raise SchemaValidationError("interrupt payload is not an approval request")
    raw = payload.get("request") or {}
    try:
        return ApprovalRequest.model_validate(raw), dict(payload.get(_METADATA_KEY) or {})
    except Exception as exc:
        raise SchemaValidationError(f"malformed approval request: {exc}") from exc


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #
def request_decision(
    request: ApprovalRequest,
    *,
    secret: str = "",
    default_reviewer: str = "system",
) -> ApprovalDecision:
    """Pause the graph until a human resolves *request*.

    First call: raises LangGraph's ``GraphInterrupt`` carrying
    :func:`encode_interrupt` of *request*. Second call (after a
    ``Command(resume=...)``): returns the validated decision.

    Args:
        request: The approval request to surface.
        secret: Shared secret used to sign the returned decision.
        default_reviewer: Reviewer recorded when the resume value omits one.

    Returns:
        The human's validated decision.

    Raises:
        ApprovalRejectedError: If the human chose ``reject``.
        ApprovalExpiredError: If the approval window elapsed.
        InvalidStateError: If the resume value does not match this gate.
    """
    from langgraph.types import interrupt

    # --- pass 1: raise the interrupt ---------------------------------- #
    raw = interrupt(encode_interrupt(request))

    # --- pass 2: a human answered ------------------------------------- #
    decision = _coerce_decision(raw, request, default_reviewer)

    if decision.approval_id and decision.approval_id != request.approval_id:
        # A stale client answered a gate that has since been re-issued. Failing
        # loudly is far safer than applying an approval to the wrong action.
        raise InvalidStateError(
            "decision does not match the pending approval",
            expected=request.approval_id,
            received=decision.approval_id,
        )

    if request.is_expired:
        raise ApprovalExpiredError(
            "approval window elapsed before a decision was recorded",
            approval_id=request.approval_id,
            expires_at=request.expires_at.isoformat() if request.expires_at else None,
        )

    if secret and not decision.signature:
        decision.signature = sign_decision(decision, secret)
    elif decision.signature and secret and not verify_decision(decision, secret):
        raise InvalidStateError(
            "decision signature failed verification",
            approval_id=decision.approval_id,
        )

    # Logged before the rejection check: a rejection ends the run, and the log
    # line is the only place the decision is guaranteed to be observed even when
    # the caller lets the exception propagate.
    log.info(
        "hitl.decision_recorded",
        approval_id=decision.approval_id,
        decision=decision.decision.value,
        reviewer=decision.reviewer,
        stage=request.stage,
    )

    if decision.decision is Decision.REJECT:
        raise ApprovalRejectedError(
            decision.comment or "human rejected the proposed action",
            approval_id=decision.approval_id,
            reviewer=decision.reviewer,
            stage=request.stage,
            decision=decision,
        )

    return decision


def _coerce_decision(
    raw: Any,
    request: ApprovalRequest,
    default_reviewer: str,
) -> ApprovalDecision:
    """Normalise whatever the client sent back into an :class:`ApprovalDecision`.

    Accepts, in order of preference: a fully-formed decision dict, a bare
    ``{"decision": "approve"}`` dict, or a plain string such as ``"approve"``.

    Raises:
        InvalidStateError: If the value cannot be interpreted at all.
    """
    if isinstance(raw, ApprovalDecision):
        return raw

    if isinstance(raw, str):
        # Trimmed: a bare string is the shape a human types, and "approve " is
        # unambiguously the same intent as "approve". Refusing it with an enum
        # error would read as a server fault rather than a typo.
        try:
            return ApprovalDecision(
                approval_id=request.approval_id,
                decision=Decision(raw.strip().lower()),
                reviewer=default_reviewer,
            )
        except ValueError as exc:
            raise InvalidStateError(f"uninterpretable decision string {raw!r}: {exc}") from exc

    if isinstance(raw, list):
        # LangGraph delivers a list when a node raises several interrupts; the
        # first entry belongs to this call site.
        if not raw:
            raise InvalidStateError("empty resume value")
        return _coerce_decision(raw[0], request, default_reviewer)

    if isinstance(raw, dict):
        # LangGraph may wrap the value under the interrupt index.
        if "value" in raw and "approval_id" not in raw:
            return _coerce_decision(raw["value"], request, default_reviewer)
        payload = dict(raw)
        payload.setdefault("approval_id", request.approval_id)
        payload.setdefault("reviewer", default_reviewer)
        if "decision" in payload and not isinstance(payload["decision"], str):
            payload["decision"] = str(payload["decision"])
        try:
            return ApprovalDecision.model_validate(payload)
        except Exception as exc:
            raise InvalidStateError(f"malformed decision payload: {exc}") from exc

    raise InvalidStateError(f"cannot interpret resume value of type {type(raw).__name__}")


# --------------------------------------------------------------------------- #
# Stage helpers — one function per gate, so the nodes stay declarative
# --------------------------------------------------------------------------- #
def gate_payload_for_stage(
    stage: Stage,
    *,
    review: ReviewResult | None = None,
    tests: TestReport | None = None,
    patch: Patch | None = None,
    report: FinalReport | None = None,
) -> dict[str, Any]:
    """Build the inspection payload a human sees for *stage*.

    Every stage shows the *evidence* needed to decide, and nothing else. A
    reviewer who has to hunt for a diff will rubber-stamp it instead of reading.
    """
    payload: dict[str, Any] = {"stage": stage.value}
    if review is not None:
        payload["review"] = review.model_dump(mode="json")
    if tests is not None:
        payload["tests"] = tests.model_dump(mode="json")
    if patch is not None:
        payload["patch"] = {
            "summary": patch.summary,
            "strategy": patch.strategy,
            "files_changed": patch.files_changed,
            "diff_size": patch.diff_size,
        }
    if report is not None:
        payload["report"] = {
            "report_id": report.report_id,
            "decision": report.decision.value,
            "citations": report.citations[:50],
        }
    return payload


def diff_preview(patch: Patch | None, *, limit: int = 8_000) -> str:
    """Return a truncated unified diff suitable for embedding in a request."""
    if patch is None or not patch.diff:
        return ""
    if len(patch.diff) <= limit:
        return patch.diff
    half = limit // 2
    return f"{patch.diff[:half]}\n... [{len(patch.diff) - limit} chars truncated] ...\n{patch.diff[-half:]}"


def record_decision_log(
    decision: ApprovalDecision,
    *,
    run_id: str,
    stage: str,
    created_at: Any = None,
    resolved_at: Any = None,
) -> dict[str, Any]:
    """Build the immutable audit-log entry stored in the graph state.

    Args:
        decision: The resolved decision.
        run_id: Owning run.
        stage: Graph stage the gate belonged to.
        created_at: When the gate was raised; used to compute response latency.
        resolved_at: When the decision was recorded. Defaults to the decision's
            own timestamp.

    Returns:
        A JSON-serialisable audit record.
    """
    raised = created_at or decision.decided_at
    record: dict[str, Any] = {
        "approval_id": decision.approval_id,
        "stage": stage,
        "decision": decision.decision.value,
        "reviewer": decision.reviewer,
        "comment": decision.comment,
        "signature": decision.signature,
        "decided_at": (resolved_at or decision.decided_at).isoformat(),
        "latency_seconds": max(
            0.0, ((resolved_at or decision.decided_at) - raised).total_seconds()
        ),
        "run_id": run_id,
    }
    if decision.payload:
        # An `edit` decision's payload *is* the human's contribution. An audit
        # trail that records "alice edited" without recording what she changed
        # answers "was a human involved?" and not "what did they author?", which
        # is the only reason to keep an audit trail at all. Bounded, because the
        # payload may carry a whole diff and the log lives in every checkpoint.
        record["payload"] = _bounded_payload(decision.payload)
    return record


#: Cap on the serialised human payload stored in the decision log. Large enough
#: to hold a real edit request, small enough that a run's checkpoint history
#: stays readable and cheap to page.
PAYLOAD_LOG_LIMIT: Final = 4_000


def _bounded_payload(payload: dict[str, Any], *, limit: int = PAYLOAD_LOG_LIMIT) -> dict[str, Any]:
    """Return a log-sized view of a human decision payload.

    Long strings are truncated rather than dropped: *what* the human wrote is the
    evidence, so the head of a long message is far more useful than its absence.

    Args:
        payload: The decision payload as submitted.
        limit: Maximum characters retained per string value.

    Returns:
        A JSON-serialisable copy with every string bounded.
    """
    bounded: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            bounded[key] = value if len(value) <= limit else f"{value[:limit]}… [truncated]"
        elif isinstance(value, list | dict):
            encoded = json.dumps(value, default=str)
            bounded[key] = value if len(encoded) <= limit else encoded[:limit] + "… [truncated]"
        else:
            bounded[key] = value
    return bounded


def assert_not_already_resolved(
    decision: ApprovalDecision, previous: Sequence[dict[str, Any]]
) -> None:
    """Guard against a double-submit from a double-clicking operator.

    Raises:
        ApprovalAlreadyResolvedError: If *approval_id* already has an entry.
    """
    for entry in previous:
        if entry.get("approval_id") == decision.approval_id:
            raise ApprovalAlreadyResolvedError(
                "this approval was already resolved",
                approval_id=decision.approval_id,
                previous_decision=entry.get("decision"),
            )


__all__ = [
    "APPROVAL_ID_PREFIX",
    "PAYLOAD_LOG_LIMIT",
    "approval_id_for",
    "assert_not_already_resolved",
    "build_approval_request",
    "decode_interrupt",
    "diff_preview",
    "encode_interrupt",
    "gate_payload_for_stage",
    "record_decision_log",
    "request_decision",
    "run_id_from_approval_id",
    "sign_decision",
    "verify_decision",
]
