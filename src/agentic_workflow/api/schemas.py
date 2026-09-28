"""Pydantic request/response models for the control plane.

These are deliberately **separate** from the domain schemas in
:mod:`agentic_workflow.domain.schemas`. The domain models describe what the graph
consumes and produces; these describe what crosses a network boundary.

Keeping them apart means a wire-format change (renaming a field, adding a
deprecated alias) never silently rewrites the contract the agents exchange, and
vice versa. The cost is a mapping function per direction, which is cheap and
entirely mechanical.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_workflow.domain.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    Decision,
    FinalReport,
    NodeTiming,
    ReviewRequest,
    RunSummary,
    SourceFile,
)

#: Identifiers are interpolated into checkpoint thread keys and log fields, so the
#: wire layer enforces the same safe alphabet the domain does.
SafeId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]


class APIModel(BaseModel):
    """Base model for the wire format.

    ``extra="forbid"`` is a security decision, not a style one: a silently ignored
    typo in a client's payload turns "reviewer: alice" into "anonymous decision"
    without a word of warning.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class StartRunRequest(APIModel):
    """Body of ``POST /v1/runs``.

    Attributes:
        run_id: Caller-chosen unique id. Doubles as the checkpoint thread key.
        request_id: Business identifier (PR number, ticket, ...).
        title: One-line subject.
        description: Problem statement.
        language: Primary language of the change under review.
        files: Source files to review.
        acceptance_criteria: Conditions the report must satisfy.
        constraints: Hard limits the agents must respect.
        metadata: Correlation data propagated end-to-end.
        auto_resolve: Answer human gates automatically instead of parking.
        auto_decision: The verdict used when ``auto_resolve`` is set.
        max_gates: Safety bound on auto-resolved gates.
    """

    run_id: SafeId
    request_id: SafeId
    title: str = Field(min_length=1, max_length=256)
    description: str = Field(default="", max_length=20_000)
    language: str = Field(default="python", min_length=1, max_length=32)
    files: list[SourceFile] = Field(default_factory=list, max_length=200)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=50)
    constraints: list[str] = Field(default_factory=list, max_length=50)
    metadata: dict[str, Any] = Field(default_factory=dict)
    auto_resolve: bool = Field(
        default=False,
        description="Answer every human gate automatically and run to completion.",
    )
    auto_decision: Decision = Field(
        default=Decision.APPROVE,
        description="Verdict applied to each gate when `auto_resolve` is set.",
    )
    max_gates: int = Field(default=32, ge=1, le=1_000)

    def to_domain(self) -> ReviewRequest:
        """Project onto the immutable domain request."""
        return ReviewRequest(
            run_id=self.run_id,
            request_id=self.request_id,
            title=self.title,
            description=self.description,
            language=self.language,
            files=self.files,
            acceptance_criteria=self.acceptance_criteria,
            constraints=self.constraints,
            metadata=self.metadata,
        )


class ResolveApprovalRequest(APIModel):
    """Body of ``POST /v1/approvals/{approval_id}/resolve``.

    Attributes:
        decision: Approve, edit or reject.
        reviewer: Who decided. Recorded in the immutable audit log.
        comment: Free-text justification.
        payload: Replacement content, required when ``decision`` is ``edit``.
    """

    decision: Decision
    reviewer: str = Field(min_length=1, max_length=128)
    comment: str = Field(default="", max_length=8_000)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _edit_needs_content(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Accept an empty payload; the service enforces the edit/payload pairing."""
        return value


class ResumeRunRequest(APIModel):
    """Body of ``POST /v1/runs/{run_id}/resume``.

    The raw decision is accepted so the graph's own stale-answer guard stays the
    single authority on whether a decision belongs to the pending gate.
    """

    approval_id: str = Field(min_length=1, max_length=128)
    decision: Decision
    reviewer: str = Field(min_length=1, max_length=128)
    comment: str = Field(default="", max_length=8_000)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> ApprovalDecision:
        """Project onto the domain decision (unsigned; the graph verifies it)."""
        return ApprovalDecision(
            approval_id=self.approval_id,
            decision=self.decision,
            reviewer=self.reviewer,
            comment=self.comment,
            payload=self.payload,
        )


class ReplayRequest(APIModel):
    """Body of ``POST /v1/threads/{run_id}/replay``."""

    checkpoint_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=1_000)


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class HealthStatus(StrEnum):
    """Coarse health verdict."""

    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class HealthResponse(APIModel):
    """Body of ``/health/live`` and ``/health/ready``."""

    status: HealthStatus
    version: str
    environment: str
    checks: dict[str, Any] = Field(default_factory=dict)
    detail: str | None = None
    timestamp: datetime


class RunDetail(APIModel):
    """Full view of one run."""

    run_id: str
    status: str
    iteration: int
    next_node: str | None = None
    checkpoint_id: str | None = None
    pending_approval: ApprovalRequest | None = None
    report: FinalReport | None = None
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    timings: list[NodeTiming] = Field(default_factory=list)
    error: str | None = None
    is_parked: bool = False
    is_finished: bool = False
    usage: dict[str, float | int] = Field(
        default_factory=dict,
        description=(
            "LLM usage attributed to the operation that produced this view: "
            "the whole run for an auto-resolved start, that drive's share for "
            "a parked start or a resolve. Reads served straight from the "
            "checkpoint (``GET /v1/runs/{id}``) carry no attribution; use "
            "``GET /v1/runs/{id}/usage`` for the run-wide total to date."
        ),
    )


class RunListResponse(APIModel):
    """Body of ``GET /v1/runs``."""

    items: list[RunSummary] = Field(default_factory=list)
    count: int = 0
    total: int = 0
    limit: int = 50
    offset: int = 0


class ApprovalListResponse(APIModel):
    """Body of ``GET /v1/approvals``."""

    items: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class CheckpointResponse(APIModel):
    """One entry of a run's checkpoint history."""

    checkpoint_id: str
    step: int
    source: str
    next_nodes: list[str] = Field(default_factory=list)
    created_at: str | None = None
    pending_approval: str | None = None


class HistoryResponse(APIModel):
    """Body of ``GET /v1/threads/{run_id}/history``."""

    run_id: str
    items: list[CheckpointResponse] = Field(default_factory=list)
    count: int = 0


class ErrorBody(APIModel):
    """Machine-readable error payload.

    The shape is stable API surface: clients branch on ``code``, never on
    ``message``. ``retryable`` tells a client whether a retry can plausibly
    succeed, which keeps "the run is waiting for a human" distinguishable from
    "the run failed".
    """

    code: str
    message: str
    retryable: bool = False
    run_id: str | None = None
    thread_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(APIModel):
    """Envelope for every error the API returns."""

    error: ErrorBody
    request_id: str | None = None


class ResolutionResponse(APIModel):
    """Body returned after an approval is answered."""

    run_id: str
    status: str
    replayed: bool = False
    approval_id: str
    decision: str
    pending_approval: ApprovalRequest | None = None
    audit_verified: bool | None = None


__all__ = [
    "APIModel",
    "ApprovalListResponse",
    "CheckpointResponse",
    "ErrorBody",
    "ErrorResponse",
    "HealthResponse",
    "HealthStatus",
    "HistoryResponse",
    "ReplayRequest",
    "ResolutionResponse",
    "ResolveApprovalRequest",
    "ResumeRunRequest",
    "RunDetail",
    "RunListResponse",
    "SafeId",
    "StartRunRequest",
]
