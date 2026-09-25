"""Domain schemas: the typed vocabulary shared by every agent in the graph.

Agents exchange *structured* objects rather than free text. This module is the
single source of truth for that vocabulary. Keeping it in one place means:

* the prompts in :mod:`agentic_workflow.prompts` can be generated from the
  schemas (guaranteeing they never drift),
* the API can reuse the exact models the graph validates against,
* the evaluation suite can score a *typed* answer instead of parsing prose.

All models are strict (``extra="forbid"``) so a hallucinated field is an
immediate, attributable error rather than silent data corruption.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #
Confidence = Annotated[float, Field(ge=0.0, le=1.0, description="Calibrated 0-1 confidence.")]


def _path_parts(value: str) -> list[str]:
    """Split *value* into POSIX path components (no filesystem access)."""
    return [part for part in value.split("/") if part]


class StrictModel(BaseModel):
    """Base model rejecting unknown fields and normalising whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False,
        ser_json_timedelta="iso8601",
    )


def utcnow() -> datetime:
    """Timezone-aware current time. Centralised for deterministic test patching."""
    return datetime.now(UTC)


class Severity(StrEnum):
    """Finding severity, ordered from informational to critical."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric rank enabling ``max()`` aggregation across findings."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Verdict(StrEnum):
    """Outcome of a review or validation gate."""

    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"
    NEEDS_HUMAN = "needs_human"
    BLOCKED = "blocked"


class Category(StrEnum):
    """Coarse finding category used for metrics and dashboards."""

    SECURITY = "security"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    TESTING = "testing"
    ARCHITECTURE = "architecture"


class Decision(StrEnum):
    """Human verdict returned through a HITL gate."""

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class ReviewRequest(StrictModel):
    """The business request that kicks off a workflow run.

    This is the graph's *input* channel: everything else is derived.

    Attributes:
        run_id: Globally unique identifier, used as the checkpoint thread key.
        request_id: Business-level identifier (PR number, invoice id, ...).
        title: Human-readable subject line.
        description: Free-form problem statement supplied by the requester.
        language: Primary programming language of the change under review.
        files: Relevant source files with their content.
        acceptance_criteria: Conditions the final report must satisfy.
        constraints: Hard limits the agents must respect.
        metadata: Arbitrary correlation data propagated end-to-end.
    """

    run_id: str = Field(
        min_length=1,
        max_length=128,
        description="Unique run identifier; also the checkpoint thread key.",
    )
    request_id: str = Field(
        min_length=1,
        max_length=128,
        description="Business identifier such as a pull-request number.",
    )
    title: str = Field(min_length=1, max_length=256)
    description: str = Field(default="", max_length=20_000)
    language: str = Field(default="python", min_length=1, max_length=32)
    files: list[SourceFile] = Field(default_factory=list, max_length=200)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=50)
    constraints: list[str] = Field(default_factory=list, max_length=50)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "request_id")
    @classmethod
    def _validate_identifiers(cls, value: str) -> str:
        """Reject whitespace and path traversal in identifiers.

        Identifiers are interpolated into checkpoint keys and log fields; keeping
        them to a safe alphabet removes an entire class of injection bugs.
        """
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
            raise ValueError(
                f"invalid identifier {value!r}: only letters, digits and '.', '_', ':', '-' allowed"
            )
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """Stable SHA-256 over the reviewable content.

        Used for prompt caching, dedup of identical runs, and as an
        ``X-Content-Hash`` idempotency key on the REST API.
        """
        hasher = hashlib.sha256()
        for src in self.files:
            hasher.update(src.path.encode())
            hasher.update(b"\0")
            hasher.update(src.content.encode())
            hasher.update(b"\0")
        hasher.update(self.description.encode())
        return f"sha256:{hasher.hexdigest()[:32]}"


class SourceFile(StrictModel):
    """A single source file under review."""

    path: str = Field(min_length=1, max_length=512, description="Repo-relative POSIX path.")
    content: str = Field(default="", max_length=200_000, description="Full file content.")
    language: str | None = Field(default=None, max_length=32)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        """Reject absolute paths and parent traversal."""
        if value.startswith("/") or ".." in _path_parts(value):
            raise ValueError(f"unsafe path {value!r}: must be repo-relative without '..'")
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def line_count(self) -> int:
        """Number of lines, used for effort estimation and diff ratios."""
        return self.content.count("\n") + 1 if self.content else 0


# --------------------------------------------------------------------------- #
# Agent outputs
# --------------------------------------------------------------------------- #
class Finding(StrictModel):
    """A single issue raised by the reviewer agent.

    Attributes:
        id: Stable, content-derived identifier. Deterministic so re-running the
            same review produces the same id, which lets the evaluation suite
            diff runs meaningfully.
        title: Short imperative summary.
        detail: Explanation of the problem and why it matters.
        severity: Impact classification.
        category: Coarse grouping for dashboards.
        file: File the finding refers to, if any.
        line: 1-indexed line number, if any.
        recommendation: Concrete remediation advice.
        confidence: Reviewer's own confidence in the finding.
    """

    id: str = Field(default="", max_length=64)
    title: str = Field(min_length=1, max_length=256)
    detail: str = Field(default="", max_length=8_000)
    severity: Severity = Severity.MEDIUM
    category: Category = Category.CORRECTNESS
    file: str | None = Field(default=None, max_length=512)
    line: int | None = Field(default=None, ge=1)
    recommendation: str = Field(default="", max_length=4_000)
    confidence: Confidence = 0.5

    @field_validator("id", mode="before")
    @classmethod
    def _default_id(cls, value: Any) -> Any:
        """Keep a caller-supplied id, otherwise derive one after validation."""
        return value or ""

    def compute_id(self) -> str:
        """Derive a deterministic identifier from the finding's content.

        Returns:
            A 16-hex-character fingerprint of ``(file, line, title)``.
        """
        basis = f"{self.file or ''}:{self.line or 0}:{self.title.strip().lower()}"
        return hashlib.sha256(basis.encode()).hexdigest()[:16]

    def model_post_init(self, _context: Any) -> None:  # noqa: D105
        """Fill an empty ``id`` deterministically (see :meth:`compute_id`)."""
        if not self.id:
            # Bypass validate_assignment re-entry by writing to __dict__.
            self.__dict__["id"] = self.compute_id()


class Patch(StrictModel):
    """A proposed code change produced by the programmer agent.

    Attributes:
        diff: Unified diff text.
        files_changed: Paths touched by the diff.
        summary: One-paragraph rationale.
        strategy: High-level approach taken (``direct``, ``refactor``, ...).
        confidence: Author's confidence that the patch resolves the findings.
    """

    diff: str = Field(default="", max_length=200_000)
    files_changed: list[str] = Field(default_factory=list, max_length=200)
    summary: str = Field(default="", max_length=8_000)
    strategy: str = Field(default="direct", max_length=64)
    confidence: Confidence = 0.5

    @computed_field  # type: ignore[prop-decorator]
    @property
    def diff_size(self) -> int:
        """Number of changed lines (additions + removals) in the unified diff."""
        added = len(re.findall(r"^\+(?!\+\+)", self.diff, flags=re.MULTILINE))
        removed = len(re.findall(r"^-(?!!)", self.diff, flags=re.MULTILINE))
        return added + removed

    @property
    def is_empty(self) -> bool:
        """``True`` when the patch contains no actual change."""
        return not self.files_changed or self.diff_size == 0


class ReviewResult(StrictModel):
    """Output of the reviewer agent for one iteration."""

    verdict: Verdict = Verdict.CHANGES_REQUESTED
    findings: list[Finding] = Field(default_factory=list, max_length=200)
    summary: str = Field(default="", max_length=8_000)
    confidence: Confidence = 0.5
    blocking_findings: list[str] = Field(
        default_factory=list,
        max_length=200,
        description="Ids of findings that must be fixed before approval.",
    )

    @property
    def highest_severity(self) -> Severity:
        """Most severe finding severity, or ``INFO`` when there are none."""
        return max((f.severity for f in self.findings), key=lambda s: s.rank, default=Severity.INFO)

    @property
    def blocking_count(self) -> int:
        """Number of findings that block approval."""
        return len(self.blocking_findings)


class TestReport(StrictModel):
    """Output of the tester agent."""

    passed: bool = False
    total: int = Field(default=0, ge=0, le=100_000)
    failed: int = Field(default=0, ge=0, le=100_000)
    skipped: int = Field(default=0, ge=0, le=100_000)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    failing_tests: list[str] = Field(default_factory=list, max_length=200)
    generated_tests: list[str] = Field(default_factory=list, max_length=200)
    regressions: list[Finding] = Field(default_factory=list, max_length=200)
    summary: str = Field(default="", max_length=8_000)

    @property
    def pass_rate(self) -> float:
        """Fraction of executed tests that passed (1.0 when nothing ran)."""
        executed = self.total - self.skipped
        return 1.0 if executed == 0 else max(0.0, (executed - self.failed) / executed)


class TaskBrief(StrictModel):
    """Normalised task description produced by the triage analyst."""

    objective: str = Field(min_length=1, max_length=2_000)
    scope: list[str] = Field(default_factory=list, max_length=50)
    out_of_scope: list[str] = Field(default_factory=list, max_length=50)
    risks: list[str] = Field(default_factory=list, max_length=50)
    success_criteria: list[str] = Field(default_factory=list, max_length=50)
    estimated_effort: str = Field(default="unknown", max_length=64)
    confidence: Confidence = 0.5


class FinalReport(StrictModel):
    """Deliverable produced by the reporter agent.

    Attributes:
        report_id: Identifier derived from the run id for traceability.
        markdown: Human-readable report body.
        decision: Overall recommendation.
        findings: Findings that survived the full loop.
        metrics: Aggregate numbers consumed by the evaluation suite.
        citations: File/line references backing each claim. The faithfulness
            metric scores against this list, so agents must populate it.
    """

    report_id: str = Field(default="", max_length=128)
    markdown: str = Field(default="", max_length=100_000)
    decision: Verdict = Verdict.BLOCKED
    findings: list[Finding] = Field(default_factory=list, max_length=200)
    metrics: dict[str, float] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list, max_length=500)
    generated_at: datetime = Field(default_factory=utcnow)

    @field_validator("generated_at")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        """Normalise naive datetimes to UTC so comparisons are always safe."""
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    @property
    def claim_units(self) -> list[str]:
        """Split the report into sentence-like claims for faithfulness scoring."""
        return [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n{1,}", self.markdown) if c.strip()]


# --------------------------------------------------------------------------- #
# Human-in-the-Loop payloads
# --------------------------------------------------------------------------- #
class ApprovalRequest(StrictModel):
    """The payload a graph node surfaces when it pauses for a human.

    This object is (a) embedded in the interrupt value, (b) served over REST and
    (c) pushed over WebSocket, so it must be fully self-describing: a reviewer
    should be able to decide without reading the source code.

    Attributes:
        approval_id: Unique id for this gate.
        node: Graph node that requested approval.
        stage: Coarse stage label used for routing and dashboards.
        title: One-line summary shown in the approval inbox.
        rationale: Why the agent believes human judgement is required.
        payload: Node-specific data for the human to inspect.
        options: Available actions.
        confidence: Agent confidence; below the escalation threshold forces a gate.
        expires_at: UTC deadline after which the request is void.
    """

    approval_id: str = Field(default="", max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    node: str = Field(min_length=1, max_length=64)
    stage: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=256)
    rationale: str = Field(default="", max_length=4_000)
    payload: dict[str, Any] = Field(default_factory=dict)
    options: list[Decision] = Field(
        default_factory=lambda: [Decision.APPROVE, Decision.EDIT, Decision.REJECT],
        max_length=10,
    )
    confidence: Confidence = 0.0
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = Field(default=None)
    diff_preview: str = Field(default="", max_length=20_000)
    requested_by: str = Field(default="system", max_length=128)

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        node: str,
        stage: str,
        title: str,
        approval_id: str,
        **kwargs: Any,
    ) -> Self:
        """Build a request, generating a default expiry when none is given."""
        kwargs.setdefault("approval_id", approval_id)
        return cls(run_id=run_id, node=node, stage=stage, title=title, **kwargs)

    @property
    def is_expired(self) -> bool:
        """``True`` when the approval window has elapsed."""
        return self.expires_at is not None and utcnow() > self.expires_at

    def allowed_options(self, *, allow_edit: bool, allow_reject: bool) -> list[Decision]:
        """Filter the offered actions according to policy."""
        allowed = [Decision.APPROVE]
        if allow_edit:
            allowed.append(Decision.EDIT)
        if allow_reject:
            allowed.append(Decision.REJECT)
        return allowed


class ApprovalDecision(StrictModel):
    """A human's response to an :class:`ApprovalRequest`.

    Attributes:
        approval_id: Gate being resolved.
        decision: Approve, edit or reject.
        reviewer: Identity of the human who decided.
        comment: Free-text justification, retained in the audit log.
        payload: Edited values, only meaningful for :attr:`Decision.EDIT`.
        signature: HMAC over the canonical decision, when required by policy.
    """

    approval_id: str = Field(min_length=1, max_length=128)
    decision: Decision
    reviewer: str = Field(min_length=1, max_length=128)
    comment: str = Field(default="", max_length=8_000)
    payload: dict[str, Any] = Field(default_factory=dict)
    signature: str | None = Field(default=None, max_length=512)
    decided_at: datetime = Field(default_factory=utcnow)

    @field_validator("decided_at")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        """Normalise naive datetimes to UTC."""
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    def model_post_init(self, _context: Any) -> None:  # noqa: D105
        """Validate the decision/payload pairing on construction."""
        if self.decision is Decision.EDIT and not self.payload:
            raise ValueError("decision='edit' requires a non-empty `payload`")


# --------------------------------------------------------------------------- #
# Run-level bookkeeping
# --------------------------------------------------------------------------- #
class NodeTiming(StrictModel):
    """Wall-clock accounting for a single graph node invocation."""

    node: str
    started_at: datetime = Field(default_factory=utcnow)
    duration_ms: float = Field(default=0.0, ge=0.0)
    ok: bool = True
    error: str | None = None

    @field_validator("started_at")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        """Normalise naive datetimes to UTC."""
        return value if value.tzinfo else value.replace(tzinfo=UTC)


class RunSummary(StrictModel):
    """Immutable snapshot of a run returned to callers."""

    run_id: str
    status: str
    iteration: int = Field(default=0, ge=0)
    next_node: str | None = None
    updated_at: datetime = Field(default_factory=utcnow)
    checkpoint_id: str | None = None
    pending_approval: ApprovalRequest | None = None
    timings: list[NodeTiming] = Field(default_factory=list)
    error: str | None = None

    @field_validator("updated_at")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        """Normalise naive datetimes to UTC."""
        return value if value.tzinfo else value.replace(tzinfo=UTC)


__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "Category",
    "Confidence",
    "Decision",
    "Finding",
    "FinalReport",
    "NodeTiming",
    "Patch",
    "ReviewRequest",
    "ReviewResult",
    "RunSummary",
    "Severity",
    "SourceFile",
    "StrictModel",
    "TaskBrief",
    "TestReport",
    "Verdict",
    "utcnow",
]
