"""Policy engine deciding *when* a human must be involved.

Separating "should we ask?" from "how do we ask?" is what keeps HITL systems
maintainable: :mod:`agentic_workflow.human.policy` is pure, deterministic and
unit-testable, while :mod:`agentic_workflow.human.gates` contains the
LangGraph-specific interrupt plumbing.

Escalation triggers
-------------------
===========================  ==========================================
Trigger                      Rationale
===========================  ==========================================
``ALWAYS``                   Irreversible action (applying a patch).
``LOW_CONFIDENCE``           Agent self-reported confidence below threshold.
``HIGH_SEVERITY``            A critical/high finding is still unresolved.
``POLICY``                   Operator-configured rule, e.g. always gate.
``NEVER``                    Autonomous batch processing.
===========================  ==========================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from agentic_workflow.config import Settings
from agentic_workflow.domain.schemas import (
    Decision,
    ReviewResult,
    Severity,
    Verdict,
)
from agentic_workflow.logging import get_logger

log = get_logger(__name__)


class EscalationReason(StrEnum):
    """Why a human gate was raised. Persisted and surfaced in the UI."""

    ALWAYS = "always"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_SEVERITY = "high_severity"
    POLICY = "policy"
    NEVER = "never"
    DISABLED = "disabled"
    AGENT_REQUESTED = "agent_requested"


class Stage(StrEnum):
    """The pipeline stages that can expose a human gate.

    The value is what the UI routes on, so it is part of the public contract.
    """

    TRIAGE = "triage"
    PATCH_REVIEW = "patch_review"
    PATCH_APPLY = "patch_apply"
    TEST_REVIEW = "test_review"
    FINAL_REPORT = "final_report"
    ON_CALL = "on_call"


#: Stages whose action is hard to reverse. These always require a human.
IRREVERSIBLE_STAGES: frozenset[Stage] = frozenset({Stage.PATCH_APPLY, Stage.FINAL_REPORT})


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Outcome of evaluating the escalation policy for one stage.

    Attributes:
        required: Whether execution must pause.
        reason: Why the decision was made.
        severity: Highest severity observed, used to justify escalation.
        confidence: Lowest confidence observed across agents.
        rationale: Human-readable explanation shown in the approval inbox.
        metadata: Extra context persisted with the approval request.
    """

    required: bool
    reason: EscalationReason
    severity: Severity = Severity.INFO
    confidence: float = 0.0
    rationale: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def skip(cls, reason: EscalationReason = EscalationReason.NEVER) -> GateDecision:
        """Build a "no human needed" decision."""
        return cls(required=False, reason=reason, rationale="autonomous execution permitted")


@dataclass(frozen=True, slots=True)
class EscalationPolicy:
    """Configurable rules mapping run state to a human gate.

    The policy is evaluated *before* every potentially expensive or irreversible
    node, which means a run that never crosses a threshold completes with zero
    human interaction — exactly what you want for high-volume batch processing.
    """

    enabled: bool = True
    confidence_threshold: float = 0.7
    escalate_severity: frozenset[Severity] = frozenset({Severity.HIGH, Severity.CRITICAL})
    always_gate_stages: frozenset[Stage] = IRREVERSIBLE_STAGES
    allow_edit: bool = True
    allow_reject: bool = True
    default_timeout_seconds: float = 86_400.0

    @classmethod
    def from_settings(cls, settings: Settings) -> EscalationPolicy:
        """Build a policy from application settings."""
        always = set(IRREVERSIBLE_STAGES)
        if settings.hitl_require_approval_before_apply:
            always.add(Stage.PATCH_APPLY)
        return cls(
            enabled=settings.hitl_enabled,
            confidence_threshold=settings.hitl_escalation_threshold,
            allow_edit=settings.hitl_allow_edit,
            allow_reject=settings.hitl_allow_reject,
            default_timeout_seconds=settings.hitl_default_timeout_seconds,
            always_gate_stages=frozenset(always),
        )

    # --------------------------------------------------------- evaluation #
    def evaluate(
        self,
        stage: Stage,
        *,
        review: ReviewResult | None = None,
        confidence: float | None = None,
        iteration: int = 0,
        max_iterations: int = 1,
    ) -> GateDecision:
        """Decide whether *stage* must pause for a human.

        Args:
            stage: The stage about to execute.
            review: Latest review result, used for severity/confidence signals.
            confidence: Explicit confidence override for the pending node.
            iteration: Current loop iteration.
            max_iterations: Configured iteration cap.

        Returns:
            A :class:`GateDecision` describing whether to pause and why.

        Note:
            The order of the checks is deliberate: an explicit ``ALWAYS`` gate
            wins over a low-confidence heuristic, because a policy gate is a
            deliberate operator decision and must not be silently downgraded.
        """
        if not self.enabled:
            return GateDecision(
                required=False,
                reason=EscalationReason.DISABLED,
                rationale="human-in-the-loop disabled by configuration",
            )

        observed_confidence = confidence
        if observed_confidence is None:
            observed_confidence = min(
                (c for c in (review.confidence if review else None,) if c is not None),
                default=1.0,
            )

        severity = review.highest_severity if review else Severity.INFO

        # 1. Explicit, irreversible-stage gate.
        if stage in self.always_gate_stages:
            return GateDecision(
                required=True,
                reason=EscalationReason.ALWAYS,
                severity=severity,
                confidence=observed_confidence,
                rationale=(
                    f"stage {stage.value!r} is irreversible or explicitly gated by "
                    "operator policy; a human authorises the action"
                ),
                metadata={"stage": stage.value, "iteration": iteration},
            )

        # 2. Agent confidence below threshold.
        if observed_confidence < self.confidence_threshold:
            return GateDecision(
                required=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                severity=severity,
                confidence=observed_confidence,
                rationale=(
                    f"agent confidence {observed_confidence:.2f} is below the "
                    f"{self.confidence_threshold:.2f} escalation threshold"
                ),
                metadata={"stage": stage.value, "threshold": self.confidence_threshold},
            )

        # 3. Unresolved high-severity findings on a human-facing stage.
        if stage in {Stage.PATCH_REVIEW, Stage.TEST_REVIEW, Stage.ON_CALL}:
            blocking_high = [
                f
                for f in (review.findings if review else [])
                if f.severity in self.escalate_severity
                and (not review or f.id in set(review.blocking_findings))
            ]
            if blocking_high:
                return GateDecision(
                    required=True,
                    reason=EscalationReason.HIGH_SEVERITY,
                    severity=severity,
                    confidence=observed_confidence,
                    rationale=(
                        f"{len(blocking_high)} blocking finding(s) at severity "
                        f"{severity.value} remain unresolved"
                    ),
                    metadata={"stage": stage.value, "findings": [f.id for f in blocking_high]},
                )

        # 4. Final report on a run that never reached approval.
        if stage is Stage.FINAL_REPORT and review and review.verdict is not Verdict.APPROVED:
            return GateDecision(
                required=True,
                reason=EscalationReason.POLICY,
                severity=severity,
                confidence=observed_confidence,
                rationale=(
                    f"run finished with verdict {review.verdict.value!r} rather than "
                    "'approved'; a human must sign off on the report"
                ),
                metadata={"stage": stage.value, "verdict": review.verdict.value},
            )

        # 5. Loop exhaustion: a run that burned its budget is a human problem.
        if iteration >= max_iterations and stage in {Stage.PATCH_REVIEW, Stage.ON_CALL}:
            return GateDecision(
                required=True,
                reason=EscalationReason.POLICY,
                severity=severity,
                confidence=observed_confidence,
                rationale=(
                    f"feedback loop exhausted after {iteration} iteration(s); escalating "
                    "instead of looping again"
                ),
                metadata={"stage": stage.value, "iteration": iteration},
            )

        return GateDecision.skip()


def allowed_decisions(policy: EscalationPolicy) -> list[Decision]:
    """Return the actions a human may take under *policy*."""
    options = [Decision.APPROVE]
    if policy.allow_edit:
        options.append(Decision.EDIT)
    if policy.allow_reject:
        options.append(Decision.REJECT)
    return options


def summarise_review(review: ReviewResult | None) -> str:
    """One-line human summary of a review, used in approval titles."""
    if review is None:
        return "no review available"
    blocking = review.blocking_count
    return (
        f"verdict={review.verdict.value} "
        f"findings={len(review.findings)} "
        f"blocking={blocking} "
        f"highest={review.highest_severity.value} "
        f"confidence={review.confidence:.2f}"
    )


def highest_severity(findings: Sequence[object]) -> Severity:
    """Return the most severe severity in *findings* (``INFO`` when empty)."""
    severities = [getattr(f, "severity", Severity.INFO) for f in findings]
    return max(severities, key=lambda s: Severity(s).rank, default=Severity.INFO)


__all__ = [
    "IRREVERSIBLE_STAGES",
    "EscalationPolicy",
    "EscalationReason",
    "GateDecision",
    "Stage",
    "allowed_decisions",
    "highest_severity",
    "summarise_review",
]
