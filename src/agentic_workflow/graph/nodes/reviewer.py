"""Code reviewer: audit the patch and decide whether the loop may close.

Two responsibilities beyond the LLM call:

* **Normalise the verdict.** A model that returns ``"approved"`` while listing
  three critical findings is a contradiction, not a judgement. This node
  reconciles the verdict with the findings rather than trusting the prose.
* **Gate to a human.** Low confidence or unresolved high-severity findings pause
  the graph instead of burning the remaining iteration budget.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agentic_workflow.domain.schemas import Finding, Patch, ReviewResult, Severity, Verdict
from agentic_workflow.domain.state import WorkflowState, as_model
from agentic_workflow.errors import ApprovalRejectedError
from agentic_workflow.graph.common import node
from agentic_workflow.graph.context import context_from_runtime
from agentic_workflow.graph.nodes.base import AgentNode
from agentic_workflow.human.policy import Stage
from agentic_workflow.logging import get_logger
from agentic_workflow.prompts import REVIEWER_SYSTEM, render_request

log = get_logger(__name__)

#: Severities that can block approval on their own.
BLOCKING_SEVERITIES: frozenset[Severity] = frozenset({Severity.HIGH, Severity.CRITICAL})


class ReviewerAgent(AgentNode):
    """Produce a reconciled :class:`ReviewResult` and escalate when unsure."""

    name: ClassVar[str] = "reviewer"
    role: ClassVar[str] = "code reviewer"
    owns: ClassVar[frozenset[str]] = frozenset({"review", "findings", "iteration"})
    system_prompt: ClassVar[str] = REVIEWER_SYSTEM

    async def __call__(self, state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
        """Review the current patch.

        Args:
            state: Workflow state.
            runtime: LangGraph runtime carrying the context.

        Returns:
            State update with ``review``, accumulated ``findings``, and either a
            human-gate decision log or a routing hint.
        """
        context = context_from_runtime(runtime)
        request = self.request(state)
        patch = as_model(state, "patch", Patch)
        iteration = int(state.get("iteration", 0)) + 1

        review = await self.ask(
            context,
            ReviewResult,
            render_request(request),
            f"iteration: {iteration}",
            f"candidate patch:\n{patch.diff[:8_000] if patch and patch.diff else '(empty)'}",
        )
        review = _reconcile(review, state)

        log.info(
            "reviewer.completed",
            run_id=request.run_id,
            verdict=review.verdict.value,
            findings=len(review.findings),
            blocking=review.blocking_count,
            confidence=review.confidence,
        )
        await context.publish(
            "reviewer.completed",
            run_id=request.run_id,
            verdict=review.verdict.value,
            findings=len(review.findings),
        )

        update: dict[str, Any] = {
            "review": review,
            "iteration": iteration,
            "findings": review.findings,
            "transcript": [self.trace(review.summary, verdict=review.verdict.value)],
        }

        # Escalate when the policy says a human is needed.
        gate = self.maybe_gate(
            context,
            stage=Stage.PATCH_REVIEW,
            review=review,
            iteration=iteration,
        )
        if gate.required:
            try:
                decision = await self.ask_human(
                    context,
                    state,
                    stage=Stage.PATCH_REVIEW,
                    title=f"Review iteration {iteration}: {review.verdict.value} "
                    f"({review.blocking_count} blocking)",
                    gate=gate,
                    payload=self.stage_payload(Stage.PATCH_REVIEW, review=review),
                    diff_preview=patch.diff[:4_000] if patch else "",
                    iteration=iteration,
                )
            except ApprovalRejectedError as exc:
                # A reviewer who rejects the patch ends the run; the rejection
                # becomes the verdict and the reporter documents it. The review
                # is passed explicitly because this node re-runs on resume and the
                # state it receives predates the review it is holding right here.
                return self.rejection_update(state, exc, stage=Stage.PATCH_REVIEW, review=review)
            update["human_decisions"] = [
                self.decision_log(state, decision, Stage.PATCH_REVIEW.value)
            ]
            if decision.decision.value == "edit":
                update.update(self.edited(decision))
        else:
            update["next_action"] = "continue" if review.verdict is not Verdict.APPROVED else "test"

        self.assert_owns(update)
        return update


def _reconcile(review: ReviewResult, state: WorkflowState) -> ReviewResult:
    """Force the verdict to agree with the findings.

    Three normalisations, in order of importance:

    1. ``approved`` with blocking findings is downgraded — a contradiction
       between prose and data always resolves in favour of the data.
    2. A finding whose id is missing from ``blocking_findings`` but is
       high/critical severity is promoted to blocking, so a model cannot
       accidentally wave through a critical issue.
    3. Already-fixed findings are dropped, so the loop terminates: a finding the
       patch demonstrably addressed must not be re-reported forever.
    """
    findings = list(review.findings)
    blocking = set(review.blocking_findings)

    promoted = [
        f.id for f in findings if f.severity in BLOCKING_SEVERITIES and f.id not in blocking
    ]
    if promoted:
        log.info("reviewer.promoted_findings", promoted=promoted)

    unresolved = _unresolved_ids(state)
    surviving = [f for f in findings if f.id not in unresolved]

    blocking = {f.id for f in surviving if f.id in blocking or f.id in promoted}
    verdict = review.verdict
    if blocking:
        if verdict is Verdict.APPROVED:
            log.warning("reviewer.verdict_downgraded", blocking=len(blocking))
        verdict = Verdict.CHANGES_REQUESTED
    elif verdict is Verdict.CHANGES_REQUESTED and surviving:
        verdict = Verdict.CHANGES_REQUESTED
    elif not surviving and not blocking:
        verdict = Verdict.APPROVED

    return review.model_copy(
        update={
            "findings": surviving,
            "blocking_findings": sorted(blocking),
            "verdict": verdict,
        }
    )


def _unresolved_ids(state: WorkflowState) -> set[str]:
    """Ids the previous iteration already carried forward as unresolved."""
    review = as_model(state, "review", ReviewResult)
    if review is None:
        return set()
    return {f.id for f in review.findings} - set(review.blocking_findings)


def blocking_findings(review: ReviewResult | None) -> list[Finding]:
    """Return the findings that block approval, or an empty list."""
    if review is None:
        return []
    blocking = set(review.blocking_findings)
    return [f for f in review.findings if f.id in blocking]


@node("reviewer")
async def reviewer_node(state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
    """LangGraph entry point for the code reviewer.

    Args:
        state: Current workflow state.
        runtime: LangGraph runtime.

    Returns:
        The agent's state update.
    """
    return await ReviewerAgent()(state, runtime)


__all__ = ["BLOCKING_SEVERITIES", "ReviewerAgent", "blocking_findings", "reviewer_node"]
