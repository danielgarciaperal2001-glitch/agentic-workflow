"""Report writer: compose the human-facing deliverable.

This node is where the pipeline's quality is actually judged: the report is the
product. Two properties are enforced here rather than trusted from the model:

* **Citations are verified.** Every ``file:line`` claim in the markdown must
  exist in the state. The native faithfulness metric (:mod:`agentic_workflow.
  evals.metrics`) reports the ratio, and the node refuses to ship a report whose
  groundedness is below the configured floor.
* **Metrics are computed, not narrated.** Token counts, iteration count and
  finding statistics come from the state, so two reports of the same run are
  always comparable.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agentic_workflow.domain.schemas import (
    FinalReport,
    Finding,
    Patch,
    ReviewRequest,
    ReviewResult,
    TestReport,
    Verdict,
    utcnow,
)
from agentic_workflow.domain.state import WorkflowState, as_model, as_models
from agentic_workflow.graph.common import node
from agentic_workflow.graph.context import context_from_runtime
from agentic_workflow.graph.nodes.base import AgentNode
from agentic_workflow.human.policy import Stage
from agentic_workflow.logging import get_logger
from agentic_workflow.prompts import (
    REPORTER_SYSTEM,
    render_request,
    render_review,
    render_test_report,
)

log = get_logger(__name__)

#: Minimum groundedness (fraction of verifiable claims) for a report to ship.
#: Below this the report is still produced, but marked ``NEEDS_HUMAN`` so the
#: gate raises and a person reviews the claims.
MIN_GROUNDEDNESS = 0.5


class ReporterAgent(AgentNode):
    """Compose the final :class:`FinalReport` and verify its evidence."""

    name: ClassVar[str] = "reporter"
    role: ClassVar[str] = "report writer"
    owns: ClassVar[frozenset[str]] = frozenset({"report"})
    system_prompt: ClassVar[str] = REPORTER_SYSTEM

    async def __call__(self, state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
        """Write, verify and route the final report.

        Args:
            state: Workflow state.
            runtime: LangGraph runtime carrying the context.

        Returns:
            State update with ``report``, ``status`` and the human-gate log.
        """
        context = context_from_runtime(runtime)
        request = self.request(state)
        review = as_model(state, "review", ReviewResult)
        tests = as_model(state, "test_report", TestReport)
        patch = as_model(state, "patch", Patch)
        iteration = int(state.get("iteration", 0))

        report = await self.ask(
            context,
            FinalReport,
            render_request(request),
            f"iteration: {iteration}",
            f"review:\n{render_review(review) if review else '(none)'}",
            f"tests:\n{render_test_report(tests) if tests else '(none)'}",
            "Cite every claim as `path:line` and list them in `citations`.",
        )

        report = _finalise(report, state, run_id=request.run_id)

        groundedness = grounded_claim_ratio(report, state)

        # The model does not get to decide how well-grounded its own report is.
        # Below the floor we still ship the report — an operator needs to see it —
        # but we downgrade the verdict so the escalation policy raises a gate and
        # a person checks the claims by hand.
        if groundedness < MIN_GROUNDEDNESS and report.decision is Verdict.APPROVED:
            log.warning(
                "reporter.low_groundedness",
                run_id=request.run_id,
                groundedness=round(groundedness, 3),
                floor=MIN_GROUNDEDNESS,
                citations=len(report.citations),
            )
            report = report.model_copy(
                update={
                    "decision": Verdict.NEEDS_HUMAN,
                    "metrics": {**report.metrics, "groundedness": round(groundedness, 3)},
                }
            )
        else:
            report = report.model_copy(
                update={"metrics": {**report.metrics, "groundedness": round(groundedness, 3)}}
            )

        log.info(
            "reporter.completed",
            run_id=request.run_id,
            decision=report.decision.value,
            findings=len(report.findings),
            groundedness=round(groundedness, 3),
        )
        await context.publish(
            "reporter.completed",
            run_id=request.run_id,
            decision=report.decision.value,
            groundedness=round(groundedness, 3),
        )

        update: dict[str, Any] = {
            "report": report,
            "status": "completed",
            "next_action": "end",
            "pending_approval": None,
            "transcript": [self.trace(f"decision={report.decision.value}")],
        }

        gate = self.maybe_gate(
            context,
            stage=Stage.FINAL_REPORT,
            review=review,
            confidence=min(groundedness, report.metrics.get("confidence", 1.0)),
            iteration=iteration,
        )
        if gate.required:
            decision = await self.ask_human(
                context,
                state,
                stage=Stage.FINAL_REPORT,
                title=f"Sign off report {report.report_id} ({report.decision.value})",
                gate=gate,
                payload=self.stage_payload(Stage.FINAL_REPORT, report=report, review=review),
                diff_preview=(patch.diff[:4_000] if patch else ""),
                iteration=iteration,
            )
            update["human_decisions"] = [
                self.decision_log(state, decision, Stage.FINAL_REPORT.value)
            ]
            if decision.decision.value == "edit":
                update.update(self.edited(decision))

        self.assert_owns(update)
        return update


# --------------------------------------------------------------------------- #
# Verification helpers (also used by the evaluation suite)
# --------------------------------------------------------------------------- #
def _finalise(report: FinalReport, state: WorkflowState, *, run_id: str) -> FinalReport:
    """Fill in derived fields and recompute metrics from the state.

    Nothing here trusts the model's own numbers: the decision, the finding list
    and every metric are recomputed from the authoritative state.
    """
    review = as_model(state, "review", ReviewResult)
    tests = as_model(state, "test_report", TestReport)
    findings = as_models(state, "findings", Finding)

    decision = _decide(review, tests)
    metrics = {
        "iterations": float(state.get("iteration", 0)),
        "findings_total": float(len(findings)),
        "findings_blocking": float(review.blocking_count if review else 0),
        "human_decisions": float(len(state.get("human_decisions") or [])),
        "tests_pass_rate": float(tests.pass_rate) if tests else 0.0,
        "test_coverage": float(tests.coverage) if tests else 0.0,
    }
    highest = max((f.severity for f in findings), key=lambda s: s.rank, default=None)
    if highest is not None:
        metrics["highest_severity_rank"] = float(highest.rank)

    markdown = report.markdown.rstrip() or _fallback_markdown(state)
    citations = _merge_citations(report.citations, findings)

    return report.model_copy(
        update={
            "report_id": report.report_id or f"rpt_{run_id}",
            "markdown": markdown,
            "decision": decision,
            "findings": findings,
            "metrics": metrics,
            "citations": citations,
            "generated_at": utcnow(),
        }
    )


def _decide(review: ReviewResult | None, tests: TestReport | None) -> Verdict:
    """Derive the overall decision from the objective signals."""
    if review is None:
        return Verdict.BLOCKED
    if review.verdict is Verdict.REJECTED:
        return Verdict.REJECTED
    if review.blocking_count > 0:
        return Verdict.CHANGES_REQUESTED
    if tests is not None and not tests.passed:
        return Verdict.BLOCKED
    if review.verdict is Verdict.APPROVED:
        return Verdict.APPROVED
    return Verdict.NEEDS_HUMAN


def _merge_citations(citations: list[str], findings: list[Finding]) -> list[str]:
    """Union the model's citations with the locations the findings reference.

    Order is preserved and duplicates removed, so the citation list is a stable,
    diffable artefact across runs.
    """
    out: list[str] = []
    seen: set[str] = set()
    for citation in [*citations, *(f"{f.file}:{f.line}" for f in findings if f.file)]:
        text = str(citation).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out[:500]


def _fallback_markdown(state: WorkflowState) -> str:
    """Deterministic report body used when the model returned nothing.

    A pipeline must never ship an empty report: an operator needs *something* to
    look at, even if the writer failed.
    """
    review = as_model(state, "review", ReviewResult)
    tests = as_model(state, "test_report", TestReport)
    findings = as_models(state, "findings", Finding)
    lines = [
        "# Automated Review Report",
        "",
        "The report writer produced no narrative; the structured result follows.",
        "",
        "## Decision",
        _decide(review, tests).value,
        "",
        "## Findings",
    ]
    lines += [f"- [{f.severity.value}] {f.title} ({f.file or 'n/a'})" for f in findings] or [
        "- none"
    ]
    if tests is not None:
        lines += ["", "## Validation", f"- pass rate: {tests.pass_rate:.1%}"]
    return "\n".join(lines)


def extract_citations(text: str) -> list[str]:
    """Pull ``path:line`` references out of free text.

    Args:
        text: Markdown or prose potentially containing references.

    Returns:
        Deduplicated citation strings, in first-seen order.
    """
    import re

    found = re.findall(r"([\w./-]+\.[A-Za-z0-9]+):(\d+)", text)
    out: list[str] = []
    for path, line in found:
        ref = f"{path}:{line}"
        if ref not in out:
            out.append(ref)
    return out


def grounded_claim_ratio(report: FinalReport, state: WorkflowState) -> float:
    """Fraction of the report's citations that resolve against the request.

    Args:
        report: The generated report.
        state: Workflow state holding the source files.

    Returns:
        A float in ``[0, 1]``. Returns ``1.0`` when the report makes no
        verifiable claims, because an empty claim set cannot be unfaithful.
    """
    citations = report.citations or extract_citations(report.markdown)
    if not citations:
        return 1.0

    request = as_model(state, "request", ReviewRequest)
    if request is None:
        return 0.0

    known: dict[str, int] = {src.path: src.line_count for src in request.files}
    if not known:
        return 0.0

    verified = 0
    for citation in citations:
        path, _, line = citation.rpartition(":")
        if path in known and line.isdigit() and int(line) <= max(1, known[path]):
            verified += 1
    return verified / len(citations)


@node("reporter")
async def reporter_node(state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
    """LangGraph entry point for the report writer.

    Args:
        state: Current workflow state.
        runtime: LangGraph runtime.

    Returns:
        The agent's state update.
    """
    return await ReporterAgent()(state, runtime)


__all__ = [
    "MIN_GROUNDEDNESS",
    "ReporterAgent",
    "extract_citations",
    "grounded_claim_ratio",
    "reporter_node",
]
