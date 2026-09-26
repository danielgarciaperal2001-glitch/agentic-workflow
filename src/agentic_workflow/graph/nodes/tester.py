"""Validation engineer: derive and report the tests that guard the change.

The tester is the objective counterweight to the reviewer: where the reviewer
opines, the tester measures. Its report is what allows the router to close the
loop on evidence rather than on a self-assessment.

In this reference implementation the tester *derives* the tests and reports a
structured verdict rather than executing a real test runner. Swapping in a real
executor is a single-node change — see the ``TesterAgent.run_tests`` extension
point documented on the class.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from agentic_workflow.domain.schemas import ReviewResult, TestReport
from agentic_workflow.domain.state import WorkflowState, as_model
from agentic_workflow.errors import ApprovalRejectedError
from agentic_workflow.graph.common import node
from agentic_workflow.graph.context import context_from_runtime
from agentic_workflow.graph.nodes.base import AgentNode
from agentic_workflow.human.policy import Stage
from agentic_workflow.logging import get_logger
from agentic_workflow.prompts import TESTER_SYSTEM, render_findings, render_request

log = get_logger(__name__)


class TestExecutor(Protocol):
    """Extension point for a real test runner.

    Implement this protocol to replace the LLM-derived report with an actual
    ``pytest``/``jest`` invocation, then inject it via
    :attr:`TesterAgent.executor`. The node is written so the executor is the only
    thing that needs to change.
    """

    async def __call__(self, state: WorkflowState) -> TestReport:  # pragma: no cover - protocol
        """Run the relevant tests and return a structured report."""
        ...


class TesterAgent(AgentNode):
    """Validate the patch and report a :class:`TestReport`.

    Attributes:
        executor: Optional real test runner. When ``None`` the report is derived
            by the LLM, which is appropriate for review-only pipelines.
    """

    name: ClassVar[str] = "tester"
    role: ClassVar[str] = "validation engineer"
    owns: ClassVar[frozenset[str]] = frozenset({"test_report"})
    system_prompt: ClassVar[str] = TESTER_SYSTEM

    executor: ClassVar[TestExecutor | None] = None

    async def __call__(self, state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
        """Run validation for the current patch.

        Args:
            state: Workflow state.
            runtime: LangGraph runtime carrying the context.

        Returns:
            State update with ``test_report`` and a routing hint.
        """
        context = context_from_runtime(runtime)
        request = self.request(state)
        review = as_model(state, "review", ReviewResult)
        iteration = int(state.get("iteration", 0))

        if self.executor is not None:
            report = await self.executor(state)
        else:
            report = await self.ask(
                context,
                TestReport,
                render_request(request),
                f"iteration: {iteration}",
                f"findings to guard against:\n{render_findings(review.findings) if review else '(none)'}",
            )

        log.info(
            "tester.completed",
            run_id=request.run_id,
            passed=report.passed,
            total=report.total,
            failed=report.failed,
            pass_rate=round(report.pass_rate, 3),
        )
        await context.publish(
            "tester.completed",
            run_id=request.run_id,
            passed=report.passed,
            failed=report.failed,
        )

        update: dict[str, Any] = {
            "test_report": report,
            "next_action": "route" if report.passed else "repair",
            "transcript": [self.trace(report.summary, pass_rate=round(report.pass_rate, 3))],
        }

        # A failing suite is a safety-relevant signal: gate it to a human rather
        # than letting the loop retry blindly.
        if not report.passed:
            gate = self.maybe_gate(
                context,
                stage=Stage.TEST_REVIEW,
                review=review,
                confidence=report.coverage,
                iteration=iteration,
            )
            if gate.required and report.failed > 0 and review is not None:
                try:
                    decision = await self.ask_human(
                        context,
                        state,
                        stage=Stage.TEST_REVIEW,
                        title=f"{report.failed} test(s) failing after iteration {iteration}",
                        gate=gate,
                        payload=self.stage_payload(Stage.TEST_REVIEW, tests=report, review=review),
                        iteration=iteration,
                    )
                except ApprovalRejectedError as exc:
                    # Refusing to keep repairing a change whose tests will not go
                    # green is a legitimate answer; the run ends with the refusal
                    # on the record rather than looping to the iteration ceiling.
                    return self.rejection_update(state, exc, stage=Stage.TEST_REVIEW)
                update["human_decisions"] = [
                    self.decision_log(state, decision, Stage.TEST_REVIEW.value)
                ]
                update.update(self.edited(decision))

        self.assert_owns(update)
        return update


@node("tester")
async def tester_node(state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
    """LangGraph entry point for the validation engineer.

    Args:
        state: Current workflow state.
        runtime: LangGraph runtime.

    Returns:
        The agent's state update.
    """
    return await TesterAgent()(state, runtime)


__all__ = ["TestExecutor", "TesterAgent", "tester_node"]
