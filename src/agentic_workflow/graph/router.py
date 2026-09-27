"""The router: a pure function that closes the feedback loop.

The router is deliberately *not* an agent. Keeping the loop-closing decision in
a deterministic, side-effect-free function has three benefits: it is trivially
testable, it cannot hallucinate a route, and the control flow of the whole system
fits in one readable table.

Routing table (``iteration`` is the number of *completed* loops)
----------------------------------------------------------------
The first three rows are *setup*: they can only fire on a fresh state, or before
the corresponding agent has produced its output.

=========================================  =========================
Condition                                  Destination
=========================================  =========================
no task brief                              ``triage``
no patch                                   ``programmer``
review not yet produced                    ``reviewer``
patch approved, not yet validated          ``tester``
approved and validated, not applied        ``apply_patch`` (human-gated)
approved, validated and applied            ``reporter``
review not approved                        ``programmer`` (feedback loop)
validation failed                          ``programmer`` (feedback loop)
patch rejected                             ``reporter`` with escalation
iteration exhausted                        ``reporter`` with escalation
=========================================  =========================

The loop is *bounded* twice: by ``settings.max_iterations`` and by the stall
detector in the programmer node. A cyclic graph with only one bound is a
runaway-cost bug waiting to happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentic_workflow.domain.schemas import Patch, ReviewResult, TestReport, Verdict, utcnow
from agentic_workflow.domain.state import WorkflowState
from agentic_workflow.graph.common import node
from agentic_workflow.graph.nodes import (
    APPLY_PATCH,
    PROGRAMMER,
    REPORTER,
    REVIEWER,
    ROUTER,
    TESTER,
    TRIAGE,
)
from agentic_workflow.logging import get_logger

log = get_logger(__name__)


class Route(StrEnum):
    """Possible destinations of the router."""

    TRIAGE = TRIAGE
    PROGRAMMER = PROGRAMMER
    REVIEWER = REVIEWER
    TESTER = TESTER
    APPLY_PATCH = APPLY_PATCH
    REPORTER = REPORTER
    END = "end"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The router's verdict plus the reasoning exposed to operators.

    Attributes:
        target: Node to execute next.
        reason: Human-readable justification, stored in the transcript.
        escalate: Whether the run should be flagged for human attention.
        exhausted: Whether the iteration budget has been consumed.
    """

    target: Route
    reason: str
    escalate: bool = False
    exhausted: bool = False

    def to_update(self) -> dict[str, Any]:
        """Render as a state update."""
        return {"next_action": self.target.value, "status": "routing"}


def decide_route(state: WorkflowState, *, max_iterations: int = 6) -> RoutingDecision:
    """Compute the next node from *state*.

    The key invariant this relies on: **the programmer clears ``test_report``
    whenever it writes a new patch**. So ``test_report is None`` is not "no tests
    exist" — it is precisely "the current patch has not been validated yet",
    which is what lets the router send an approved patch through the tester
    before anything irreversible happens to the repository.

    Args:
        state: Current workflow state.
        max_iterations: Iteration budget; reaching it forces the terminal route.

    Returns:
        The :class:`RoutingDecision` for the next hop.

    Example:
        --------
        >>> from agentic_workflow.domain import initial_state, ReviewRequest
        >>> s = initial_state(ReviewRequest(run_id="r", request_id="p", title="t"))
        >>> decide_route(s, max_iterations=6).target
        <Route.TRIAGE: 'triage'>
    """
    iteration = int(state.get("iteration", 0))

    if state.get("task_brief") is None:
        return RoutingDecision(Route.TRIAGE, "no task brief yet")

    from agentic_workflow.domain.state import as_model

    patch = as_model(state, "patch", Patch)
    if patch is None:
        return RoutingDecision(Route.PROGRAMMER, "no candidate patch yet")

    review = as_model(state, "review", ReviewResult)
    tests = as_model(state, "test_report", TestReport)

    if review is None:
        return RoutingDecision(Route.REVIEWER, "patch awaiting review")

    applied = bool(state.get("context", {}).get("patch_applied"))

    if review.verdict is Verdict.APPROVED:
        if patch.is_empty:
            return RoutingDecision(Route.REPORTER, "approved, but the patch is empty")
        if tests is None:
            # Never apply an unvalidated change: the tester is the objective
            # counterweight to the reviewer's opinion.
            return RoutingDecision(
                Route.TESTER, "review approved; validating the change before applying it"
            )
        if not tests.passed:
            return _repair_or_escalate(
                iteration,
                max_iterations,
                f"approved but validation failed ({tests.failed} test(s)); repairing",
                escalate=tests.failed > 3,
            )
        if not applied:
            return RoutingDecision(
                Route.APPLY_PATCH,
                "review approved and validated; applying the change (human-gated)",
            )
        return RoutingDecision(Route.REPORTER, "change applied; composing the report")

    if review.verdict is Verdict.REJECTED:
        return RoutingDecision(
            Route.REPORTER,
            "reviewer rejected the change; documenting the decision",
            escalate=True,
        )

    # --- feedback path: verdict is changes_requested / needs_human -------- #
    if tests is not None and not tests.passed:
        return _repair_or_escalate(
            iteration,
            max_iterations,
            f"validation failed ({tests.failed} test(s)); returning to the patch author",
            escalate=tests.failed > 3,
        )

    return _repair_or_escalate(
        iteration,
        max_iterations,
        f"review requested changes ({review.blocking_count} blocking); iteration {iteration + 1}",
        escalate=review.confidence < 0.5,
    )


def _repair_or_escalate(
    iteration: int,
    max_iterations: int,
    reason: str,
    *,
    escalate: bool,
) -> RoutingDecision:
    """Hand the work back to the patch author, unless the budget is spent.

    Every decision that routes to the programmer goes through here, and the
    budget lives here rather than in any one branch. That placement is the whole
    point: the three repair paths are reached by different conditions, so a check
    inside a single branch of :func:`decide_route` is only consulted when that
    branch's condition happens to hold. A failing test suite is the most common
    repair of all, and it is exactly the one that used to loop without end.

    A run with no iterations left stops and escalates instead. Continuing would
    spend money on a repair the configuration has already declined to buy, and
    the operator is the one who can decide whether to grant more.

    Args:
        iteration: Completed repair loops so far.
        max_iterations: The configured ceiling.
        reason: Why the work is going back, recorded in the transcript.
        escalate: Whether to flag the loop for human attention.

    Returns:
        A route to the programmer, or a terminal report when out of budget.
    """
    if iteration >= max_iterations:
        return RoutingDecision(
            Route.REPORTER,
            f"iteration budget exhausted after {iteration} loop(s); "
            "escalating to a human instead of looping again",
            escalate=True,
            exhausted=True,
        )
    return RoutingDecision(Route.PROGRAMMER, reason, escalate=escalate)


@node(ROUTER)
async def router_node(state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
    """LangGraph entry point for the router.

    The conditional edges read ``next_action`` from the returned update, so the
    router node itself never needs the LLM or the context.

    Args:
        state: Current workflow state.
        runtime: LangGraph runtime (unused).

    Returns:
        State update with ``next_action`` and a transcript entry.
    """
    from agentic_workflow.graph.context import context_from_runtime

    # The router must never be the reason a run fails: if the context is missing
    # or malformed we fall back to a conservative budget rather than propagating.
    # `decide_route` bounds the loop either way, so a wrong guess here can waste
    # iterations but cannot hang the graph.
    max_iterations = 3
    try:
        max_iterations = context_from_runtime(runtime).settings.max_iterations
    except Exception as exc:
        log.warning("router.context_unavailable", error=str(exc), fallback=max_iterations)

    decision = decide_route(state, max_iterations=max_iterations)
    log.info("router.decision", target=decision.target.value, reason=decision.reason)

    update: dict[str, Any] = {
        "next_action": decision.target.value,
        "transcript": [
            {
                "node": ROUTER,
                "role": "router",
                "summary": f"-> {decision.target.value}: {decision.reason}",
                "payload": {"exhausted": decision.exhausted, "escalate": decision.escalate},
                "at": utcnow().isoformat(),
            }
        ],
    }
    if decision.exhausted:
        update["context"] = {"iteration_exhausted": True}
    return update


#: Human-readable routing table, exported for the documentation generator.
#: Order matters: it is the literal evaluation order of :func:`decide_route`.
ROUTING_TABLE: tuple[tuple[str, str], ...] = (
    ("task_brief is None", Route.TRIAGE.value),
    ("patch is None", Route.PROGRAMMER.value),
    ("review is None", Route.REVIEWER.value),
    ("verdict is approved and patch is empty", Route.REPORTER.value),
    ("verdict is approved and test_report is None", Route.TESTER.value),
    ("verdict is approved and tests failed", Route.PROGRAMMER.value),
    ("verdict is approved, tests passed, not applied", Route.APPLY_PATCH.value),
    ("verdict is approved, tests passed, already applied", Route.REPORTER.value),
    ("verdict is rejected", Route.REPORTER.value),
    ("tests failed", Route.PROGRAMMER.value),
    ("iteration >= max_iterations", Route.REPORTER.value),
    ("otherwise (changes requested)", Route.PROGRAMMER.value),
)

__all__ = ["ROUTING_TABLE", "Route", "RoutingDecision", "decide_route", "router_node"]
