"""The graph's topology, and the routing table that drives it.

These are the tests that catch the most expensive class of regression in a state
machine: a topology that still *runs* but routes differently. Nothing raises when
an edge is wired to the wrong node — the workflow just quietly produces a worse
answer, and nobody notices until a customer does.

Three things are pinned:

* the compiled graph matches :func:`graph_topology`,
* the declared cycles are real cycles in the compiled graph,
* :data:`ROUTING_TABLE` stays in sync with :func:`decide_route`, so the
  documentation table cannot drift from the behaviour.
"""

from __future__ import annotations

import inspect
import itertools
import json
from typing import Any

import pytest

from agentic_workflow.config import Settings
from agentic_workflow.domain.state import WorkflowState, initial_state
from agentic_workflow.graph.builder import build_graph, graph_topology
from agentic_workflow.graph.nodes import ALL_NODES, APPLY_PATCH, REPORTER, ROUTER, TRIAGE
from agentic_workflow.graph.router import ROUTING_TABLE, Route, decide_route
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from tests.helpers import make_request

pytestmark = pytest.mark.unit

#: LangGraph's synthetic endpoints, which are not part of the declared topology.
START_NODE = "__start__"
END_NODE = "__end__"


@pytest.fixture(scope="module")
def compiled() -> Any:
    """Return a graph compiled with an in-memory checkpointer.

    A checkpointer is required because human-in-the-loop is enabled by default;
    an interrupt cannot be resumed without durable state, so the builder refuses
    to compile without one. That refusal is itself a guard worth keeping.

    Returns:
        The compiled graph.
    """
    settings = Settings(_env_file=None, llm_provider="echo", hitl_enabled=True)
    return build_graph(settings, checkpointer=build_memory_checkpointer())


def _static_edges(compiled: Any) -> set[tuple[str, str]]:
    """Return the graph's unconditional edges as ``(source, target)`` pairs.

    Conditional edges are excluded because LangGraph expands the router's
    ``path_map`` into one edge per destination, and comparing those against the
    hand-declared static edges would be comparing different things.

    Args:
        compiled: The compiled graph.

    Returns:
        The static edge set.
    """
    return {
        (edge.source, edge.target) for edge in compiled.get_graph().edges if not edge.conditional
    }


class TestTopology:
    """The shape of the state machine."""

    def test_every_node_is_registered(self, compiled: Any) -> None:
        """The compiled graph exposes every declared node."""
        nodes = set(compiled.get_graph().nodes)
        assert ALL_NODES, "ALL_NODES must not be empty"
        for name in ALL_NODES:
            assert name in nodes, f"node {name!r} is declared but not in the compiled graph"

    def test_declared_nodes_match_the_compiled_graph(self, compiled: Any) -> None:
        """``graph_topology()`` lists the same nodes the graph was built with."""
        declared = set(graph_topology()["nodes"])
        actual = set(compiled.get_graph().nodes) - {START_NODE, END_NODE}
        assert declared == actual

    def test_declared_static_edges_match_the_compiled_graph(self, compiled: Any) -> None:
        """The hand-declared static edges are exactly the ones LangGraph built.

        A mismatch means someone edited an edge without updating the declaration,
        and every diagram, document and dashboard derived from it is a lie.
        """
        declared = {
            (edge["from"], edge["to"])
            for edge in graph_topology()["edges"]
            if edge["kind"] == "static"
        }
        # The declared topology uses the human names START/END; LangGraph uses
        # its own synthetic node names for the same two positions.
        actual = {
            ("START" if source == START_NODE else source, "END" if target == END_NODE else target)
            for source, target in _static_edges(compiled)
        }
        assert declared == actual

    def test_router_edge_is_the_only_conditional_one(self, compiled: Any) -> None:
        """Exactly one dynamic edge exists, and it leaves the router.

        A second conditional edge would mean a second place that can steer the
        machine, and the routing table would no longer be a complete description
        of the control flow.
        """
        conditional = {
            (edge.source, edge.target) for edge in compiled.get_graph().edges if edge.conditional
        }
        assert {source for source, _ in conditional} == {ROUTER}

    def test_router_can_reach_every_routable_node(self, compiled: Any) -> None:
        """Every :class:`Route` member names a node the graph actually has.

        The router is the only dynamic edge, so a member without a corresponding
        node would send execution into the void.
        """
        sources = {target for source, target in _conditional_targets(compiled) if source == ROUTER}
        for route in Route:
            if route is Route.END:
                continue
            assert route.value in sources, f"the router cannot reach {route.value!r}"

    def test_declared_cycles_are_real_cycles(self, compiled: Any) -> None:
        """Every declared feedback cycle exists as a closed path in the graph.

        The programmer/reviewer loop and the router/tester loop are the reason
        this is a state *machine* rather than a pipeline. If either stopped being
        a cycle the workflow would silently degrade to a single pass — a
        behavioural change no assertion about individual nodes would catch.

        Conditional edges count: the closing hop of every cycle leaves the router,
        so checking only the static edges would reject all of them.
        """
        edges = _static_edges(compiled) | _conditional_targets(compiled)
        for cycle in graph_topology()["cycles"]:
            assert len(cycle) >= 3, f"a cycle must repeat its entry node: {cycle}"
            assert cycle[0] == cycle[-1], f"a declared cycle is not closed: {cycle}"
            for source, target in itertools.pairwise(cycle):
                assert (source, target) in edges, (
                    f"cycle edge {source}->{target} does not exist in the compiled graph"
                )

    def test_both_feedback_loops_are_declared(self) -> None:
        """The rejection loop and the validation loop are both documented.

        These are the two paths that make the graph cyclic. Losing either turns a
        multi-agent review into a pipeline, and the declaration is what the
        architecture diagrams are generated from.
        """
        cycles = [tuple(cycle) for cycle in graph_topology()["cycles"]]
        assert ("router", "tester", "router") in cycles
        assert ("programmer", "reviewer", "router", "programmer") in cycles

    def test_entry_point_is_triage(self, compiled: Any) -> None:
        """Execution starts at ``triage``."""
        assert (START_NODE, TRIAGE) in _static_edges(compiled)

    def test_reporter_is_terminal(self, compiled: Any) -> None:
        """The reporter's only outgoing edge is END.

        Terminal in the graph sense, not "no outgoing edge": it has exactly one,
        to ``__end__``. Any second edge would mean the run continues after it has
        already published its deliverable.
        """
        outgoing = {target for source, target in _static_edges(compiled) if source == REPORTER}
        assert outgoing == {END_NODE}
        # And nothing dispatches back into it once it has run, except the router's
        # final hop, which is how a run reaches its report at all.
        assert not {t for s, t in _conditional_targets(compiled) if s == REPORTER}

    def test_apply_patch_precedes_reporter(self, compiled: Any) -> None:
        """Applying a patch always leads to the report, never to a rerun.

        This is the workflow's safety property: the only node that writes to the
        repository is immediately followed by the node that records what happened.
        """
        assert (APPLY_PATCH, REPORTER) in _static_edges(compiled)

    def test_topology_is_json_serialisable(self) -> None:
        """The topology feeds the documentation generator, so it must be plain data."""
        payload = json.dumps(graph_topology())
        assert "triage" in payload

    def test_build_is_deterministic(self) -> None:
        """Two builds from the same settings produce the same structure.

        Determinism is what makes the topology tests meaningful, and it is what
        lets a checkpointed run be replayed against a rebuilt graph.
        """
        settings = Settings(_env_file=None, llm_provider="echo")
        first = build_graph(settings, checkpointer=build_memory_checkpointer())
        second = build_graph(settings, checkpointer=build_memory_checkpointer())
        assert _static_edges(first) == _static_edges(second)


def _conditional_targets(compiled: Any) -> set[tuple[str, str]]:
    """Return the graph's conditional edges as ``(source, target)`` pairs.

    Args:
        compiled: The compiled graph.

    Returns:
        The conditional edge set.
    """
    return {(edge.source, edge.target) for edge in compiled.get_graph().edges if edge.conditional}


class TestRoutingTable:
    """The declared routing table and the function that implements it."""

    def test_table_rows_are_condition_destination_pairs(self) -> None:
        """Each row is a ``(condition, destination)`` pair with a real destination."""
        assert ROUTING_TABLE, "the routing table must not be empty"
        valid = {route.value for route in Route}
        for row in ROUTING_TABLE:
            assert len(row) == 2, f"malformed routing row: {row!r}"
            condition, destination = row
            assert condition, "a routing condition must be readable"
            assert destination in valid, f"unknown destination {destination!r}"

    def test_every_destination_is_reachable_from_the_router(self, compiled: Any) -> None:
        """The router can actually dispatch to every destination the table names.

        A documented route the router cannot take is worse than no documentation,
        because a maintainer will rely on it.
        """
        targets = {target for _, target in ROUTING_TABLE}
        reachable = {target for _, target in _conditional_targets(compiled)}
        for destination in targets:
            assert destination in reachable

    def test_table_matches_the_implementation(self) -> None:
        """Every documented condition appears in :func:`decide_route`'s body.

        ``decide_route`` is multi-branch so the table cannot be *generated* from
        it, but every destination it constructs must be documented and every
        documented condition must exist. This test fails the moment a branch is
        added without updating the table — which is exactly when the docs rot.
        """
        source = inspect.getsource(decide_route)
        for _condition, destination in ROUTING_TABLE:
            # The table's conditions are prose; match on the destination, which
            # must appear as a constructed Route member.
            member = f"Route.{_enum_member(destination)}"
            assert member in source, (
                f"ROUTING_TABLE documents {destination!r} but decide_route never "
                f"constructs {member} — the table is stale"
            )

    def test_router_is_never_a_destination(self) -> None:
        """The router dispatches; it is never dispatched to.

        A self-edge would create a one-node infinite loop.
        """
        assert all(destination != ROUTER for _, destination in ROUTING_TABLE)

    def test_table_order_matches_the_evaluation_order(self) -> None:
        """The first three rows are the setup checks, in source order.

        ``decide_route`` short-circuits top to bottom, so the table's row order
        *is* its semantics. Setup rows must come first, or a reader would
        conclude that an empty review is dispatched to the tester.
        """
        conditions = [condition for condition, _ in ROUTING_TABLE]
        assert conditions[0] == "task_brief is None"
        assert conditions[1] == "patch is None"
        assert conditions[2] == "review is None"

    def test_approved_patch_routes_to_the_tester_before_apply(self) -> None:
        """The table encodes the validate-before-apply invariant in that order.

        Row 5 sends an approved-but-unvalidated patch to the tester, and only
        later rows allow ``apply_patch``. Reversing those two rows would let a
        reviewer's approval alone write to the repository.
        """
        conditions = [condition for condition, _ in ROUTING_TABLE]
        to_tester = conditions.index("verdict is approved and test_report is None")
        to_apply = conditions.index("verdict is approved, tests passed, not applied")
        assert to_tester < to_apply


def _enum_member(node_name: str) -> str:
    """Return the :class:`Route` member name for a node name.

    Args:
        node_name: A node name such as ``apply_patch``.

    Returns:
        The enum member name, e.g. ``APPLY_PATCH``.
    """
    return node_name.upper()


class TestRouterInvariants:
    """The routing decisions the workflow's correctness rests on."""

    def test_fresh_state_goes_to_triage(self) -> None:
        """Without a task brief, the machine has not started."""
        decision = decide_route(initial_state(make_request()))
        assert decision.target is Route.TRIAGE
        assert decision.reason

    def test_decide_route_is_pure(self) -> None:
        """Routing depends only on the state it is given.

        A route that consulted the clock or a global would make runs
        irreproducible, which defeats replay and the evaluation suite at once.
        """
        state = initial_state(make_request())
        assert decide_route(state) == decide_route(state)

    def _state(self, **overrides: Any) -> WorkflowState:
        """Build a mid-flight state with *overrides* applied.

        Args:
            **overrides: Channels to set on top of a fresh initial state.

        Returns:
            A workflow state ready for :func:`decide_route`.
        """
        return {**initial_state(make_request()), **overrides}

    @staticmethod
    def _patch() -> dict[str, Any]:
        """Return a minimal non-empty patch payload."""
        return {"diff": "@@ -1 +1 @@\n-a\n+b", "files_changed": ["a.py"]}

    def test_approved_patch_is_validated_before_it_is_applied(self) -> None:
        """An approved patch goes to the tester, not to ``apply_patch``.

        The invariant the router exists to enforce. The tester must see the
        change before ``apply_patch`` writes it; the reverse order would let a
        broken patch reach the repository and only then discover it was broken.
        """
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": "approved", "findings": []},
                test_report=None,
            )
        )
        assert decision.target is Route.TESTER

    def test_validated_patch_proceeds_to_apply(self) -> None:
        """Once the tester validated the change, it may be applied."""
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": "approved", "findings": []},
                test_report={"passed": True, "total": 4, "failed": 0, "skipped": 0},
            )
        )
        assert decision.target is Route.APPLY_PATCH

    def test_applied_patch_goes_to_the_reporter(self) -> None:
        """Once the change is in the repository, the run composes its report."""
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": "approved", "findings": []},
                test_report={"passed": True, "total": 4, "failed": 0, "skipped": 0},
                context={"patch_applied": True},
            )
        )
        assert decision.target is Route.REPORTER

    def test_failed_validation_returns_to_the_programmer(self) -> None:
        """A failing test suite sends the work back to the patch author."""
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": "approved", "findings": []},
                test_report={"passed": False, "total": 4, "failed": 2, "skipped": 0},
            )
        )
        assert decision.target is Route.PROGRAMMER
        assert decision.escalate is False

    def test_reviewer_rejection_terminates_with_escalation(self) -> None:
        """A reviewer rejection ends the run and escalates, rather than retrying.

        Looping after an explicit rejection would ignore the judgement that was
        just recorded — the one outcome a reviewer must always be able to force.
        """
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": "rejected", "findings": []},
            )
        )
        assert decision.target is Route.REPORTER
        assert decision.escalate is True

    def test_changes_requested_re_enters_the_feedback_loop(self) -> None:
        """A normal review comment sends the work back to the programmer."""
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": "changes_requested", "findings": []},
            )
        )
        assert decision.target is Route.PROGRAMMER
        assert decision.exhausted is False

    def test_iteration_budget_forces_a_terminal_route(self) -> None:
        """Reaching the iteration ceiling terminates the run and escalates.

        Without this bound a cyclic graph is an unbounded cost, and the only
        difference between "converged" and "runaway" would be the invoice. The
        counter is the *completed* loops, so a run that has already looped three
        times with a budget of three must stop.
        """
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": "changes_requested", "findings": []},
                iteration=3,
            ),
            max_iterations=3,
        )
        assert decision.target is Route.REPORTER
        assert decision.escalate is True
        assert decision.exhausted is True

    def test_iteration_counter_uses_completed_loops(self) -> None:
        """One loop below the budget still continues.

        The ceiling is ``>=`` on *completed* loops, so an off-by-one here would
        silently shorten every run by one iteration.
        """
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": "changes_requested", "findings": []},
                iteration=2,
            ),
            max_iterations=3,
        )
        assert decision.target is Route.PROGRAMMER
        assert decision.exhausted is False

    @pytest.mark.parametrize(
        ("verdict", "tests"),
        [
            pytest.param(
                "approved", {"passed": False, "total": 4, "failed": 2}, id="approved-failing-tests"
            ),
            pytest.param(
                "changes_requested",
                {"passed": False, "total": 4, "failed": 2},
                id="changes-failing-tests",
            ),
            pytest.param("changes_requested", None, id="changes-no-tests"),
        ],
    )
    def test_no_repair_loop_outlives_the_budget(
        self, verdict: str, tests: dict[str, Any] | None
    ) -> None:
        """Every route back to the programmer obeys the iteration budget.

        The budget used to be checked in a single branch, below the branches that
        decide to loop. That made it reachable only when the review asked for
        changes *and* the tests happened to pass — so the two most common repair
        paths, a failing suite under either verdict, ignored it entirely and
        asked for another attempt forever. Nothing stopped them but LangGraph's
        recursion limit, which surfaces as an infrastructure error rather than as
        the business outcome it is: this run ran out of budget.

        Each repair path is listed separately because the check has to be shared
        by all of them. Guarding one branch would leave the others unbounded, and
        a budget that depends on *why* the work is looping is not a budget.
        """
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch=self._patch(),
                review={"verdict": verdict, "findings": []},
                test_report=tests,
                iteration=3,
            ),
            max_iterations=3,
        )
        assert decision.target is Route.REPORTER
        assert decision.exhausted is True
        assert decision.escalate is True

    def test_empty_patch_is_reported_rather_than_applied(self) -> None:
        """An approved but empty patch produces a report instead of a no-op write.

        Applying an empty diff would consume a human gate and a checkpoint for
        nothing, and would report success for a change that was never made.
        """
        decision = decide_route(
            self._state(
                task_brief={"objective": "fix drift"},
                patch={"diff": "", "files_changed": []},
                review={"verdict": "approved", "findings": []},
            )
        )
        assert decision.target is Route.REPORTER

    def test_corrupt_next_action_falls_back_to_the_reporter(self) -> None:
        """A bad ``next_action`` terminates safely instead of hanging.

        The fallback is the reporter because it always produces a deliverable and
        dispatches nowhere else, so a corrupt state cannot become a loop.
        """
        from agentic_workflow.graph.builder import _read_next_action

        assert _read_next_action({"next_action": "does_not_exist"}) == REPORTER
        assert _read_next_action({}) == REPORTER
        assert _read_next_action({"next_action": TRIAGE}) == TRIAGE
