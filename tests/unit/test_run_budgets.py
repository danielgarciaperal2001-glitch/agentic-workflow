"""The two budgets on a run, and why one has to clear the other.

``max_iterations`` is a business rule: the router enforces it and the outcome is
a run that asks for a human. LangGraph's recursion limit is a process-level
backstop against a graph that is not converging at all. They are not
alternatives, and the second is useless if it fires while the first still has
room — a run that has spent its allowance should say so, not die of
infrastructure exhaustion.

The derivation lived in two places at once: a documented function in
``builder.py`` that nothing called, and the same arithmetic inlined in
``create_run_config`` with no explanation. Editing the documented one changed
nothing, which is the failure mode a duplicated constant always has.
"""

from __future__ import annotations

import pytest

from agentic_workflow.config import Settings
from agentic_workflow.graph.runtime import create_run_config, recursion_limit_for
from agentic_workflow.human.gates import approval_id_for
from agentic_workflow.human.policy import Stage

pytestmark = pytest.mark.unit


def _settings(max_iterations: int) -> Settings:
    """Build an offline configuration with a given iteration budget.

    Args:
        max_iterations: The budget to configure.

    Returns:
        The settings.
    """
    return Settings(_env_file=None, llm_provider="echo", max_iterations=max_iterations)


class TestRecursionLimit:
    """The backstop has to sit above the allowance it is a backstop for."""

    @pytest.mark.parametrize("budget", [1, 6, 50, 100])
    def test_the_limit_clears_the_budget_it_protects(self, budget: int) -> None:
        """One loop must never exhaust the graph limit on its own.

        If the limit were at or below the budget, the last permitted iteration
        would be killed by the backstop instead of being spent, and the run
        would fail with a recursion error while it still had iterations left.
        The headroom also has to absorb the nodes outside the loop and the
        re-execution an interrupt causes.
        """
        limit = recursion_limit_for(_settings(budget))

        assert limit > budget

    def test_the_limit_rises_with_the_budget(self) -> None:
        """Raising the budget has to raise what bounds the graph.

        A limit computed once from a constant would make ``max_iterations=100``
        unreachable: the run would be cut at the default before the router could
        spend the allowance the operator asked for.
        """
        assert recursion_limit_for(_settings(50)) > recursion_limit_for(_settings(6))

    def test_the_run_config_carries_the_derived_limit(self) -> None:
        """The limit the run is actually invoked with is the derived one.

        Asserting the helper alone would pass while the run config quietly used
        something else — which is precisely the split this module had.
        """
        settings = _settings(12)

        config = create_run_config("run-1", settings=settings)

        assert config["recursion_limit"] == recursion_limit_for(settings)

    def test_an_explicit_override_still_wins(self) -> None:
        """A caller that knows its graph's shape can set the limit itself.

        The override exists for embedding callers and for tests that need to
        observe the backstop tripping; deriving the value must not quietly
        discard it.
        """
        config = create_run_config("run-1", settings=_settings(6), recursion_limit=7)

        assert config["recursion_limit"] == 7


class TestApprovalIdentity:
    """Gate ids stay distinct across the iterations a loop can reach.

    The recursion limit only works as a backstop if the router's budget is the
    authority, and the router's budget is only meaningful if each iteration
    presents a distinguishable gate. A collision would let a decision for one
    iteration be accepted for the next.
    """

    def test_each_iteration_gets_its_own_gate_id(self) -> None:
        """Ids are a function of the coordinates, including the iteration.

        Re-entering the same stage in a later iteration is a different gate
        asking the same question, and a human answering the first one must not
        be taken as having answered the second.
        """
        ids = {
            approval_id_for("run-1", Stage.PATCH_REVIEW, iteration)
            for iteration in range(1, _settings(50).max_iterations + 1)
        }

        assert len(ids) == _settings(50).max_iterations
