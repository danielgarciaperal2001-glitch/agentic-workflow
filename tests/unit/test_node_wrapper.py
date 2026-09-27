"""The wrapper every graph node runs inside.

``@node`` is the last thing between a node's code and the engine's bookkeeping:
it stamps correlation ids, enforces the node budget, records timings and turns
unexpected exceptions into domain errors. Every node in the system passes
through it, and none of it was covered — a wrapper that is never executed by a
test is a wrapper whose behaviour is whatever the code happens to do.

These tests concentrate on the budget, because it is the part with a stated
contract and no implementation behind it: ``node_timeout_seconds`` is a
documented setting, ``@node`` documents a fallback to it, and the two were never
connected. Every node therefore ran unbounded, and a node waiting on a provider
that never answered held its run — and its share of the concurrency budget —
until the run-level deadline, which is fifteen minutes by default and is
intended as a backstop rather than as a limit.

The fallback is resolved per invocation rather than when the module is imported,
and that is asserted directly: the decorator runs at import time, so reading the
settings there would freeze the process configuration into every node and make
the setting unreachable for anyone who built their own ``Settings``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentic_workflow.config import Settings, reset_settings_cache
from agentic_workflow.domain.schemas import utcnow
from agentic_workflow.errors import RunTimeoutError
from agentic_workflow.graph.common import node
from agentic_workflow.graph.context import AgentContext
from tests.helpers import make_request

pytestmark = pytest.mark.unit


class _Runtime:
    """A stand-in for the ``Runtime`` LangGraph passes as a node's second argument.

    Args:
        context: The runtime context a real invocation would carry.
    """

    def __init__(self, context: AgentContext) -> None:
        self.context = context


#: Short enough that a missing budget fails in seconds rather than sitting out
#: `pytest-timeout`, long enough that a loaded machine does not produce a
#: spurious failure.
BUDGET = 0.05


def _context(node_timeout_seconds: float = BUDGET) -> AgentContext:
    """Build a runtime context carrying a per-node budget.

    Args:
        node_timeout_seconds: The budget the node should observe.

    Returns:
        The context.
    """
    return AgentContext(
        settings=Settings(
            _env_file=None,
            llm_provider="echo",
            node_timeout_seconds=node_timeout_seconds,
        )
    )


def _state() -> dict[str, Any]:
    """Return a minimal graph state carrying a request.

    Returns:
        A state the wrapper can read a run id from.
    """
    return {"request": make_request()}


async def _ok(state: dict[str, Any], runtime: Any = None) -> dict[str, Any]:
    """A node that does nothing and succeeds.

    Args:
        state: Ignored.
        runtime: Ignored.

    Returns:
        An empty update.
    """
    return {}


async def _hangs(state: dict[str, Any], runtime: Any = None) -> dict[str, Any]:
    """A node that never returns, standing in for a provider that never answers.

    Args:
        state: Ignored.
        runtime: Ignored.

    Yields:
        Nothing; this coroutine only ever suspends.
    """
    await asyncio.sleep(3600)
    return {}


class TestNodeBudget:
    """A node cannot hold a run open forever."""

    async def test_a_hanging_node_is_stopped_by_the_configured_budget(self) -> None:
        """The documented default is actually applied to every node.

        The failure this pins is unbounded, not slow: a node awaiting a provider
        that never responds holds a concurrency slot for the whole run deadline
        — fifteen minutes by default, and configurable upward. A handful of such
        runs exhaust ``max_parallel_runs`` and the deployment stops accepting
        work, with the cause sitting in whichever node is stuck.
        """
        hanging = node("hanging")(_hangs)

        with pytest.raises(RunTimeoutError) as caught:
            await asyncio.wait_for(hanging(_state(), _Runtime(_context())), timeout=10)

        assert caught.value.context["node"] == "hanging"
        # The run id and node travel in the error, so a failure is attributable
        # without correlating against the log. A bare `TimeoutError` would carry
        # neither, which is what made the budget invisible while it was enforced.
        assert caught.value.run_id
        assert caught.value.context["timeout_seconds"] == pytest.approx(BUDGET)

    async def test_an_explicit_budget_wins_over_the_default(self) -> None:
        """A node may set its own budget without editing global configuration.

        Some work legitimately needs longer than the default — a tester running
        a slow suite, say — and the alternative would be raising the ceiling for
        every node to accommodate one.
        """
        slow = node("slow", timeout_seconds=BUDGET)(_hangs)

        with pytest.raises(RunTimeoutError) as caught:
            await asyncio.wait_for(
                slow(_state(), _Runtime(_context(node_timeout_seconds=3600.0))), timeout=10
            )

        assert caught.value.context["timeout_seconds"] == pytest.approx(BUDGET)

    async def test_the_budget_follows_the_context_not_the_import(self) -> None:
        """Two invocations of one node can carry different budgets.

        The decorator executes when its module is imported, which is long before
        any settings are known. Resolving the fallback there would bind every
        node in the process to whatever was configured at import — so a caller
        that constructed its own ``Settings``, which is every embedding
        application and every test, could not change it.
        """
        patient = node("patient")(_hangs)
        impatient = node("impatient")(_hangs)

        with pytest.raises(RunTimeoutError):
            await asyncio.wait_for(impatient(_state(), _Runtime(_context(BUDGET))), timeout=10)

        # Same process, a budget set 70_000x higher. Only the caller's own
        # deadline should end this, which is what distinguishes a budget read
        # per invocation from one frozen when the module was imported.
        with pytest.raises(TimeoutError) as outer:
            await asyncio.wait_for(patient(_state(), _Runtime(_context(3600.0))), timeout=0.5)
        assert not isinstance(outer.value, RunTimeoutError)

    async def test_a_node_within_budget_returns_its_update(self) -> None:
        """The budget does not interfere with a node that finishes in time.

        A wrapper that leaked the timeout's ``CancelledError`` or swallowed the
        return value would pass every test above while breaking the graph, so
        the happy path is asserted rather than assumed.
        """
        quick = node("quick")(_ok)

        update = await quick(_state(), _Runtime(_context()))

        assert [t.node for t in update["node_timings"]] == ["quick"]
        assert update["node_timings"][0].ok is True
        assert update["updated_at"] <= utcnow().isoformat()

    async def test_a_node_runs_without_a_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare ``graph.ainvoke`` still gets a budget.

        Notebooks and embedding callers invoke the graph with no context at all,
        so the fallback reads the process settings. The guarantee has to hold
        there too: a safety limit that only exists once somebody wires up a
        runtime context is not a limit.

        The setting is narrowed through the environment and the settings cache
        cleared, because that is the only way to reach the branch where no
        context exists at all.
        """
        hanging = node("no-runtime")(_hangs)
        monkeypatch.setenv("AWF_NODE_TIMEOUT_SECONDS", str(BUDGET))
        reset_settings_cache()
        try:
            with pytest.raises(RunTimeoutError):
                await asyncio.wait_for(hanging(_state()), timeout=10)
        finally:
            reset_settings_cache()
