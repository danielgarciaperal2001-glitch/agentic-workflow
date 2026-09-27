"""Tests for the batch runner.

The runner's contract is unusual and worth stating before testing it: it
promises not to raise. ``run_pipeline``'s docstring says ``Raises: Nothing``
and captures every per-request failure in the summary instead, because the
question it answers is "what happened when two hundred requests landed at
once" — and a runner that aborts on the first bad request cannot answer that
at all.

That is a promise a caller relies on, so it is asserted directly rather than
inferred from the happy path. It is also why the batch runner is the one
module in the package with no test: a bug here loses work silently, because a
batch that "succeeded" while quietly dropping half its requests looks exactly
like a batch that succeeded.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentic_workflow.config import load_settings
from agentic_workflow.domain.schemas import ApprovalRequest, ReviewRequest
from agentic_workflow.errors import InvalidStateError, ProviderTimeoutError
from agentic_workflow.services.engine import RunOutcome
from agentic_workflow.services.runner import (
    BatchSummary,
    run_pipeline,
    sequential_runner,
)
from tests.helpers import make_request


def _outcome(run_id: str, status: str = "completed", **kwargs: Any) -> RunOutcome:
    return RunOutcome(run_id=run_id, status=status, state={}, **kwargs)


class _Engine:
    """Engine double recording how the runner drove it.

    The runner reads ``settings.max_parallel_runs`` and calls either ``start``
    or ``run_until_done``; nothing else. Reaching for a real engine would pull
    a graph and a store into a test about batch accounting.
    """

    def __init__(self, outcome: Any = None, error: Exception | None = None) -> None:
        self.settings = load_settings()
        self._outcome = outcome
        self._error = error
        self.started: list[str] = []
        self.driven: list[tuple[str, bool]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def _run(self, request: ReviewRequest) -> RunOutcome:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # Yield so genuinely concurrent requests overlap. Without a real
            # suspension point the gate would be untestable: every request
            # would complete before the next one started.
            await asyncio.sleep(0)
            if self._error is not None:
                raise self._error
            if callable(self._outcome):
                return self._outcome(request)
            return self._outcome or _outcome(request.run_id)
        finally:
            self.in_flight -= 1

    async def start(self, request: ReviewRequest) -> RunOutcome:
        self.started.append(request.run_id)
        return await self._run(request)

    async def run_until_done(self, request: ReviewRequest, *, decide: Any = None) -> RunOutcome:
        self.driven.append((request.run_id, decide is not None))
        return await self._run(request)


def _requests(count: int) -> list[ReviewRequest]:
    return [make_request(f"run-{i:03d}") for i in range(count)]


# --------------------------------------------------------------------------- #
class TestBatchSummary:
    def test_an_empty_batch_does_not_divide_by_zero(self) -> None:
        """``success_rate`` on an empty batch is 0.0, not a ZeroDivisionError
        escaping from a summary nobody thought would throw."""
        assert BatchSummary(total=0).success_rate == 0.0

    def test_the_rate_is_completed_over_submitted(self) -> None:
        """Parked runs count against the rate: they are unfinished, and calling
        a batch 100% successful while every run waits on a human is the kind of
        number that gets a system trusted that does not work."""
        summary = BatchSummary(total=4, completed=1, parked=3)
        assert summary.success_rate == 0.25

    def test_as_dict_is_json_shaped_and_carries_the_run_ids(self) -> None:
        summary = BatchSummary(
            total=2,
            completed=1,
            failed=1,
            outcomes=[_outcome("a"), _outcome("b", "failed")],
            errors={"b": "boom"},
            duration_seconds=1.23456,
        )
        rendered = summary.as_dict()
        assert rendered["runs"] == ["a", "b"]
        assert rendered["success_rate"] == 0.5
        assert rendered["duration_seconds"] == 1.235
        assert rendered["errors"] == {"b": "boom"}

    def test_the_errors_mapping_is_copied_not_aliased(self) -> None:
        """Returning the live dict would let a caller mutate the summary it was
        handed, which is a subtle corruption rather than an obvious one."""
        summary = BatchSummary(total=1, errors={"a": "boom"})
        summary.as_dict()["errors"]["a"] = "tampered"
        assert summary.errors == {"a": "boom"}

    def test_the_str_form_leads_with_what_succeeded(self) -> None:
        text = str(BatchSummary(total=10, completed=7, parked=2, failed=1))
        assert "7/10 completed" in text
        assert "2 parked" in text
        assert "1 failed" in text


# --------------------------------------------------------------------------- #
class TestRunPipeline:
    async def test_an_empty_batch_returns_an_empty_summary(self) -> None:
        summary = await run_pipeline(_Engine(), [])
        assert summary.total == 0
        assert summary.outcomes == []

    async def test_every_request_produces_an_outcome(self) -> None:
        engine = _Engine()
        summary = await run_pipeline(engine, _requests(5))
        assert summary.total == 5
        assert len(summary.outcomes) == 5
        assert summary.completed == 5
        assert {o.run_id for o in summary.outcomes} == {f"run-{i:03d}" for i in range(5)}

    async def test_outcome_statuses_are_counted_separately(self) -> None:
        """The counters are the whole product of the call: an operator reads
        "7 completed, 2 parked" and knows whether humans are the bottleneck."""
        plan = {
            0: "completed",
            1: "completed",
            2: "completed",
            3: "waiting_human",
            4: "waiting_human",
            5: "cancelled",
        }
        engine = _Engine(outcome=lambda r: _outcome(r.run_id, plan[int(r.run_id[-3:])]))
        summary = await run_pipeline(engine, _requests(6))
        assert (summary.completed, summary.parked, summary.cancelled) == (3, 2, 1)

    async def test_a_workflow_error_is_captured_rather_than_raised(self) -> None:
        """The contract in one assertion: a failing request must not take the
        batch down, or two hundred requests become "stopped at the first
        bad one"."""
        engine = _Engine(error=InvalidStateError("bad state"))
        summary = await run_pipeline(engine, _requests(3))
        assert summary.failed == 3
        assert summary.completed == 0
        assert set(summary.errors) == {"run-000", "run-001", "run-002"}
        assert "bad state" in summary.errors["run-000"]

    async def test_an_unexpected_exception_also_survives(self) -> None:
        """``except WorkflowError`` alone is not enough: a provider adapter
        raising a bare ``ValueError`` would otherwise abort the batch."""
        engine = _Engine(error=ValueError("something unforeseen"))
        summary = await run_pipeline(engine, _requests(2))
        assert summary.failed == 2
        assert summary.errors["run-000"].startswith("ValueError: ")

    async def test_a_rejected_decision_is_not_counted_as_a_failure(self) -> None:
        """A human saying no is a verdict, not an error. Reporting it under
        ``failed`` would tell an operator to go and debug their own team."""
        engine = _Engine(outcome=lambda r: _outcome(r.run_id, "rejected"))
        summary = await run_pipeline(engine, _requests(2))
        assert summary.failed == 0
        assert summary.parked == 2

    async def test_one_failure_does_not_stop_the_others(self) -> None:
        engine = _Engine(error=ProviderTimeoutError("slow"))
        summary = await run_pipeline(engine, _requests(4))
        assert summary.failed == 4, "the default is to keep going"

    async def test_stop_on_error_abandons_the_rest(self) -> None:
        """Opt-in because the opposite is usually right; but when a batch is
        burning real money on a broken deployment, stopping is the point."""
        engine = _Engine(error=InvalidStateError("bad state"))
        summary = await run_pipeline(engine, _requests(20), concurrency=1, stop_on_error=True)
        assert summary.failed == 1
        assert len(engine.started) == 1
        assert summary.total == 20, "the submitted count is not rewritten"

    async def test_the_concurrency_gate_is_honoured(self) -> None:
        """Unbounded concurrency against a rate-limited provider is how one
        workflow takes down everyone else's quota."""
        engine = _Engine()
        await run_pipeline(engine, _requests(12), concurrency=3)
        assert engine.max_in_flight <= 3

    async def test_a_single_slot_serialises_the_batch(self) -> None:
        engine = _Engine()
        await run_pipeline(engine, _requests(4), concurrency=1)
        assert engine.max_in_flight == 1

    async def test_the_engine_limit_is_the_default_gate(self) -> None:
        engine = _Engine()
        await run_pipeline(engine, _requests(2))
        assert engine.max_in_flight <= engine.settings.max_parallel_runs

    async def test_on_event_sees_every_completed_outcome(self) -> None:
        seen: list[str] = []
        await run_pipeline(_Engine(), _requests(4), on_event=lambda o: seen.append(o.run_id))
        assert len(seen) == 4

    async def test_a_broken_on_event_does_not_lose_the_batch(self) -> None:
        """A progress callback that raises is a caller bug; letting it abort a
        batch of paid work would make the callback a liability."""

        def _boom(outcome: RunOutcome) -> None:
            raise RuntimeError("callback exploded")

        engine = _Engine()
        with pytest.raises(RuntimeError, match="callback exploded"):
            await run_pipeline(engine, _requests(3), on_event=_boom)
        # The work itself was done; only the callback failed afterwards.
        assert len(engine.started) == 3


# --------------------------------------------------------------------------- #
class TestDeciderRouting:
    async def test_without_a_decider_the_runner_only_starts_runs(self) -> None:
        """Parked is the right default for a human-in-the-loop system: a batch
        that auto-approves everything is a batch nobody reviewed."""
        engine = _Engine(outcome=lambda r: _outcome(r.run_id, "waiting_human"))
        summary = await run_pipeline(engine, _requests(2))
        assert engine.driven == []
        assert len(engine.started) == 2
        assert summary.parked == 2

    async def test_a_decider_drives_runs_to_a_decision(self) -> None:
        engine = _Engine()
        await run_pipeline(engine, _requests(2), decide=lambda _: {"decision": "approve"})
        assert engine.started == []
        assert [run_id for run_id, _ in engine.driven] == ["run-000", "run-001"]

    async def test_the_decider_is_forwarded_to_the_engine_intact(self) -> None:
        """The runner does not invoke the decider — the engine does, once per
        gate. What the runner owes it is passing the same callable through, so
        a policy that closes over state behaves identically inside a batch."""
        received: list[Any] = []

        async def run_until_done(request: ReviewRequest, *, decide: Any = None) -> RunOutcome:
            received.append(decide)
            return _outcome(request.run_id)

        def policy(_approval: ApprovalRequest) -> dict[str, Any] | None:
            return None

        engine = _Engine()
        engine.run_until_done = run_until_done  # type: ignore[method-assign]
        await run_pipeline(engine, _requests(3), decide=policy)
        assert received == [policy, policy, policy]


# --------------------------------------------------------------------------- #
class TestSequentialRunner:
    async def test_the_bound_runner_reuses_the_engine(self) -> None:
        engine = _Engine()
        run = sequential_runner(engine)
        first = await run(_requests(2))
        second = await run(_requests(1))
        assert first.total == 2
        assert second.total == 1
        assert len(engine.started) == 3

    async def test_the_decider_stays_bound_across_batches(self) -> None:
        """A caller submitting several batches wants identical semantics each
        time, which is the whole reason this factory exists."""
        engine = _Engine()
        run = sequential_runner(engine, decide=lambda _: {"decision": "approve"})
        await run(_requests(2))
        await run(_requests(1))
        assert len(engine.driven) == 3
        assert all(flag for _, flag in engine.driven)
