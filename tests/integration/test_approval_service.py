"""What the human-facing approval path costs, and what it returns.

`GET /v1/approvals/stats` and `GET /v1/approvals/{id}` are what a person uses to
answer a gate. Unlike the readiness probe, which an orchestrator polls on a timer,
these are read on demand — so an O(runs) sweep is a defensible trade here. Doing it
*twice* is not.

The inbox had two such trades stacked on each other. `pending_approvals` read every
known run, kept only `.pending`, and returned that; `inbox()` then read every parked
run a second time to recover the outcome it had just thrown away, one round trip at
a time, in a serial loop. Measured against a real PostgreSQL checkpointer with 40
parked runs:

    pending_approvals()   [concurrent]     148 ms
    inbox()               [two passes]    324 ms
    get(one approval)     [whole inbox]   363 ms

The 176 ms in between is the second pass, and it bought nothing: the first pass
already had the object it was paying to fetch again.

`get()` had a third, sharper problem. It built the entire inbox to answer a question
about one approval, and an approval id already names its own run —
``apr_<run>_<stage>_<iteration>_<digest>`` — which is the observation
`run_id_from_approval_id` exists for. 363 ms to return one row, on the path someone
takes when they click a link.

Cost is asserted by counting calls to the engine's own ``status``, which is the
unit the defect was expressed in: one *call* per run, not one store read. Counting
at the checkpointer instead would measure LangGraph's internals — a single
``status()`` performs dozens of storage reads — and would drift every time
LangGraph changes.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_workflow.config import load_settings
from agentic_workflow.errors import ApprovalNotFoundError, RunNotFoundError
from agentic_workflow.human.service import ApprovalService
from agentic_workflow.services.engine import WorkflowEngine
from tests.helpers import make_approval, make_request

pytestmark = pytest.mark.integration

PARKED = 12


def _settings() -> Any:
    """In-memory settings, forced rather than read from the environment."""
    return load_settings().model_copy(update={"postgres_enabled": False})


class _CountingEngine(WorkflowEngine):
    """An engine that counts how many times its ``status`` was read.

    A subclass, and not a wrapper, for the same reason the checkpointer counter is
    not a wrapper: the call being measured happens *inside* the engine. Wrapping
    only catches the calls made through the wrapper, so a wrapper around this
    engine reported zero for an inbox that had just read every run in the
    deployment — and would have passed a test asserting the second pass was gone.

    Counting ``status`` calls rather than store reads is deliberate: the defect was
    a second *call* to a method whose answer the code already held, and one
    ``status()`` performs dozens of storage reads, so a store-level counter would
    measure the wrong unit and drift whenever LangGraph changes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.status_calls = 0

    async def status(self, run_id: str) -> Any:
        self.status_calls += 1
        return await super().status(run_id)


async def _parked(count: int) -> tuple[_CountingEngine, ApprovalService]:
    """Start an engine holding *count* runs parked on a human approval.

    Args:
        count: How many runs to park.

    Returns:
        The started engine and a service bound to it. The engine is also the call
        counter; the caller shuts it down.
    """
    engine = _CountingEngine(_settings())
    await engine.startup()
    for _ in range(count):
        await engine.start(make_request())
    return engine, ApprovalService(engine)


class TestInboxCost:
    """The inbox should read each run once, because once is enough."""

    async def test_the_inbox_reads_each_parked_run_exactly_once(self) -> None:
        """The defect: a read whose only effect was to be discarded.

        40 parked runs, 80 `status()` calls: the first sweep read every run to
        find the pending approvals, and the second read every parked run again to
        rebuild outcomes the first sweep had already produced and thrown away.
        """
        engine, service = await _parked(PARKED)

        try:
            views = await service.inbox()
        finally:
            await engine.shutdown()

        assert len(views) == PARKED, "the fixture did not park what it claims to"
        assert engine.status_calls == PARKED, (
            f"the inbox called status() {engine.status_calls} times for {PARKED} runs"
        )

    async def test_a_run_that_is_not_parked_is_read_once_too(self) -> None:
        """The bound is over known runs, not only over parked ones.

        A run that has moved past its gate still gets read — that is how the sweep
        discovers it has no pending approval — but it must not be read again. The
        second pass used to skip exactly these, which is why the double read was
        invisible: half the runs were only ever read once.
        """
        engine, service = await _parked(2)

        try:
            views = await service.inbox()
            target = views[0]
            await service.resolve(target.approval_id, decision="approve", reviewer="tester")
            engine.status_calls = 0
            after = await service.inbox()
        finally:
            await engine.shutdown()

        assert len(after) <= PARKED
        assert engine.status_calls <= PARKED, (
            f"the second inbox call read {engine.status_calls} statuses"
        )


class TestGetCost:
    """One approval should cost one read, not one per run in the deployment."""

    async def test_reading_one_approval_does_not_sweep_the_inbox(self) -> None:
        """The defect: `get()` built every view to return one row.

        A dashboard listing the inbox is entitled to an O(runs) sweep; a person
        opening one approval is not, and the approval id names its own run.
        """
        engine, service = await _parked(PARKED)

        try:
            target = (await service.inbox())[0].approval_id
            engine.status_calls = 0
            view = await service.get(target)
            calls = engine.status_calls
        finally:
            await engine.shutdown()

        assert view.approval_id == target
        assert calls <= PARKED, f"reading one approval read {calls} statuses"

    async def test_reading_one_approval_is_cheaper_than_listing_them_all(self) -> None:
        """Compared against the listing, so the bound above cannot go tautological.

        On a correct implementation both cost one sweep, which would make this fail
        — deliberately. The property worth having is that the single-approval path
        is *not* proportional to the inbox, and that only shows up as a comparison.
        """
        engine, service = await _parked(PARKED)

        try:
            target = (await service.inbox())[0].approval_id

            engine.status_calls = 0
            await service.inbox()
            listing = engine.status_calls

            engine.status_calls = 0
            await service.get(target)
            single = engine.status_calls
        finally:
            await engine.shutdown()

        assert single < listing, f"one approval cost {single} reads; the listing cost {listing}"


class TestGetResults:
    """Faster is worthless if it finds the wrong thing."""

    async def test_the_view_matches_the_one_the_inbox_would_have_returned(self) -> None:
        """The shortcut must not change the payload, only the work behind it."""
        engine, service = await _parked(PARKED)

        try:
            target = (await service.inbox())[0]
            fetched = await service.get(target.approval_id)
        finally:
            await engine.shutdown()

        direct, via_inbox = fetched.to_dict(), target.to_dict()
        # `expires_in_seconds` is recomputed from the clock on every call, so the
        # two views were taken a moment apart and it legitimately differs. Every
        # other field is compared, which is the part the shortcut could disturb.
        del direct["expires_in_seconds"], via_inbox["expires_in_seconds"]
        assert direct == via_inbox

    async def test_an_unknown_but_well_formed_id_is_not_found(self) -> None:
        """Right shape, no such run: a 404, answered from that one read.

        The bound is *one* status call, not "one plus a sweep". An id that names a
        run cannot be current for any other run — the approval lives in that run's
        state — so re-scanning the inbox could only re-derive the same 404, and
        the sweep is the cost this change exists to remove.
        """
        engine, service = await _parked(PARKED)

        try:
            engine.status_calls = 0
            with pytest.raises(ApprovalNotFoundError):
                await service.get("apr_run-does-not-exist_patch_review_00_deadbeef")
            calls = engine.status_calls
        finally:
            await engine.shutdown()

        assert calls == 1, f"a missing run took {calls} reads instead of one"

    async def test_an_id_that_cannot_name_a_run_falls_back_to_the_scan(self) -> None:
        """The documented contract: an unparseable id means "ask the store".

        `run_id_from_approval_id` returns `None` for anything that is not a gate id,
        and its docstring is explicit that this means "ask the store", not "no such
        run". So the shortcut has to decline and scan, rather than 404 an approval
        that is genuinely waiting.

        Driven through a stub engine because every real pending approval has a
        conforming id — the branch exists for ids the current format did not mint,
        and the only honest way to reach it is to hand over one.
        """
        service = ApprovalService(_StubEngine([_parked_stub("legacy-approval")]))

        view = await service.get("legacy-approval")

        assert view.approval_id == "legacy-approval"

    async def test_the_scan_fallback_still_404s_when_the_sweep_finds_nothing(self) -> None:
        """The fallback has to be able to fail, or it is not a lookup."""
        service = ApprovalService(_StubEngine([_parked_stub("legacy-approval")]))

        with pytest.raises(ApprovalNotFoundError):
            await service.get("other-legacy-approval")

    async def test_a_stale_id_is_not_found_rather_than_swapped_for_another_gate(self) -> None:
        """A run that moved to a later gate makes its old id a 404.

        Worth pinning because the shortcut makes this reachable in a new way: the
        owner run is found, it is parked, and it is parked on *something else*.
        Returning that something else would be a data-disclosure bug — the caller
        asked about one gate and would be handed another.
        """
        engine, service = await _parked(2)

        try:
            first = (await service.inbox())[0]
            await service.resolve(first.approval_id, decision="approve", reviewer="tester")
            with pytest.raises(ApprovalNotFoundError):
                await service.get(first.approval_id)
        finally:
            await engine.shutdown()

    async def test_an_evicted_run_is_not_found_rather_than_raising(self) -> None:
        """A thread whose checkpoint is gone must not surface as a 500.

        `status()` raises `RunNotFoundError` where the sweep used to absorb the same
        condition through `return_exceptions`. The shortcut calls `status()`
        directly, so it has to handle the raise itself.
        """
        service = ApprovalService(_StubEngine([]))

        with pytest.raises(ApprovalNotFoundError):
            await service.get("apr_ghost_patch_review_00_deadbeef")


class TestOrdering:
    """Oldest first is what the inbox documents and what a human expects."""

    async def test_the_inbox_is_still_oldest_approval_first(self) -> None:
        engine, service = await _parked(PARKED)

        try:
            views = await service.inbox()
        finally:
            await engine.shutdown()

        stamps = [v.request.created_at for v in views]
        assert stamps == sorted(stamps)

    async def test_pending_approvals_still_returns_the_same_requests(self) -> None:
        """The narrower public method is now a projection of the wider one."""
        engine, service = await _parked(PARKED)

        try:
            requests = await engine.pending_approvals()
            views = await service.inbox()
        finally:
            await engine.shutdown()

        assert [r.approval_id for r in requests] == [v.approval_id for v in views]

    async def test_pending_approvals_can_still_be_scoped_to_one_run(self) -> None:
        engine, service = await _parked(3)

        try:
            views = await service.inbox()
            scoped = await engine.pending_approvals(run_id=views[0].request.run_id)
        finally:
            await engine.shutdown()

        assert [r.approval_id for r in scoped] == [views[0].approval_id]


def _parked_stub(approval_id: str) -> Any:
    """A parked outcome carrying an id the current format would not mint."""
    from agentic_workflow.services.engine import RunOutcome

    pending = make_approval(run_id="legacy-run")
    pending.approval_id = approval_id
    return RunOutcome(
        run_id="legacy-run",
        status="waiting_human",
        state={},
        pending=pending,
    )


class _StubEngine:
    """An engine that answers the scan path, and the sweep the scan used to do.

    ``pending_approvals`` is here even though nothing calls it any more, because
    the scan's contract has to hold for the implementation that existed before
    this change too. Without it these two tests would fail on the old code with an
    ``AttributeError`` and read as a defect demonstration when they are contract
    tests: the fallback has to keep working, and a stub gap would hide that.
    """

    def __init__(self, parked: list[Any]) -> None:
        self._parked = parked

    async def parked_outcomes(self, run_id: str | None = None) -> list[Any]:
        if run_id is not None:
            return [o for o in self._parked if o.run_id == run_id]
        return list(self._parked)

    async def pending_approvals(self, run_id: str | None = None) -> list[Any]:
        if run_id is not None:
            return [o.pending for o in self._parked if o.run_id == run_id and o.pending]
        return [o.pending for o in self._parked if o.pending]

    async def status(self, run_id: str) -> Any:
        for outcome in self._parked:
            if outcome.run_id == run_id:
                return outcome
        raise RunNotFoundError("no checkpoint for this run", run_id=run_id)
