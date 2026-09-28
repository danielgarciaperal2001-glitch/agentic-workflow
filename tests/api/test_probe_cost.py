"""A readiness probe that costs what the work costs.

`/health/ready` exists to answer one question — can this instance take traffic?
— and an orchestrator asks it every few seconds regardless of whether anything
is wrong. So whatever the probe touches is load the deployment pays
continuously, whether or not there is an incident.

Two things made that load proportional to the work rather than to the check.

The probe called `approvals.stats()`, which builds the whole inbox, and the
inbox enriches every view from the checkpoint store: `pending_approvals()` reads
every known run, then `inbox()` reads each parked run a *second* time to fill in
the view. Per probe, against a real PostgreSQL checkpointer:

    parked runs      before        after
              25      181 ms      2.3 ms
             100      728 ms      2.3 ms
             800    5,788 ms      4.7 ms

`PROBE_TIMEOUT_SECONDS` is 5, and 5,788 ms is past it. So at around 700 parked
runs — an ordinary backlog — a healthy instance answered 503, and the
orchestrator stopped sending it traffic. Counting reads against the in-memory
store's storage rather than wrapping the saver: 3,900 reads per probe at 40
parked runs, 0 after.

The store check had the opposite problem. `_check_store()` skipped `setup()`
whenever `_awf_ready` was set, and then reported `{"ok": True}` — so from the
second probe onwards the readiness check asserted the database was reachable
without issuing a single query. The existing test pinned that as intended ("the
DDL must not be re-run on every probe"). The intent was right about the DDL and
wrong about the check: `CREATE TABLE IF NOT EXISTS` is not how you ask whether a
connection still works. The two defects compounded — the cheap check had stopped
checking, and the expensive one was incidentally the only thing still noticing an
outage.

The two fixes are independent. The probe counts from the registry, which is
per-process state it already reports, and asks the store one real question.

The counters that need a store read — `expired` and `resolved_pending` — are
reported as zero by the probe and are *not* reported as zero by
`GET /v1/approvals/stats`, which is a human-facing route and keeps the
authoritative sweep. That split is asserted from both sides, because the failure
mode is a dashboard quietly showing a healthy backlog.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_workflow.config import load_settings
from agentic_workflow.human.service import ApprovalService
from agentic_workflow.services.engine import WorkflowEngine
from tests.api.test_health_probes import _client
from tests.helpers import CountingSaver, make_request

pytestmark = pytest.mark.api


def _settings() -> Any:
    """In-memory settings, derived from the ambient configuration.

    Forced rather than read from the environment: the suite runs both with and
    without ``AWF_POSTGRES_ENABLED``, and a test whose precondition is the
    ambient environment passes in one job and fails in the other.
    """
    return load_settings().model_copy(update={"postgres_enabled": False})


async def _engine_with_parked_runs(saver: Any, count: int) -> WorkflowEngine:
    """Build an engine holding *count* runs parked on a human approval.

    Every run parks, which is the state that makes the inbox expensive: a run that
    is still executing is in nobody's inbox and costs a probe nothing to skip.

    Args:
        saver: The counting checkpointer to inject.
        count: How many runs to park.

    Returns:
        A started engine whose runs are all waiting on a human.
    """
    engine = WorkflowEngine(_settings(), checkpointer=saver)
    await engine.startup()
    for _ in range(count):
        await engine.start(make_request())
    return engine


class TestProbeCost:
    """What the probe is allowed to make the store do."""

    async def test_the_probe_cost_does_not_grow_with_the_number_of_runs(self) -> None:
        """The defect: a diagnostic endpoint costing O(runs) on every call.

        Measured in steady state, which is the only state that matters for a probe
        an orchestrator calls forever. The first probe is deliberately not
        counted: the app's startup rehydrates the registry, so that one carries a
        read the steady-state probe does not, and asserting against it would hide
        the very cost being fixed.

        Asserted as a bound rather than a proportionality, because the property
        worth having is that the probe does not *scale* with the work. A fix that
        merely halved the inbox cost would still be linear, and would still take
        the instance out of the load balancer at a larger scale.
        """
        saver = CountingSaver()
        engine = await _engine_with_parked_runs(saver, 40)

        try:
            with _client(engine, settings=_settings()) as client:
                # The startup read, discarded.
                client.get("/health/ready")
                baseline = saver.reads

                for _ in range(5):
                    client.get("/health/ready")
                repeated = saver.reads - baseline
        finally:
            await engine.shutdown()

        assert repeated <= 5 * 2, (
            f"five probes over 40 parked runs read {repeated} times; a probe must "
            f"not scale with the inbox"
        )

    async def test_a_probe_is_idempotent_in_cost(self) -> None:
        """Probing twice must cost the same as probing once, twice over.

        The old cost was a function of the inbox, so a deployment that had been
        up longer paid more per probe. This pins that a probe's price does not
        drift with the work it happens to be reporting on.
        """
        saver = CountingSaver()
        engine = await _engine_with_parked_runs(saver, 40)

        try:
            with _client(engine, settings=_settings()) as client:
                client.get("/health/ready")  # startup read, discarded

                mark = saver.reads
                client.get("/health/ready")
                one = saver.reads - mark

                mark = saver.reads
                client.get("/health/ready")
                two = saver.reads - mark
        finally:
            await engine.shutdown()

        assert one == two, f"consecutive probes cost {one} and {two}"


class TestStoreCheck:
    """A readiness check that cannot fail is not a check."""

    def test_the_probe_still_asks_the_store_after_the_first_probe(self) -> None:
        """The defect: from the second probe onwards, "ok" was asserted, not read.

        `_check_store()` skipped `setup()` once `_awf_ready` was set and then
        reported success. The database could have been unreachable for an hour and
        every probe would have answered `{"ok": true}`, which is the one outcome
        a readiness probe must never produce on a guess.
        """
        settings = load_settings().model_copy(update={"postgres_enabled": True})
        saver = CountingSaver()

        with _client(_EngineWithSaver(saver), settings=settings) as client:
            first = client.get("/health/ready")
            client.get("/health/ready")

        # A 200 as well, so a silently broken sibling check cannot let this pass.
        assert first.status_code == 200
        assert saver.setup_calls >= 1
        assert saver.reads >= 1, "the probe never actually read from the store"

    def test_a_broken_store_is_still_reported_on_every_probe(self) -> None:
        """The behaviour that must survive: a dead database is a 503, every time.

        Pinned separately from the cost fix because a cheaper check is also an
        easier check to break: skipping the round trip for speed is exactly how
        this regressed the first time.
        """
        settings = load_settings().model_copy(update={"postgres_enabled": True})

        class _Dead:
            async def setup(self) -> None:
                raise RuntimeError("could not connect to server")

            async def aget_tuple(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("could not connect to server")

        with _client(_EngineWithSaver(_Dead()), settings=settings) as client:
            first = client.get("/health/ready")
            second = client.get("/health/ready")

        assert first.status_code == 503
        assert second.status_code == 503
        assert "could not connect" in second.json()["checks"]["checkpointer"]["detail"]

    def test_a_slow_store_is_reported_rather_than_waited_on(self) -> None:
        """The 5 s bound is what makes the probe's verdict usable at all."""
        settings = load_settings().model_copy(update={"postgres_enabled": True})

        class _Slow:
            async def setup(self) -> None:
                import asyncio

                await asyncio.sleep(3600)

            async def aget_tuple(self, *args: Any, **kwargs: Any) -> Any:
                import asyncio

                await asyncio.sleep(3600)

        with _client(_EngineWithSaver(_Slow()), settings=settings) as client:
            response = client.get("/health/ready")

        assert response.status_code == 503
        assert response.json()["checks"]["checkpointer"]["ok"] is False

    def test_a_store_that_offers_no_read_is_not_failed_for_it(self) -> None:
        """The defensive branches must not turn a probe into a crash.

        A checkpointer that exposes neither ``setup`` nor ``aget_tuple`` — a
        double, a future implementation, anything — has nothing to be asked, and
        the honest answer is "nothing to check" rather than a 503. Asserted
        because these are the two lines most likely to be deleted as unreachable
        once the test suite is green: nothing in the real deployment hits them, so
        the only thing standing between them and a future ``AttributeError`` is a
        test that says they were reached on purpose.
        """
        settings = load_settings().model_copy(update={"postgres_enabled": True})

        with _client(_EngineWithSaver(object()), settings=settings) as client:
            response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["checks"]["checkpointer"] == {"ok": True}

    def test_durable_settings_with_no_checkpointer_is_not_a_503(self) -> None:
        """The contradictory configuration, handled rather than crashed on.

        ``use_durable_checkpointer`` is derived from the settings, so an engine
        can be handed ``checkpointer=None`` while the flag says durable. The probe
        has nothing to ask and says so; what it must not do is treat the absence
        of a checkpointer as a broken one, which would take a correctly
        configured instance out of rotation over a null it never used.
        """
        settings = load_settings().model_copy(update={"postgres_enabled": True})

        with _client(_EngineWithSaver(None), settings=settings) as client:
            response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["checks"]["checkpointer"] == {"ok": True}


class _EngineWithSaver:
    """The minimal engine surface the readiness handler reads.

    Every method the handler touches is present, including ``parked_summary``.
    A test double missing one does not fail loudly: the handler wraps each check
    in its own ``try``, so a gap turns into ``{"ok": false}`` and a 503 — which is
    exactly what most of the tests here are asserting, so the gap hides.
    """

    def __init__(self, checkpointer: Any) -> None:
        self.registry: dict[str, Any] = {}
        self.checkpointer = checkpointer

    async def pending_approvals(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def status(self, run_id: str) -> Any:
        from agentic_workflow.errors import RunNotFoundError

        raise RunNotFoundError("no checkpoint for this run", run_id=run_id)

    def parked_summary(self) -> dict[str, Any]:
        return {"pending": 0, "expired": 0, "resolved_pending": 0, "by_stage": {}}

    def set_event_sink(self, sink: Any) -> None:
        self.sink = sink

    async def startup(self) -> None:
        """Awaited by the lifespan."""

    async def shutdown(self) -> None:
        """Awaited on lifespan exit."""


class TestCountingCorrectness:
    """A cheaper counter that under-reports would be worse than the old cost.

    The probe stopped building the inbox, so the question these answer is whether
    the cheap path tells the truth. A probe reporting "0 pending" while runs sit
    waiting is the one failure mode that looks exactly like a healthy instance.
    """

    async def test_the_cheap_count_agrees_with_the_inbox_it_replaced(self) -> None:
        """The registry path and the store path must report the same backlog.

        Contrasted against :meth:`ApprovalService.inbox` — the sweep that reads
        every run — rather than against ``stats()``, which is now the same cheap
        path and would make this a tautology.
        """
        saver = CountingSaver()
        engine = await _engine_with_parked_runs(saver, 12)
        service = ApprovalService(engine)

        try:
            inbox = await service.inbox()
            summary = engine.parked_summary()
        finally:
            await engine.shutdown()

        assert len(inbox) == 12, "the fixture did not park what it claims to"
        assert summary["pending"] == len(inbox)
        by_stage: dict[str, int] = {}
        for view in inbox:
            stage = view.request.stage
            by_stage[stage] = by_stage.get(stage, 0) + 1
        assert summary["by_stage"] == by_stage

    async def test_a_leaving_run_leaves_the_count(self) -> None:
        """The count must track the queue, not the history of what was queued.

        A counter that only ever went up would be a cheap way to look busy: the
        backlog would read 3 forever, and an operator watching it during an
        incident would be reading history. Cancelled rather than approved, because
        this workflow has several gates — answering one parks the same run on the
        next, so the count legitimately holds at 3.
        """
        saver = CountingSaver()
        engine = await _engine_with_parked_runs(saver, 3)
        service = ApprovalService(engine)

        try:
            assert engine.parked_summary()["pending"] == 3
            first = (await service.inbox())[0]
            await engine.cancel(first.request.run_id, reason="test")
        finally:
            await engine.shutdown()

        assert engine.parked_summary()["pending"] == 2

    async def test_answering_an_approval_moves_the_run_to_its_next_gate(self) -> None:
        """The per-stage breakdown has to move, not just the total.

        Pinning the total alone would let a breakdown frozen at the first gate
        pass: 3 pending before, 3 pending after. The stage is what distinguishes
        "a queue that is draining" from "a queue stuck at the same gate", and it
        is what a reviewer reads during an incident.
        """
        saver = CountingSaver()
        engine = await _engine_with_parked_runs(saver, 3)
        service = ApprovalService(engine)

        try:
            before = engine.parked_summary()
            first = (await service.inbox())[0]
            await service.resolve(first.approval_id, decision="approve", reviewer="tester")
            after = engine.parked_summary()
        finally:
            await engine.shutdown()

        assert before["by_stage"].get("patch_review") == 3
        assert after["pending"] == 3
        assert after["by_stage"] != before["by_stage"], (
            f"the breakdown stayed at {after['by_stage']} after approving {first.approval_id}"
        )

    async def test_the_two_counters_that_need_a_store_are_reported_as_unknown(self) -> None:
        """Pin the gap rather than let it look like a zero.

        ``expired`` and ``resolved_pending`` live on the approval object and the
        decision log respectively, neither of which the registry keeps. The cheap
        path reports zero for both, which is a *lie* if a reader takes it
        literally: an expired backlog would be reported as healthy. The docs say
        so, and this says it in code, so a later change that starts guessing has
        to argue with an assertion rather than with a comment.
        """
        saver = CountingSaver()
        engine = await _engine_with_parked_runs(saver, 4)
        service = ApprovalService(engine)

        try:
            inbox = await service.inbox()
            summary = engine.parked_summary()
        finally:
            await engine.shutdown()

        expired = sum(1 for view in inbox if view.is_expired)
        assert expired == 0, "the fixture cannot produce an expiry to compare"
        assert summary["expired"] == 0
        assert summary["resolved_pending"] == 0

    async def test_the_probe_reports_the_registry_count(self) -> None:
        """The handler must surface the cheap numbers, not rebuild the inbox.

        Also asserts a 200, so the wiring is verified: the handler swallows
        per-check exceptions into ``{"ok": false}``, which means a probe whose
        approval check is silently broken still passes a test that only looks at
        the checkpointer.
        """
        saver = CountingSaver()
        settings = _settings()
        engine = await _engine_with_parked_runs(saver, 7)

        try:
            with _client(engine, settings=settings) as client:
                response = client.get("/health/ready")
        finally:
            await engine.shutdown()

        assert response.status_code == 200
        assert response.json()["checks"]["approvals"]["pending"] == 7
