"""The engine's relationship with durable state it did not create in this process.

Every test here is about a run that exists in the checkpointer but *not* in this
engine's in-process registry. That combination is the normal one in production —
a restarted process, a second replica behind a load balancer, a queue worker
picking up a run another process started — and it is invisible in a test suite
that only ever drives runs from the process that created them.

All three tests were written after the corresponding bug was found by running
the real thing against a live PostgreSQL, which is worth saying plainly: the
in-memory suite passed with the registry-only behaviour in place, because a
single process registering its own runs never hits it. They run without a
database by sharing one in-memory saver between two engines, which reproduces
the shape of the problem — two engines, one store — even though the store is not
durable.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_workflow.config import Settings
from agentic_workflow.services.engine import WorkflowEngine
from tests.helpers import make_decision, make_request

pytestmark = pytest.mark.integration


class TestAnotherProcesssRun:
    """A run this process did not start is still a real run."""

    async def test_a_second_engine_can_read_it(self, settings: Settings, checkpointer: Any) -> None:
        """The checkpointer is the authoritative read, and it is one.

        Two engines, one saver, one run. The second engine has an empty registry
        and must still answer from the store. The assertion is on
        :meth:`WorkflowEngine.status` specifically because the *registry-backed*
        paths are the ones that can silently agree — a lookup that falls back to
        the registry would pass this test and fail the next one.
        """
        first = WorkflowEngine(settings, checkpointer=checkpointer)
        await first.startup()
        parked = await first.start(make_request(run_id="foreign-run"))
        await first.shutdown()

        second = WorkflowEngine(settings, checkpointer=checkpointer)
        await second.startup()
        try:
            assert second._registry.find("foreign-run") is None, (
                "the fixture is not testing a foreign run: the registry is shared"
            )
            observed = await second.status("foreign-run")
            assert observed.run_id == "foreign-run"
            assert observed.status == parked.status
        finally:
            await second.shutdown()

    async def test_a_second_engine_can_resume_it(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """Human-in-the-loop across a process boundary.

        The whole point of a durable checkpointer: a human answers a gate hours
        later, and the process that answers is not the one that asked. Before
        the registry was hydrated on boot this raised ``RunNotFoundError`` from
        inside the error handler, so the traceback named the registry and never
        mentioned the workflow.
        """
        first = WorkflowEngine(settings, checkpointer=checkpointer)
        await first.startup()
        parked = await first.start(make_request(run_id="foreign-resume"))
        assert parked.is_parked and parked.pending is not None
        await first.shutdown()

        second = WorkflowEngine(settings, checkpointer=checkpointer)
        await second.startup()
        try:
            resumed = await second.resume(
                "foreign-resume",
                make_decision(parked.pending.approval_id).model_dump(mode="json"),
            )
            assert resumed.error is None, resumed.error
            # The run parks again at the *next* gate. What matters is that the
            # gate this process answered is gone and a different one replaced it,
            # which can only be true if the decision reached the checkpoint.
            assert resumed.is_parked, resumed.status
            assert resumed.pending is not None
            assert resumed.pending.approval_id != parked.pending.approval_id
        finally:
            await second.shutdown()

    async def test_a_second_engine_can_cancel_it(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """Cancelling is the one action a human takes about *somewhere else*.

        An operator watching a run in another replica clicks stop. With the
        existence check reading the registry, that got a 404 for a run that was
        demonstrably running, and "I cannot stop it" is the worst answer a
        cancellation endpoint can give.
        """
        first = WorkflowEngine(settings, checkpointer=checkpointer)
        await first.startup()
        await first.start(make_request(run_id="foreign-cancel"))
        await first.shutdown()

        second = WorkflowEngine(settings, checkpointer=checkpointer)
        await second.startup()
        try:
            await second.cancel("foreign-cancel", reason="cancelled by the test")
            status = await second.status("foreign-cancel")
            assert status.status == "cancelled", status.status
        finally:
            await second.shutdown()

    async def test_cancelling_a_run_that_does_not_exist_still_404s(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """The existence check must not have been removed, only moved.

        Removing the guard would have "fixed" the previous test. Asserted
        explicitly, because the failure mode of over-correcting here is a cancel
        that silently succeeds against a typo'd run id and the operator believes
        they stopped something.
        """
        from agentic_workflow.errors import RunNotFoundError

        engine = WorkflowEngine(settings, checkpointer=checkpointer)
        await engine.startup()
        try:
            with pytest.raises(RunNotFoundError):
                await engine.cancel("no-such-run")
        finally:
            await engine.shutdown()


class TestFailureRecording:
    """An error path that raises is a diagnostic dead end."""

    async def test_recording_a_failure_for_an_unknown_run_does_not_raise(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """``_fail`` sits inside an ``except`` block, so it must never throw.

        The registry is per-process, so a run started elsewhere has no record
        here, and ``registry.update`` raised ``RunNotFoundError`` from inside the
        handler for a genuine workflow error. The original cause was replaced
        before it could be logged or returned: the traceback named the registry
        and said nothing about what had actually gone wrong.
        """
        engine = WorkflowEngine(settings, checkpointer=checkpointer)
        await engine.startup()
        try:
            engine._fail("not-registered", "the real cause")
            record = engine._registry.find("not-registered")
            assert record is not None, "the failure was dropped instead of recorded"
            assert record.status == "failed"
            assert record.error == "the real cause"
        finally:
            await engine.shutdown()

    async def test_recording_a_failure_twice_is_not_an_error(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """The happy path still works, and a repeat is not a crash.

        ``_fail`` creating a missing record must not turn a *known* run's failure
        into a duplicate-registration error, since the second registration is
        the thing the guard is protecting against.
        """
        engine = WorkflowEngine(settings, checkpointer=checkpointer)
        await engine.startup()
        try:
            engine._fail("twice", "first cause")
            engine._fail("twice", "second cause")
            record = engine._registry.find("twice")
            assert record is not None
            assert record.error == "second cause"
        finally:
            await engine.shutdown()


class TestBootHydration:
    """The registry is a projection, so it has to be rebuildable."""

    async def test_boot_rehydrates_runs_the_database_already_holds(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """A restart must not produce a process that has forgotten every run.

        Driven through a stub checkpointer rather than PostgreSQL, because the
        behaviour under test is the *engine's* side of the contract: it asks the
        store what threads exist and populates the registry from them. The real
        implementation of the other half is covered by the ``postgres`` suite.
        """
        store = _StubDurableStore(checkpointer)
        seeder = WorkflowEngine(settings, checkpointer=store)
        await seeder.startup()
        await seeder.start(make_request(run_id="hydrated-run", title="A durable run"))
        await seeder.shutdown()

        fresh = WorkflowEngine(settings, checkpointer=_StubDurableStore(checkpointer))
        await fresh.startup()
        try:
            record = fresh._registry.find("hydrated-run")
            assert record is not None, "boot forgot a run the database still holds"
            # Not merely present: the projection has to *agree with the
            # authoritative read*. Compared rather than hardcoded, because a
            # hydration that registered every run as `pending` looks identical
            # from the outside while being wrong about all of them — and a
            # hardcoded `iteration == 0` would have passed against a store that
            # had never been written to at all.
            authoritative = await fresh.status("hydrated-run")
            assert record.status == authoritative.status
            assert record.iteration == authoritative.iteration
            assert record.pending_approval == (
                authoritative.pending.approval_id if authoritative.pending else None
            )
        finally:
            await fresh.shutdown()

    async def test_hydration_passes_the_configured_budget(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """The engine's own limit must reach the store, not a hardcoded default.

        A restart has to be bounded, or a large ``checkpoints`` table turns a
        deploy into a long stall. The bound is a setting, so the engine has to
        forward it; a stub that ignores the argument would let that regress
        silently.
        """
        store = _StubDurableStore(checkpointer)
        seeder = WorkflowEngine(settings, checkpointer=store)
        await seeder.startup()
        await seeder.start(make_request(run_id="budgeted-run"))
        await seeder.shutdown()

        fresh_settings = Settings(**{**settings.model_dump(), "recovery_max_runs": 7})
        store.list_calls.clear()
        fresh = WorkflowEngine(fresh_settings, checkpointer=store)
        await fresh.startup()
        try:
            assert store.list_calls == [7], store.list_calls
        finally:
            await fresh.shutdown()

    async def test_a_store_that_cannot_be_listed_does_not_block_boot(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """Degraded, not failed.

        The checkpointer is already open by the time hydration runs, so a listing
        failure means the run *listing* is degraded — never that the service
        refuses to start. Every authoritative read still goes to the store.
        """
        broken = _StubDurableStore(checkpointer, fail_listing=True)
        engine = WorkflowEngine(settings, checkpointer=broken)
        await engine.startup()  # must not raise
        try:
            assert engine._registry.count() == 0
        finally:
            await engine.shutdown()

    async def test_an_in_memory_store_is_left_alone(
        self, settings: Settings, checkpointer: Any
    ) -> None:
        """No ``list_thread_ids`` means nothing to hydrate, and that is fine.

        The in-memory saver is per-process, so its contents *are* this process's
        contents. A hydration step that invented threads for it would be
        reporting runs that do not exist.
        """
        engine = WorkflowEngine(settings, checkpointer=checkpointer)
        await engine.startup()
        try:
            assert engine._registry.count() == 0
        finally:
            await engine.shutdown()


class _StubDurableStore:
    """An in-memory saver that also answers ``list_thread_ids``.

    Wraps the real saver rather than faking checkpoints, so the engine drives
    genuine LangGraph machinery. Only the enumeration is added, which is the one
    capability the engine needs and the one the in-memory saver lacks.
    """

    def __init__(self, saver: Any, *, fail_listing: bool = False) -> None:
        """Store the wrapped saver and the failure mode.

        Args:
            saver: The real in-memory checkpointer to delegate to.
            fail_listing: Make :meth:`list_thread_ids` raise, to exercise the
                degraded path.
        """
        self._saver = saver
        self._fail_listing = fail_listing
        self.list_calls: list[int] = []

    def __getattr__(self, name: str) -> Any:
        """Delegate everything unknown to the wrapped saver.

        A ``durable`` checkpointer exposes ``list_thread_ids``; the in-memory one
        does not. The engine probes for it, so a stub that always had it would
        make the "nothing to hydrate" test meaningless.

        Raises:
            AttributeError: Propagated from the wrapped saver, so the probe
                behaves the way it does against a real store.
        """
        return getattr(self._saver, name)

    @property
    def saver(self) -> Any:
        """The wrapped saver, for :attr:`WorkflowEngine.graph`.

        Same indirection the real ``PostgresCheckpointer`` provides: the engine
        unwraps with ``getattr(saver, "saver", saver)`` because LangGraph's
        ``compile`` rejects anything that is not a ``BaseCheckpointSaver``. The
        stub has to carry the same shape or it is rejected at compile time
        rather than exercising the hydration path.
        """
        return self._saver

    async def setup(self) -> None:
        """Open the wrapped saver, if it has a lifecycle of its own.

        The in-memory saver has no ``setup``; only the durable wrapper does.
        Delegating blindly would raise ``AttributeError`` from a test about
        hydration, which is how the first draft of this file failed.
        """
        opener = getattr(self._saver, "setup", None)
        if opener is not None:
            await opener()

    async def close(self) -> None:
        """Close the wrapped saver, if it has a lifecycle of its own."""
        closer = getattr(self._saver, "close", None)
        if closer is not None:
            await closer()

    async def list_thread_ids(self, *, limit: int = 10_000) -> list[str]:
        """Enumerate the wrapped saver's threads, as a durable store would.

        Enumerated through the saver's own ``alist`` for the same reason the
        PostgreSQL implementation is: the checkpoint tuple layout belongs to the
        saver, and reimplementing it in the test is how the test ends up
        asserting against its own idea of the data rather than the real thing.

        Args:
            limit: The bound the engine asked for; recorded so a test can assert
                the engine passed its configured budget through.

        Returns:
            Distinct thread ids, most recently checkpointed first.

        Raises:
            RuntimeError: When the stub was built to fail, on purpose.
        """
        self.list_calls.append(limit)
        if self._fail_listing:
            raise RuntimeError("the listing index is unavailable")
        threads: list[str] = []
        seen: set[str] = set()
        async for item in self._saver.alist(None, limit=limit):
            thread_id = (
                (getattr(item, "config", None) or {}).get("configurable", {}).get("thread_id")
            )
            if isinstance(thread_id, str) and thread_id not in seen:
                seen.add(thread_id)
                threads.append(thread_id)
        return threads
