"""Checkpoint retention, against the real LangGraph saver protocol.

The janitor's first version probed ``checkpointer.alist()`` with no arguments
and was only ever tested against a hand-written fake whose ``alist`` accepted
whatever it was given. Both faults were invisible: the shipped
:class:`~agentic_workflow.persistence.checkpointer.PostgresCheckpointer` is a
*wrapper* that exposes no ``alist`` at all, and LangGraph's protocol declares
``config`` as a required positional. So the sweep could not have run against
either real checkpointer, and the failure mode was an error at the first pass
rather than a wrong answer — which is why nobody noticed a command that only
ever runs on a schedule.

These tests therefore use a real ``InMemorySaver`` and a real wrapper, so the
signature and the attribute lookup are both part of what is pinned.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_workflow.config import Settings, load_settings
from agentic_workflow.errors import PersistenceError
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.persistence.retention import (
    CheckpointJanitor,
    _checkpoint_timestamp,
)
from agentic_workflow.services.engine import WorkflowEngine
from tests.helpers import make_request, unique_run_id

pytestmark = pytest.mark.integration


class _Stamped:
    """A checkpoint entry carrying only a timestamp."""

    def __init__(self, ts: str) -> None:
        self.checkpoint = {"ts": ts}
        self.config = {"configurable": {"thread_id": "t"}}
        self.metadata: dict[str, Any] = {}


class _Blanked:
    """A real ``CheckpointTuple`` re-issued with its timestamp removed."""

    def __init__(self, source: Any, checkpoint: dict[str, Any]) -> None:
        self.__dict__.update(
            {
                "config": getattr(source, "config", None),
                "metadata": getattr(source, "metadata", None),
                "parent_config": getattr(source, "parent_config", None),
                "checkpoint": checkpoint,
                "pending_writes": getattr(source, "pending_writes", None),
            }
        )


async def _seed(runs: int) -> tuple[WorkflowEngine, list[str]]:
    """Run *runs* workflows to completion against a memory checkpointer.

    Args:
        runs: How many runs to create.

    Returns:
        The engine and the run ids, so a test can assert on specific threads.
    """
    settings = load_settings(
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        hitl_enabled=True,
        log_level="WARNING",
    )
    engine = WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
    await engine.startup()
    ids: list[str] = []
    try:
        for _ in range(runs):
            request = make_request(run_id=unique_run_id("janitor"))
            await engine.run_until_done(request, max_gates=8)
            ids.append(request.run_id)
    finally:
        await engine.shutdown()
    return engine, ids


class TestListing:
    """The sweep has to be able to *see* what it is meant to delete."""

    async def test_a_sweep_over_a_populated_store_reports_what_it_saw(
        self,
    ) -> None:
        """Zero scanned is the signature of the ``alist`` signature bug.

        A janitor that reports nothing to do is indistinguishable, from the log,
        from a janitor with nothing to do. Anything less than the seeded count
        means the listing silently failed.
        """
        engine, run_ids = await _seed(2)
        settings: Settings = load_settings(postgres_enabled=False, log_level="WARNING")

        report = await CheckpointJanitor(engine.checkpointer, settings).run(
            retention_days=0, dry_run=True
        )

        assert report.examined >= len(run_ids)

    async def test_listing_works_against_a_live_engine(self) -> None:
        """The sweep must find threads through the saver, not a side channel."""
        engine, _ = await _seed(1)
        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        report = await CheckpointJanitor(engine.checkpointer, settings).run(
            retention_days=0, dry_run=True
        )

        assert report.examined >= 1
        assert report.stale >= 1, "a zero-day window makes everything stale"
        assert report.deleted == 0, "a dry run must not delete"

    async def test_a_wrapped_checkpointer_is_unwrapped(self) -> None:
        """``PostgresCheckpointer`` forwards nothing, so probing it finds no ``alist``.

        This is the shape the production deployment uses, and it is the reason
        the janitor could not run at all against Postgres.
        """
        engine, _ = await _seed(1)

        class _Wrapper:
            """A stand-in with the same forwarding surface as the real wrapper."""

            def __init__(self, saver: Any) -> None:
                self.saver = saver

        settings = load_settings(postgres_enabled=False, log_level="WARNING")
        report = await CheckpointJanitor(_Wrapper(engine.checkpointer), settings).run(
            retention_days=0, dry_run=True
        )

        assert report.examined >= 1

    async def test_a_saver_that_cannot_be_listed_is_a_typed_error(self) -> None:
        """A janitor with no listing capability must say so, not delete nothing."""

        class _Opaque:
            """Implements nothing the janitor needs."""

        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        with pytest.raises(PersistenceError):
            await CheckpointJanitor(_Opaque(), settings).run(dry_run=True)

    async def test_a_saver_that_cannot_delete_is_a_typed_error(self) -> None:
        """Deleting must not be silently skipped once threads are selected."""

        class _ListOnly:
            """Can be listed but not deleted."""

            def alist(self, config: Any) -> Any:
                raise AssertionError("not reached: the list is empty")

        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        with pytest.raises(PersistenceError):
            await CheckpointJanitor(_ListOnly(), settings).run(retention_days=0)


class TestDeletion:
    async def test_a_dry_run_changes_nothing(self) -> None:
        engine, run_ids = await _seed(1)
        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        report = await CheckpointJanitor(engine.checkpointer, settings).run(
            retention_days=0, dry_run=True
        )

        assert report.deleted == 0
        # The run must still be there — that is the whole point of a dry run.
        assert (await engine.status(run_ids[0])).run_id == run_ids[0]

    async def test_retention_zero_removes_the_history(self) -> None:
        engine, run_ids = await _seed(1)
        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        report = await CheckpointJanitor(engine.checkpointer, settings).run(retention_days=0)

        assert report.deleted >= 1
        with pytest.raises(Exception):  # noqa: B017 - any typed miss is acceptable
            await engine.history(run_ids[0])

    async def test_stale_reports_the_whole_backlog_beyond_the_pass_limit(self) -> None:
        """`stale` is how old the store is; the limit only bounds the delete.

        A pass bounded at ``limit=2`` over a store with five old threads used to
        report ``stale=2`` because the report was built from the *selected*
        threads. That made ``stale`` read "how stale is the store" but mean "how
        many did this pass reach", so an operator with a backlog could not see
        that one existed — precisely the situation where the number matters. The
        field's own documentation always promised the full count; the report only
        disagreed with it under limit pressure.
        """
        engine, run_ids = await _seed(5)
        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        report = await CheckpointJanitor(engine.checkpointer, settings).run(
            retention_days=0, dry_run=True, limit=2
        )

        assert report.stale >= len(run_ids)
        assert report.deleted == 0

    async def test_the_delete_is_still_bounded_by_the_limit(self) -> None:
        """The fix must not turn the cap into decoration."""
        engine, run_ids = await _seed(5)
        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        report = await CheckpointJanitor(engine.checkpointer, settings).run(
            retention_days=0, limit=2
        )

        assert report.stale >= len(run_ids)
        assert report.deleted == min(report.stale, 2)

    async def test_a_nonzero_window_spares_recent_work(self) -> None:
        """The common case: a sweep on a healthy system must delete nothing.

        A default that deleted recent checkpoints would be a data-loss bug that
        only shows up on a nightly cron, so the safe direction is pinned here.
        This is the assertion that caught the ``meta.ts`` bug: the timestamp is
        not an attribute of ``CheckpointTuple``, so every thread looked like it
        was written at the epoch, and a 30-day window deleted the whole store.
        """
        engine, run_ids = await _seed(1)
        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        report = await CheckpointJanitor(engine.checkpointer, settings).run(retention_days=30)

        assert report.examined >= 1
        assert report.deleted == 0
        # And the work is still addressable afterwards, not merely uncounted.
        assert (await engine.status(run_ids[0])).run_id == run_ids[0]

    async def test_an_undated_checkpoint_is_never_deleted(self) -> None:
        """An entry the sweep cannot age is kept, loudly.

        Failing open is the one unforgivable mistake here: the work would be gone
        irrecoverably, on a schedule, behind a log line saying ``deleted=N``. So
        "cannot tell" is resolved as "do not touch", and the gap is reported.
        """
        engine, _ = await _seed(1)
        settings = load_settings(postgres_enabled=False, log_level="WARNING")

        class _UndatedSaver:
            """Yields the real entries with the timestamp field removed."""

            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def alist(self, config: Any) -> Any:
                return self._strip(self._inner.alist(config))

            async def _strip(self, source: Any) -> Any:
                async for meta in source:
                    checkpoint = dict(getattr(meta, "checkpoint", {}) or {})
                    checkpoint.pop("ts", None)
                    yield _Blanked(meta, checkpoint)

            async def adelete_thread(self, thread_id: str) -> None:
                await self._inner.adelete_thread(thread_id)

        report = await CheckpointJanitor(_UndatedSaver(engine.checkpointer), settings).run(
            retention_days=0
        )

        assert report.examined >= 1
        assert report.deleted == 0

    async def test_a_utcoffset_timestamp_is_understood(self) -> None:
        """LangGraph writes tz-aware ISO-8601; a naive parse misjudges age.

        ``Z`` and ``+00:00`` denote the same instant, so a comparison that
        disagrees between them makes the retention window depend on the saver.
        """
        offset = _checkpoint_timestamp(_Stamped("2020-01-01T00:00:00+00:00"))
        zulu = _checkpoint_timestamp(_Stamped("2020-01-01T00:00:00Z"))

        assert offset == zulu
        assert offset is not None
        assert offset > 0
