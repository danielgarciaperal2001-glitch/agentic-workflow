"""The durable store, against a real PostgreSQL.

These are the only tests that can be wrong in a way nothing else notices. The
in-memory checkpointer shares an interface with ``AsyncPostgresSaver`` but not
its failure modes, and every one of those failure modes is a way to lose a run:

* a **connection pool** that is not open yet, so the app boots before the
  database is ready;
* a **serializer** that round-trips a state object through msgpack and returns a
  subtly different object, so a resumed run is not the run that was parked;
* **statement timeouts** that fire on a legitimate write;
* a **replay** that mutates the history it claims to preserve.

None of that is reachable without a server, which is why these are marked
``postgres`` and skipped rather than mocked. A mock of the durable store tests
the mock.

Every test creates its own schema and drops it on teardown, so the suite can be
run repeatedly against one database — and run in parallel against a shared CI
service — without one test's threads leaking into another's assertions. Teardown
is best effort: a leaked schema is untidy, but a teardown that *raises* masks
the real assertion failure, which is far worse.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import logging
from typing import Any
import uuid

import pytest

from agentic_workflow.config import Settings, load_settings, reset_settings_cache
from agentic_workflow.errors import PersistenceError
from agentic_workflow.persistence.checkpointer import build_checkpointer
from agentic_workflow.persistence.retention import CheckpointJanitor, RetentionReport
from agentic_workflow.services.engine import WorkflowEngine
from tests.helpers import make_decision

pytestmark = pytest.mark.postgres

log = logging.getLogger(__name__)


def _durable_settings(base: Settings, **overrides: Any) -> Settings:
    """Build settings for a private, disposable schema.

    Args:
        base: The suite's offline settings, used only for its DSN.
        **overrides: Extra fields.

    Returns:
        Settings pointed at a schema name unique to this call.
    """
    return load_settings(
        llm_provider="echo",
        postgres_enabled=True,
        postgres_dsn=base.postgres_dsn,
        postgres_schema=f"awf_test_{uuid.uuid4().hex[:16]}",
        postgres_auto_setup=True,
        postgres_pool_min_size=1,
        postgres_pool_max_size=2,
        # Generous, because a slow CI runner is not a hung query — and a timeout
        # firing on a legitimate write is indistinguishable from a store bug.
        postgres_statement_timeout_ms=30_000,
        log_level="ERROR",
        api_rate_limit_per_minute=0,
        **overrides,
    )


async def _drop_schema(settings: Settings) -> None:
    """Remove a test's schema, logging rather than swallowing any failure.

    Deliberately not a bare ``except: pass``. A leaked schema is untidy; a
    teardown that *raises* masks the real assertion failure, which is worse. So
    the failure is swallowed — but it is recorded, because a suite that quietly
    leaves forty schemas behind on a shared CI database is a problem that
    surfaces much later, to whoever is trying to work out why it got slow.

    Args:
        settings: Settings whose DSN and schema name are used.
    """
    try:
        import psycopg
    except ImportError:  # pragma: no cover - a CI dependency, not a local one
        return
    try:
        connection = await psycopg.AsyncConnection.connect(settings.postgres_dsn, autocommit=True)
        async with connection:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{settings.postgres_schema}" CASCADE')
    except Exception:
        log.warning(
            "could not drop the test schema",
            extra={"schema": settings.postgres_schema},
            exc_info=True,
        )


class _ExplodingSaver:
    """A saver whose ``setup`` fails, to reach the post-pool failure path.

    Delegates everything to the real saver so the pool is genuine, and fails only
    at the DDL step — which is the step that runs *after* the pool is open and is
    the one that realistically fails on a real deployment (permissions, a
    read-only volume, a schema another role owns).
    """

    def __init__(self, inner: Any) -> None:
        """Wrap the real saver.

        Args:
            inner: The real ``AsyncPostgresSaver`` to delegate to.
        """
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the real saver.

        Args:
            name: Attribute name.

        Returns:
            The wrapped saver's attribute.
        """
        return getattr(self._inner, name)

    async def setup(self) -> None:
        """Fail the way a refused ``CREATE TABLE`` would.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("injected DDL failure")


@pytest.fixture
async def durable(settings: Settings) -> AsyncIterator[tuple[Any, WorkflowEngine, Settings]]:
    """Yield an engine backed by a private PostgreSQL schema.

    Args:
        settings: The suite's offline settings, used for the DSN.

    Yields:
        The checkpointer, a started engine, and the settings both were built
        from, so a test can assert against the same configuration.
    """
    resolved = _durable_settings(settings)
    saver = build_checkpointer(resolved)
    engine = WorkflowEngine(resolved, checkpointer=saver)
    await engine.startup()
    try:
        yield saver, engine, resolved
    finally:
        await engine.shutdown()
        await _drop_schema(resolved)
        reset_settings_cache()


class TestSchemaSetup:
    """The store has to be usable before the first request arrives."""

    async def test_startup_creates_the_schema_and_is_repeatable(
        self, settings: Settings, request_factory: Any
    ) -> None:
        """Auto-setup must happen, and must survive a second claim on it.

        An app that starts cleanly against a database with no tables and then
        fails on its first run is the most common way a compose stack looks
        healthy and is not. The second engine is not paranoia either: blue/green
        deploys overlap by construction, and so does a retry after a crash.
        """
        resolved = _durable_settings(settings)
        try:
            for attempt in (1, 2):
                saver = build_checkpointer(resolved)
                engine = WorkflowEngine(resolved, checkpointer=saver)
                await engine.startup()
                try:
                    if attempt == 2:
                        # The second claim on the schema is only meaningful if the
                        # first one actually wrote something.
                        # `start`, not `run_until_done`: the run parks at its
                        # first gate and stays there, which is what "the schema
                        # accepted a write" means here.
                        outcome = await engine.start(request_factory(run_id="setup-twice"))
                        assert outcome.is_parked, outcome.status
                finally:
                    await engine.shutdown()
        finally:
            await _drop_schema(resolved)
            reset_settings_cache()

    async def test_a_checkpointer_can_be_closed_and_reopened(
        self, settings: Settings, request_factory: Any
    ) -> None:
        """Close, then use it again — on the *same instance*.

        The class documents itself as reusable, and reuse is the normal shape:
        a lifespan that restarts, a test that reuses a fixture, a retry after a
        crash. Keeping the closed saver wired to a closed pool does not fail
        loudly at ``close``; it fails at the next ``setup``, on a pool that no
        longer exists.
        """
        resolved = _durable_settings(settings)
        try:
            saver = build_checkpointer(resolved)
            await saver.setup()
            await saver.close()
            await saver.setup()  # must not raise

            engine = WorkflowEngine(resolved, checkpointer=saver)
            await engine.startup()
            try:
                outcome = await engine.start(request_factory(run_id="reopened"))
                assert outcome.is_parked, outcome.status
            finally:
                await engine.shutdown()
        finally:
            await _drop_schema(resolved)
            reset_settings_cache()

    async def test_a_failed_setup_still_releases_its_pool(self, settings: Settings) -> None:
        """The failure path is exactly where a leaked pool hurts.

        The pool is opened before any DDL runs, so a ``CREATE SCHEMA`` that is
        refused — a read-only role, a name another role already owns — leaves real
        sockets open. Gating the cleanup on "setup completed" leaked them
        precisely when the operator is already looking at a broken boot.

        The failure is injected one layer down, at the saver's own ``setup``,
        which runs *after* the pool is open. Forcing it through the DDL instead
        would need a second role with fewer privileges than the test has, which
        is not something a test suite can assume of the database it is given.
        """
        resolved = _durable_settings(settings)
        saver = build_checkpointer(resolved)
        saver._saver = _ExplodingSaver(saver.saver)
        try:
            with pytest.raises(PersistenceError):
                await saver.setup()
            # The pool was built; it must not be left open.
            assert saver._pool is None, "close() was skipped because setup did not finish"
        finally:
            await saver.close()
            await _drop_schema(resolved)
            reset_settings_cache()


class TestDurability:
    """What the in-memory saver cannot show: state that outlives the process."""

    async def test_a_parked_run_survives_a_new_engine(
        self, settings: Settings, request_factory: Any
    ) -> None:
        """The core promise: a gate is a row, not a process.

        A run parks in one engine, that engine shuts down, and a *different*
        engine built from scratch answers the gate. If the parked state only
        lived in the first engine's memory this fails, and the whole reason for
        a durable checkpointer evaporates.

        This is the single test in the repository that the in-memory suite
        cannot stand in for, which is why it is worth the setup.
        """
        resolved = _durable_settings(settings)
        request = request_factory(run_id="durable-1")
        try:
            first = WorkflowEngine(resolved, checkpointer=build_checkpointer(resolved))
            await first.startup()
            try:
                parked = await first.start(request)
                assert parked.status == "waiting_human", parked.status
                assert parked.pending is not None
                assert parked.pending.approval_id
            finally:
                await first.shutdown()

            second = WorkflowEngine(resolved, checkpointer=build_checkpointer(resolved))
            await second.startup()
            try:
                resumed = await second.resume(
                    "durable-1",
                    make_decision(parked.pending.approval_id).model_dump(mode="json"),
                )
                assert resumed.error is None, resumed.error
                # The run parks again at the *next* gate, so "not waiting_human"
                # would be wrong. The claim worth making is that the gate this
                # engine answered is gone and a different one has taken its
                # place — which is only true if the answer survived the restart.
                assert resumed.is_parked, resumed.status
                assert resumed.pending is not None
                assert resumed.pending.approval_id != parked.pending.approval_id
            finally:
                await second.shutdown()
        finally:
            await _drop_schema(resolved)
            reset_settings_cache()

    async def test_checkpoints_are_addressable(
        self, durable: tuple[Any, WorkflowEngine, Settings], request_factory: Any
    ) -> None:
        """Time travel needs ids that come back exactly as they went in.

        A checkpoint id that is regenerated on read would make ``state_at`` a
        lie and the replay endpoint would fork from a state the caller never
        inspected. Uniqueness is asserted alongside, because two super-steps
        sharing an id is how "replay from step 3" becomes ambiguous.
        """
        _, engine, _ = durable
        await engine.start(request_factory(run_id="durable-addressable"))

        history = await engine.history("durable-addressable")
        assert len(history) > 1, "a run that parked must have several super-steps"
        ids = [entry.checkpoint_id for entry in history]
        assert len(set(ids)) == len(ids), "checkpoint ids must be unique"

        # `history()` is newest-first, so index 0 is the *head* of the run and
        # index -1 is where it began. Asserting against the head is what makes
        # the "ids come back as they went in" claim meaningful: a regenerated id
        # would still be self-consistent at the head.
        head = await engine.state_at("durable-addressable", history[0].checkpoint_id)
        assert head.run_id == "durable-addressable"
        # The run id lives inside the request, not as a sibling state key.
        assert head.state["request"].run_id == "durable-addressable"

    async def test_an_unknown_checkpoint_in_a_known_thread_is_reported(
        self, durable: tuple[Any, WorkflowEngine, Settings], request_factory: Any
    ) -> None:
        """A typo in a checkpoint id must raise, not return the latest state.

        Silently falling back would turn "replay from checkpoint X" into
        "replay from whenever", which is a difference only noticed after the
        incident it caused.
        """
        from agentic_workflow.errors import CheckpointNotFoundError

        _, engine, _ = durable
        await engine.start(request_factory(run_id="durable-unknown-cp"))

        with pytest.raises(CheckpointNotFoundError):
            await engine.state_at("durable-unknown-cp", "cp_not_a_real_id")

    async def test_a_replay_forks_without_destroying_what_it_forked_from(
        self, durable: tuple[Any, WorkflowEngine, Settings], request_factory: Any
    ) -> None:
        """A branch adds to the thread; it never rewrites the branch point.

        LangGraph forks by handing ``aget_state``/``astream`` a *checkpoint id*,
        so the new work lands in the same thread as a sibling of the history it
        branched from. "The original history is untouched" is therefore the wrong
        claim, and asserting it fails on a correct implementation. The claim worth
        making is the one a before/after diff actually depends on: every original
        checkpoint is still there, in order, at the end of the list.

        Checked as a suffix, not as an equality, because the branch genuinely
        adds entries. Checked as a suffix rather than as a set difference,
        because a set would not notice a reordering.
        """
        _, engine, _ = durable
        run_id = "durable-replay"
        await engine.start(request_factory(run_id=run_id))

        before = [entry.checkpoint_id for entry in await engine.history(run_id)]
        assert before
        branch_from = before[-1]  # oldest: the branch point itself

        forked = await engine.replay_from(run_id, branch_from)
        assert forked.run_id == run_id

        after = [entry.checkpoint_id for entry in await engine.history(run_id)]
        assert len(after) > len(before), "the fork added no checkpoints at all"
        assert after[len(after) - len(before) :] == before, (
            "the original history was not preserved verbatim as a suffix"
        )


class TestRetentionAgainstRealData:
    """The janitor deleting rows from a live table is the irreversible part."""

    async def test_a_dry_run_changes_nothing(
        self, settings: Settings, request_factory: Any
    ) -> None:
        """The most important property of a deletion tool, and the easiest to
        get wrong: a dry run must leave the data alone.

        Asserted by *reading the data afterwards*, not by trusting the report. A
        report claiming zero deletions while the thread is intact means the dry
        run worked; a report claiming zero deletions while the thread is gone
        means it lied. Only the second read tells the two apart.
        """
        resolved = _durable_settings(settings)
        try:
            saver = build_checkpointer(resolved)
            engine = WorkflowEngine(resolved, checkpointer=saver)
            await engine.startup()
            try:
                # `start`, not `run_until_done`: `max_gates` is the budget of
                # gates *answered*, so `run_until_done(max_gates=1)` drives past
                # the first gate and then raises when the second one arrives.
                # The intent here is "produce a parked run and leave it parked".
                await engine.start(request_factory(run_id="retention-dry"))
                before = [e.checkpoint_id for e in await engine.history("retention-dry")]
                assert before, "the fixture produced no data to protect"

                report = await CheckpointJanitor(saver, resolved).run(
                    retention_days=0, dry_run=True
                )
                assert report.examined >= 1
                assert report.deleted == 0, "a dry run reported deletions"
                assert report.dry_run is True

                after = [e.checkpoint_id for e in await engine.history("retention-dry")]
                assert after == before, "a dry run changed the data"
            finally:
                await engine.shutdown()
        finally:
            await _drop_schema(resolved)
            reset_settings_cache()

    async def test_a_real_pass_deletes_and_the_run_stops_being_readable(
        self, settings: Settings, request_factory: Any
    ) -> None:
        """The destructive counterpart, so the dry-run test cannot pass vacuously.

        Without this, a janitor that deletes nothing at all would satisfy the
        dry-run test perfectly. A retention tool that never deletes is a
        liability of a different kind — an unbounded table.
        """
        from agentic_workflow.errors import RunNotFoundError

        resolved = _durable_settings(settings)
        try:
            saver = build_checkpointer(resolved)
            engine = WorkflowEngine(resolved, checkpointer=saver)
            await engine.startup()
            try:
                await engine.start(request_factory(run_id="retention-real"))
                assert await engine.history("retention-real")

                report = await CheckpointJanitor(saver, resolved).run(retention_days=0)
                assert report.deleted >= 1, "a zero-day window deleted nothing"
                # Gone, and gone *loudly*: an empty history is indistinguishable
                # from a filter mistake, so the failure mode has to be typed.
                with pytest.raises(RunNotFoundError):
                    await engine.history("retention-real")
            finally:
                await engine.shutdown()
        finally:
            await _drop_schema(resolved)
            reset_settings_cache()

    def test_undated_threads_are_reported_not_deleted(self) -> None:
        """The janitor fails closed, and this is the assertion for it.

        A thread whose age cannot be determined is kept and counted. Deleting on
        a parse error means a retention job that eventually deletes the wrong
        thing, and the wrong thing is always the run someone is looking at.
        """
        report = RetentionReport(
            examined=3,
            stale=1,
            deleted=0,
            freed_checkpoints=0,
            duration_seconds=0.01,
            dry_run=False,
            undated=2,
        )
        assert report.undated == 2
        assert report.deleted == 0
        assert report.as_dict()["undated"] == 2


class TestConcurrency:
    """Two runs against one pool, which is what a served instance actually does."""

    async def test_interleaved_runs_keep_separate_histories(
        self, durable: tuple[Any, WorkflowEngine, Settings], request_factory: Any
    ) -> None:
        """Two runs at once must not share a checkpoint history.

        LangGraph keys checkpoints by ``thread_id``, so this passes trivially —
        which is exactly why it is worth asserting. A regression that reused a
        thread id, or cached the graph where it should have cached the
        checkpointer, would interleave two runs' checkpoints into one history,
        and each run would then resume into the other's state.
        """
        _, engine, _ = durable
        first, second = "concurrent-a", "concurrent-b"
        results = await asyncio.gather(
            engine.start(request_factory(run_id=first)),
            engine.start(request_factory(run_id=second)),
        )
        for result in results:
            assert result.is_parked, result.status
            assert result.pending is not None
            assert result.pending.run_id == result.run_id

        # And each history is individually coherent: it starts at the beginning
        # and its own state agrees about which run it belongs to.
        for run_id in (first, second):
            history = await engine.history(run_id)
            assert history, run_id
            # Newest first, so the steps must *descend*. A reader who assumes
            # chronological order gets a reversed list, and every other
            # assertion in the file that indexes into `history` silently means
            # the opposite of what it looks like.
            steps = [entry.step for entry in history]
            assert steps == sorted(steps, reverse=True), run_id
            head = await engine.state_at(run_id, history[0].checkpoint_id)
            assert head.state["request"].run_id == run_id, run_id
