"""The run registry: a per-process projection of durable state.

Two contracts matter more than the rest, and both were violated before they were
tested.

**A record cannot hold a status outside the lifecycle vocabulary.** The field is
annotated ``RunStatus``, and a ``dataclass`` enforces nothing, so a value copied
out of the checkpoint's own ``status`` field — where the *graph's* vocabulary
differs from the engine's — landed in the registry as ``"triaged"``. It matched
no status filter, was not in ``TERMINAL_STATUSES``, and so was invisible to
every poller waiting for the run to finish. A poller would have waited forever on
a run that had already completed.

**The registry is rebuildable.** It is per-process, so after a restart it knows
nothing. :meth:`RunRegistry.rebuild_from` is the mechanism, and it must take the
*engine's* projection of a run rather than raw checkpoint state, so that there is
exactly one place in the codebase that knows how to turn a LangGraph snapshot
into a status.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest

from agentic_workflow.persistence.repository import (
    RUN_STATUSES,
    TERMINAL_STATUSES,
    RunRecord,
    RunRegistry,
    RunStatus,
)

pytestmark = pytest.mark.unit


class TestStatusVocabulary:
    """The promises a ``Literal`` annotation cannot keep on its own."""

    def test_the_runtime_set_matches_the_annotation(self) -> None:
        """One source of truth.

        ``RUN_STATUSES`` is derived from ``RunStatus`` rather than restated, so a
        new member of the ``Literal`` cannot be accepted by the type checker and
        then rejected at runtime with no way to tell the two lists apart.
        """
        assert frozenset(get_args(RunStatus)) == RUN_STATUSES

    def test_every_terminal_status_is_a_known_status(self) -> None:
        """A poller asking "has it finished?" must not meet an unknown value.

        ``is_terminal`` is a membership test, so a status outside both sets is
        reported as "still going" — the failure mode is a hang, not an error.
        """
        assert TERMINAL_STATUSES <= RUN_STATUSES

    def test_a_record_refuses_a_status_it_does_not_recognise(self) -> None:
        """Checked at construction, so a bad value never exists at all.

        The alternative — a record that answers ``is_terminal`` and every filter
        with a silent ``False`` — is indistinguishable from a healthy record
        until someone is waiting on it.
        """
        with pytest.raises(ValueError, match="invalid run status"):
            RunRecord(run_id="r1", thread_id="r1", status="triaged")  # type: ignore[arg-type]

    def test_the_graph_vocabulary_is_rejected_explicitly(self) -> None:
        """The exact value that arrived by accident, named in the test.

        Worth its own case: ``"triaged"`` is a perfectly reasonable status for
        the *graph* to report, which is why it is such a plausible thing to
        copy into the wrong field.
        """
        with pytest.raises(ValueError):
            RunRecord(run_id="r1", thread_id="r1", status="triaged")  # type: ignore[arg-type]

    @pytest.mark.parametrize("status", sorted(RUN_STATUSES))
    def test_every_declared_status_is_accepted(self, status: str) -> None:
        """The constraint must not be tighter than the vocabulary.

        A pattern that only allowed one member would satisfy the rejection test
        above and make every other status unreachable.
        """
        assert RunRecord(run_id="r1", thread_id="r1", status=status).status == status  # type: ignore[arg-type]


class TestTerminalStatuses:
    """``is_terminal`` is what a poller uses to stop waiting."""

    @pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
    def test_a_terminal_status_reports_itself(self, status: str) -> None:
        record = RunRecord(run_id="r1", thread_id="r1", status=status)  # type: ignore[arg-type]
        assert record.is_terminal

    @pytest.mark.parametrize("status", sorted(RUN_STATUSES - TERMINAL_STATUSES))
    def test_a_non_terminal_status_does_not(self, status: str) -> None:
        record = RunRecord(run_id="r1", thread_id="r1", status=status)  # type: ignore[arg-type]
        assert not record.is_terminal

    def test_rejection_is_terminal_and_distinct_from_failure(self) -> None:
        """A human's "no" is a conclusion, and it is a *final* one.

        Both halves matter. Terminal, because a rejected run must not leave a
        poller waiting for a decision that has already been made. Distinct from
        ``failed``, because reporting a person's decision as a system fault
        blames the system for the one thing it was built to let them decide.
        """
        record = RunRecord(run_id="r1", thread_id="r1", status="rejected")
        assert record.is_terminal
        assert record.status != "failed"


class TestLifecycle:
    """Create, update, delete."""

    def test_creating_the_same_run_twice_is_refused(self) -> None:
        """Re-registration would silently discard the first record's state."""
        from agentic_workflow.errors import RunAlreadyExistsError

        registry = RunRegistry()
        registry.create("r1")
        with pytest.raises(RunAlreadyExistsError):
            registry.create("r1")

    def test_an_unknown_run_is_not_silently_invented(self) -> None:
        """``get`` raising is what stops a write from creating a phantom run.

        This is the property the engine's ``_record`` helper works around: the
        strict behaviour is correct, and every caller that might be writing
        about a run it did not create has to opt into creating it deliberately.
        """
        from agentic_workflow.errors import RunNotFoundError

        registry = RunRegistry()
        with pytest.raises(RunNotFoundError):
            registry.get("never-created")

    def test_update_clears_a_field_only_when_asked(self) -> None:
        """Omitted and ``None`` mean different things.

        ``status=None`` is not a request to clear the status — it is a caller
        that had nothing to say about it. Collapsing the two would erase a run's
        status every time something updated its iteration.
        """
        registry = RunRegistry()
        registry.create("r1")
        registry.update("r1", status="running", pending_approval="apr_1")

        registry.update("r1", iteration=3)
        assert registry.get("r1").status == "running"
        assert registry.get("r1").pending_approval == "apr_1"

        registry.update("r1", pending_approval=None)
        assert registry.get("r1").pending_approval is None
        assert registry.get("r1").status == "running"


class TestRebuild:
    """Boot-time hydration: the projection has to be reconstructible."""

    def test_it_registers_every_projection_it_is_given(self) -> None:
        registry = RunRegistry()
        created = registry.rebuild_from(
            {
                "r1": {"status": "waiting_human", "iteration": 2, "pending_approval": "apr_1"},
                "r2": {"status": "completed", "iteration": 5},
            }
        )
        assert created == 2
        assert registry.get("r1").status == "waiting_human"
        assert registry.get("r1").pending_approval == "apr_1"
        assert registry.get("r2").iteration == 5

    def test_a_thread_with_no_projection_is_registered_as_pending(self) -> None:
        """Unreadable is not nonexistent, and only the listing tells them apart.

        Registering nothing would make a run we failed to read disappear, which
        is indistinguishable from a run that was never started — and the operator
        is looking at a real one.
        """
        registry = RunRegistry()
        registry.rebuild_from({"r1": {}})
        assert registry.get("r1").status == "pending"
        assert registry.get("r1").iteration == 0

    def test_it_never_overwrites_a_run_this_process_already_tracks(self) -> None:
        """Live state beats a reconstruction of it.

        Hydration runs at boot before anything else, but it is still a
        projection: if this process knows something newer, the projection is the
        stale copy and must lose.
        """
        registry = RunRegistry()
        registry.create("r1")
        registry.update("r1", status="running", iteration=7, pending_approval="live")

        registry.rebuild_from({"r1": {"status": "pending", "iteration": 0}})

        record = registry.get("r1")
        assert record.status == "running"
        assert record.iteration == 7
        assert record.pending_approval == "live"

    def test_hydration_is_idempotent(self) -> None:
        """Boot happens more than once in a test and in a retry.

        Calling it twice must not duplicate the work or change the result, or the
        ``created`` count — the number a boot log reports — becomes a lie.
        """
        registry = RunRegistry()
        entries: dict[str, dict[str, Any]] = {"r1": {"status": "completed"}}
        assert registry.rebuild_from(entries) == 1
        assert registry.rebuild_from(entries) == 0
        assert len(registry) == 1


class TestEviction:
    """A bounded registry must drop something predictably."""

    def test_the_oldest_run_is_evicted_first(self) -> None:
        """Recency, not insertion order.

        Touching a run is what protects it, and the eviction order is what makes
        "touch it to keep it" true. Sorted by run id instead, a busy run would be
        evicted while an abandoned one survived.
        """
        registry = RunRegistry(max_entries=2)
        registry.create("r1")
        registry.create("r2")
        registry.update("r2", iteration=1)  # r2 is now the most recent
        registry.create("r3")

        assert "r1" not in registry
        assert {record.run_id for record in registry.list_runs(limit=10)} == {"r2", "r3"}

    def test_a_ttl_expires_a_run_that_was_never_touched(self) -> None:
        """Age is measured from ``updated_at`` so a heartbeat keeps a run alive.

        A long run that is still working must survive a sweep that a run nobody
        has looked at in an hour would not.
        """
        registry = RunRegistry(ttl_seconds=3600)
        registry.create("r1")
        assert registry.purge_expired(now=0.0) == 0, "nothing has aged yet"
        assert len(registry) == 1
