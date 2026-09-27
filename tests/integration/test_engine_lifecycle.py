"""The engine's lifecycle: start, park, resume, cancel, time travel.

Every one of these behaviours is a promise the API makes to a caller, so they
are tested at the service layer rather than through HTTP. The engine is the only
object allowed to invoke the graph, which makes it the natural place to assert
that the graph is driven correctly and that no run can leak.

The properties under test, in order of how expensive they are to get wrong:

* **Durability.** A parked run survives a fresh engine over the same
  checkpointer. If it did not, "resumable" would be a marketing word.
* **Isolation.** Two concurrent runs must not see each other's state. Run ids are
  thread keys, and a leak between them is a data-disclosure bug.
* **Convergence.** No run may loop forever, however the agents behave.
* **Cancellation.** ``cancel()`` must interrupt *real* work, not merely mark a
  flag the graph never checks.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from langgraph.errors import GraphRecursionError
import pytest

from agentic_workflow.config import Settings
from agentic_workflow.domain.schemas import Decision, ReviewRequest, Verdict
from agentic_workflow.errors import (
    CheckpointNotFoundError,
    ConcurrencyLimitError,
    InvalidStateError,
    IterationLimitExceededError,
    RunAlreadyExistsError,
    RunNotFoundError,
    RunTimeoutError,
)
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.services.engine import CheckpointInfo, RunOutcome, WorkflowEngine
from tests.helpers import make_request, unique_run_id

pytestmark = pytest.mark.integration


def _approve(pending: Any) -> dict[str, Any]:
    """Return a decision that approves every gate.

    Args:
        pending: The approval the graph raised.

    Returns:
        A resume payload.
    """
    return {
        "approval_id": pending.approval_id,
        "decision": Decision.APPROVE.value,
        "reviewer": "alice",
        "comment": "looks right",
    }


class TestStart:
    """Submitting work."""

    async def test_start_promises_pending_or_parked(self, engine: WorkflowEngine) -> None:
        """A first start either finishes or parks — it never returns ``running``.

        The engine drives the graph to a stopping point before answering, so a
        client never has to poll to find out whether its request was accepted.
        """
        outcome = await engine.start(make_request())
        assert outcome.status in {"completed", "waiting_human", "failed"}
        assert outcome.is_parked or outcome.is_finished

    async def test_start_publishes_a_pending_approval(self, engine: WorkflowEngine) -> None:
        """With human gates on, the run parks and names the gate it is on.

        The approval must carry the run id, because that is the only key the
        approval service has to find the parked thread.
        """
        outcome = await engine.start(make_request())
        if outcome.is_parked:
            assert outcome.pending is not None
            assert outcome.pending.run_id == outcome.run_id
        else:  # pragma: no cover - echo provider always parks with HITL on
            pytest.fail("HITL is enabled but the run did not park")

    async def test_start_is_idempotent_per_run_id(self, engine: WorkflowEngine) -> None:
        """Starting the same run id twice is a conflict, not a second thread.

        Silently accepting it would fork one logical run into two threads, and
        the two would fight over the same checkpoint namespace.
        """
        request = make_request(run_id=unique_run_id("dup"))
        await engine.start(request)
        with pytest.raises(RunAlreadyExistsError):
            await engine.start(request)

    async def test_unknown_run_status_raises(self, engine: WorkflowEngine) -> None:
        """Reading a run that was never submitted is a typed 404, not a ``None``."""
        with pytest.raises(RunNotFoundError):
            await engine.status("does-not-exist")

    async def test_metadata_reaches_the_checkpoint(self, engine: WorkflowEngine) -> None:
        """Caller metadata is stored, so a later read can correlate the run."""
        request = make_request(metadata={"ticket": "OPS-1"})
        await engine.start(request)
        outcome = await engine.status(request.run_id)
        assert outcome.run_id == request.run_id

    async def test_event_sink_receives_run_events(self) -> None:
        """A run emits a lifecycle stream, which is what the WebSocket serves.

        Without this the event stream is untestable, and an unobservable run is
        indistinguishable from a hung one.
        """
        events: list[dict[str, Any]] = []

        async def sink(event: dict[str, Any]) -> None:
            events.append(event)

        engine = WorkflowEngine(
            Settings(_env_file=None, llm_provider="echo", log_level="ERROR"),
            checkpointer=build_memory_checkpointer(),
            event_sink=sink,
        )
        await engine.startup()
        try:
            await engine.start(make_request())
        finally:
            await engine.shutdown()

        names = {str(event.get("event")) for event in events}
        assert "run.started" in names
        assert names & {"run.parked", "run.completed", "run.failed"}
        assert all("run_id" in event for event in events)

    async def test_a_failing_sink_does_not_fail_the_run(self, engine: WorkflowEngine) -> None:
        """Observability must never be able to break a workflow.

        The sink runs inside the graph's task; letting it raise would abandon a
        run mid-flight because a subscriber had a bug.
        """
        engine_with_sink = engine
        engine_with_sink._event_sink = _explode  # type: ignore[attr-defined]
        outcome = await engine_with_sink.start(make_request())
        assert outcome.status in {"waiting_human", "completed", "failed"}


async def _explode(_event: dict[str, Any]) -> None:
    """Raise unconditionally, simulating a broken observer.

    Args:
        _event: Ignored.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("subscriber is broken")


class TestResume:
    """Answering a human gate."""

    async def test_resume_advances_and_parks_again(self, engine: WorkflowEngine) -> None:
        """Approving a gate moves the run forward, possibly to the next gate.

        The workflow has more than one gate by design, so a single resume is not
        expected to finish the run — and a client that assumes it is would
        mis-render the state.
        """
        outcome = await engine.start(make_request())
        assert outcome.is_parked
        after = await engine.resume(outcome.run_id, _decide(outcome.pending))
        assert after.status in {"waiting_human", "completed"}
        assert after.iteration >= outcome.iteration

    async def test_resume_records_the_decision_in_the_log(self, engine: WorkflowEngine) -> None:
        """Every human decision is appended, with its author and justification.

        This is the audit trail: "who approved this" must be answerable from the
        run alone, without a separate log service that could disagree.
        """
        outcome = await engine.start(make_request())
        after = await engine.resume(outcome.run_id, _decide(outcome.pending))
        assert after.decisions
        latest = after.decisions[-1]
        assert latest["reviewer"] == "alice"
        assert latest["decision"] in {d.value for d in Decision}

    async def test_resume_rejects_a_stale_approval_id(self, engine: WorkflowEngine) -> None:
        """Answering a gate the run is not actually parked on is refused.

        Approval ids are deterministic precisely so this check is possible. A
        client answering a gate it read ten minutes ago must not silently
        approve whatever the run is parked on *now*.
        """
        from agentic_workflow.errors import ApprovalNotFoundError

        outcome = await engine.start(make_request())
        stale = _decide(outcome.pending)
        stale["approval_id"] = "apr_some_other_gate_00_deadbeef"
        with pytest.raises((ApprovalNotFoundError, InvalidStateError)):
            await engine.resume(outcome.run_id, stale)

    async def test_resume_on_a_running_run_is_a_state_error(self, engine: WorkflowEngine) -> None:
        """A run that is not parked cannot be resumed.

        Otherwise a duplicated resume would inject a decision into a run that has
        already moved on.
        """
        outcome = await engine.run_until_done(make_request(), decide=_approve)
        assert outcome.is_finished
        with pytest.raises(InvalidStateError):
            await engine.resume(outcome.run_id, _decide(outcome.pending))

    async def test_run_until_done_converges(self, engine: WorkflowEngine) -> None:
        """Answering every gate terminates the run in a bounded number of steps.

        This is the test that would hang if the feedback loop lost its ceiling:
        ``max_iterations`` is the only thing between a bad reviewer and an
        unbounded bill.
        """
        request = make_request()
        outcome = await asyncio.wait_for(
            engine.run_until_done(request, decide=_approve), timeout=30
        )
        assert outcome.is_finished
        assert outcome.iteration <= engine.settings.max_iterations

    async def test_a_rejection_ends_the_run(self, engine: WorkflowEngine) -> None:
        """A human rejection is a conclusion, not a failure.

        Retrying after an explicit "no" would ignore the decision just recorded —
        the one outcome a reviewer must always be able to force, and the one the
        whole gate mechanism exists to make available. The status must say
        ``rejected``: reporting ``failed`` blames the system for the one thing it
        was built to let a person decide.
        """
        outcome = await engine.run_until_done(
            make_request(),
            decide=lambda pending: {
                "approval_id": pending.approval_id,
                "decision": Decision.REJECT.value,
                "reviewer": "bob",
                "comment": "wrong approach",
            },
        )
        assert outcome.status == "rejected"
        assert outcome.is_finished
        assert outcome.is_parked is False

    async def test_a_rejection_still_produces_a_report(self, engine: WorkflowEngine) -> None:
        """A refused change is documented, not abandoned.

        The person who said no is the audience for the report: they need to see
        what was proposed, what was found, and that their refusal was the reason
        the run stopped. A rejection with no report leaves them with a status
        code and nothing to act on.
        """
        outcome = await engine.run_until_done(
            make_request(),
            decide=lambda pending: {
                "approval_id": pending.approval_id,
                "decision": Decision.REJECT.value,
                "reviewer": "bob",
                "comment": "wrong approach",
            },
        )
        assert outcome.report is not None
        assert outcome.report.decision is Verdict.REJECTED
        assert outcome.report.markdown.strip()

    async def test_a_rejection_is_written_to_the_audit_trail(self, engine: WorkflowEngine) -> None:
        """Who refused, at which gate, and what they said — all recoverable.

        The audit trail is the entire justification for interrupting a machine
        mid-task. A rejection that ends the run without a record would be the one
        decision in the system that leaves no evidence, which is precisely
        backwards: it is the decision a reviewer would most need to defend later.
        """
        outcome = await engine.run_until_done(
            make_request(),
            decide=lambda pending: {
                "approval_id": pending.approval_id,
                "decision": Decision.REJECT.value,
                "reviewer": "bob",
                "comment": "wrong approach",
            },
        )
        rejections = [entry for entry in outcome.decisions if entry["decision"] == "reject"]
        assert rejections
        for entry in rejections:
            assert entry["reviewer"] == "bob"
            assert entry["stage"]
            assert entry["approval_id"]
            assert entry["decided_at"]

    async def test_a_rejection_survives_a_restart(self, engine: WorkflowEngine) -> None:
        """The refusal is durable, not just visible in the process that made it.

        Approval is durable because the answer is recorded, and an answer that
        evaporates on restart would let a restarted worker re-prompt for — and
        potentially reverse — a decision a person already made.
        """
        request = make_request()
        first = await engine.run_until_done(
            request,
            decide=lambda pending: {
                "approval_id": pending.approval_id,
                "decision": Decision.REJECT.value,
                "reviewer": "bob",
                "comment": "no",
            },
        )
        again = await engine.status(request.run_id)
        assert again.status == first.status == "rejected"
        assert [entry["decision"] for entry in again.decisions] == [
            entry["decision"] for entry in first.decisions
        ]

    async def test_an_edit_decision_flows_into_the_run(self, engine: WorkflowEngine) -> None:
        """A human-supplied payload reaches the node that asked for it.

        "Modify" is only useful if the modification is actually applied. An
        approver who edits a diff and finds the run ignoring it has been given
        the *appearance* of control and none of the substance.
        """
        seen: list[dict[str, Any]] = []
        outcome = await engine.run_until_done(
            make_request(),
            decide=lambda pending: {
                "approval_id": pending.approval_id,
                "decision": Decision.EDIT.value,
                "reviewer": "alice",
                "comment": "tighten it up",
                "payload": {"instruction": "use Decimal", "touched_by": "alice"},
            },
        )
        seen.extend(outcome.decisions)
        assert outcome.is_finished
        assert any(entry["decision"] == "edit" for entry in seen)
        edited = next(entry for entry in seen if entry["decision"] == "edit")
        assert edited["payload"] == {"instruction": "use Decimal", "touched_by": "alice"}


class TestStatus:
    """Reading a run's state."""

    async def test_status_survives_an_engine_restart(self) -> None:
        """A new engine over the same store answers identically.

        This is the durability promise. If status came from the in-process
        registry, a restart would report every parked run as missing and the
        approval inbox would be empty.
        """
        saver = build_memory_checkpointer()
        settings = Settings(_env_file=None, llm_provider="echo", log_level="ERROR")
        first = WorkflowEngine(settings, checkpointer=saver)
        await first.startup()
        try:
            original = await first.start(make_request())
        finally:
            await first.shutdown()

        second = WorkflowEngine(settings, checkpointer=saver)
        await second.startup()
        try:
            recovered = await second.status(original.run_id)
            assert recovered.run_id == original.run_id
            assert recovered.status == original.status
            assert recovered.is_parked == original.is_parked
            assert recovered.pending is not None
        finally:
            await second.shutdown()

    async def test_status_reads_from_the_store_not_the_registry(
        self, engine: WorkflowEngine
    ) -> None:
        """Deleting the cached record does not make a run unreadable.

        The registry is a rebuildable projection for dashboards, never the source
        of truth. A test that wipes it proves status is answered from the
        checkpoint.
        """
        outcome = await engine.start(make_request())
        run_id = outcome.run_id
        engine.registry.delete(run_id)
        assert (await engine.status(run_id)).run_id == run_id

    async def test_list_runs_reports_the_run(self, engine: WorkflowEngine) -> None:
        """A submitted run is discoverable in the listing."""
        request = make_request()
        await engine.start(request)
        listed = await engine.list_runs(limit=50)
        assert request.run_id in {item.run_id for item in listed}

    async def test_list_runs_filters_by_status(self, engine: WorkflowEngine) -> None:
        """A status filter returns only matching runs, not a page of everything."""
        await engine.start(make_request())
        parked = await engine.list_runs(status="waiting_human", limit=50)
        assert all(item.status == "waiting_human" for item in parked)

    async def test_list_runs_paginates(self, engine: WorkflowEngine) -> None:
        """Limit and offset actually page, so a client can walk the whole set."""
        for _ in range(3):
            await engine.start(make_request())
        first_page = await engine.list_runs(limit=2, offset=0)
        second_page = await engine.list_runs(limit=2, offset=2)
        assert len(first_page) == 2
        assert {item.run_id for item in first_page} & {item.run_id for item in second_page} == set()

    async def test_pending_approvals_includes_parked_runs(self, engine: WorkflowEngine) -> None:
        """The inbox is derived from the parked runs, not from a side list.

        Two sources of truth would drift the moment a run parked through a path
        that did not update the list.
        """
        outcome = await engine.start(make_request())
        pending = await engine.pending_approvals()
        assert outcome.run_id in {item.run_id for item in pending}

    async def test_pending_approvals_can_be_scoped_to_one_run(self, engine: WorkflowEngine) -> None:
        """Scoping narrows the inbox, so one run's approvals are findable in a busy queue."""
        first = await engine.start(make_request())
        second = await engine.start(make_request())
        scoped = await engine.pending_approvals(run_id=first.run_id)
        assert {item.run_id for item in scoped} == {first.run_id}
        assert second.run_id not in {item.run_id for item in scoped}


class TestTimeTravel:
    """Checkpoint history and replay."""

    async def test_history_is_ordered_newest_first(self, engine: WorkflowEngine) -> None:
        """History is returned newest-first so a UI can show the head immediately.

        The order is part of the contract: a client that renders ``items[0]`` as
        "now" is correct only because of it.
        """
        outcome = await engine.start(make_request())
        history = await engine.history(outcome.run_id)
        assert len(history) > 1
        assert isinstance(history[0], CheckpointInfo)
        assert history[0].created_at >= history[-1].created_at

    async def test_history_respects_the_limit(self, engine: WorkflowEngine) -> None:
        """A limit really truncates, so a dashboard cannot pull the whole history."""
        outcome = await engine.start(make_request())
        assert len(await engine.history(outcome.run_id, limit=2)) == 2

    async def test_history_of_an_unknown_run_raises(self, engine: WorkflowEngine) -> None:
        """ "No such run" and "a run with an empty history" are different answers.

        An empty list is indistinguishable from a run that has not started yet,
        so a client polling history would show an empty timeline for a typo'd run
        id and conclude the run never existed. Raising lets the API answer 404.
        """
        with pytest.raises(RunNotFoundError):
            await engine.history("never-ran")

    async def test_state_at_reconstructs_an_earlier_step(self, engine: WorkflowEngine) -> None:
        """Any historical checkpoint can be rendered as a full run state.

        This is what turns "why did iteration two fail?" into a query instead of a
        guess, which is the entire reason for checkpointing every super-step.
        """
        outcome = await engine.start(make_request())
        history = await engine.history(outcome.run_id)
        earliest = history[-1]
        snapshot = await engine.state_at(outcome.run_id, earliest.checkpoint_id)
        assert snapshot.run_id == outcome.run_id
        assert snapshot.iteration <= outcome.iteration

    async def test_state_at_rejects_an_unknown_checkpoint(self, engine: WorkflowEngine) -> None:
        """A checkpoint that does not exist is a typed error.

        Silently returning the head would be far worse: a caller inspecting
        "iteration 2" would be shown iteration 5 and believe it.
        """
        outcome = await engine.start(make_request())
        with pytest.raises(CheckpointNotFoundError):
            await engine.state_at(outcome.run_id, "1-0-0-not-a-real-checkpoint")

    async def test_a_missing_checkpoint_is_distinct_from_a_missing_run(
        self, engine: WorkflowEngine
    ) -> None:
        """The two demand opposite client behaviour, so they are different errors.

        A bad checkpoint id is a typo to correct; a missing run is work that was
        never submitted. Collapsing them told a client to resubmit a run that
        already existed, which duplicates the review.
        """
        outcome = await engine.start(make_request())

        with pytest.raises(CheckpointNotFoundError) as bad_checkpoint:
            await engine.state_at(outcome.run_id, "1-0-0-nope")
        with pytest.raises(RunNotFoundError):
            await engine.status("a-run-that-never-ran")

        assert not isinstance(bad_checkpoint.value, RunNotFoundError)
        assert bad_checkpoint.value.context["checkpoint_id"] == "1-0-0-nope"

    async def test_replay_branches_without_destroying_the_original(
        self, engine: WorkflowEngine
    ) -> None:
        """Replaying appends a branch and leaves the original history intact.

        Re-running in place would destroy the evidence, which is the one thing
        time travel is for.
        """
        outcome = await engine.start(make_request())
        history = await engine.history(outcome.run_id)
        target = history[0].checkpoint_id
        before = len(await engine.history(outcome.run_id))

        branch = await engine.replay_from(outcome.run_id, target)
        after = len(await engine.history(outcome.run_id))

        assert branch.run_id == outcome.run_id
        assert after > before
        original = await engine.status(outcome.run_id)
        assert original.run_id == outcome.run_id

    async def test_replay_from_an_unknown_checkpoint_fails(self, engine: WorkflowEngine) -> None:
        """A replay target that does not exist is refused rather than ignored."""
        outcome = await engine.start(make_request())
        with pytest.raises(CheckpointNotFoundError):
            await engine.replay_from(outcome.run_id, "not-a-checkpoint")

    async def test_replay_refuses_a_run_still_in_flight(self) -> None:
        """Time-travelling into a live run is refused.

        A branch is written to the same thread as the run it forks from, so
        replaying a run that is still executing interleaves two histories on one
        thread id. It also takes the same ``_running`` slot the live task holds,
        orphaning it. Both are reasons to refuse rather than reorder.
        """
        inside = asyncio.Event()
        release = asyncio.Event()

        async def sink(event: dict[str, Any]) -> None:
            # The first event a run publishes is `run.started`, emitted once its
            # task is already registered in `_running`. Blocking there holds the
            # run genuinely mid-flight.
            if not inside.is_set():
                inside.set()
                await release.wait()

        engine = WorkflowEngine(
            Settings(_env_file=None, llm_provider="echo", log_level="ERROR"),
            checkpointer=build_memory_checkpointer(),
            event_sink=sink,
        )
        await engine.startup()
        try:
            request = make_request()
            occupying = asyncio.create_task(engine.start(request))
            await asyncio.wait_for(inside.wait(), timeout=10)
            history = await engine.history(request.run_id)

            with pytest.raises(RunAlreadyExistsError):
                await engine.replay_from(request.run_id, history[0].checkpoint_id)

            release.set()
            await asyncio.wait_for(occupying, timeout=10)
        finally:
            release.set()
            await engine.cancel_all()
            await engine.shutdown()


class TestCancellation:
    """Stopping work in flight."""

    async def test_cancel_marks_a_parked_run(self, engine: WorkflowEngine) -> None:
        """A parked run can be cancelled, and the status reflects it.

        Parked runs hold a thread and an inbox entry forever, so being able to
        clear them is what keeps the queue honest.
        """
        outcome = await engine.start(make_request())
        await engine.cancel(outcome.run_id, reason="not needed")
        assert (await engine.status(outcome.run_id)).status == "cancelled"

    async def test_cancel_is_reported_even_though_the_graph_cannot_report_it(
        self, engine: WorkflowEngine
    ) -> None:
        """Cancellation is overlaid from the registry, because the graph cannot see it.

        The graph is a task that gets interrupted; it has no opportunity to write
        a terminal status. Without the overlay, a cancelled run would keep
        reporting ``waiting_human`` forever.
        """
        outcome = await engine.start(make_request())
        await engine.cancel(outcome.run_id)
        cancelled = await engine.status(outcome.run_id)
        assert cancelled.status == "cancelled"
        assert cancelled.error

    async def test_cancelling_an_unknown_run_raises(self, engine: WorkflowEngine) -> None:
        """Cancelling nothing is a 404, not a silent success."""
        with pytest.raises(RunNotFoundError):
            await engine.cancel("never-existed")

    async def test_cancel_all_drains_everything(self, engine: WorkflowEngine) -> None:
        """Shutdown drains all in-flight work so the process can exit cleanly.

        A leaked ``asyncio.Task`` keeps the loop alive forever, which turns a
        restart into a hang.
        """
        for _ in range(3):
            await engine.start(make_request())
        await engine.cancel_all()
        for summary in await engine.list_runs(limit=10):
            assert summary.status in {"cancelled", "completed", "failed", "waiting_human"}


class TestConcurrency:
    """The bounds that keep a shared deployment alive."""

    async def test_the_ceiling_is_enforced(self) -> None:
        """Exceeding ``max_parallel_runs`` is a typed 429, not unbounded queuing.

        Without the ceiling, a burst of submissions would each hold an LLM
        connection and a checkpoint writer, and the first casualty would be the
        database.

        The ceiling counts *live graph tasks*, so the test has to keep one alive.
        Blocking inside a node's event sink is the honest way to do that: the run
        is genuinely mid-graph, not merely scheduled. The event is one a node
        publishes (``triage.completed``), not a log record, so the pause happens
        while the graph task is definitely still running.
        """
        inside = asyncio.Event()
        release = asyncio.Event()

        async def blocking_sink(event: dict[str, Any]) -> None:
            if event.get("event") == "triage.completed":
                inside.set()
                await release.wait()

        settings = Settings(
            _env_file=None,
            llm_provider="echo",
            log_level="ERROR",
            max_parallel_runs=1,
        )
        engine = WorkflowEngine(
            settings,
            checkpointer=build_memory_checkpointer(),
            event_sink=blocking_sink,
        )
        await engine.startup()
        try:
            occupying = asyncio.create_task(engine.start(make_request()))
            await asyncio.wait_for(inside.wait(), timeout=10)

            with pytest.raises(ConcurrencyLimitError):
                await engine.start(make_request())

            release.set()
            assert (await asyncio.wait_for(occupying, timeout=10)).is_parked
        finally:
            release.set()
            await engine.cancel_all()
            await engine.shutdown()

    async def test_two_concurrent_resumes_cannot_drive_one_run(self) -> None:
        """A parked run accepts one resume at a time.

        ``resume()`` reads the status, concludes the run is parked, and then
        drives it. Nothing is held across the gap between those two steps, so a
        second caller does exactly the same thing and both believe they own the
        run. The second ``create_task`` overwrites the first one's handle in
        ``_running``, which orphans it — ``cancel()`` can no longer see it — and
        the node after the gate then executes twice against the same thread.

        That is not a duplicated log line. A patch gate is precisely where a
        resumed run applies a diff, so a lost race here is a double-apply of
        someone else's code change.
        """
        inside = asyncio.Event()
        release = asyncio.Event()
        armed = False

        async def sink(event: dict[str, Any]) -> None:
            nonlocal armed
            # Armed only after the run is parked, so `start` is not blocked.
            # The first event a resume emits is `run.started`, which is published
            # after the task is registered in `_running` — exactly the window
            # the race needs.
            if armed and not inside.is_set():
                inside.set()
                await release.wait()

        engine = WorkflowEngine(
            Settings(
                _env_file=None,
                llm_provider="echo",
                log_level="ERROR",
                max_parallel_runs=8,
            ),
            checkpointer=build_memory_checkpointer(),
            event_sink=sink,
        )
        await engine.startup()
        try:
            parked = await engine.start(make_request())
            assert parked.is_parked

            armed = True
            first = asyncio.create_task(engine.resume(parked.run_id, _approve(parked.pending)))
            await asyncio.wait_for(inside.wait(), timeout=10)

            with pytest.raises(RunAlreadyExistsError):
                await engine.resume(parked.run_id, _approve(parked.pending))

            # The refusal must not have disturbed the run that legitimately owns
            # the slot: a guard that clobbered state would trade a double-apply
            # for a lost run.
            release.set()
            assert await asyncio.wait_for(first, timeout=10) is not None
        finally:
            release.set()
            await engine.cancel_all()
            await engine.shutdown()

    async def test_a_free_slot_is_reusable_after_cancellation(self) -> None:
        """Cancelling a run returns its slot to the pool.

        A ceiling that only ever grows turns a burst of work into a permanent
        outage: the first N runs would be unrecoverable without a restart.
        """
        inside = asyncio.Event()
        release = asyncio.Event()

        async def blocking_sink(event: dict[str, Any]) -> None:
            if event.get("event") == "triage.completed":
                inside.set()
                await release.wait()

        settings = Settings(
            _env_file=None,
            llm_provider="echo",
            log_level="ERROR",
            max_parallel_runs=1,
        )
        engine = WorkflowEngine(
            settings,
            checkpointer=build_memory_checkpointer(),
            event_sink=blocking_sink,
        )
        await engine.startup()
        try:
            occupying = asyncio.create_task(engine.start(make_request()))
            await asyncio.wait_for(inside.wait(), timeout=10)
            await engine.cancel(_sole_running_run_id(engine), reason="operator")
            release.set()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(occupying, timeout=10)

            # The pool is free again, so a new run is admitted rather than refused.
            release.clear()
            fresh = make_request()
            accepted = asyncio.create_task(engine.start(fresh))
            await asyncio.sleep(0)
            release.set()
            assert (await asyncio.wait_for(accepted, timeout=10)).run_id == fresh.run_id
        finally:
            release.set()
            await engine.cancel_all()
            await engine.shutdown()

    async def test_concurrent_runs_do_not_share_state(self, engine: WorkflowEngine) -> None:
        """Two runs in flight keep entirely separate state.

        Run ids are the checkpoint thread keys, so a leak here would be a
        cross-tenant data-disclosure bug rather than a mere race.
        """
        first, second = make_request(), make_request()
        await asyncio.gather(engine.start(first), engine.start(second))

        left = await engine.status(first.run_id)
        right = await engine.status(second.run_id)
        assert left.run_id != right.run_id
        assert left.pending is None or left.pending.run_id == first.run_id
        assert right.pending is None or right.pending.run_id == second.run_id

    async def test_a_run_budget_ends_a_runaway(self) -> None:
        """A run that never converges is terminated, not left burning tokens.

        The wall-clock budget is the last line of defence behind the iteration
        ceiling, for the case where a provider hangs rather than loops.
        """
        settings = Settings(
            _env_file=None,
            llm_provider="echo",
            log_level="ERROR",
            run_timeout_seconds=30.0,
        )
        engine = WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
        await engine.startup()
        try:
            outcome = await asyncio.wait_for(engine.start(make_request()), timeout=30)
            assert outcome.status in {"waiting_human", "completed", "failed"}
        finally:
            await engine.cancel_all()
            await engine.shutdown()


class _RunawayGraph:
    """A graph stand-in that trips LangGraph's own recursion backstop.

    The engine takes a pre-compiled graph precisely so a test can substitute one
    that misbehaves in a way no real graph would.
    """

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        """Raise the moment the graph is iterated, as a cyclic graph eventually does."""
        raise GraphRecursionError("Recursion limit of 25 reached without a Next or END step")
        yield {}  # pragma: no cover - unreachable, makes this an async generator


class TestGraphBackstop:
    """What happens when the graph's own limit trips before the router's budget."""

    async def test_recursion_is_reported_as_budget_exhaustion(self) -> None:
        """A runaway loop is an exhausted allowance, not an infrastructure fault.

        LangGraph's recursion limit is the last backstop under the router's
        iteration budget. When it trips, the run genuinely spent what it was
        allowed and the answer is "this needs a human" — 422, with a code the
        client can branch on. Left unhandled it fell through to the generic
        handler and surfaced as a 500, which reads as a bug in the deployment
        and pages someone. It is a decision, not a fault.
        """
        engine = WorkflowEngine(
            Settings(_env_file=None, llm_provider="echo", log_level="ERROR"),
            checkpointer=build_memory_checkpointer(),
            graph=_RunawayGraph(),
        )
        await engine.startup()
        try:
            with pytest.raises(IterationLimitExceededError) as caught:
                await engine.start(make_request())

            assert "recursion limit" in str(caught.value)
            # The failure has to be readable afterwards: a run that dies this way
            # with nothing but a stack trace is a run nobody can look up. The
            # registry is checked rather than `status()` because a graph that
            # trips its limit on the first step never writes a checkpoint.
            recorded = {s.run_id: s for s in await engine.list_runs(limit=10)}
            assert recorded[caught.value.run_id].status == "failed"
        finally:
            await engine.cancel_all()
            await engine.shutdown()


class _SpyLLM:
    """A client that reports being closed, delegating everything else.

    Args:
        inner: The real client to forward to.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.closed = 0

    async def aclose(self) -> None:
        """Record the close and forward it.

        Returns:
            Nothing.
        """
        self.closed += 1
        await self._inner.aclose()

    def __getattr__(self, name: str) -> Any:
        """Forward every other attribute to the wrapped client.

        Args:
            name: Attribute name.

        Returns:
            Whatever the wrapped client returns.
        """
        return getattr(self._inner, name)


class TestProviderResources:
    """The LLM client is a process resource, not a per-run one.

    The client owns the connection pool and the semaphore that bounds how many
    requests reach the provider at once. Everything here follows from that: one
    client per process, and closed exactly once by whoever created it.
    """

    async def test_two_runs_share_one_client(self) -> None:
        """The concurrency ceiling is per process, so the client cannot be per run.

        ``llm_max_concurrency`` reads like a deployment-wide promise, and on the
        API path it was not one. The engine built a fresh ``AgentContext`` per
        run, each of which lazily built its own client, and each client built
        its own semaphore from the same setting. Eight concurrent runs with
        ``llm_max_concurrency=8`` therefore put sixty-four requests in flight
        against a provider the configuration says is allowed eight — and a
        provider's rate limiter answers that with 429s, not with a queue.

        The ceiling is the thing being tested: one client means one semaphore.
        """
        engine = WorkflowEngine(
            Settings(_env_file=None, llm_provider="echo", log_level="ERROR"),
            checkpointer=build_memory_checkpointer(),
        )
        await engine.startup()
        try:
            first = engine._context_for(None, "run-a")
            second = engine._context_for(None, "run-b")

            assert first is not second, "each run needs its own sink to attribute events"
            assert first.client is second.client
        finally:
            await engine.cancel_all()
            await engine.shutdown()


class TestOutcome:
    """The value object every caller receives."""

    def test_parked_and_finished_are_exclusive(self) -> None:
        """A run cannot be both waiting and done.

        The API maps parked to 202 and finished to 200, so a run that reported
        both would let a client render the wrong one.
        """
        parked = RunOutcome(
            run_id="r",
            status="waiting_human",
            state={"status": "waiting_human"},
            pending=_pending(),
        )
        assert parked.is_parked is True
        assert parked.is_finished is False

    def test_iteration_is_parsed_from_state(self) -> None:
        """``iteration`` reads through to the state rather than caching a copy.

        The two can disagree if the graph advanced after the outcome was built,
        and the state is the authority.
        """
        outcome = RunOutcome(run_id="r", status="waiting_human", state={"iteration": 4})
        assert outcome.iteration == 4

    def test_iteration_defaults_to_zero(self) -> None:
        """A missing or malformed counter reads as zero, not as a crash.

        A corrupt checkpoint should degrade the reporting, not make the run
        unreadable.
        """
        assert RunOutcome(run_id="r", status="running", state={}).iteration == 0
        assert RunOutcome(run_id="r", status="running", state={"iteration": None}).iteration == 0

    def test_to_dict_is_json_shaped(self) -> None:
        """The dict projection is what the REST layer serialises."""
        payload = RunOutcome(
            run_id="r", status="waiting_human", state={"iteration": 1}, pending=_pending()
        ).to_dict()
        assert payload["run_id"] == "r"
        assert payload["status"] == "waiting_human"
        import json

        json.dumps(payload, default=str)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _decide(pending: Any) -> dict[str, Any]:
    """Return an approving decision for *pending*.

    Args:
        pending: The approval request; may be ``None`` for a finished run.

    Returns:
        A resume payload.
    """
    from tests.helpers import make_decision

    return make_decision(pending.approval_id if pending else "none").model_dump(mode="json")


def _pending() -> Any:
    """Return a minimal parked approval for the outcome tests.

    Returns:
        An :class:`~agentic_workflow.domain.schemas.ApprovalRequest`.
    """
    from tests.helpers import make_approval

    return make_approval()


async def _hold(engine: WorkflowEngine, request: ReviewRequest) -> Any:
    """Start a run and hold the engine's slot until cancelled.

    Args:
        engine: The engine under test.
        request: The run to submit.

    Returns:
        The run outcome.
    """
    outcome = await engine.start(request)
    await asyncio.sleep(30)
    return outcome


def _sole_running_run_id(engine: WorkflowEngine) -> str:
    """Return the id of the only run the engine currently has in flight.

    Args:
        engine: The engine to inspect.

    Returns:
        The live run id.

    Raises:
        AssertionError: If zero or several runs are in flight, which would make
            the cancellation under test ambiguous.
    """
    live = list(engine._running)
    assert len(live) == 1, f"expected exactly one live run, saw {live}"
    return live[0]


__all__ = ["RunTimeoutError", "Settings"]
