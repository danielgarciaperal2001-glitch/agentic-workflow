"""The workflow engine: one object that owns a run's whole lifecycle.

Why a service layer at all
--------------------------
The graph is pure orchestration. Everything *around* it — enforcing a wall-clock
deadline, bounding concurrency, recording status, cancelling cleanly, exposing
time travel and publishing the parked state to the control plane — is cross-cutting
and would otherwise be copy-pasted into the API, the CLI and the tests.
:class:`WorkflowEngine` is that shared layer, and it is the only object in the
project allowed to invoke the graph. One owner means one definition of "what a
run is", whatever the caller.

Run lifecycle
-------------
::

    start ──▶ running ──┬─▶ completed
              │        ├─▶ failed
              │        ├─▶ cancelled
              └──▶ waiting_human ──(resume)──▶ running

``waiting_human`` is a *first-class* status, not an error. A run parked on an
approval is healthy, durable and resumable — that is the whole point of using a
checkpointer, and the API reflects it with ``202 Accepted`` rather than a 5xx.

Example:
    --------
    >>> engine = WorkflowEngine()  # doctest: +SKIP
    >>> outcome = await engine.start(request)  # doctest: +SKIP
    >>> outcome.is_parked
    True
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
import inspect
from typing import Any, Final, cast, get_args

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agentic_workflow.config import Settings, load_settings
from agentic_workflow.domain.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    FinalReport,
    NodeTiming,
    ReviewRequest,
    RunSummary,
    utcnow,
)
from agentic_workflow.domain.state import WorkflowState, as_model, as_models, initial_state
from agentic_workflow.errors import (
    ApprovalRejectedError,
    CheckpointNotFoundError,
    ConcurrencyLimitError,
    InvalidStateError,
    IterationLimitExceededError,
    RunAlreadyExistsError,
    RunNotFoundError,
    RunTimeoutError,
    WorkflowError,
)
from agentic_workflow.graph.builder import build_graph
from agentic_workflow.graph.context import AgentContext, EventSink
from agentic_workflow.graph.runtime import (
    THREAD_ID,
    create_run_config,
    extract_interrupts,
    thread_config,
)
from agentic_workflow.human.gates import decode_interrupt
from agentic_workflow.logging import bind_context, get_logger
from agentic_workflow.persistence.checkpointer import build_checkpointer
from agentic_workflow.persistence.repository import RunRegistry, RunStatus

log = get_logger(__name__)

#: Answers a human gate. May be sync or async; returning ``None`` parks the run
#: instead of deciding it, which is how a caller opts out of auto-resolution.
Decider = Callable[[ApprovalRequest], Any]

#: Statuses a :class:`RunOutcome` may report, mirroring the registry vocabulary.
RUN_STATUSES: Final[frozenset[str]] = frozenset(get_args(RunStatus))

#: Statuses that will not change again.
FINAL_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed", "cancelled", "rejected"})

#: Terminal statuses the *graph* cannot report about itself, because the node
#: that would have written them raised or was interrupted instead. They are
#: recorded on the registry and overlaid on every subsequent read.
_OUTSIDE_OVERLAY_STATUSES: Final[frozenset[str]] = frozenset({"cancelled", "rejected"})


@dataclass(slots=True)
class CheckpointInfo:
    """One entry of a run's checkpoint history.

    Attributes:
        checkpoint_id: Opaque identifier, passed back to :meth:`WorkflowEngine.state_at`
            or :meth:`WorkflowEngine.replay_from`.
        step: LangGraph super-step counter.
        source: What created the checkpoint (``loop``, ``update``, ``input``…).
        next_nodes: Nodes that were about to run — non-empty means the run is
            parked or resumable at this point.
        created_at: ISO-8601 timestamp reported by the store.
        pending: The approval blocking execution at this checkpoint, if any.
    """

    checkpoint_id: str
    step: int = 0
    source: str = ""
    next_nodes: tuple[str, ...] = ()
    created_at: str | None = None
    pending: ApprovalRequest | None = None

    @property
    def is_parked(self) -> bool:
        """Whether execution is waiting on a human at this checkpoint."""
        return self.pending is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the API and the CLI."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "source": self.source,
            "next_nodes": list(self.next_nodes),
            "created_at": self.created_at,
            "pending_approval": self.pending.approval_id if self.pending else None,
        }


@dataclass(slots=True)
class RunOutcome:
    """Result of driving a run to a stopping point.

    Attributes:
        run_id: The run this outcome describes.
        status: Terminal or parked status.
        state: Final workflow state (``WorkflowState`` shape).
        report: The final report, when the run produced one.
        pending: The approval blocking the run, if it is parked.
        decisions: Every human decision recorded so far.
        timings: Per-node wall-clock accounting.
        error: Terminal error message, if the run failed.
        checkpoint_id: Identifier of the checkpoint the outcome was read from.
        next_nodes: Nodes LangGraph intends to execute next (non-empty when
            parked on an interrupt).
    """

    run_id: str
    status: RunStatus
    state: dict[str, Any]
    report: FinalReport | None = None
    pending: ApprovalRequest | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    timings: list[NodeTiming] = field(default_factory=list)
    error: str | None = None
    checkpoint_id: str | None = None
    next_nodes: tuple[str, ...] = ()

    @property
    def is_parked(self) -> bool:
        """Whether the run is waiting on a human decision."""
        return self.pending is not None or self.status == "waiting_human"

    @property
    def is_finished(self) -> bool:
        """Whether the run reached an immutable final status."""
        return self.status in FINAL_STATUSES

    def to_summary(self) -> RunSummary:
        """Project onto the public :class:`RunSummary` model."""
        return RunSummary(
            run_id=self.run_id,
            status=self.status,
            iteration=self.iteration,
            next_node=self.next_nodes[0] if self.next_nodes else None,
            updated_at=utcnow(),
            checkpoint_id=self.checkpoint_id,
            pending_approval=self.pending,
            timings=self.timings,
            error=self.error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the API and the CLI."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "iteration": self.iteration,
            "pending_approval": self.pending.model_dump(mode="json") if self.pending else None,
            "decisions": list(self.decisions),
            "error": self.error,
            "report": self.report.model_dump(mode="json") if self.report else None,
            "next_nodes": list(self.next_nodes),
        }

    @property
    def iteration(self) -> int:
        """Number of completed feedback-loop iterations, never negative."""
        try:
            return max(0, int(self.state.get("iteration") or 0))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0


class WorkflowEngine:
    """Owns the compiled graph, the checkpointer and every run's lifecycle.

    The engine is the only component that invokes the graph. Everything else
    (API routers, CLI, tests) goes through its methods, which is what guarantees
    run semantics are identical everywhere — in particular that every run is
    bounded, tracked, cancellable and inspectable.

    Example:
        --------
        >>> engine = WorkflowEngine()  # doctest: +SKIP
        >>> await engine.startup()  # doctest: +SKIP
        >>> outcome = await engine.start(request)  # doctest: +SKIP
        >>> outcome.status  # doctest: +SKIP
        'waiting_human'
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        checkpointer: Any = None,
        context: AgentContext | None = None,
        registry: RunRegistry | None = None,
        graph: Any = None,
        event_sink: EventSink | None = None,
    ) -> None:
        """Wire the engine.

        Args:
            settings: Application configuration; defaults to the process settings.
            checkpointer: A LangGraph checkpointer. When ``None`` one is built from
                the configuration (PostgreSQL if enabled, in-memory otherwise).
            context: Runtime context template injected into every node. A
                per-run copy is derived from it so an event sink can attribute
                events to a run without cross-talk.
            registry: Run registry. Defaults to a fresh :class:`RunRegistry`.
            graph: A pre-compiled graph, mainly for tests. When given, the engine
                does not compile its own.
            event_sink: Optional async callback receiving run and node events.
                This is what the API wires to its WebSocket broadcaster.
        """
        self._settings = settings or load_settings()
        self._checkpointer = (
            checkpointer if checkpointer is not None else build_checkpointer(self._settings)
        )
        self._owns_checkpointer = checkpointer is None
        self._registry = registry or RunRegistry(
            max_entries=max(self._settings.max_parallel_runs * 64, 1_000),
            ttl_seconds=float(self._settings.state_retention_days) * 86_400.0,
        )
        self._graph = graph
        self._context_template = context
        self._event_sink = event_sink
        self._running: dict[str, asyncio.Task[list[ApprovalRequest]]] = {}
        self._cancelled: set[str] = set()
        self._started = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ wiring #
    @property
    def settings(self) -> Settings:
        """The engine's configuration."""
        return self._settings

    def set_event_sink(self, sink: EventSink | None) -> None:
        """Attach or replace the engine's event sink.

        The sink is a *transport* detail, not a lifecycle one, so it is wired per
        app instance rather than fixed at construction. An engine injected into
        :func:`~agentic_workflow.api.app.create_app` was built with no sink, and
        without this the application would come up healthy, accept the handshake
        and then never emit a single event — the entire real-time surface dead,
        with nothing logged. Re-pointing the sink replaces whatever the engine
        already had; a caller that needs two consumers should compose them.

        Args:
            sink: Async callback receiving run and node events, or ``None`` to
                detach the current one.
        """
        self._event_sink = sink

    @property
    def registry(self) -> RunRegistry:
        """The engine's run registry."""
        return self._registry

    @property
    def checkpointer(self) -> Any:
        """The checkpointer backing every run."""
        return self._checkpointer

    @property
    def graph(self) -> Any:
        """The compiled graph, compiled on first access."""
        if self._graph is None:
            saver = self._checkpointer
            # `build_checkpointer` returns a PostgresCheckpointer *wrapper* for the
            # durable case; the graph needs the saver itself.
            saver = getattr(saver, "saver", saver)
            self._graph = build_graph(self._settings, checkpointer=saver)
        return self._graph

    async def startup(self) -> None:
        """Open durable resources and compile the graph. Idempotent.

        Must be awaited before the first run when a durable checkpointer is
        configured: the pool is opened here rather than lazily inside a request
        so that a misconfigured database fails at boot, where an operator can see
        it, instead of on the first customer request.
        """
        if self._started:
            return
        setup = getattr(self._checkpointer, "setup", None)
        if setup is not None:
            await setup()
        # Touching `.graph` compiles it, which is where wiring mistakes surface.
        _ = self.graph
        await self._rehydrate_registry()
        self._started = True
        log.info(
            "engine.started",
            durable=self._settings.use_durable_checkpointer,
            hitl=self._settings.hitl_enabled,
            runs=self._registry.count(),
        )

    async def _rehydrate_registry(self) -> None:
        """Rebuild the run registry from durable checkpoint threads.

        The registry is a *projection* of durable state, and a projection is only
        a projection if it can be rebuilt. Without this, a process that restarts
        — or a second replica joining a pool — has an empty registry while the
        database is full of live runs, and every registry-backed path reports
        them as nonexistent. ``RunRegistry.rebuild_from`` exists for exactly this
        and nothing was calling it, so the boot-time claim in that method's
        docstring was not true.

        Bounded and best effort. A large ``checkpoints`` table must not turn a
        restart into a long stall, and a database that cannot be listed is not a
        reason to refuse to start: the checkpointer itself is already open by
        this point, so a failure here is a degraded listing, not a lost run, and
        every authoritative read still goes to the checkpointer.
        """
        lister = getattr(self._checkpointer, "list_thread_ids", None)
        if lister is None:
            return
        try:
            thread_ids = await lister(limit=self._settings.recovery_max_runs)
        except Exception as exc:
            log.warning("engine.registry_rehydrate_failed", error=str(exc))
            return
        if not thread_ids:
            return
        # Every thread is read *before* anything is handed over, and what is
        # handed over is the engine's own projection of the run — never the raw
        # checkpoint values. Two things went wrong when this passed state
        # through, and both are worth recording:
        #
        # 1. `rebuild_from` was synchronous, so an async lookup gave it a
        #    coroutine it could not await. It stored the truthy coroutine
        #    object, fell back to `{}`, and registered every run as `pending` —
        #    a registry that looked correctly populated and was wrong about all
        #    of it, which is the worst shape a bug can take because the presence
        #    check it exists to enable passes.
        # 2. It then read `state["status"]`, which is the *graph's* status, not
        #    the engine's. A run parked on a human gate came back as
        #    `"triaged"`: not in the lifecycle vocabulary, matched by no status
        #    filter, and not in TERMINAL_STATUSES, so a poller waited forever on
        #    a run that had already finished.
        #
        # Reading first also lets the reads go concurrently, and going through
        # `_outcome_from_snapshot` is what keeps the projection and the
        # authoritative read derived from exactly the same code.
        snapshots = await asyncio.gather(
            *(self._aget_state(thread_id) for thread_id in thread_ids),
            return_exceptions=True,
        )
        entries: dict[str, dict[str, Any]] = {}
        for thread_id, snap in zip(thread_ids, snapshots, strict=True):
            if isinstance(snap, BaseException):
                log.warning("engine.registry_read_failed", thread_id=thread_id, error=str(snap))
                continue
            try:
                outcome = self._outcome_from_snapshot(thread_id, snap)
            except Exception as exc:
                log.warning("engine.registry_project_failed", thread_id=thread_id, error=str(exc))
                continue
            entries[thread_id] = {
                "status": outcome.status,
                "iteration": outcome.iteration,
                "pending_approval": outcome.pending.approval_id if outcome.pending else None,
                "created_at": (outcome.state or {}).get("started_at"),
            }
        self._registry.rebuild_from(entries)
        log.info(
            "engine.registry_rehydrated",
            runs=self._registry.count(),
            unreadable=len(thread_ids) - len(entries),
        )

    async def shutdown(self) -> None:
        """Cancel in-flight runs and release durable resources."""
        await self.cancel_all()
        if self._owns_checkpointer:
            close = getattr(self._checkpointer, "close", None)
            if close is not None:
                with suppress(Exception):
                    await close()
        llm = self._context_template.llm if self._context_template else None
        if llm is not None and hasattr(llm, "aclose"):
            with suppress(Exception):
                await llm.aclose()
        self._started = False
        log.info("engine.stopped")

    # ------------------------------------------------------------- runs  #
    async def start(
        self,
        request: ReviewRequest,
        *,
        context: AgentContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunOutcome:
        """Start a new run and drive it until it finishes or parks on a gate.

        Args:
            request: The review request to process.
            context: Per-run runtime context. Defaults to one derived from the
                engine's settings.
            metadata: Annotations stored alongside the run in the registry.

        Returns:
            A :class:`RunOutcome`. It is *not* an error when ``is_parked`` is
            true — that is the normal outcome of a gated run.

        Raises:
            ConcurrencyLimitError: If too many runs are already in flight.
            RunAlreadyExistsError: If the run id is already registered.
        """
        await self._guard_reentry(request.run_id)
        return await self._drive(request, context=context, resume=None, metadata=metadata)

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision | dict[str, Any],
        *,
        context: AgentContext | None = None,
    ) -> RunOutcome:
        """Resume a parked run with a human decision.

        Args:
            run_id: The parked run.
            decision: The human's decision, as a model or a client payload.
            context: Per-run runtime context.

        Returns:
            The :class:`RunOutcome` after resuming — usually either the next
            pending gate or a finished run.

        Raises:
            RunNotFoundError: If the run is unknown.
            InvalidStateError: If the run is not parked on a human gate.
        """
        outcome = await self.status(run_id)
        if not outcome.is_parked:
            raise InvalidStateError(
                "run is not waiting on a human decision",
                run_id=run_id,
                status=outcome.status,
            )
        return await self._drive(None, context=context, resume=decision, run_id=run_id)

    async def run_until_done(
        self,
        request: ReviewRequest,
        *,
        decide: Decider | None = None,
        context: AgentContext | None = None,
        max_gates: int = 32,
    ) -> RunOutcome:
        """Run to completion, auto-answering every gate with *decide*.

        This is the loop a batch caller wants: it must not have to know about
        interrupts at all. *decide* receives the :class:`ApprovalRequest` and
        returns the resume value, so tests and the CLI can drive a full run with
        one function. It may be sync or async, and returning ``None`` parks the
        run and stops the loop.

        Args:
            request: The review request.
            decide: Callable returning the resume value for a gate. ``None``
                selects the built-in auto-approve policy.
            context: Per-run runtime context.
            max_gates: Safety bound on the number of gates answered, so a
                misbehaving graph cannot spin forever.

        Returns:
            The final :class:`RunOutcome`.

        Raises:
            InvalidStateError: If the gate budget is exhausted.
        """
        outcome = await self.start(request, context=context)
        answered = 0
        while outcome.is_parked and outcome.pending is not None:
            if answered >= max_gates:
                raise InvalidStateError(
                    "auto-resolution gate budget exhausted",
                    run_id=request.run_id,
                    gates=answered,
                )
            value = (
                _default_decision(outcome.pending) if decide is None else decide(outcome.pending)
            )
            if inspect.isawaitable(value):
                value = await value
            if value is None:
                break
            outcome = await self.resume(request.run_id, value, context=context)
            answered += 1
        return outcome

    # ------------------------------------------------------------ status #
    async def status(self, run_id: str) -> RunOutcome:
        """Read the current state of a run from the checkpoint store.

        This is the *authoritative* read: it goes to the checkpointer, not to the
        registry, so a process restart — or a read served by a different replica
        after a load balancer moved the request — returns the same answer.

        Args:
            run_id: The run to inspect.

        Returns:
            The current :class:`RunOutcome`.

        Raises:
            RunNotFoundError: If the checkpoint thread does not exist.
        """
        snap = await self._aget_state(run_id)
        if snap is None:
            raise RunNotFoundError("no checkpoint for this run", run_id=run_id)
        return self._outcome_from_snapshot(run_id, snap)

    async def list_runs(
        self, *, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[RunSummary]:
        """List known runs, most recently updated first.

        Authoritative state is fetched for every record, concurrently: a listing
        that reported "running" for a run that finished an hour ago would be worse
        than no listing at all. Registry records the checkpointer has since
        evicted (or a foreign replica owns) still appear, marked with the last
        status the registry knew.

        Args:
            status: Optional status filter applied by the registry.
            limit: Maximum number of runs to return.
            offset: Pagination offset.

        Returns:
            The matching run summaries.
        """
        records = self._registry.list_runs(status=status, limit=limit, offset=offset)
        if not records:
            return []
        outcomes = await asyncio.gather(
            *(self.status(record.run_id) for record in records), return_exceptions=True
        )
        summaries: list[RunSummary] = []
        for record, outcome in zip(records, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                log.debug("engine.listing_fallback", run_id=record.run_id, error=str(outcome))
                summaries.append(record.to_summary())
            else:
                summaries.append(outcome.to_summary())
        return summaries

    async def pending_approvals(self, run_id: str | None = None) -> list[ApprovalRequest]:
        """Return every approval currently blocking a run.

        Args:
            run_id: Restrict to one run. When ``None``, every known run is
                scanned.

        Returns:
            The pending approval requests, oldest first.
        """
        candidates = [run_id] if run_id else self._known_run_ids()
        found = await asyncio.gather(
            *(self.status(candidate) for candidate in candidates), return_exceptions=True
        )
        pending = [o.pending for o in found if isinstance(o, RunOutcome) and o.pending]
        return sorted(pending, key=lambda request: request.created_at)

    # ------------------------------------------------------- time travel #
    async def history(self, run_id: str, *, limit: int = 50) -> list[CheckpointInfo]:
        """Return the checkpoint history of a run, newest first.

        This is the backbone of *time travel*: every super-step LangGraph took is
        addressable, which is what makes "why did the reviewer reject iteration
        two?" a question with an answer instead of a guess.

        Args:
            run_id: The run to inspect.
            limit: Maximum number of checkpoints to return.

        Returns:
            The checkpoint descriptors, newest first.

        Raises:
            RunNotFoundError: If the run has no checkpoints at all.
        """
        entries: list[CheckpointInfo] = []
        async for snap in self.graph.aget_state_history(thread_config(run_id), limit=max(1, limit)):
            entries.append(_checkpoint_info(snap))
        if not entries:
            raise RunNotFoundError("run has no checkpoint history", run_id=run_id)
        return entries

    async def state_at(self, run_id: str, checkpoint_id: str) -> RunOutcome:
        """Project the outcome of a run *as of* a historical checkpoint.

        Read-only: nothing is written and the current head of the run is
        untouched, so this is safe to call from a UI that lets an operator scrub
        through a run.

        Args:
            run_id: The run to inspect.
            checkpoint_id: Checkpoint to read, as returned by :meth:`history`.

        Returns:
            The :class:`RunOutcome` at that point in time.

        Raises:
            CheckpointNotFoundError: If the run exists but the checkpoint does not.
        """
        snap = await self._read_checkpoint(run_id, checkpoint_id)
        return self._outcome_from_snapshot(run_id, snap)

    async def replay_from(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        context: AgentContext | None = None,
    ) -> RunOutcome:
        """Re-execute a run from a historical checkpoint.

        LangGraph treats an invocation carrying an older ``checkpoint_id`` as a
        *fork*: the new work is appended to the same thread as a sibling of the
        original history, so the branch that misbehaved stays intact and
        comparable. That is the property that makes debugging possible — you can
        re-run the interesting half of a run with a fixed agent and diff the two
        histories.

        Args:
            run_id: The run to replay.
            checkpoint_id: Checkpoint to branch from.
            context: Per-run runtime context.

        Returns:
            The :class:`RunOutcome` of the branch.

        Raises:
            CheckpointNotFoundError: If the run exists but the checkpoint does not.
            RunTimeoutError: If the branch exceeds the run budget.
        """
        snap = await self._read_checkpoint(run_id, checkpoint_id)
        log.info(
            "engine.replay",
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            next_nodes=list(snap.next or ()),
        )
        return await self._drive(
            None,
            context=context,
            resume=None,
            run_id=run_id,
            config=_branch_config(run_id, checkpoint_id),
        )

    # ------------------------------------------------------------ cancel #
    async def cancel(self, run_id: str, *, reason: str = "cancelled by operator") -> None:
        """Cancel a run.

        Cancellation is cooperative: the in-flight task is cancelled and the
        status recorded. Because LangGraph checkpoints after every super-step,
        cancelling never loses completed work — a later ``resume`` picks up from
        the last checkpoint.

        Args:
            run_id: The run to cancel.
            reason: Recorded on the run for the audit trail.

        Raises:
            RunNotFoundError: If the run is unknown.
            InvalidStateError: If the run already reached a terminal status.
        """
        # Existence is decided by the checkpointer, not by the registry. The
        # registry is per-process, so consulting it here made a run started by
        # another replica impossible to cancel: the operator got a 404 for a
        # run that demonstrably existed. `_aget_state` is the documented
        # authoritative read, and this is exactly the moment that has to be
        # true — cancelling is the one action a human takes to stop something
        # they can see running somewhere else.
        if await self._aget_state(run_id) is None:
            raise RunNotFoundError("no checkpoint for this run", run_id=run_id)
        # A run that already finished cannot be cancelled, and forcing it would
        # be destructive rather than defensive: the terminal status is the
        # record of what happened, so a late "cancel" — a user clicking the
        # button as a review completed, a retried request arriving after the
        # fact — would rewrite "completed" to "cancelled" and leave the audit
        # trail claiming an operator stopped work that had already shipped.
        settled = await self._aget_state(run_id)
        if settled is not None and not settled.next:
            current = self._outcome_from_snapshot(run_id, settled)
            if current.status in FINAL_STATUSES:
                raise InvalidStateError(
                    "run has already finished and cannot be cancelled",
                    run_id=run_id,
                    status=current.status,
                )
        async with self._lock:
            # Marked *before* cancelling: the driving coroutine checks this flag to
            # tell "an operator cancelled me" from "my caller went away".
            self._cancelled.add(run_id)
            task = self._running.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        await self._settle_cancelled(run_id, reason)
        log.info("run.cancelled", run_id=run_id, reason=reason)

    async def cancel_all(self) -> None:
        """Cancel every in-flight run. Used on shutdown."""
        for run_id in list(self._running):
            with suppress(Exception):
                await self.cancel(run_id, reason="engine shutting down")

    # --------------------------------------------------------- internals #
    async def _guard_reentry(self, run_id: str) -> None:
        """Refuse to re-drive a thread that is already mid-flight.

        Feeding a fresh ``initial_state`` into a thread that is parked on a human
        gate does not restart the run — LangGraph merges the input into the
        existing checkpoint, so the parked run would resume under a different
        request. Failing loudly is the only safe answer.
        """
        if self._registry.find(run_id) is not None:
            raise RunAlreadyExistsError("run already registered", run_id=run_id)
        snap = await self._aget_state(run_id)
        if snap is None or not snap.next:
            return
        raise RunAlreadyExistsError(
            "run already exists and has not finished; resume or cancel it instead",
            run_id=run_id,
            next_nodes=list(snap.next),
        )

    def _admit(self, run_id: str) -> None:
        """Enforce per-run exclusion and the global ceiling.

        Caller must hold ``self._lock``.

        The per-run check has to live here rather than in :meth:`resume`.
        ``resume`` decides whether the run is parked by reading the checkpointer,
        and that read is not held across the gap before ``_drive`` claims the run,
        so two callers can both observe "parked" and both drive. Here the check
        and the registration happen under one lock, which is what makes
        claiming atomic.

        The global ceiling alone is not sufficient. ``_running`` is keyed by run
        id, so a second task for the same id replaces the first one's handle
        rather than coexisting with it: the original becomes unreachable —
        ``cancel`` can no longer see it, so it could not be interrupted — and the
        node after the gate executes twice against the same thread. Replaying a
        run that is still in flight is the same hazard, and the same check
        rejects it, which is right in its own terms: the history being branched
        from is still being written.
        """
        existing = self._running.get(run_id)
        if existing is not None and not existing.done():
            raise RunAlreadyExistsError(
                "run is already in flight and cannot be driven twice at once",
                run_id=run_id,
            )
        live = [task for task in self._running.values() if not task.done()]
        if len(live) >= self._settings.max_parallel_runs:
            raise ConcurrencyLimitError(
                "too many runs in flight",
                run_id=run_id,
                limit=self._settings.max_parallel_runs,
            )
        self._cancelled.discard(run_id)

    def _context_for(self, context: AgentContext | None, run_id: str) -> AgentContext:
        """Build the runtime context for one run.

        A per-run context is the point: a single global event sink serves many
        runs, and every event it receives has to be attributable. Copying the
        template keeps the run id out of the shared object.
        """
        if context is not None:
            return context
        template = self._context_template
        if template is None:
            return AgentContext(settings=self._settings, emit=self._event_sink)
        if self._event_sink is None or template.emit is not None:
            return template
        return replace(template, emit=self._scoped_sink(run_id))

    async def _drive(
        self,
        request: ReviewRequest | None,
        *,
        context: AgentContext | None,
        resume: ApprovalDecision | dict[str, Any] | None,
        run_id: str | None = None,
        config: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunOutcome:
        """Invoke the graph, translating interrupts into a parked outcome.

        Args:
            request: Seed state for a new run; ``None`` when resuming or replaying.
            context: Per-run runtime context.
            resume: Human decision driving the resume, if any.
            run_id: Existing run identifier. Defaults to ``request.run_id``.
            config: Explicit invocation config, used by :meth:`replay_from` to
                branch from a historical checkpoint.
            metadata: Registry annotations for a newly created run.

        Returns:
            The settled :class:`RunOutcome`.

        Raises:
            ConcurrencyLimitError: If the concurrency budget is exhausted.
            RunAlreadyExistsError: If the run id is taken.
            RunTimeoutError: If the run exceeds its wall-clock budget.
        """
        resolved_run_id = run_id or (request.run_id if request is not None else "")
        if not resolved_run_id:
            raise InvalidStateError("a run id is required to drive the graph")
        ctx = self._context_for(context, resolved_run_id)
        if config is None:
            config = create_run_config(
                resolved_run_id, run_id=resolved_run_id, settings=self._settings
            )
        if resume is not None:
            payload: Any = Command(resume=_resume_value(resume))
        elif request is not None:
            payload = initial_state(request)
        else:
            # `replay_from` resumes a run from history without seeding a request.
            payload = None
        timeout = self._settings.run_timeout_seconds

        # `bind_context` must wrap task *creation*: a task snapshots the current
        # contextvars at creation time, so binding afterwards would leave every
        # log record emitted inside the graph without its run_id.
        with bind_context(run_id=resolved_run_id):
            # The graph runs in its own task so `cancel()` can interrupt it from
            # another coroutine, and so the concurrency ceiling counts real work
            # rather than coroutines that merely got scheduled.
            async with self._lock:
                self._admit(resolved_run_id)
                if request is not None:
                    self._registry.create(
                        resolved_run_id,
                        metadata={**self._safe_metadata(request), **(metadata or {})},
                    )
                task = asyncio.get_running_loop().create_task(
                    self._stream(payload, config, ctx, resolved_run_id, timeout=timeout),
                    name=f"awf-run:{resolved_run_id}",
                )
                self._running[resolved_run_id] = task

            try:
                self._record(resolved_run_id, status="running")
                await self._emit("run.started", run_id=resolved_run_id)
                pending = await task
            except asyncio.CancelledError:
                await _abort(task)
                if resolved_run_id in self._cancelled:
                    # Operator-driven: settle as cancelled rather than tearing
                    # down the caller's coroutine, which is not itself cancelled.
                    self._running.pop(resolved_run_id, None)
                    return await self._settle_cancelled(
                        resolved_run_id,
                        "cancelled by operator",
                        snap=await self._aget_state(resolved_run_id),
                    )
                raise
            except TimeoutError as exc:
                self._fail(resolved_run_id, f"run exceeded its {timeout:g}s budget")
                raise RunTimeoutError(
                    f"run exceeded its {timeout:g}s budget",
                    run_id=resolved_run_id,
                    timeout_seconds=timeout,
                ) from exc
            except ApprovalRejectedError as exc:
                # A human said no. That is the workflow reaching a conclusion, not
                # a malfunction, so it is settled as a terminal `rejected` run
                # and *not* re-raised: the caller asked a question ("what happened
                # to this run?") and deserves an answer, not a stack trace. The
                # REST layer already maps the error to 200 for the same reason.
                self._running.pop(resolved_run_id, None)
                rejected = await self._settle_rejected(
                    resolved_run_id,
                    str(exc),
                    snap=await self._aget_state(resolved_run_id),
                )
                await self._emit_run_end(rejected)
                return rejected
            except WorkflowError as exc:
                self._fail(resolved_run_id, str(exc))
                raise
            except GraphRecursionError as exc:
                # The graph hit its own backstop. The router's iteration budget is
                # meant to stop a runaway repair loop long before this, so reaching
                # it means the loop was not the one the budget counts — but the
                # business answer is the same: this run used up its allowance and
                # needs a human. Reported as an infrastructure error it reads like
                # a fault, and nobody pages anyone for a fault that is in fact a
                # decision.
                raised = IterationLimitExceededError(
                    f"the graph exhausted its recursion limit: {exc}",
                    run_id=resolved_run_id,
                )
                self._fail(resolved_run_id, str(raised))
                raise raised from exc
            except Exception as exc:
                # A node raising something outside the taxonomy (a provider
                # blowing up, a malformed response) must still leave a durable,
                # explainable record rather than an orphaned thread.
                self._fail(resolved_run_id, f"{type(exc).__name__}: {exc}")
                raise
            finally:
                self._running.pop(resolved_run_id, None)

            outcome = await self._settle(resolved_run_id, pending)
            await self._emit_run_end(outcome)
            return outcome

    async def _stream(
        self,
        payload: Any,
        config: Any,
        ctx: AgentContext,
        run_id: str,
        *,
        timeout: float,
    ) -> list[ApprovalRequest]:
        """Stream the graph, collecting any interrupts it raises.

        ``astream`` (not ``ainvoke``) is used because the interrupt payload is only
        reliably observable in the stream: ``ainvoke`` returns the final values
        with the interrupt tucked into a private key whose name differs between
        LangGraph versions.
        """
        pending: list[ApprovalRequest] = []

        async def consume() -> None:
            async for chunk in self.graph.astream(
                payload, config=config, stream_mode="updates", context=ctx
            ):
                gates = extract_interrupts(chunk)
                if gates:
                    pending.extend(gates)
                    for gate in gates:
                        await self._emit(
                            "approval.requested",
                            run_id=run_id,
                            approval_id=gate.approval_id,
                            stage=gate.stage,
                            title=gate.title,
                            confidence=gate.confidence,
                        )
                else:
                    await self._emit_node_updates(run_id, chunk)

        if timeout > 0:
            async with asyncio.timeout(timeout):
                await consume()
        else:
            await consume()
        return pending

    async def _settle(self, run_id: str, pending: list[ApprovalRequest]) -> RunOutcome:
        """Read the post-invocation checkpoint and build the outcome."""
        snap = await self._aget_state(run_id)
        if snap is None:
            raise InvalidStateError("graph produced no checkpoint", run_id=run_id)

        if pending:
            outcome = self._outcome_from_snapshot(run_id, snap, pending_override=pending[0])
        else:
            if snap.next:
                # LangGraph still intends to run more nodes but raised nothing we
                # recognise. Report it rather than claiming the run is done.
                log.warning("engine.unexpected_next", run_id=run_id, next=list(snap.next))
            outcome = self._outcome_from_snapshot(run_id, snap)

        self._record(
            run_id,
            status=outcome.status,
            iteration=outcome.iteration,
            pending_approval=outcome.pending.approval_id if outcome.pending else None,
            error=outcome.error,
        )
        return outcome

    async def _settle_rejected(
        self, run_id: str, reason: str, *, snap: Any | None = None
    ) -> RunOutcome:
        """Project a rejected run, forcing the terminal ``rejected`` status.

        The graph cannot record this itself: the node that asked for the decision
        raised, so it never returned an update. The registry is the only place
        the fact can live, which is why :meth:`_outcome_from_snapshot` overlays
        it on every later read — the same mechanism cancellation relies on.

        Args:
            run_id: The rejected run.
            reason: Human-readable justification, from the rejection itself.
            snap: Post-raise checkpoint, when one could be read.

        Returns:
            The settled outcome, carrying everything the run achieved up to the
            point it was rejected.
        """
        if snap is None:
            outcome = RunOutcome(run_id=run_id, status="rejected", state={}, error=reason)
        else:
            outcome = self._outcome_from_snapshot(run_id, snap)
            outcome.status = "rejected"
            outcome.error = reason
            # The gate that was rejected is still sitting in the checkpoint as
            # "pending", so the projection above reports it. Clearing it here is
            # what makes the value returned by `resume` agree with the value
            # `status()` returns afterwards: a terminal run is not waiting on
            # anybody, and an outcome claiming otherwise would make the auto-
            # resolution loop try to answer a gate that is already closed.
            outcome.pending = None
        self._record(
            run_id,
            status="rejected",
            error=reason,
            pending_approval=outcome.pending.approval_id if outcome.pending else None,
        )
        await self._emit("run.rejected", run_id=run_id, reason=reason, iteration=outcome.iteration)
        return outcome

    async def _settle_cancelled(
        self, run_id: str, reason: str, *, snap: Any | None = None
    ) -> RunOutcome:
        """Project a cancelled run, forcing the terminal ``cancelled`` status.

        The pending gate is cleared for the same reason as in
        :meth:`_settle_rejected`: the checkpoint still lists the interrupted
        approval, so the projection would otherwise hand back an outcome that
        both says ``cancelled`` and says "waiting on a human".
        """
        if snap is None:
            outcome = RunOutcome(run_id=run_id, status="cancelled", state={}, error=reason)
        else:
            outcome = self._outcome_from_snapshot(run_id, snap)
            outcome.status = "cancelled"
            outcome.error = reason
            outcome.pending = None
        self._record(
            run_id,
            status="cancelled",
            error=reason,
            pending_approval=outcome.pending.approval_id if outcome.pending else None,
        )
        await self._emit("run.cancelled", run_id=run_id, reason=reason, iteration=outcome.iteration)
        return outcome

    def _record(self, run_id: str, **fields: Any) -> None:
        """Write to the run registry, creating the record when this process lacks one.

        The registry is a *per-process* projection of durable state, so a run
        this process did not start has no record here: it may have been inherited
        from the database on boot, or created by another replica entirely. Every
        write path must therefore create before it updates.

        Routing all of them through this one method is the point. Each write site
        was individually reasonable and individually broken — the first
        incarnation of this only guarded the failure path, and resuming or
        cancelling a run from a second process still raised ``RunNotFoundError``
        from ``update``, because three other call sites had the same assumption
        written into them. A shared entry point is what makes the invariant
        checkable; five copies of the same guard is five chances to forget it.

        Args:
            run_id: The run to write about.
            **fields: Field updates, passed through to ``registry.update``.
        """
        if self._registry.find(run_id) is None:
            self._registry.create(run_id)
        self._registry.update(run_id, **fields)

    def _fail(self, run_id: str, message: str) -> None:
        """Record a terminal failure on the registry.

        Must not raise, and must not be the thing that decides whether a failure
        is reported. It sits in the ``except`` block of :meth:`_drive`, so an
        exception here replaces the real error with a second, unrelated one and
        the original cause is lost.

        That is not hypothetical: the ``RunRegistry`` is per-process, so a run
        that started in a *different* process has no record here, and
        ``registry.update`` raised ``RunNotFoundError`` from inside the handler
        for a genuine workflow error. The traceback named the registry and
        nothing about what actually went wrong. An error path that can raise is
        a diagnostic dead end, so the log line — the only place the truth
        survives — always happens, whatever the registry does.
        """
        self._record(run_id, status="failed", error=message, pending_approval=None)
        log.error("run.failed", run_id=run_id, error=message)

    def _outcome_from_snapshot(
        self,
        run_id: str,
        snap: Any,
        *,
        pending_override: ApprovalRequest | None = None,
    ) -> RunOutcome:
        """Project a LangGraph snapshot onto a :class:`RunOutcome`.

        Cancellation is the one status the graph cannot report about itself: the
        task was interrupted from outside, so no node ever ran to write it. The
        registry holds that fact, so it is overlaid here rather than left to be
        silently overwritten by the checkpoint's view of a run that is still,
        from LangGraph's perspective, waiting for a human.
        """
        values: WorkflowState = cast("WorkflowState", dict(snap.values or {}))
        pending = pending_override or _pending_from(values, snap)
        state_status = str(values.get("status") or "")

        if pending is not None:
            status: RunStatus = "waiting_human"
        elif values.get("error"):
            status = "failed"
        elif not snap.next and state_status in FINAL_STATUSES:
            status = cast("RunStatus", state_status)
        elif not snap.next:
            status = "completed"
        else:
            status = _run_status(state_status)

        record = self._registry.find(run_id)
        if record is not None and record.status in _OUTSIDE_OVERLAY_STATUSES:
            status = record.status
            pending = None
            values = {**values, "status": record.status}

        return RunOutcome(
            run_id=run_id,
            status=status,
            state=dict(values),
            report=as_model(values, "report", FinalReport),
            pending=pending,
            decisions=[dict(entry) for entry in (values.get("human_decisions") or [])],
            timings=as_models(values, "node_timings", NodeTiming),
            error=values.get("error") or (record.error if record is not None else None),
            checkpoint_id=_checkpoint_id(snap),
            next_nodes=tuple(snap.next or ()),
        )

    async def _read_checkpoint(self, run_id: str, checkpoint_id: str) -> Any:
        """Load a specific historical checkpoint, or fail loudly.

        LangGraph does **not** treat an unknown ``checkpoint_id`` as an error. It
        echoes the requested id back in ``config`` and returns a snapshot with
        empty ``values`` — indistinguishable at a glance from a real checkpoint.
        A guard written as ``not (snap.values or snap.config)`` therefore passes
        for a checkpoint that does not exist, and ``state_at``/``replay_from``
        silently answer with the head of the run.

        For a UI that scrubs through a run, that is the worst possible failure:
        the operator inspects "iteration 1", is shown "iteration 4", and
        believes it. So the discriminator is ``values`` alone — a snapshot with no
        state is not a state anyone can look at — plus a check that the store
        really resolved the id it was asked for.

        Args:
            run_id: Thread the checkpoint belongs to.
            checkpoint_id: The checkpoint to read.

        Returns:
            The resolved ``StateSnapshot``.

        Raises:
            RunNotFoundError: If the checkpoint thread does not exist.
            CheckpointNotFoundError: If the thread exists but the checkpoint does not.
        """
        try:
            snap = await self.graph.aget_state(_branch_config(run_id, checkpoint_id))
        except Exception as exc:
            # The store itself refused the thread: nothing was ever written under
            # this run id, so the run — not merely one of its checkpoints — is
            # unknown. Conflating the two would tell a client to resubmit a run
            # that already exists.
            log.debug("engine.checkpoint_unavailable", run_id=run_id, error=str(exc))
            raise RunNotFoundError(
                "unknown checkpoint thread", run_id=run_id, checkpoint_id=checkpoint_id
            ) from exc
        resolved = _checkpoint_id(snap) if snap is not None else None
        if snap is None or not snap.values or resolved != checkpoint_id:
            # The thread answered, so the run exists and the *checkpoint* does
            # not. A distinct code is what lets a client tell a typo in the
            # checkpoint id (retry with a corrected id) apart from a run that was
            # never submitted.
            raise CheckpointNotFoundError(
                "unknown checkpoint",
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                resolved_checkpoint_id=resolved,
            )
        return snap

    async def _aget_state(self, run_id: str, checkpoint_id: str | None = None) -> Any | None:
        """Read a checkpoint, returning ``None`` when the thread is unknown.

        A snapshot for a thread that does not exist comes back with empty values
        and a populated config, so emptiness of ``values`` — not of the config —
        is the signal that must be treated as "no such run".
        """
        config = (
            thread_config(run_id)
            if checkpoint_id is None
            else _branch_config(run_id, checkpoint_id)
        )
        try:
            snap = await self.graph.aget_state(config)
        except Exception as exc:  # an unknown thread is not fatal
            log.debug("engine.state_unavailable", run_id=run_id, error=str(exc))
            return None
        if snap is None or not (snap.values or snap.next or snap.interrupts):
            return None
        return snap

    def _known_run_ids(self) -> list[str]:
        """Every run id the registry knows about, registry-only (no I/O)."""
        return [record.run_id for record in self._registry.list_runs(limit=10_000)]

    def _scoped_sink(self, run_id: str) -> EventSink:
        """Wrap the global event sink so every event carries its run id."""

        async def sink(event: dict[str, Any]) -> None:
            if self._event_sink is not None:
                await self._event_sink({"run_id": run_id, **event})

        return sink

    async def _emit(self, event: str, **fields: Any) -> None:
        """Publish a lifecycle event, swallowing sink failures.

        Observability must never break a run: a WebSocket that just went away is
        an ops problem, not a business one.
        """
        if self._event_sink is None:
            return
        payload = {"event": event, "ts": utcnow().isoformat(), **fields}
        try:
            await self._event_sink(payload)
        except Exception as exc:  # an event sink must never break a run
            log.warning("engine.emit_failed", failed_event=event, error=str(exc))

    async def _emit_node_updates(self, run_id: str, chunk: Any) -> None:
        """Turn one ``updates`` chunk into per-node progress events.

        Only the *shape* of the update is published — the changed channel names
        and the lifecycle fields — because a full state delta contains the full
        patch diff. Clients that need the detail fetch it over REST.
        """
        if self._event_sink is None or not isinstance(chunk, dict):
            return
        for node, delta in chunk.items():
            # `__interrupt__` is a LangGraph bookkeeping key, not a node.
            if node.startswith("__") or not isinstance(delta, dict):
                continue
            await self._emit(
                "node.completed",
                run_id=run_id,
                node=node,
                updated=sorted(delta),
                status=delta.get("status"),
                iteration=delta.get("iteration"),
            )

    async def _emit_run_end(self, outcome: RunOutcome) -> None:
        """Publish the terminal (or parked) event for a run."""
        event = {
            "completed": "run.completed",
            "failed": "run.failed",
            "cancelled": "run.cancelled",
            "rejected": "run.rejected",
        }.get(outcome.status, "run.parked")
        await self._emit(
            event,
            run_id=outcome.run_id,
            status=outcome.status,
            iteration=outcome.iteration,
            error=outcome.error,
            pending_approval=outcome.pending.approval_id if outcome.pending else None,
        )

    @staticmethod
    def _safe_metadata(request: ReviewRequest) -> dict[str, Any]:
        """Extract log-safe annotations from a request."""
        return {
            "request_id": request.request_id,
            "title": request.title[:120],
            "files": len(request.files),
            "content_hash": request.content_hash,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _branch_config(run_id: str, checkpoint_id: str) -> Any:
    """Config addressing one historical checkpoint of a thread.

    LangGraph reads the checkpoint out of ``configurable``, so the id must go
    there and not into the metadata dict.
    """
    return RunnableConfig({"configurable": {THREAD_ID: run_id, "checkpoint_id": checkpoint_id}})


async def _abort(task: asyncio.Task[Any]) -> None:
    """Cancel *task* and wait for it to actually stop.

    Waiting matters: the caller goes on to read a checkpoint, and a task still
    mid-write would make that read a lie.
    """
    if not task.done():
        task.cancel()
    with suppress(BaseException):
        await task


def _run_status(value: str) -> RunStatus:
    """Coerce an arbitrary state string to a registry status."""
    return cast("RunStatus", value) if value in RUN_STATUSES else "running"


def _pending_from(values: WorkflowState, snap: Any) -> ApprovalRequest | None:
    """Recover the pending approval from the state or the snapshot interrupts."""
    stored = as_model(values, "pending_approval", ApprovalRequest)
    if stored is not None:
        return stored
    for interrupt in snap.interrupts or ():
        value = getattr(interrupt, "value", None)
        if value is None:
            continue
        try:
            request, _meta = decode_interrupt(value)
        except Exception as exc:  # an unknown interrupt is not fatal here
            log.warning("engine.undecodable_interrupt", error=str(exc))
            continue
        return request
    return None


def _resume_value(resume: ApprovalDecision | dict[str, Any]) -> Any:
    """Convert a decision into the JSON-safe value LangGraph stores."""
    if isinstance(resume, ApprovalDecision):
        return resume.model_dump(mode="json")
    return resume


def _checkpoint_id(snap: Any) -> str | None:
    """Extract the checkpoint identifier from a snapshot's config."""
    config = getattr(snap, "config", None) or {}
    value = config.get("configurable", {}).get("checkpoint_id")
    return str(value) if value else None


def _checkpoint_info(snap: Any) -> CheckpointInfo:
    """Project a history snapshot onto a :class:`CheckpointInfo`."""
    metadata = dict(getattr(snap, "metadata", None) or {})
    return CheckpointInfo(
        checkpoint_id=_checkpoint_id(snap) or "",
        step=int(metadata.get("step") or 0),
        source=str(metadata.get("source") or ""),
        next_nodes=tuple(snap.next or ()),
        created_at=getattr(snap, "created_at", None),
        pending=_pending_from(cast("WorkflowState", dict(snap.values or {})), snap),
    )


def _default_decision(pending: ApprovalRequest) -> dict[str, Any]:
    """Auto-approve a gate (used when the caller supplies no policy)."""
    return {
        "approval_id": pending.approval_id,
        "decision": "approve",
        "reviewer": "auto",
        "comment": "auto-approved by run_until_done",
    }


__all__ = [
    "FINAL_STATUSES",
    "RUN_STATUSES",
    "CheckpointInfo",
    "Decider",
    "RunOutcome",
    "WorkflowEngine",
]
