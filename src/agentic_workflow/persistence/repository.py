"""Run registry: live bookkeeping that is *not* durable state.

Design decision
---------------
There are two very different kinds of data in this system, and conflating them is
the classic mistake:

* **Durable state** (graph values, node order, interrupts) lives in the
  checkpointer. It survives restarts, supports time travel and is the single
  source of truth for "where is this run?".
* **Control-plane bookkeeping** (which runs exist, their coarse status, when they
  last ticked) is a *projection*. It must be cheap to read on every list
  endpoint, and it must never be able to disagree with the graph.

So the registry is an intentionally simple, in-process, TTL-bounded cache. It is
rebuildable: :func:`RunRegistry.rebuild_from` repopulates it from a checkpointer,
and every authoritative read (status, state, approvals) goes through the graph's
``aget_state``, never through this cache. That is what keeps the two layers from
drifting.

For a multi-process deployment, put the registry behind a shared store (Redis, or
a small table) — the interface below is deliberately narrow so that is a drop-in
change, and the checkpoint store already provides the durable audit trail.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Literal

from agentic_workflow.domain.schemas import (
    ApprovalRequest,
    NodeTiming,
    RunSummary,
    utcnow,
)
from agentic_workflow.errors import RunAlreadyExistsError, RunNotFoundError
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

RunStatus = Literal[
    "pending",
    "running",
    "waiting_human",
    "completed",
    "failed",
    "cancelled",
]

#: Statuses that will not change again. Used by pollers to stop waiting.
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

#: Distinguishes "argument omitted, leave the field alone" from
#: "argument is None, clear the field". ``...`` would do it at runtime but
#: ``Ellipsis`` is a real value, so a caller could accidentally collide with it.
_UNSET: Final = object()


@dataclass(slots=True)
class RunRecord:
    """Mutable metadata about one run.

    Attributes:
        run_id: Durable run identifier, also the checkpoint thread id.
        thread_id: Checkpoint thread. Equal to ``run_id`` unless the caller
            chooses to group runs under one thread.
        status: Coarse lifecycle status.
        iteration: Last observed feedback-loop iteration.
        pending_approval: Approval id blocking the run, if any.
        created_at / updated_at: ISO-8601 lifecycle markers.
        error: Terminal error message, if the run failed.
        metadata: Free-form, log-safe annotations (request id, actor, …).
    """

    run_id: str
    thread_id: str
    status: RunStatus = "pending"
    iteration: int = 0
    pending_approval: str | None = None
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: utcnow().isoformat())
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Whether the run has reached an immutable final status."""
        return self.status in TERMINAL_STATUSES

    def to_summary(
        self,
        *,
        next_node: str | None = None,
        checkpoint_id: str | None = None,
        pending: ApprovalRequest | None = None,
        timings: Sequence[NodeTiming] = (),
    ) -> RunSummary:
        """Project onto the public :class:`RunSummary` model.

        The registry deliberately stores only the *id* of a pending approval, so
        the full object is injected by whoever read it from the checkpoint. That
        keeps this cache small and guarantees the approval the client sees is the
        one the graph is actually parked on.

        Args:
            next_node: Node the graph is about to execute (from ``aget_state``).
            checkpoint_id: Identifier of the checkpoint this view describes.
            pending: The full pending approval, when known.
            timings: Per-node timings from the checkpoint.

        Returns:
            A validated summary.
        """
        return RunSummary(
            run_id=self.run_id,
            status=self.status,
            iteration=self.iteration,
            next_node=next_node,
            updated_at=_parse_iso(self.updated_at),
            checkpoint_id=checkpoint_id,
            pending_approval=pending,
            timings=list(timings),
            error=self.error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (used by the WebSocket payloads)."""
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "status": self.status,
            "iteration": self.iteration,
            "pending_approval": self.pending_approval,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class RunRegistry:
    """Bounded, concurrency-safe, in-process registry of runs.

    Args:
        max_entries: Hard cap on retained runs. The least recently updated run is
            evicted once the cap is reached, which bounds memory without needing
            a background janitor.
        ttl_seconds: Optional wall-clock TTL; :meth:`purge_expired` drops runs
            older than this. ``0`` disables expiry.

    Example:
        --------
        >>> registry = RunRegistry(max_entries=2)
        >>> _ = registry.create("run-1")
        >>> registry.get("run-1").status
        'pending'
    """

    def __init__(self, *, max_entries: int = 1_000, ttl_seconds: float = 0.0) -> None:
        self._max_entries = max(1, max_entries)
        self._ttl_seconds = ttl_seconds
        self._runs: OrderedDict[str, RunRecord] = OrderedDict()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- writes #
    def create(
        self,
        run_id: str,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Register a new run.

        Args:
            run_id: Unique run identifier.
            thread_id: Checkpoint thread; defaults to ``run_id``.
            metadata: Free-form annotations to expose on the run listing.

        Returns:
            The created record.

        Raises:
            RunAlreadyExistsError: If ``run_id`` is already registered.
        """
        if run_id in self._runs:
            raise RunAlreadyExistsError("run already registered", run_id=run_id)
        record = RunRecord(
            run_id=run_id, thread_id=thread_id or run_id, metadata=dict(metadata or {})
        )
        self._runs[run_id] = record
        self._evict_if_needed()
        log.info("run.registered", run_id=run_id, thread_id=record.thread_id)
        return record

    def update(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        iteration: int | None = None,
        pending_approval: str | object | None = _UNSET,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Patch a run's metadata.

        Only the fields explicitly provided are touched, which is what lets a
        node-local event (iteration tick) avoid clobbering a status set by the
        run manager.

        Args:
            run_id: Run to patch.
            status: New status, if it changed.
            iteration: New iteration counter.
            pending_approval: Approval id, or ``None`` to clear it. Omit
                entirely (the default) to leave the field as-is.
            error: Terminal error message.
            metadata: Annotations to merge in.

        Returns:
            The updated record.

        Raises:
            RunNotFoundError: If the run is unknown.
        """
        record = self.get(run_id)
        if status is not None:
            record.status = status
        if iteration is not None:
            record.iteration = iteration
        if pending_approval is not _UNSET:
            record.pending_approval = pending_approval  # type: ignore[assignment]
        if error is not None:
            record.error = error
        if metadata:
            record.metadata.update(metadata)
        record.updated_at = utcnow().isoformat()
        self._runs.move_to_end(run_id)
        return record

    def touch(self, run_id: str) -> None:
        """Refresh ``updated_at`` without changing anything else."""
        try:
            record = self.get(run_id)
        except RunNotFoundError:
            return
        record.updated_at = utcnow().isoformat()
        self._runs.move_to_end(run_id)

    def delete(self, run_id: str) -> None:
        """Forget a run entirely.

        Raises:
            RunNotFoundError: If the run is unknown.
        """
        if self._runs.pop(run_id, None) is None:
            raise RunNotFoundError("unknown run", run_id=run_id)
        log.info("run.deleted", run_id=run_id)

    # -------------------------------------------------------------- reads #
    def get(self, run_id: str) -> RunRecord:
        """Return the record for *run_id*.

        Raises:
            RunNotFoundError: If the run is unknown.
        """
        record = self._runs.get(run_id)
        if record is None:
            raise RunNotFoundError("unknown run", run_id=run_id)
        return record

    def find(self, run_id: str) -> RunRecord | None:
        """Return the record for *run_id*, or ``None``."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunRecord]:
        """List runs, newest first.

        Named ``list_runs`` rather than ``list`` on purpose: a method called
        ``list`` shadows the builtin for every annotation in the class body,
        which mypy rightly rejects.

        Args:
            status: Optional status filter.
            limit: Maximum records to return.
            offset: Pagination offset.

        Returns:
            The matching records, most recently updated first.
        """
        records = list(reversed(self._runs.values()))
        if status is not None:
            records = [r for r in records if r.status == status]
        start = max(0, offset)
        return records[start : start + max(0, limit)]

    def count(self, *, status: str | None = None) -> int:
        """Count registered runs, optionally filtered by status."""
        if status is None:
            return len(self._runs)
        return sum(1 for r in self._runs.values() if r.status == status)

    def __len__(self) -> int:
        return len(self._runs)

    def __contains__(self, run_id: object) -> bool:
        return run_id in self._runs

    # ----------------------------------------------------------- recovery #
    def rebuild_from(
        self,
        thread_ids: list[str],
        *,
        lookup: Any = None,
    ) -> int:
        """Repopulate the registry from durable checkpoint threads.

        Called on boot when the registry is empty but the database is not: a
        process that restarted must be able to show runs that were in flight.

        Args:
            thread_ids: Thread identifiers to hydrate.
            lookup: Optional callable ``(thread_id) -> dict`` returning the
                checkpoint's values (iteration, status…). Failures are tolerated:
                a thread we cannot read is registered as ``pending`` rather than
                crashing boot.

        Returns:
            The number of records created.
        """
        created = 0
        for thread_id in thread_ids:
            if thread_id in self._runs:
                continue
            state: dict[str, Any] = {}
            if lookup is not None:
                try:
                    state = dict(lookup(thread_id) or {})
                except Exception as exc:
                    log.warning("run.rebuild_failed", thread_id=thread_id, error=str(exc))
                    state = {}
            record = RunRecord(
                run_id=thread_id,
                thread_id=thread_id,
                status=state.get("status") or "pending",
                iteration=int(state.get("iteration") or 0),
                pending_approval=state.get("pending_approval"),
                created_at=state.get("started_at") or utcnow().isoformat(),
            )
            self._runs[thread_id] = record
            created += 1
        if created:
            log.info("run.rebuilt", restored=created)
        return created

    # ----------------------------------------------------------- eviction #
    def purge_expired(self, now: float | None = None) -> int:
        """Drop runs older than the configured TTL.

        Args:
            now: Reference timestamp (epoch seconds). Defaults to the clock.

        Returns:
            The number of records removed.
        """
        if self._ttl_seconds <= 0:
            return 0
        import time

        reference = now if now is not None else time.time()
        cutoff = reference - self._ttl_seconds
        doomed = [
            run_id
            for run_id, record in self._runs.items()
            if _epoch(record.updated_at) < cutoff and record.is_terminal
        ]
        for run_id in doomed:
            del self._runs[run_id]
        if doomed:
            log.info("run.purged", count=len(doomed))
        return len(doomed)

    def _evict_if_needed(self) -> None:
        while len(self._runs) > self._max_entries:
            run_id, _ = self._runs.popitem(last=False)
            log.warning("run.evicted", run_id=run_id, reason="registry_full")


def _parse_iso(iso: str) -> datetime:
    """Parse an ISO-8601 marker, falling back to *now* if it is unreadable."""
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return utcnow()


def _epoch(iso: str) -> float:
    """Best-effort ISO-8601 to epoch conversion (0.0 when unparseable)."""
    try:
        return _parse_iso(iso).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


__all__ = ["TERMINAL_STATUSES", "RunRecord", "RunRegistry", "RunStatus"]
