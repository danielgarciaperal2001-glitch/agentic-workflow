"""Checkpoint retention.

Every checkpoint a run writes is a full copy of the state, so a chatty
multi-agent run produces a lot of rows. Left alone the checkpoint table becomes
the largest table in the database and slows down the listing queries the
approval inbox depends on.

LangGraph exposes the primitives needed to prune safely:

* ``adelete_thread(thread_id)`` — drop a whole run's history at once.
* ``alist`` with a ``filter`` — enumerate checkpoint metadata cheaply, without
  deserialising the state blobs.

The janitor is idempotent and safe to run concurrently with live runs: it only
deletes threads whose *last update* is older than the TTL, so a run that is
currently being worked on is never touched.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agentic_workflow.config import Settings, load_settings
from agentic_workflow.domain.schemas import utcnow
from agentic_workflow.errors import PersistenceError
from agentic_workflow.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Selection:
    """What one pass decided to do, before it does any of it.

    Kept separate from :class:`RetentionReport` so a dry run and a real run
    cannot disagree about what was considered: the selection is computed once
    and both report the same numbers.

    Attributes:
        threads: Thread ids selected for deletion, bounded by the pass limit.
        checkpoints: Checkpoints those threads are expected to free.
        examined: Threads considered in this pass.
        undated: Checkpoints whose age could not be established.
    """

    threads: list[str]
    checkpoints: int
    examined: int
    undated: int


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """Outcome of a janitor pass.

    ``examined`` and ``stale`` are separate fields because "the sweep looked at
    nothing" and "the sweep looked at everything and nothing was old" are
    opposite conditions that a single counter renders identically. The first is a
    broken sweep; the second is a healthy one. Reporting only the count of
    selected threads meant a sweep that could not list anything logged
    ``scanned=0`` — identical to a quiet night — and that is precisely how a
    sweep deleting the entire store went unnoticed.

    Attributes:
        examined: Threads inspected in this pass.
        stale: Threads whose newest checkpoint predates the cutoff.
        deleted: Threads actually removed.
        freed_checkpoints: Approximate checkpoints removed (LangGraph does not
            return row counts, so this is the number of threads times the
            observed history depth).
        duration_seconds: Wall-clock cost of the pass.
        dry_run: Whether the pass only reported what it *would* delete.
        undated: Checkpoints whose age could not be established, and which were
            therefore never deleted.
    """

    examined: int
    stale: int
    deleted: int
    freed_checkpoints: int
    duration_seconds: float
    dry_run: bool
    undated: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for logging and the CLI."""
        return {
            "examined": self.examined,
            "stale": self.stale,
            "deleted": self.deleted,
            "freed_checkpoints": self.freed_checkpoints,
            "duration_seconds": round(self.duration_seconds, 3),
            "dry_run": self.dry_run,
            "undated": self.undated,
        }


class CheckpointJanitor:
    """Delete checkpoint histories older than the configured retention window.

    Example:
        --------
        >>> report = await CheckpointJanitor(checkpointer).run()  # doctest: +SKIP
        >>> report.deleted
        12
    """

    def __init__(self, checkpointer: Any, settings: Settings | None = None) -> None:
        """Bind the janitor to a checkpointer.

        Args:
            checkpointer: A compiled saver (async or sync).
            settings: Configuration; ``state_retention_days`` drives the TTL.
        """
        self._checkpointer = checkpointer
        self._settings = settings or load_settings()

    async def run(
        self,
        *,
        retention_days: int | None = None,
        dry_run: bool = False,
        limit: int = 1_000,
    ) -> RetentionReport:
        """Perform one cleanup pass.

        Args:
            retention_days: Override the configured TTL. ``0`` deletes every
                thread older than the current instant, i.e. everything.
            dry_run: Report what would be deleted without deleting it.
            limit: Maximum threads examined in one pass, so the janitor cannot
                monopolise the database.

        Returns:
            A :class:`RetentionReport`.

        Raises:
            PersistenceError: If the checkpointer cannot be read or written.
        """
        import time

        started = time.perf_counter()
        days = self._settings.state_retention_days if retention_days is None else retention_days
        cutoff = utcnow().timestamp() - max(0, days) * 86_400.0

        selection = await self._collect(cutoff, limit)
        threads = selection.threads
        deleted = 0
        if not dry_run:
            for thread_id in threads:
                try:
                    await self._delete_thread(thread_id)
                    deleted += 1
                except Exception as exc:
                    log.warning("janitor.delete_failed", thread_id=thread_id, error=str(exc))
        report = RetentionReport(
            examined=selection.examined,
            stale=len(threads),
            deleted=deleted,
            freed_checkpoints=selection.checkpoints,
            duration_seconds=time.perf_counter() - started,
            dry_run=dry_run,
            undated=selection.undated,
        )
        log.info("janitor.pass", **report.as_dict())
        return report

    # ------------------------------------------------------------------ #
    async def _collect(self, cutoff: float, limit: int) -> _Selection:
        """Find threads whose newest checkpoint predates *cutoff*.

        A thread whose age cannot be established is **kept**. Failing open here
        is the one unforgivable mistake this component could make: the sweep
        would delete work that was never stale, irrecoverably, on a schedule
        nobody was watching.
        """
        # thread -> (newest known timestamp, checkpoint count)
        per_thread: dict[str, tuple[float | None, int]] = {}
        undated = 0
        try:
            async for meta in self._alist():
                thread_id = _thread_id_of(meta)
                if not thread_id:
                    continue
                stamp = _checkpoint_timestamp(meta)
                if stamp is None:
                    undated += 1
                previous, depth = per_thread.get(thread_id, (None, 0))
                if stamp is not None:
                    stamp = max(previous, stamp) if previous is not None else stamp
                per_thread[thread_id] = (stamp, depth + 1)
        except Exception as exc:
            raise PersistenceError(f"janitor could not list checkpoints: {exc}") from exc

        if undated:
            # Loud, because it means the deletion decision is being made on
            # incomplete information for part of the store.
            log.warning(
                "janitor.undated_checkpoints",
                undated=undated,
                hint="checkpoints without a usable `ts` are never deleted",
            )

        stale = [
            (thread_id, depth)
            for thread_id, (newest, depth) in per_thread.items()
            # `newest is None` → undecidable age → keep. A float of 0.0 is a
            # real timestamp (the epoch) and is genuinely stale.
            if newest is not None and newest < cutoff
        ]
        stale.sort()
        selected = stale[:limit]
        return _Selection(
            threads=[t for t, _ in selected],
            checkpoints=sum(d for _, d in selected),
            examined=len(per_thread),
            undated=undated,
        )

    def _saver(self) -> Any:
        """Resolve the object that actually implements the saver protocol.

        Both shapes of checkpointer reach this code. A raw LangGraph saver
        implements ``alist`` itself; :class:`~agentic_workflow.persistence.
        checkpointer.PostgresCheckpointer` is a *wrapper* that owns the saver,
        its pool and its schema, and forwards the handful of methods the engine
        needs. Probing the wrapper found no ``alist`` at all, so the janitor
        reported "the configured checkpointer cannot be listed" for the one
        checkpointer it exists to clean up.

        Returns:
            The object exposing ``alist``.

        Raises:
            PersistenceError: If neither the checkpointer nor a wrapped ``saver``
                implements ``alist``.
        """
        for candidate in (self._checkpointer, getattr(self._checkpointer, "saver", None)):
            if candidate is not None and hasattr(candidate, "alist"):
                return candidate
        raise PersistenceError("the configured checkpointer cannot be listed")

    def _alist(self) -> Any:
        """Return an async iterator over checkpoint metadata.

        ``config`` is a *required positional* argument on LangGraph's saver
        protocol; passing nothing raised ``TypeError: alist() missing 1 required
        positional argument: 'config'`` for every saver. ``None`` is the value
        that means "every thread", which is exactly what a retention sweep wants
        — the alternative, walking a registry of known run ids, would miss
        threads whose run has already been evicted from memory.
        """
        return self._saver().alist(None)

    async def _delete_thread(self, thread_id: str) -> None:
        """Remove every checkpoint belonging to *thread_id*."""
        for name in ("adelete_thread", "delete_thread"):
            for candidate in (self._checkpointer, getattr(self._checkpointer, "saver", None)):
                if candidate is None:
                    continue
                method = getattr(candidate, name, None)
                if method is None:
                    continue
                result = method(thread_id)
                if asyncio.iscoroutine(result):
                    await result
                return
        raise PersistenceError("the configured checkpointer cannot delete threads")


def _thread_id_of(meta: Any) -> str | None:
    """Read a checkpoint's thread id from its ``RunnableConfig``.

    Args:
        meta: One ``CheckpointTuple`` from ``alist``.

    Returns:
        The thread id, or ``None`` when the entry carries none.
    """
    config = getattr(meta, "config", None)
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else None


def _checkpoint_timestamp(meta: Any) -> float | None:
    """Read when a checkpoint was written, in epoch seconds.

    The timestamp is **not** an attribute of ``CheckpointTuple`` — reading
    ``meta.ts`` returned ``None`` for every real entry, which the sweep then
    interpreted as the epoch. Since the epoch is older than any cutoff, every
    thread looked stale and a ``retention_days=30`` pass deleted the whole store
    on its first run. It lives in ``meta.checkpoint["ts"]`` as an ISO-8601
    string, and ``meta.metadata`` is checked as a fallback for savers that put
    it there instead.

    Args:
        meta: One ``CheckpointTuple`` from ``alist``.

    Returns:
        Epoch seconds, or ``None`` when the age cannot be established. ``None``
        must be read as "do not delete".
    """
    checkpoint = getattr(meta, "checkpoint", None)
    if isinstance(checkpoint, dict):
        stamp = _as_timestamp(checkpoint.get("ts"))
        if stamp:
            return stamp
    metadata = getattr(meta, "metadata", None)
    if isinstance(metadata, dict):
        stamp = _as_timestamp(metadata.get("ts"))
        if stamp:
            return stamp
    return None


def _as_timestamp(value: Any) -> float:
    """Coerce a LangGraph checkpoint timestamp into epoch seconds.

    A value that cannot be read becomes ``0.0``, which callers must treat as
    "no information" — see :func:`_checkpoint_timestamp`, which is the only
    place that decision is made.
    """
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return _dt.datetime.fromisoformat(value).timestamp()
        except ValueError:
            return 0.0
    if isinstance(value, int | float):
        return float(value) / (1_000 if value > 1e11 else 1)
    return 0.0


__all__ = ["CheckpointJanitor", "RetentionReport"]
