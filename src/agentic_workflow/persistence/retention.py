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
class RetentionReport:
    """Outcome of a janitor pass.

    Attributes:
        scanned: Threads inspected.
        deleted: Threads removed.
        freed_checkpoints: Approximate checkpoints removed (LangGraph does not
            return row counts, so this is the number of threads times the
            observed history depth).
        duration_seconds: Wall-clock cost of the pass.
        dry_run: Whether the pass only reported what it *would* delete.
    """

    scanned: int
    deleted: int
    freed_checkpoints: int
    duration_seconds: float
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for logging and the CLI."""
        return {
            "scanned": self.scanned,
            "deleted": self.deleted,
            "freed_checkpoints": self.freed_checkpoints,
            "duration_seconds": round(self.duration_seconds, 3),
            "dry_run": self.dry_run,
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

        threads, checkpoints = await self._collect(cutoff, limit)
        deleted = 0
        if not dry_run:
            for thread_id in threads:
                try:
                    await self._delete_thread(thread_id)
                    deleted += 1
                except Exception as exc:
                    log.warning("janitor.delete_failed", thread_id=thread_id, error=str(exc))
        report = RetentionReport(
            scanned=len(threads),
            deleted=deleted,
            freed_checkpoints=checkpoints,
            duration_seconds=time.perf_counter() - started,
            dry_run=dry_run,
        )
        log.info("janitor.pass", **report.as_dict())
        return report

    # ------------------------------------------------------------------ #
    async def _collect(self, cutoff: float, limit: int) -> tuple[list[str], int]:
        """Find threads whose newest checkpoint predates *cutoff*."""
        per_thread: dict[str, tuple[float, int]] = {}
        try:
            async for meta in self._alist():
                thread_id = getattr(meta, "config", {}).get("configurable", {}).get("thread_id")
                if not thread_id:
                    continue
                stamp = _as_timestamp(getattr(meta, "ts", None))
                depth = per_thread.get(thread_id, (0.0, 0))[1] + 1
                previous = per_thread.get(thread_id)
                per_thread[thread_id] = (
                    max(previous[0], stamp) if previous else stamp,
                    depth,
                )
        except Exception as exc:
            raise PersistenceError(f"janitor could not list checkpoints: {exc}") from exc

        stale = [
            (thread_id, depth)
            for thread_id, (newest, depth) in per_thread.items()
            if newest < cutoff
        ]
        stale.sort()
        selected = stale[:limit]
        return [t for t, _ in selected], sum(d for _, d in selected)

    def _alist(self) -> Any:
        """Return an async iterator over checkpoint metadata."""
        saver = self._checkpointer
        lister = getattr(saver, "alist", None)
        if lister is None:  # pragma: no cover - defensive
            raise PersistenceError("the configured checkpointer cannot be listed")
        return lister()

    async def _delete_thread(self, thread_id: str) -> None:
        """Remove every checkpoint belonging to *thread_id*."""
        saver = self._checkpointer
        for name in ("adelete_thread", "delete_thread"):
            method = getattr(saver, name, None)
            if method is None:
                continue
            result = method(thread_id)
            if asyncio.iscoroutine(result):
                await result
            return
        raise PersistenceError("the configured checkpointer cannot delete threads")


def _as_timestamp(value: Any) -> float:
    """Coerce a LangGraph checkpoint timestamp into epoch seconds."""
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
