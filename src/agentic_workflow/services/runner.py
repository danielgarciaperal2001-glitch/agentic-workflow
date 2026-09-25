"""Batch entry point: process many requests through one engine.

The API is optimised for one run at a time, but the interesting production
question is usually "what happens when two hundred pull requests land at once".
This module answers it: bounded concurrency, per-request isolation, and a summary
that tells the operator what actually happened rather than what was requested.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentic_workflow.domain.schemas import ApprovalRequest, ReviewRequest
from agentic_workflow.errors import WorkflowError
from agentic_workflow.logging import get_logger
from agentic_workflow.services.engine import RunOutcome, WorkflowEngine

log = get_logger(__name__)

#: Decides a gate. Return ``None`` to leave the run parked.
Decider = Callable[[ApprovalRequest], dict[str, Any] | None]


@dataclass(slots=True)
class BatchSummary:
    """Aggregated outcome of a batch.

    Attributes:
        total: Requests submitted.
        completed: Runs that reached ``completed``.
        parked: Runs waiting on a human decision.
        failed: Runs that ended in an error.
        cancelled: Runs cancelled by the caller.
        outcomes: Per-run outcomes, in submission order.
        errors: ``run_id -> message`` for every failure.
        duration_seconds: Wall-clock cost of the batch.
    """

    total: int
    completed: int = 0
    parked: int = 0
    failed: int = 0
    cancelled: int = 0
    outcomes: list[RunOutcome] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of runs that completed without a human or an error."""
        if self.total == 0:
            return 0.0
        return self.completed / self.total

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "total": self.total,
            "completed": self.completed,
            "parked": self.parked,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "success_rate": round(self.success_rate, 4),
            "errors": dict(self.errors),
            "duration_seconds": round(self.duration_seconds, 3),
            "runs": [o.run_id for o in self.outcomes],
        }

    def __str__(self) -> str:
        return (
            f"{self.completed}/{self.total} completed "
            f"({self.parked} parked, {self.failed} failed) "
            f"in {self.duration_seconds:.1f}s"
        )


async def run_pipeline(
    engine: WorkflowEngine,
    requests: Sequence[ReviewRequest],
    *,
    decide: Decider | None = None,
    concurrency: int | None = None,
    stop_on_error: bool = False,
    on_event: Callable[[RunOutcome], None] | None = None,
) -> BatchSummary:
    """Process a batch of requests through *engine*.

    Concurrency is bounded twice — by the engine's own semaphore and by the local
    ``concurrency`` gate — so a batch of ten thousand requests cannot exhaust the
    provider's rate limit or the process's memory.

    Args:
        engine: The engine that owns the checkpointer and graph.
        requests: The requests to process.
        decide: Auto-resolution policy for human gates. ``None`` leaves runs
            parked, which is the right default for a human-in-the-loop system.
        concurrency: Maximum simultaneous runs. Defaults to the configured
            ``max_parallel_runs``.
        stop_on_error: Abort the batch after the first failure. Off by default:
            one bad request should not take down the other ninety-nine.
        on_event: Called with each outcome as it completes, for progress
            reporting.

    Returns:
        A :class:`BatchSummary` covering every submitted request.

    Raises:
        Nothing: per-request failures are captured in
            :attr:`BatchSummary.errors` so the caller sees a complete picture.
    """
    import time

    started = time.perf_counter()
    limit = max(1, concurrency or engine.settings.max_parallel_runs)
    semaphore = asyncio.Semaphore(limit)
    summary = BatchSummary(total=len(requests))
    abort = asyncio.Event()

    async def one(request: ReviewRequest) -> RunOutcome | None:
        if abort.is_set():
            return None
        async with semaphore:
            if abort.is_set():
                return None
            try:
                outcome = await _run_one(engine, request, decide)
            except WorkflowError as exc:
                summary.failed += 1
                summary.errors[request.run_id] = str(exc)
                log.warning("batch.run_failed", run_id=request.run_id, error=str(exc))
                if stop_on_error:
                    abort.set()
                return None
            except Exception as exc:  # a batch must survive anything
                summary.failed += 1
                summary.errors[request.run_id] = f"{type(exc).__name__}: {exc}"
                log.error("batch.run_crashed", run_id=request.run_id, error=str(exc))
                if stop_on_error:
                    abort.set()
                return None

            summary.outcomes.append(outcome)
            if outcome.status == "completed":
                summary.completed += 1
            elif outcome.status == "cancelled":
                summary.cancelled += 1
            else:
                summary.parked += 1
            if on_event is not None:
                on_event(outcome)
            return outcome

    await asyncio.gather(*(one(request) for request in requests))
    summary.duration_seconds = time.perf_counter() - started
    log.info("batch.completed", **summary.as_dict())
    return summary


async def _run_one(
    engine: WorkflowEngine,
    request: ReviewRequest,
    decide: Decider | None,
) -> RunOutcome:
    """Run a single request to a stopping point."""
    if decide is None:
        return await engine.start(request)
    return await engine.run_until_done(request, decide=decide)


def sequential_runner(
    engine: WorkflowEngine,
    *,
    decide: Decider | None = None,
) -> Callable[[Sequence[ReviewRequest]], Awaitable[BatchSummary]]:
    """Return a reusable runner bound to *engine*.

    Convenience for callers that submit several batches and want identical
    semantics each time (the CLI, the demo, the evaluation harness).
    """

    async def run(requests: Sequence[ReviewRequest]) -> BatchSummary:
        """Run *requests* through the bound engine."""
        return await run_pipeline(engine, requests, decide=decide)

    return run


__all__ = ["BatchSummary", "Decider", "run_pipeline", "sequential_runner"]
