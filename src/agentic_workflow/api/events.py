"""In-process fan-out of run events to WebSocket subscribers.

The engine emits structured events (node completed, approval requested, run
parked…) through a single :data:`~agentic_workflow.graph.context.EventSink`
callback. :class:`EventHub` is that callback, and it owns one bounded queue per
subscriber.

Three properties matter more than features here, because this is the component
most likely to take the whole service down:

1. **A slow reader must never block a run.** Each subscriber has a bounded
   queue; when it overflows the *oldest* event is dropped and a counter is
   incremented. A browser tab on hotel wifi cannot apply backpressure to an
   LLM-backed workflow.
2. **A dead reader must never leak.** Delivery is best-effort and a failed send
   unsubscribes immediately instead of retrying against a closed socket.
3. **Events are notifications, not the source of truth.** Every event can be
   recovered over REST, so dropping one degrades the UI's latency, never its
   correctness. That is what makes lossy delivery acceptable.

Subscribers register under a run id, or under :data:`ANY_RUN` to observe every
run at once (dashboards). A wildcard subscriber receives the union of all
buckets, so a single socket can replace N per-run ones.

This hub is intentionally in-process. A multi-replica deployment swaps it for a
Redis pub/sub adapter with the same three-method interface; see
``docs/architecture.md``.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Final

from agentic_workflow.config import Settings
from agentic_workflow.domain.schemas import utcnow
from agentic_workflow.errors import ErrorCode, WorkflowError
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

#: Per-subscriber queue depth. Deep enough to absorb a burst of node events
#: during a reconnect, shallow enough that a wedged client cannot pin memory.
DEFAULT_QUEUE_SIZE: Final[int] = 256

#: Bucket key for a subscriber that observes every run.
ANY_RUN: Final[str] = "*"


class ConnectionLimitError(WorkflowError):
    """A run already has the maximum number of observers attached.

    Every socket is a coroutine held for the life of a connection. Without a cap
    one misbehaving client could occupy the whole event loop's task budget, so the
    limit is a correctness property rather than a nicety.

    This is a :class:`~agentic_workflow.errors.WorkflowError` rather than a bare
    ``ConnectionError`` so the WebSocket handler's ``except`` clause can name it
    with context, and so a future HTTP caller gets the standard error envelope
    instead of a string.
    """

    code = ErrorCode.RATE_LIMITED

    def __init__(self, message: str = "", /, **context: Any) -> None:
        """Build the error.

        Args:
            message: Human-readable explanation.
            **context: Log-safe context such as ``run_id`` and ``limit``.
        """
        self.retryable = True
        super().__init__(message, **context)


@dataclass(slots=True)
class Subscription:
    """One WebSocket client's view of the event stream.

    Attributes:
        run_id: The run being observed, or :data:`ANY_RUN`.
        queue: Bounded event queue, read with :meth:`get`.
        dropped: Number of events discarded because the reader fell behind.
        key: Monotonic identity, for logging.
    """

    run_id: str
    queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=DEFAULT_QUEUE_SIZE)
    )
    dropped: int = 0
    key: int = 0

    @property
    def is_wildcard(self) -> bool:
        """Whether this subscriber observes every run."""
        return self.run_id == ANY_RUN

    async def get(self, *, timeout: float) -> dict[str, Any] | None:
        """Await the next event.

        Args:
            timeout: Seconds to wait before giving up.

        Returns:
            The next event, or ``None`` when *timeout* elapsed — which the reader
            turns into a heartbeat frame rather than a disconnect.
        """
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    def put(self, event: dict[str, Any]) -> bool:
        """Enqueue an event, dropping the oldest one if the reader is behind.

        Args:
            event: The event to deliver.

        Returns:
            ``True`` if it was enqueued, ``False`` if the queue was still full.
        """
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop-oldest, not drop-newest: the newest event is the one a UI
            # needs most in order to converge on the right state.
            with suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self.dropped += 1
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - only under a reentrant send
                return False
        return True


class EventHub:
    """Broadcasts engine events to subscribers, bucketed by run id.

    Example:
        --------
        >>> hub = EventHub(Settings(_env_file=None))  # doctest: +SKIP
        >>> sub = await hub.subscribe("run-1")  # doctest: +SKIP
        >>> await hub.publish({"run_id": "run-1", "event": "run.started"})  # doctest: +SKIP
        >>> (await sub.get(timeout=0.1))["event"]  # doctest: +SKIP
        'run.started'
    """

    def __init__(self, settings: Settings, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        """Wire the hub.

        Args:
            settings: Supplies the per-run connection limit.
            queue_size: Per-subscriber queue depth.
        """
        self._settings = settings
        self._queue_size = max(1, queue_size)
        self._subscribers: dict[str, set[Subscription]] = {}
        self._counter = 0
        self._published = 0
        self._delivered = 0
        self._dropped = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ publish #
    async def publish(self, event: dict[str, Any]) -> None:
        """Deliver an event to the subscribers of its run and to every wildcard.

        This is the ``EventSink`` the engine is constructed with. It never
        raises: a hub failure must not be able to fail a workflow run.

        Args:
            event: Structured event. ``run_id`` selects the bucket; an event
                without one goes to wildcard subscribers only.
        """
        try:
            run_id = str(event.get("run_id") or "")
            payload = {**event, "seq": self._published}
            targets: list[Subscription] = list(self._subscribers.get(ANY_RUN, ()))
            if run_id:
                targets.extend(self._subscribers.get(run_id, ()))

            delivered = 0
            for sub in targets:
                if sub.put(payload):
                    delivered += 1
                else:
                    self._dropped += 1
            self._published += 1
            self._delivered += delivered
        except Exception as exc:  # an observability bug must not fail a run
            log.warning("hub.publish_failed", error=str(exc))

    # ----------------------------------------------------------- subscribe #
    async def subscribe(self, run_id: str = ANY_RUN) -> Subscription:
        """Register a subscriber.

        Args:
            run_id: Run to observe, or :data:`ANY_RUN` for every run. The cap
                applies to per-run buckets; a wildcard subscription is limited by
                the same number so a dashboard cannot bypass it either.

        Returns:
            The new :class:`Subscription`.

        Raises:
            ConnectionLimitError: If the bucket is already at its limit.
        """
        async with self._lock:
            bucket = self._subscribers.setdefault(run_id, set())
            limit = self._settings.ws_max_connections_per_run
            if len(bucket) >= limit:
                raise ConnectionLimitError(
                    f"too many observers for {run_id!r} (limit {limit})",
                    run_id=run_id,
                    limit=limit,
                )
            self._counter += 1
            sub = Subscription(run_id=run_id, key=self._counter)
            bucket.add(sub)
            subscribers = len(bucket)
        log.info("hub.subscribed", run_id=run_id, subscribers=subscribers)
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        """Release a subscriber's slot. Safe to call twice.

        Args:
            sub: The subscription to drop.
        """
        async with self._lock:
            bucket = self._subscribers.get(sub.run_id)
            if bucket is not None:
                bucket.discard(sub)
                if not bucket:
                    del self._subscribers[sub.run_id]
        log.info("hub.unsubscribed", run_id=sub.run_id, dropped=sub.dropped)

    async def close(self) -> None:
        """Drop every subscriber. Used on shutdown."""
        async with self._lock:
            self._subscribers.clear()

    # -------------------------------------------------------------- stats #
    def subscriber_count(self, run_id: str | None = None) -> int:
        """Number of active subscribers, optionally for one bucket.

        Args:
            run_id: Restrict the count to a single run. ``None`` counts them all.

        Returns:
            The subscriber count.
        """
        if run_id is not None:
            return len(self._subscribers.get(run_id, ()))
        return sum(len(bucket) for bucket in self._subscribers.values())

    def observed_runs(self) -> list[str]:
        """Return the per-run buckets that currently have at least one observer.

        Returns:
            The observed run ids, excluding the wildcard bucket.
        """
        return [key for key, bucket in self._subscribers.items() if key != ANY_RUN and bucket]

    def stats(self) -> dict[str, Any]:
        """Return counters for the readiness probe and the metrics endpoint.

        Returns:
            Published / delivered / dropped event counts, subscriber totals and
            the current UTC timestamp.
        """
        return {
            "published": self._published,
            "delivered": self._delivered,
            "dropped": self._dropped,
            "subscribers": self.subscriber_count(),
            "runs_observed": len(self.observed_runs()),
            "ts": utcnow().isoformat(),
        }


__all__ = ["ANY_RUN", "DEFAULT_QUEUE_SIZE", "ConnectionLimitError", "EventHub", "Subscription"]
