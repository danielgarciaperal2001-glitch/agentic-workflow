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
   correctness. That is what makes lossy delivery acceptable — but it holds
   only for a client that can *tell* it dropped something, which is why the
   losses are counted: per subscription in :attr:`Subscription.dropped`, and
   process-wide as ``awf_events_dropped``. A drop that is neither counted nor
   reported leaves a frozen dashboard indistinguishable from a healthy one,
   and the recovery in point 3 becomes advice the client has no way to act on.

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
from typing import Any, Final, NamedTuple

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


class PutResult(NamedTuple):
    """What one enqueue attempt did to a subscriber's queue.

    The two fields are independent by design. Under the drop-oldest policy a
    full queue both discards an event and accepts the new one, so any single
    return value loses half of what happened: a bool that means "enqueued"
    cannot report the loss, and a loss count cannot report whether the new
    event arrived. :meth:`Subscription.put` returns this so the hub can keep
    ``awf_events_delivered`` and ``awf_events_dropped`` each honest.

    Attributes:
        enqueued: Whether the new event reached the queue.
        lost: How many queued events were discarded to make room.
    """

    enqueued: bool
    lost: int


#: ``eq=False`` is load-bearing, not a style choice. The hub holds subscribers in
#: a ``set`` keyed by identity, and a dataclass's generated ``__eq__`` sets
#: ``__hash__`` to ``None`` — so the default made every ``subscribe()`` raise
#: ``TypeError: unhashable type: 'Subscription'``. It would also have been the
#: wrong comparison: two subscriptions with identical fields hold distinct
#: queues, so structural equality would collapse them into one bucket entry and
#: a client's events would be delivered to nobody.
@dataclass(slots=True, eq=False)
class Subscription:
    """One WebSocket client's view of the event stream.

    Compared by identity, never by value: this object *is* the subscription, and
    two clients watching the same run are two different subscribers.

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
    #: Loss the reader has not been told about yet. Distinct from ``dropped``,
    #: which is a running total for the logs: a notice is a delta, and it is
    #: consumed by whoever delivers it, so the two cannot be the same field.
    _untold: int = field(default=0, repr=False)
    key: int = 0

    @property
    def is_wildcard(self) -> bool:
        """Whether this subscriber observes every run."""
        return self.run_id == ANY_RUN

    def take_loss(self) -> int:
        """Collect the losses this client has not been told about, and reset them.

        The delta rather than the running total, because the client's question
        is "how much did I miss since you last spoke", and a client that
        reconnects — the normal way to recover — has no baseline to difference a
        cumulative count against.

        Consumed rather than read, so a notice is delivered exactly once. The
        loss is noticed by the writer and delivered by the reader, which are
        different tasks: a client that is mid-frame when the drop happens is
        precisely the client that dropped events, so the notice has to survive
        until somebody collects it.

        Returns:
            How many events were lost since the previous call, or ``0`` when
            nothing has been lost. The caller leaves the field off the frame
            entirely in that case, rather than sending a zero nobody needs.
        """
        lost, self._untold = self._untold, 0
        return lost

    async def get(self, *, timeout: float) -> dict[str, Any] | None:
        """Await the next event.

        Uses :func:`asyncio.timeout` rather than :func:`asyncio.wait_for`, and the
        distinction is load-bearing on Python 3.11. There, ``wait_for`` wraps its
        argument in a task and, on cancellation, does this::

            except CancelledError:
                if fut.done():
                    return fut.result()   # the cancellation is discarded
                else:
                    ...
                    raise

        This method is the reason a subscription's queue is worth filling: a full
        queue is what a *lagging* reader has, and a full queue makes ``Queue.get()``
        return without suspending, so ``fut`` is always already done and the
        cancellation that ends the pump is the one ``wait_for`` throws away. The
        reader then loops forever on a queue nobody is draining, and while the queue
        stays full its loop has no suspension point left, so it starves the event
        loop rather than merely leaking its own slot. ``asyncio.timeout`` has no
        wrapped future to inspect and re-raises an external ``CancelledError``
        untouched; catching ``TimeoutError`` alone is what already lets it through.

        Args:
            timeout: Seconds to wait before giving up.

        Returns:
            The next event, or ``None`` when *timeout* elapsed — which the reader
            turns into a heartbeat frame rather than a disconnect.
        """
        try:
            async with asyncio.timeout(timeout):
                return await self.queue.get()
        except TimeoutError:
            return None

    def put(self, event: dict[str, Any]) -> PutResult:
        """Enqueue an event, dropping the oldest one if the reader is behind.

        Args:
            event: The event to deliver.

        Returns:
            Whether the new event reached the queue, and how many events were
            discarded to make room for it. Both facts are needed and neither
            implies the other.

            This used to return a single ``True`` for "the event was enqueued",
            which cannot express what actually happened: dropping the oldest
            event and enqueueing the new one is a successful put *and* a lost
            event. The caller counted losses only when the return was ``False``
            — a reentrant send marked ``pragma: no cover`` — so the hub's
            dropped total was structurally pinned at zero under the drop-oldest
            policy. That total is what ``awf_events_dropped`` reports, and it
            was answering zero while subscribers were losing events.
        """
        try:
            self.queue.put_nowait(event)
            return PutResult(enqueued=True, lost=0)
        except asyncio.QueueFull:
            # Drop-oldest, not drop-newest: the newest event is the one a UI
            # needs most in order to converge on the right state.
            with suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self.dropped += 1
            self._untold += 1
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - only under a reentrant send
                return PutResult(enqueued=False, lost=1)
            return PutResult(enqueued=True, lost=1)


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
            settings: Supplies the per-run and total connection limits.
            queue_size: Per-subscriber queue depth.
        """
        self._settings = settings
        self._queue_size = max(1, queue_size)
        self._subscribers: dict[str, set[Subscription]] = {}
        self._counter = 0
        # Running total, rather than a sum over the buckets, because the total
        # is checked on every handshake and the sum is O(buckets) — which is
        # exactly the quantity an attacker is inflating. `unsubscribe` is the
        # only other writer, and `close` zeroes it.
        self._live = 0
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
                # A slow subscriber loses its oldest queued event *and* receives
                # the new one. Both hold at once, which is why `put` reports
                # them separately: the enqueue succeeding is not evidence that
                # nothing was lost, and a loss is not evidence that the enqueue
                # failed.
                result = sub.put(payload)
                if result.enqueued:
                    delivered += 1
                self._dropped += result.lost
            self._published += 1
            self._delivered += delivered
        except Exception as exc:  # an observability bug must not fail a run
            log.warning("hub.publish_failed", error=str(exc))

    # ----------------------------------------------------------- subscribe #
    async def subscribe(self, run_id: str = ANY_RUN) -> Subscription:
        """Register a subscriber, if both the bucket and the plane have room.

        Two limits, because they answer different questions.
        ``ws_max_connections_per_run`` bounds one run's fan-out, so a dashboard
        cannot crowd out the operator watching a run parked on a human decision.
        ``ws_max_connections_total`` bounds the process, and it is the one that
        holds when a client spreads its sockets over invented run ids: the
        bucket key is a string the caller supplies and the hub never checks it
        against a run, so without a total the number of buckets is the caller's
        to choose. Measured cost of one live socket on a real server: about
        39 KiB and one file descriptor, linear in the count.

        The wildcard bucket is subject to both, so a dashboard gains nothing by
        not naming a run.

        Args:
            run_id: Run to observe, or :data:`ANY_RUN` for every run.

        Returns:
            The new :class:`Subscription`.

        Raises:
            ConnectionLimitError: If the bucket or the plane is at its limit.
        """
        async with self._lock:
            bucket = self._subscribers.get(run_id, ())
            per_run = self._settings.ws_max_connections_per_run
            if len(bucket) >= per_run:
                raise ConnectionLimitError(
                    f"too many observers for {run_id!r} (limit {per_run})",
                    run_id=run_id,
                    limit=per_run,
                    scope="run",
                )
            total = self._settings.ws_max_connections_total
            if self._live >= total:
                # Both numbers travel with the refusal because the run id in the
                # log is the one the client asked for, which is not necessarily
                # the one at its limit. An operator who sees only the total
                # cannot tell a full plane from a full run.
                raise ConnectionLimitError(
                    f"too many observers on this plane (limit {total})",
                    run_id=run_id,
                    limit=total,
                    scope="total",
                    total=total,
                    per_run=per_run,
                    subscribers=self._live,
                )
            if not bucket:
                bucket = self._subscribers.setdefault(run_id, set())
            self._counter += 1
            sub = Subscription(run_id=run_id, key=self._counter)
            bucket.add(sub)
            self._live += 1
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
            if bucket is not None and sub in bucket:
                bucket.discard(sub)
                self._live -= 1
                if not bucket:
                    del self._subscribers[sub.run_id]
        log.info("hub.unsubscribed", run_id=sub.run_id, dropped=sub.dropped)

    async def close(self) -> None:
        """Drop every subscriber. Used on shutdown."""
        async with self._lock:
            self._subscribers.clear()
            # The count is what the total cap reads, so leaving it behind would
            # leave a plane that is permanently full after a restart-in-place.
            self._live = 0

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

        The subscriber total is reported next to the cap it is measured against.
        A count on its own cannot distinguish a plane at capacity from a quiet
        one, and that is the difference between raising a limit and hunting for
        the client holding the slots.

        Returns:
            Published / delivered / dropped event counts, subscriber totals, the
            two connection caps and the current UTC timestamp.
        """
        return {
            "published": self._published,
            "delivered": self._delivered,
            "dropped": self._dropped,
            "subscribers": self.subscriber_count(),
            "subscribers_max": self._settings.ws_max_connections_total,
            "runs_observed": len(self.observed_runs()),
            "ts": utcnow().isoformat(),
        }


__all__ = [
    "ANY_RUN",
    "DEFAULT_QUEUE_SIZE",
    "ConnectionLimitError",
    "EventHub",
    "PutResult",
    "Subscription",
]
