"""The counter that answers "did a subscriber miss events?" is always zero.

``awf_events_dropped`` is the one number an operator looks at when a dashboard
freezes and the question is whether the stream lost something. The hub's own
docstring names the mechanism: a slow reader's bounded queue drops its *oldest*
event rather than applying backpressure to an LLM-backed run.

The counter never recorded that. ``Subscription.put`` drops the oldest event
and returns ``True`` -- the *new* event was enqueued, so as far as the caller
could tell the put had worked -- and ``publish`` only incremented its running
total when ``put`` returned ``False``, which happens solely on a reentrant send
the code marks ``pragma: no cover``. Under the drop-oldest policy the exported
counter is therefore structurally incapable of being anything but zero.

Measured through the real ``/metrics`` endpoint, against the hub the app's
lifespan builds, flooding one subscriber past its 256-deep queue:

    events truly discarded:   1744
    subscriber's own count:   1744   <- exact
    stats()["dropped"]:          0
    awf_events_dropped:          0   <- what the operator sees
    awf_events_published:     2010   <- the sibling counter is honest

So the loss is invisible to the operator *and* to the client. ``seq`` does not
substitute: it is a global publication counter, so a subscriber watching one
run sees holes purely because other runs were busy (measured: seqs ``[0, 1, 4]``
with zero drops), and because the drop is always the oldest event the loss
lands at the head of the queue, leaving ``seq`` perfectly contiguous across it
(measured: a subscriber that lost 144 events saw ``144..399`` with no hole at
all). The module docstring's claim that dropping "degrades the UI's latency,
never its correctness" holds only for a client that can *tell* it dropped.

The tests below pin the counter as a measurement, and audit the rest of
``stats()`` so this one cannot be fixed while a sibling lies.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import re

from fastapi.testclient import TestClient
import pytest

from agentic_workflow.api.app import create_app
from agentic_workflow.api.events import ANY_RUN, EventHub, PutResult
from agentic_workflow.api.routers.events import _pump
from agentic_workflow.config import Settings

pytestmark = pytest.mark.api

#: The shipped per-subscriber queue depth. A subscriber flooded past it is the
#: only way to reach the drop path.
QUEUE = 256


def _settings(**overrides: object) -> Settings:
    """Build an offline configuration.

    Args:
        **overrides: Settings fields to set.

    Returns:
        The configuration.
    """
    return Settings(_env_file=None, llm_provider="echo", **overrides)  # type: ignore[arg-type]


def _metric(body: str, name: str) -> str | None:
    """Read one gauge out of the exposition text.

    Args:
        body: The ``/metrics`` payload.
        name: The counter name.

    Returns:
        The rendered value, or ``None`` when absent.
    """
    match = re.search(rf'awf_metric\{{name="{name}"\}}\s+(\S+)', body)
    return match.group(1) if match else None


class TestTheDroppedCounter:
    """``dropped`` has to count the events it says it counts."""

    async def test_a_flooded_subscriber_is_counted_in_stats(self) -> None:
        """Discarding the oldest event is a drop, and the counter has to say so.

        The queue is bounded so a browser tab on hotel wifi cannot pin an
        LLM-backed run. That is the right trade, and it is only honest if the
        loss is visible afterwards -- otherwise a frozen dashboard and a healthy
        one look identical from the metrics.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")

        for step in range(QUEUE * 3):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        # Ground truth, computed from the queue depth rather than read back from
        # the hub: everything published past the queue had to be discarded.
        truth = QUEUE * 3 - QUEUE
        assert sub.dropped == truth, "the per-subscription count is the reference"
        assert hub.stats()["dropped"] == truth, "the exported total must match it"

    async def test_the_counter_only_moves_on_an_actual_drop(self) -> None:
        """A hub nobody overran has dropped nothing, and must report nothing.

        The mirror of the test above. Without it, a fix that simply incremented
        on every publish would pass the first test while making this number
        meaningless -- a rate, not a count of losses.
        """
        hub = EventHub(_settings())
        await hub.subscribe("run-1")

        for step in range(10):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        stats = hub.stats()
        assert stats["dropped"] == 0
        assert stats["published"] == 10
        assert stats["delivered"] == 10

    async def test_the_total_counts_every_subscriber_that_overran(self) -> None:
        """One flood is counted once per socket, not once per run.

        Two subscribers on the same run each fill their own queue, so the
        process lost twice as many events. Reporting one would understate the
        cost of a slow reader by the number of clients watching it.
        """
        hub = EventHub(_settings())
        first = await hub.subscribe("run-1")
        second = await hub.subscribe("run-1")

        for step in range(QUEUE * 2):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        assert first.dropped == second.dropped
        assert hub.stats()["dropped"] == first.dropped + second.dropped

    async def test_a_wildcard_subscriber_loses_events_too(self) -> None:
        """The wildcard bucket is a real queue and is counted like any other.

        A dashboard is the reader most likely to be slow -- it is one socket
        watching every run -- so exempting it from the count would hide exactly
        the case the number exists for.
        """
        hub = EventHub(_settings())
        wildcard = await hub.subscribe(ANY_RUN)

        for step in range(QUEUE * 2):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        assert wildcard.dropped > 0
        assert hub.stats()["dropped"] == wildcard.dropped


class TestTheCounterIsExported:
    """The number has to survive the trip to ``/metrics``."""

    def test_the_endpoint_reports_a_real_loss(self) -> None:
        """A flooded hub does not export ``dropped 0``.

        Measured through the endpoint against the hub the app's lifespan
        builds, because the earlier version of this defect was invisible in
        unit tests and only appeared once rendered as Prometheus text.
        """
        settings = _settings(environment="development", postgres_enabled=False)
        app = create_app(settings, configure_logs=False)

        import asyncio

        with TestClient(app) as client:
            hub = client.app.state.hub

            async def flood() -> int:
                sub = await hub.subscribe("flooded")
                for step in range(QUEUE * 3):
                    await hub.publish(
                        {"event": "node.completed", "run_id": "flooded", "step": step}
                    )
                return sub.dropped

            lost = asyncio.run(flood())
            body = client.get("/metrics").text

        assert lost > 0, "the probe must actually overrun the queue"
        assert _metric(body, "awf_events_dropped") == str(lost)

    def test_a_quiet_hub_exports_zero_rather_than_omitting_it(self) -> None:
        """The gauge is present when nothing has been lost.

        A missing series and a zero read differently in a dashboard: one is a
        broken scrape, the other is a healthy stream. The count must be
        reported either way so the panel does not go blank on a fresh process.
        """
        app = create_app(
            _settings(environment="development", postgres_enabled=False), configure_logs=False
        )
        with TestClient(app) as client:
            body = client.get("/metrics").text

        assert _metric(body, "awf_events_dropped") is not None
        assert float(_metric(body, "awf_events_dropped") or 0) == 0.0


class TestTheOtherCounters:
    """The siblings are correct, and the audit that found this should stay true.

    Every counter in ``stats()`` was measured against independently computed
    ground truth while chasing the drop bug. Only ``dropped`` was wrong. These
    pin the rest, because a counter that is accurate by coincidence is one
    commit away from being wrong, and because the two that are *not* plain
    counts are the ones a future reader will misread.
    """

    async def test_delivered_counts_fanout_not_distinct_events(self) -> None:
        """``delivered`` is a socket-delivery count, and that is what it reports.

        Ten events to a run watched by two sockets plus one wildcard is thirty
        deliveries, not ten. A reader expecting distinct events sees a number
        three times too large; the alternative -- redefining it to count
        distinct events -- would make it useless for spotting a slow reader,
        which is the thing it is exported for.
        """
        hub = EventHub(_settings())
        await hub.subscribe("run-1")
        await hub.subscribe("run-1")
        await hub.subscribe(ANY_RUN)

        for _ in range(10):
            await hub.publish({"event": "node.completed", "run_id": "run-1"})

        stats = hub.stats()
        assert stats["published"] == 10
        assert stats["delivered"] == 30

    async def test_subscribers_returns_to_zero_without_going_negative(self) -> None:
        """Unsubscribing twice is a no-op, so the count never dips below zero.

        A negative subscriber count would make the total cap in
        ``subscribe`` read as permanently free capacity, which is a far worse
        failure than a leaked slot.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")
        assert hub.stats()["subscribers"] == 1

        await hub.unsubscribe(sub)
        await hub.unsubscribe(sub)

        assert hub.stats()["subscribers"] == 0

    async def test_runs_observed_excludes_the_wildcard_bucket(self) -> None:
        """``runs_observed`` counts runs, not buckets.

        A dashboard watching everything is subscribed to the wildcard bucket.
        Counting it would report one more observed "run" than exists, and the
        number is a capacity signal: it is how an operator sees a client
        spreading sockets over invented run ids.
        """
        hub = EventHub(_settings())
        for run_id in ("run-1", "run-2", "run-3"):
            await hub.subscribe(run_id)
        await hub.subscribe(ANY_RUN)

        assert hub.stats()["runs_observed"] == 3
        assert ANY_RUN not in hub.observed_runs()

    async def test_a_raising_subscriber_cannot_fail_a_run(self) -> None:
        """The publish fail-safe holds even when a subscriber misbehaves.

        Not reachable through the shipped ``Subscription.put``, which only ever
        raises ``QueueFull`` and catches it. It is pinned anyway because the
        fail-safe is the reason a buggy subscriber cannot take down a workflow
        run, and a future change to ``put`` is exactly the kind of change that
        would quietly remove that. The assertion is on the absence of a raise:
        ``publish`` is called from inside the graph's event path, so an
        exception here would surface as a failed run.
        """
        hub = EventHub(_settings())

        class Exploding:
            """A subscriber that fails on every send, like a dead socket."""

            def put(self, event: dict[str, object]) -> PutResult:
                """Fail the way a broken socket would.

                Args:
                    event: The event being delivered.

                Raises:
                    RuntimeError: Always.
                """
                raise RuntimeError("socket exploded")

        # Replaces the bucket outright: the fault has to be inside the loop that
        # `publish` iterates, and no legitimate path can raise there.
        hub._subscribers["run-1"] = {Exploding()}  # type: ignore[set-item]

        await hub.publish({"event": "node.completed", "run_id": "run-1"})

        # And the hub is still usable afterwards, which is the point of the
        # fail-safe: one broken subscriber must not wedge the event path.
        assert hub.stats()["published"] == 0, "the swallowed event is not counted"
        assert hub.subscriber_count("run-1") == 1


class TestTheClientIsTold:
    """A client that lost events has to be able to find out, from the stream.

    Counting the loss is only half of it. The hub's contract is that events are
    notifications and REST is the source of truth, so a dropped event costs a
    client latency rather than correctness — but that recovery is advice a
    client can only act on if it knows it dropped something. Before this, a
    client had no way to learn it: ``seq`` is a global publication counter, so
    the gaps it sees belong to other runs (measured: ``[0, 1, 4]`` with zero
    drops), and because the loss is always the *oldest* event it lands at the
    head of the queue, leaving ``seq`` contiguous straight across it (measured:
    144 lost events, ``144..399``, no hole at all).

    The notice rides on the heartbeat channel rather than as a new event type.
    A client that does not know the field sees a heartbeat it already handles,
    which is what makes this safe to ship to deployments whose front-ends are
    older than the server.

    One property is load-bearing and easy to get wrong, so it is pinned
    separately: the client that most needs telling is the one whose queue is
    *full*, and a full queue means the idle heartbeat never fires. An idle-only
    notice would therefore reach exactly the clients that are keeping up, and
    never the one that fell behind.
    """

    async def test_a_loss_is_reported_on_the_heartbeat_channel(self) -> None:
        """A heartbeat following a loss carries how many events it missed.

        The count has to be there, not just the fact of the loss: a client
        deciding whether to re-fetch the whole run state or just wait depends on
        whether it missed one event or two hundred.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")

        for step in range(QUEUE * 2):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        missed = sub.take_loss()
        assert missed == QUEUE
        assert sub.take_loss() == 0, "the notice is consumed, so it is not repeated"

    async def test_a_client_that_keeps_up_is_never_told_it_lost_nothing(self) -> None:
        """A healthy stream reports zero, and the field is left off entirely.

        Sending ``dropped_since_last: 0`` on every idle heartbeat would be noise
        on the busiest socket in the fleet, and it would train a client to
        ignore the field — which is how the real value stops being noticed.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")

        for step in range(10):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        assert sub.take_loss() == 0

    async def test_two_bursts_report_separately_rather_than_cumulatively(self) -> None:
        """Each notice covers only the losses since the previous one.

        A cumulative count would force the client to keep its own baseline to
        work out what changed, and a client that reconnects — the normal way to
        recover — would have no baseline at all.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")

        for step in range(QUEUE * 2):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})
        first = sub.take_loss()

        # Drain, so the queue has depth again and the next burst is a real burst.
        while not sub.queue.empty():
            sub.queue.get_nowait()
        for step in range(QUEUE * 2):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})
        second = sub.take_loss()

        assert first == QUEUE
        assert second == QUEUE, "the second burst is the second burst, not the sum"

    async def test_the_notice_waits_for_the_reader_rather_than_being_lost_itself(self) -> None:
        """A pending notice survives until the reader actually collects it.

        The drop is noticed by the *writer* and told by the *reader*, and those
        are two different tasks. If the notice were consumed at drop time it
        would be discarded unread whenever the reader was mid-frame — which is
        precisely the state a lagging reader is in.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")

        for step in range(QUEUE * 2):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        assert sub.dropped == QUEUE, "the total keeps counting"
        assert sub.take_loss() == QUEUE
        assert sub.dropped == QUEUE, "and collecting the notice does not zero it"


class _FakeSocket:
    """A socket that records frames and never applies backpressure.

    Deliberately does not block on ``send_text``: a real slow client is slow at
    the OS layer, and a fake that blocked here would deadlock the pump instead
    of modelling it.
    """

    def __init__(self) -> None:
        """Start with no frames sent."""
        self.frames: list[dict[str, object]] = []
        self.url = type("Url", (), {"path": "/ws/runs/run-1"})()

    async def send_text(self, frame: str) -> None:
        """Record one JSON frame.

        Args:
            frame: The serialised frame.
        """
        import json

        self.frames.append(json.loads(frame))


class TestTheNoticeIsForced:
    """The notice cannot wait for an idle heartbeat, because there is no idle.

    A subscriber's queue is full *because* it fell behind, and a full queue
    means ``sub.get`` returns immediately every time — so the idle heartbeat
    never fires for exactly the client that needs telling. Delivering the
    notice only on idle would reach the clients that are keeping up and never
    the one that lost 144 events, which is the failure mode this design has to
    avoid rather than merely improve on.
    """

    async def test_a_backlogged_reader_is_told_without_waiting_for_idle(self) -> None:
        """The notice arrives while the queue is still full.

        The heartbeat interval is a full second here, so an implementation that
        waited for idle would need that second to pass. The assertion does not
        depend on timing: the socket is still being pumped and the notice is
        already there.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")
        for step in range(QUEUE * 2):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})
        assert not sub.queue.empty(), "the reader is backlogged, not idle"

        socket = _FakeSocket()
        pump = asyncio.create_task(_pump(socket, sub, heartbeat=1.0, send_timeout=1.0))  # type: ignore[arg-type]
        try:
            for _ in range(200):
                await asyncio.sleep(0)
                if any("dropped_since_last" in f for f in socket.frames):
                    break
        finally:
            pump.cancel()
            with suppress(asyncio.CancelledError):
                await pump

        notices = [f for f in socket.frames if "dropped_since_last" in f]
        assert notices, (
            "a backlogged reader must be told immediately, not on the next idle "
            f"heartbeat; frames seen: {[f.get('event') for f in socket.frames[:5]]}"
        )
        assert notices[0]["dropped_since_last"] == QUEUE
        # It rides the heartbeat event, so a client that has never heard of the
        # field handles it as the frame it already knows.
        assert notices[0]["event"] == "heartbeat"

    async def test_a_healthy_reader_is_never_sent_the_field(self) -> None:
        """No loss means no field, on a socket that is pumping normally.

        The mirror of the test above, so a fix cannot simply always send the
        notice and satisfy both.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")
        for step in range(5):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        socket = _FakeSocket()
        pump = asyncio.create_task(_pump(socket, sub, heartbeat=0.01, send_timeout=1.0))  # type: ignore[arg-type]
        try:
            for _ in range(50):
                await asyncio.sleep(0)
        finally:
            pump.cancel()
            with suppress(asyncio.CancelledError):
                await pump

        assert socket.frames, "the pump must have delivered the events at all"
        assert all("dropped_since_last" not in f for f in socket.frames)
        assert all(f.get("event") != "heartbeat" for f in socket.frames), (
            "an idle socket still gets heartbeats, but never a zero-loss notice"
        )
