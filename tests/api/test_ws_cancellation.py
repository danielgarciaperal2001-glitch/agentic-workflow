"""Cancelling a WebSocket connection has to release it, on every supported Python.

:func:`~agentic_workflow.api.routers.events._serve` frees a subscriber's slot in a
``finally``, which only runs if :func:`~agentic_workflow.api.routers.events._pump`
returns. The pump spends its life inside :func:`asyncio.wait_for`, waiting for the
next event with the heartbeat as the deadline, and on Python 3.11 that call can
*consume* a cancellation instead of honouring it. Its own source says so::

    except exceptions.CancelledError:
        if fut.done():
            return fut.result()      # the CancelledError is dropped on the floor
        else:
            ...
            raise

The branch taken is whether the wrapped awaitable had already finished. It usually
has not, so this looks unreachable — but a subscriber's queue is full *because* it
fell behind, and a full queue makes ``Queue.get()`` return without ever suspending.
That is precisely the state where ``fut.done()`` is true and the cancellation is
swallowed, and precisely the state a lagging reader is in. Every iteration of the
pump's loop is another chance at it, so over the 256 events it has queued the
outcome stops being a matter of luck.

Measured on 3.11.16, cancelling a connection whose queue is full:

    connection task          never ends, so the slot is never released
    with a publisher still flooding, the pump stops yielding at all
    a 20s deadline in the same process    never fires
    external `timeout 40`                 has to kill it (exit 124)

That second half is the serious one. The pump's loop has no suspension point left
while the queue stays full, so it starves the event loop: no other request, no
other run, no other socket is served, and the graceful shutdown uvicorn is waiting
on cannot complete either. The hub's own module docstring claims "a slow reader
must never block a run" and "a dead reader must never leak"; on 3.11 a dead reader
blocks the whole process. Measured on 3.12.14, 3.13.15 and 3.14.7, where
``wait_for`` no longer wraps its argument in a task and so has no ``fut`` to
inspect: the connection ends, the slot comes back, and the flood case takes 1.3s.

The fix is the one ``graph/common.py`` already documents and uses: an
``asyncio.timeout`` block, which reports a deadline as ``TimeoutError`` and leaves
an external ``CancelledError`` alone. Catching only ``TimeoutError`` — which both
call sites already do — is then enough to let the cancellation through.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from typing import Any

import pytest

from agentic_workflow.api.events import EventHub
from agentic_workflow.api.routers.events import _pump
from agentic_workflow.config import Settings

pytestmark = pytest.mark.api

#: The shipped per-subscriber queue depth. Filling it is what puts the pump in
#: the state where the cancellation is lost, so a smaller queue would make this
#: file test something the server never does.
QUEUE = 256


def _settings() -> Settings:
    """Build an offline configuration.

    Returns:
        The configuration.
    """
    return Settings(_env_file=None, llm_provider="echo")  # type: ignore[arg-type]


class _FakeSocket:
    """A socket that records frames and never applies backpressure.

    Deliberately does not block on ``send_text``: a real slow client is slow at
    the OS layer, and a fake that blocked here would deadlock the pump instead of
    modelling it.
    """

    def __init__(self) -> None:
        """Start with no frames sent."""
        self.frames: list[dict[str, Any]] = []
        self.url = type("Url", (), {"path": "/ws/runs/run-1"})()

    async def send_text(self, frame: str) -> None:
        """Record one JSON frame.

        Args:
            frame: The serialised frame.
        """
        self.frames.append(json.loads(frame))


async def _connection(hub: EventHub, sub: Any, socket: _FakeSocket) -> None:
    """Run one connection the way ``_serve`` does: pump, then release the slot.

    Args:
        hub: The hub holding the subscription.
        sub: The subscriber the connection owns.
        socket: The live socket.
    """
    try:
        await _pump(socket, sub, heartbeat=1.0, send_timeout=1.0)  # type: ignore[arg-type]
    finally:
        await hub.unsubscribe(sub)


async def _flood(hub: EventHub) -> None:
    """Publish continuously, so the subscriber's queue never empties.

    Args:
        hub: The hub to publish into.
    """
    step = 0
    while True:
        for _ in range(64):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})
            step += 1
        await asyncio.sleep(0)


class TestACancelledConnectionReleasesItsSlot:
    """The pump must not swallow the cancellation that ends it."""

    async def test_the_subscription_comes_back(self) -> None:
        """A connection cancelled mid-drain gives its slot to the next client.

        The pump is cancelled at the moment its queue is full, because that is
        the only moment the pump is actually behind — and a slot that is never
        released is not a slow leak but a permanent one: the per-plane cap counts
        live subscribers, so a service that is cancelled and restarted enough
        times refuses every new socket while reporting itself healthy.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")
        for step in range(QUEUE * 2):
            await hub.publish({"event": "node.completed", "run_id": "run-1", "step": step})

        task = asyncio.create_task(_connection(hub, sub, _FakeSocket()))
        await asyncio.sleep(0)  # the pump is now inside its wait, queue still full
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except TimeoutError:
            pytest.fail(
                "the connection never ended: the cancellation was consumed and the "
                "pump is still looping over a queue nobody is draining"
            )
        except asyncio.CancelledError:
            pass

        assert hub.stats()["subscribers"] == 0, "the slot must go back to the plane"


class TestACancelledConnectionDoesNotWedgeTheLoop:
    """Losing a reader may cost events. It may not cost the process."""

    async def test_other_work_is_still_served_afterwards(self) -> None:
        """A connection cancelled mid-flood leaves the loop able to run anything.

        A pump whose queue is full has no suspension point left in its loop, so if
        it refuses to end it does not merely leak one slot — it stops the event
        loop entirely, and the first casualty is the graceful shutdown waiting on
        it. The check is deliberately trivial work: under a starved loop even
        ``sleep(0)`` never completes, so the assertion cannot pass by being slow.
        """
        hub = EventHub(_settings())
        sub = await hub.subscribe("run-1")
        flood = asyncio.create_task(_flood(hub))
        task = asyncio.create_task(_connection(hub, sub, _FakeSocket()))
        try:
            await asyncio.sleep(0.05)  # a live stream, queue full, pump behind
            task.cancel()
            with suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5.0)

            await asyncio.wait_for(asyncio.sleep(0, result="served"), timeout=5.0)
        finally:
            flood.cancel()
            with suppress(asyncio.CancelledError):
                await flood

        assert hub.stats()["subscribers"] == 0
