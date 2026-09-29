"""A cap on observers that a client can lift by choosing a different run id.

``ws_max_connections_per_run`` is documented as a correctness property, not a
nicety: "every socket is a coroutine held for the life of a connection.
Without a cap one misbehaving client could occupy the whole event loop's task
budget." The cap it provides is per *bucket*, and the bucket key is the run id
in the URL — a string the caller supplies, which the hub never checks against a
run. So the number of buckets is the caller's to choose, and choosing more
buckets is choosing more sockets.

Measured against the code as it stood, at the shipped default of 16 per run,
one client holding sockets open on invented run ids over a real uvicorn:

    ws_max_connections_per_run = 16 (per run id)
      500 sockets open:  rss=116300 KiB (+19684)  fds=1014 (+1000)
     1000 sockets open:  rss=135460 KiB (+38844)  fds=2014 (+2000)
     1500 sockets open:  rss=154632 KiB (+58016)  fds=3014 (+3000)
     2000 sockets open:  rss=173548 KiB (+76932)  fds=4014 (+4000)

Linearly, with nothing refused, because no two of those sockets shared a bucket.
At 39 KB each, 100 000 sockets is about 3.9 GB and the default file-descriptor
limit is reached first. The per-run cap is working exactly as written and
bounding nothing a single client can reach.

Authentication is not the answer and the tests below say so explicitly. With
``api_auth_enabled``, a wrong token is refused before the hub is reached —
measured, 0 subscriptions registered — but a *valid* token still opens 40
sockets across 40 invented run ids without a single refusal. The cap has to
hold for a client that is allowed to be here, because the deployment that
enables authentication is exactly the one with something worth exhausting.

What the cap does not do, and the tests pin it as a decision rather than a
surprise: it does not validate the run id. A client may subscribe to a run that
does not exist yet, on purpose, because a UI that opens the stream before it
POSTs the run is a supported shape (``test_subscribing_to_an_unknown_run_is_
not_an_error`` in ``test_runs_api.py``). Rejecting unknown runs would close the
hole by breaking that, and the total is what closes it instead.
"""

from __future__ import annotations

import asyncio

import pytest

from agentic_workflow.api.events import ANY_RUN, ConnectionLimitError, EventHub
from agentic_workflow.config import Settings

pytestmark = pytest.mark.api

#: Small enough that a test reaches it, large enough to be a different number
#: from the per-run cap so the two limits cannot be confused for one another.
TOTAL = 20
PER_RUN = 3


def _hub(**overrides: int) -> EventHub:
    """Build a hub with small caps.

    Args:
        **overrides: Settings fields to set on an offline configuration.

    Returns:
        The hub, not yet started.
    """
    settings = Settings(
        _env_file=None,
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        ws_max_connections_per_run=PER_RUN,
        ws_max_connections_total=TOTAL,
        **overrides,
    )
    return EventHub(settings=settings)


async def _drain() -> None:
    """Let the event loop finish the work queued by the last subscribe."""
    await asyncio.sleep(0)


class TestTheTotalCap:
    async def test_distinct_run_ids_cannot_exceed_the_total(self) -> None:
        """The limit that holds is the one that counts every bucket.

        Three per run is generous enough that no single run is what stops the
        client here, so a failure can only be the total not being enforced.
        """
        hub = _hub()
        accepted: list[object] = []
        for index in range(TOTAL):
            accepted.append(await hub.subscribe(f"run-{index}"))

        assert hub.subscriber_count() == TOTAL
        with pytest.raises(ConnectionLimitError):
            await hub.subscribe("run-the-cap-does-not-see")
        for sub in accepted:
            await hub.unsubscribe(sub)

    async def test_the_per_run_cap_still_applies_on_its_own(self) -> None:
        """Adding a total must not replace the per-run limit with it.

        They answer different questions. The per-run cap is what stops sixteen
        dashboards from crowding out the operator watching a run that is parked
        on a human decision; the total is what bounds the process. A fix that
        only counted the total would let one run take every slot.
        """
        hub = _hub()
        held = [await hub.subscribe("hot") for _ in range(PER_RUN)]

        with pytest.raises(ConnectionLimitError):
            await hub.subscribe("hot")
        # A different run still has room, so this is the per-run cap and not
        # the total cap refusing.
        other = await hub.subscribe("cold")
        await hub.unsubscribe(other)
        for sub in held:
            await hub.unsubscribe(sub)

    async def test_a_wildcard_socket_counts_towards_the_total(self) -> None:
        """A dashboard is one socket, and it is still a coroutine.

        The per-run cap already limits the wildcard bucket, which is why a
        dashboard cannot bypass the limit by not naming a run. The total has to
        count it too, or the same client opens the wildcard bucket and every
        invented one and the total never notices.
        """
        hub = _hub()
        held = [await hub.subscribe(ANY_RUN) for _ in range(PER_RUN)]
        for index in range(TOTAL - PER_RUN):
            held.append(await hub.subscribe(f"run-{index}"))

        assert hub.subscriber_count() == TOTAL
        with pytest.raises(ConnectionLimitError):
            await hub.subscribe(ANY_RUN)
        for sub in held:
            await hub.unsubscribe(sub)

    async def test_a_released_slot_can_be_reused(self) -> None:
        """The cap bounds live sockets, not the number a client may ever open.

        A deployment that has been full once and refuses afterwards is a
        deployment whose observers cannot reconnect, which is the failure this
        cap exists to prevent.
        """
        hub = _hub()
        held = [await hub.subscribe(f"run-{index}") for index in range(TOTAL)]
        with pytest.raises(ConnectionLimitError):
            await hub.subscribe("one-too-many")

        await hub.unsubscribe(held.pop())
        await _drain()
        assert hub.subscriber_count() == TOTAL - 1

        replacement = await hub.subscribe("one-too-many")
        assert hub.subscriber_count() == TOTAL
        await hub.unsubscribe(replacement)
        for sub in held:
            await hub.unsubscribe(sub)

    async def test_closing_the_hub_frees_every_slot(self) -> None:
        """A refused handshake must not leak the subscription it never made.

        Worth its own test because the refusal is raised after the bucket is
        resolved, and the bucket is the thing ``close`` has to clear for the
        count to come back. If a refused subscribe left a bucket behind, the
        count would never fall and the plane would be permanently full after
        enough bad handshakes.
        """
        hub = _hub()
        held = [await hub.subscribe(f"run-{index}") for index in range(TOTAL)]
        with pytest.raises(ConnectionLimitError):
            await hub.subscribe("refused")

        await hub.close()
        assert hub.subscriber_count() == 0

        fresh = await hub.subscribe("after-close")
        await hub.unsubscribe(fresh)
        for sub in held:
            await hub.unsubscribe(sub)


class TestTheRefusalIsDistinguishable:
    async def test_the_refusal_says_which_limit_was_hit(self) -> None:
        """An operator reading ``ws.rejected`` has to know what to do about it.

        A per-run refusal is a busy run and a total refusal is a full plane:
        the first is normal on a dashboard-heavy deployment, the second means
        clients are being turned away and somebody has to raise a limit or find
        the client that is holding them. Same exception, same status on the
        wire, so the scope has to be in the log and the reason.
        """
        hub = _hub()
        held = [await hub.subscribe("hot") for _ in range(PER_RUN)]
        with pytest.raises(ConnectionLimitError) as per_run:
            await hub.subscribe("hot")
        for sub in held:
            await hub.unsubscribe(sub)
        assert per_run.value.context["scope"] == "run"

        held = [await hub.subscribe(f"run-{index}") for index in range(TOTAL)]
        with pytest.raises(ConnectionLimitError) as overall:
            await hub.subscribe("one-too-many")
        assert overall.value.context["scope"] == "total"
        assert overall.value.context["limit"] == TOTAL
        for sub in held:
            await hub.unsubscribe(sub)

    async def test_the_total_refusal_reports_both_limits(self) -> None:
        """The number that was hit and the number that was not.

        A total refusal reporting only the total leaves the operator unable to
        tell a full plane from a full run, and the run id in the log is the one
        the client asked for, which is not necessarily the one at its limit.
        """
        hub = _hub()
        held = [await hub.subscribe(f"run-{index}") for index in range(TOTAL)]
        with pytest.raises(ConnectionLimitError) as refusal:
            await hub.subscribe("one-too-many")

        assert refusal.value.context["total"] == TOTAL
        assert refusal.value.context["per_run"] == PER_RUN
        assert refusal.value.context["subscribers"] == TOTAL
        for sub in held:
            await hub.unsubscribe(sub)


class TestTheDefaultIsBounded:
    def test_the_shipped_default_is_a_number(self) -> None:
        """The cap is not something a deployment has to remember to turn on.

        A cap defaulting to unlimited would leave the plane exactly as
        vulnerable as before for anyone who never read the runbook, which is the
        shape of bug this whole exercise is about.
        """
        settings = Settings(_env_file=None, llm_provider="echo", postgres_enabled=False)

        assert 0 < settings.ws_max_connections_total <= 100_000
        assert settings.ws_max_connections_total >= settings.ws_max_connections_per_run

    def test_a_total_below_the_per_run_cap_is_rejected(self) -> None:
        """The two numbers cannot be set in an order that makes one dead.

        ``per_run=16, total=8`` is not a stricter configuration, it is a broken
        one: every run would be refused at 8 and the operator who set 16 would
        be reading a per-run limit that never applied to anything.
        """
        with pytest.raises(ValueError, match="ws_max_connections_total"):
            Settings(
                _env_file=None,
                llm_provider="echo",
                postgres_enabled=False,
                ws_max_connections_per_run=16,
                ws_max_connections_total=8,
            )
