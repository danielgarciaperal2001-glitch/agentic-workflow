"""Which client the limiter forgets, and what a busy client is owed.

:data:`~agentic_workflow.api.deps._RATE_LIMIT_TRACKED_CLIENTS` bounds how many
identities the limiter remembers, so it has to forget somebody. The attribute it
documents — ``hits``, per-key timestamps "least-recent first" — says which one:
the one that has gone longest without asking for anything. The code forgot a
different one. The dict was only ever written on a *miss*, so a key already
present was never moved, the order it kept was the order clients first appeared
in it, and ``popitem(last=False)`` evicted the oldest *arrival*.

The inversion that costs something is the busy client. Measured against the code
as it stood — a budget of 5 over a 60 s window, and a client called ``steady``
that has spent its whole budget:

    steady spends 5 requests at t=1         5 allowed
    5 000 strangers arrive at t=2
    steady asks again at t=3               refused  (used, so it is not the oldest)
    5 000 more strangers arrive at t=4      (the cap is now reached, one eviction)
    steady asks again at t=5               refused, tracked, budget intact

and with the arrival order it stood in, the same script ends with steady evicted
at t=4 and its next request **allowed** — a full budget refunded to the client
that has been using the service most, by traffic that had nothing to do with it.
Any deployment that sees more than 10 000 identities hits this: a load balancer
in front of many client addresses, the loopback address of a sidecar per pod.
The limiter was at its most permissive exactly where the deployment is busiest,
and it did so to its heaviest users rather than its quietest.

What this cannot do, stated plainly because the first draft of these tests
asserted it and was wrong: a bounded tracker forgets whoever has gone quiet. If
a client stops asking and 10 000 others arrive, it is genuinely the
least-recently-used entry, dropping it is correct, and it comes back with a
full budget. No eviction order fixes that, and pretending otherwise would trade
a measured defect for an unmeasured one. What recency buys is the narrower and
still-load-bearing claim the tests below make: a client that *keeps talking*
keeps its budget, and a refused request counts as talking.

The tests assert the order itself rather than the symptom, so a later change to
the container cannot reintroduce arrival order under a different type.
"""

from __future__ import annotations

import pytest

from agentic_workflow.api.deps import _RATE_LIMIT_TRACKED_CLIENTS, RateLimiter
from agentic_workflow.errors import RateLimitedError

pytestmark = pytest.mark.api

#: A budget small enough to exhaust in a handful of calls, and a window long
#: enough that nothing in these tests expires on its own.
BUDGET = 5
WINDOW = 60.0


def _spend(limiter: RateLimiter, key: str, count: int, *, at: float) -> None:
    """Spend part of a client's budget at a single instant.

    Args:
        limiter: The limiter under test.
        key: The client identity to charge.
        count: How many requests to record.
        at: The instant they are recorded at, inside the window.
    """
    for _ in range(count):
        limiter.check(key, now=at)


def _flood(limiter: RateLimiter, first: int, last: int, *, at: float) -> None:
    """Fill the tracked set with clients that each make a single request.

    Args:
        limiter: The limiter under test.
        first: First index of the one-shot identities to introduce.
        last: Index to stop before.
        at: The instant they are recorded at.
    """
    for index in range(first, last):
        limiter.check(f"flood-{index}", now=at)


class TestEvictionOrder:
    def test_the_order_follows_use_and_not_arrival(self) -> None:
        """A client that arrives first and speaks again last is at the back.

        This is the property ``hits`` documents, asserted on the container
        rather than on a symptom: a fix that special-cased one client would
        leave the order wrong, and the next caller of this dict to rely on it
        would be the thing that breaks.
        """
        limiter = RateLimiter(limit=BUDGET, window=WINDOW)
        _spend(limiter, "first", 1, at=1.0)
        _spend(limiter, "second", 1, at=2.0)
        _spend(limiter, "third", 1, at=3.0)
        # "first" speaks again, so it is now the most recent of the three.
        _spend(limiter, "first", 1, at=4.0)

        # Three clients plus this many strangers is two more insertions than the
        # cap allows, so exactly two keys are dropped and no more.
        _flood(limiter, 0, _RATE_LIMIT_TRACKED_CLIENTS - 1, at=5.0)

        # The two that go must be the two that have not spoken since t=3. Under
        # arrival order the pair would be "first" and "second" — the client that
        # just used the service, and the one after it.
        assert "first" in limiter.hits, "the most recently used client was evicted"
        assert "second" not in limiter.hits
        assert "third" not in limiter.hits


class TestWhatABusyClientIsOwed:
    def test_a_client_that_keeps_talking_keeps_its_budget(self) -> None:
        """The cap may cost a client its history, but not a client that is using it.

        The full script from the module docstring: spend the budget, be buried
        under strangers, ask again — which is refused, and still counts as asking
        — be buried again until the cap is reached, and ask once more. Under
        arrival order the first burial was enough to end it.
        """
        limiter = RateLimiter(limit=BUDGET, window=WINDOW)
        _spend(limiter, "steady", BUDGET, at=1.0)
        _flood(limiter, 0, 5_000, at=2.0)

        # Refused, and the refusal is a use: the client is still on the line.
        with pytest.raises(RateLimitedError):
            limiter.check("steady", now=3.0)

        _flood(limiter, 5_000, _RATE_LIMIT_TRACKED_CLIENTS, at=4.0)

        assert "steady" in limiter.hits, "an active client was evicted by churn"
        with pytest.raises(RateLimitedError):
            limiter.check("steady", now=5.0)

    def test_the_retry_hint_points_at_the_window_the_client_actually_spent(
        self,
    ) -> None:
        """Surviving the cap must not move the window the client is waiting out.

        The hint is computed from the oldest hit still in the bucket, so an entry
        that survived the cap and was then topped up has to keep reporting the
        start of the window it is already inside. A client told to wait sixty
        seconds when it has fifty-eight left is a client that gives up and comes
        back immediately, which is the traffic pattern that makes the limiter
        look like the outage.
        """
        limiter = RateLimiter(limit=BUDGET, window=WINDOW)
        _spend(limiter, "steady", BUDGET - 1, at=1.0)
        _flood(limiter, 0, 5_000, at=2.0)
        limiter.check("steady", now=2.5)  # still active, spends the last one
        _flood(limiter, 5_000, _RATE_LIMIT_TRACKED_CLIENTS, at=3.0)

        with pytest.raises(RateLimitedError) as refusal:
            limiter.check("steady", now=4.0)

        assert refusal.value.context["retry_after_seconds"] == pytest.approx(57.0, abs=0.1)


class TestWhatForgettingStillCosts:
    def test_a_quiet_client_is_forgiven_and_that_is_deliberate(self) -> None:
        """Bounding the tracker means a client that stops asking starts over.

        Pinned so the refund is a recorded decision rather than a surprise. The
        alternative — refusing to forget anyone — turns a memory cap into an
        unbounded dict, and an unbounded dict in front of a caller-supplied
        identity is the worse of the two. The budget that comes back is bounded
        by the window, and the window is short.
        """
        limiter = RateLimiter(limit=BUDGET, window=WINDOW)
        _spend(limiter, "quiet", BUDGET, at=1.0)
        _flood(limiter, 0, _RATE_LIMIT_TRACKED_CLIENTS, at=2.0)

        assert "quiet" not in limiter.hits
        limiter.check("quiet", now=3.0)  # allowed: the entry was dropped, not kept

    def test_the_window_still_expires_a_client_that_survived(self) -> None:
        """Recency is not immortality: an entry older than the window is spent.

        Keeping a client tracked must not extend its budget. Once its last
        request is older than the window the limiter has to treat it as a new
        arrival, or the cap would turn a long-lived deployment into one where
        nothing is ever forgotten.
        """
        limiter = RateLimiter(limit=BUDGET, window=WINDOW)
        _spend(limiter, "steady", 1, at=1.0)
        _flood(limiter, 0, 5_000, at=2.0)
        _spend(limiter, "steady", 1, at=2.0)

        # Within the window: the bucket still holds the first two hits, so the
        # third of five is over budget and the rest stay refused.
        with pytest.raises(RateLimitedError):
            _spend(limiter, "steady", BUDGET, at=WINDOW / 2)

        # Past the window: from a clean budget, whether or not it was tracked.
        _spend(limiter, "steady", BUDGET, at=WINDOW * 2)
