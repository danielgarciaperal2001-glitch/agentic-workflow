"""Counting the work the server declined to do.

Three ceilings answer ``429``: the request budget in :mod:`~agentic_workflow.api.deps`,
the concurrency budget in :mod:`~agentic_workflow.services.engine` and the token
budget in :mod:`~agentic_workflow.llm.base`. None of them was visible to an
operator, because the counters on ``/metrics`` described what *happened* rather
than what was refused — ``awf_runs_registered`` counts runs that started and
``awf_events_*`` counts events that flowed, so a plane turning away every request
looks exactly like a quiet one.

A refusal is defined here by what the client was told, not by a list of causes:
every ``429`` is counted, labelled with the error code that produced it. A new
ceiling is therefore counted the day it is added, with nobody editing this module,
and the four labels that exist today are a measurement rather than a contract.

Example
-------
>>> counters = RefusalCounters()
>>> counters.record("rate_limited")
>>> counters.record("rate_limited")
>>> counters.snapshot()
{'rate_limited': 2}
"""

from __future__ import annotations

from fastapi import Request

__all__ = ["RefusalCounters", "counters_in", "record_refusal"]


class RefusalCounters:
    """Per-application totals of refused requests, keyed by reason.

    No lock, matching :class:`~agentic_workflow.api.events.EventHub`: the
    increment happens in an ``async`` exception handler on the event loop, with no
    ``await`` between the read and the write, so a plain ``dict`` is exactly as
    safe as the plain integers the hub keeps. The synchronous dependencies that
    raise refusals run in FastAPI's threadpool, but the *handler* that catches
    their error does not, so the counter is only ever touched from one thread.

    Only reasons that actually occurred are recorded. Exporting a zero for every
    possible code would need a static list to keep in step with
    :class:`~agentic_workflow.errors.ErrorCode`, and a list that falls behind is a
    list of wrong answers; absence is also the clearer reading of a dashboard,
    since there was nothing to refuse.

    Attributes:
        _by_reason: Observed reason labels mapped to their totals.
    """

    def __init__(self) -> None:
        """Start empty: nothing has been refused yet."""
        self._by_reason: dict[str, int] = {}

    def record(self, reason: str) -> None:
        """Count one refusal of *reason*.

        Args:
            reason: The error code the refusal was reported under, such as
                ``"rate_limited"``.
        """
        self._by_reason[reason] = self._by_reason.get(reason, 0) + 1

    def snapshot(self) -> dict[str, int]:
        """Return the totals observed so far.

        Returns:
            A copy, so the rendering path cannot edit the counts it is reading.
            Empty when no refusal has been recorded.
        """
        return dict(self._by_reason)


def counters_in(request: Request) -> RefusalCounters | None:
    """Find the counter attached to the application serving *request*.

    Every lookup here is optional and returns ``None`` rather than raising. The
    error renderer is reachable without a full application behind it — from unit
    tests, and from any caller turning a domain failure into a response on its
    own — and an ``AttributeError`` raised *while counting* would replace a
    correct ``429`` with a ``500``. Losing a count is recoverable; losing the
    refusal is not.

    Args:
        request: The inbound request.

    Returns:
        The application's counter, or ``None`` when there is no application or it
        never installed one.
    """
    app = request.scope.get("app")
    found = getattr(getattr(app, "state", None), "refusals", None)
    return found if isinstance(found, RefusalCounters) else None


def record_refusal(request: Request, reason: str) -> bool:
    """Count one refusal against the serving application, if it has a counter.

    Args:
        request: The inbound request.
        reason: The error code the refusal was reported under.

    Returns:
        Whether the refusal was counted. ``False`` means the application had no
        counter attached, which is a silent under-count rather than an error.
    """
    counters = counters_in(request)
    if counters is None:
        return False
    counters.record(reason)
    return True
