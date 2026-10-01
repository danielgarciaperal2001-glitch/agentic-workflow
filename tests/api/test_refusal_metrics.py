"""Refusals are counted on ``/metrics``, because nothing else counts them.

Three ceilings exist and all three are invisible. The rate limiter answers
``429`` and keeps its own per-client window; the concurrency ceiling answers
``429`` from inside the engine; the token budget answers ``429`` from inside the
LLM client. An operator watching a control plane has no way to tell a refused
workload from a quiet one: ``awf_runs_registered`` counts runs that *started*,
``awf_events_*`` counts events that *flowed*, and neither moves when work is
turned away. The failure mode is silent in the worst direction — a client whose
budget is exhausted looks exactly like a client that stopped sending.

Measured against the code as it stood, ``GET /metrics`` exported event counters,
``awf_runs_registered`` and the four process-wide LLM usage counters. There was
no refusal counter of any kind: not the rate limiter, not the concurrency
ceiling, and not the token budget added in 29fc080.

The counter is defined by *what the client was told* rather than by a list of
causes. Every ``429`` the server answers is a refusal, labelled by the error code
that produced it, and a ``429`` code added tomorrow is counted without anyone
touching the metric. ``provider_rate_limited`` is included on purpose: it is the
one 429 whose cause is upstream, and it is exactly what an operator needs to see
when a provider quota is exhausted.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from agentic_workflow.api.app import create_app
from agentic_workflow.api.error_handlers import STATUS_MAP, error_response
from agentic_workflow.api.refusals import RefusalCounters
from agentic_workflow.config import Settings
from agentic_workflow.errors import (
    ConcurrencyLimitError,
    ProviderRateLimitedError,
    RateLimitedError,
    RunNotFoundError,
    TokenBudgetExceededError,
    WorkflowError,
)
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.services.engine import WorkflowEngine

pytestmark = pytest.mark.api

#: The exposition name. A counter, not a gauge: a refusal that has happened cannot
#: un-happen, and a dashboard that computes a delta over a gauge would read the
#: history as a rate.
FAMILY = "awf_refusals_total"

#: Every code the transport maps to 429. Asserted rather than hand-listed in the
#: implementation so a new ceiling cannot be added without appearing here.
REFUSAL_CODES = [code for code, mapped in STATUS_MAP.items() if mapped == 429]

_BUDGET = 5
_REQUESTS = 12


def _client(**overrides: Any) -> TestClient:
    """Build a bound application on an offline configuration.

    Args:
        **overrides: Settings fields to set.

    Returns:
        A test client, used as a context manager.
    """
    settings = Settings(
        _env_file=None,
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        **{"api_rate_limit_per_minute": 0, **overrides},
    )
    engine = WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
    return TestClient(create_app(settings, engine=engine, configure_logs=False))


#: One exposition line of the refusal family. Parsed with a pattern rather than by
#: splitting on ``=``, because the label's own ``reason=`` is the first equals
#: sign on the line.
_REFUSAL_LINE = re.compile(rf'^{FAMILY}\{{reason="([^"]+)"\}}\s+(\S+)$')


def _refusals(client: TestClient) -> dict[str, int]:
    """Parse the refusal family out of the metrics exposition.

    Args:
        client: The bound test client.

    Returns:
        Mapping from reason label to its count. Empty when no refusal happened.
    """
    found: dict[str, int] = {}
    for raw in client.get("/metrics").text.splitlines():
        match = _REFUSAL_LINE.match(raw)
        if match is not None:
            found[match.group(1)] = int(float(match.group(2)))
    return found


def _bare_request() -> Request:
    """Build a request whose app carries a refusal counter.

    ``error_response`` is reachable without an HTTP round trip, which is what lets
    a refusal raised three layers below the transport be counted for free.

    Returns:
        A request bound to a throwaway application with a fresh counter.
    """
    app = FastAPI()
    app.state.refusals = RefusalCounters()
    return Request({"type": "http", "app": app, "headers": [], "method": "GET", "path": "/v1/runs"})


def _counters_of(request: Request) -> RefusalCounters:
    """Read the counter back off a bare request's app.

    Args:
        request: The request handed to :func:`error_response`.

    Returns:
        The counter the response was recorded against.
    """
    counter = request.app.state.refusals
    assert isinstance(counter, RefusalCounters)
    return counter


def _refusals_of(error: WorkflowError) -> dict[str, int]:
    """Render *error* and return the refusals it recorded.

    Args:
        error: The domain error to render.

    Returns:
        Mapping from reason to count after the response was built.
    """
    request = _bare_request()
    error_response(request, error)
    return _counters_of(request).snapshot()


class TestARefusalIsCounted:
    """The counter has to move when work is turned away."""

    def test_a_rate_limited_request_is_counted(self) -> None:
        """The refusal path through a dependency reaches the counter.

        The rate limiter is a *sync* dependency, so FastAPI runs it in a
        threadpool and the error is caught by the async handler that renders it.
        That indirection is the whole risk: a counter incremented at the ``raise``
        would see it, one incremented where the handler sees it has to survive the
        hop, and the rate limiter is the only ceiling that takes that hop.
        """
        with _client(api_rate_limit_per_minute=_BUDGET) as client:
            rejected = [client.get("/v1/runs").status_code for _ in range(_REQUESTS)]
            refusals = _refusals(client)

        assert rejected.count(429) == _REQUESTS - _BUDGET, "the limiter did reject some"
        assert refusals == {"rate_limited": _REQUESTS - _BUDGET}, refusals

    def test_an_over_budget_submission_is_counted(self) -> None:
        """A refusal raised deep in the engine is counted too.

        The token budget is raised inside the LLM client and travels out through
        the node wrapper, three layers below the transport, which is the opposite
        end of the hop from the rate limiter. Both matter: counting only the
        reachable refusal would publish a number that is wrong for exactly the
        ceiling an operator is most likely to be watching for.
        """
        files = [{"path": f"m{i}.py", "content": "y = 2  # " + "z" * 20_000} for i in range(20)]
        with _client(llm_token_budget_per_run=50_000) as client:
            response = client.post(
                "/v1/runs",
                json={
                    "run_id": "refusal-budget-1",
                    "request_id": "PR-1042",
                    "title": "t",
                    "description": "d",
                    "auto_resolve": True,
                    "files": files,
                },
            )
            refusals = _refusals(client)

        assert response.status_code == 429, response.text
        assert refusals == {"token_budget_exceeded": 1}, refusals

    def test_every_429_is_counted_under_its_own_reason(self) -> None:
        """All four ceilings are covered by one rule, not by a list.

        Pinning the count as well as the label: a counter that reported every
        refusal under a single name would satisfy a weaker test and leave the
        operator unable to tell a busy server from a client overspending.
        """
        assert {code.value for code in REFUSAL_CODES} == {
            "rate_limited",
            "concurrency_limit",
            "token_budget_exceeded",
            "provider_rate_limited",
        }

        assert _refusals_of(RateLimitedError("slow down")) == {"rate_limited": 1}
        assert _refusals_of(ConcurrencyLimitError("too many", limit=8)) == {"concurrency_limit": 1}
        assert _refusals_of(TokenBudgetExceededError("too big", budget=1)) == {
            "token_budget_exceeded": 1
        }
        assert _refusals_of(ProviderRateLimitedError("upstream said no")) == {
            "provider_rate_limited": 1
        }

    def test_a_refusal_accumulates_rather_than_saturating(self) -> None:
        """Three refusals of one kind read three, not one.

        The metric exists to be graphed over time. A counter that recorded only
        *whether* anything was refused would draw a flat line at one through the
        exact incident the operator is trying to see the shape of.
        """
        request = _bare_request()
        for _ in range(3):
            error_response(request, RateLimitedError("slow down"))

        assert _counters_of(request).snapshot() == {"rate_limited": 3}

    def test_an_error_that_is_not_a_refusal_is_not_counted(self) -> None:
        """A 404 is a failure, not a ceiling, and must not inflate the count.

        The other direction of the same rule. Counting every error would make the
        metric a duplicate of the access log and would bury the refusals inside
        client mistakes that nobody needs to alert on.
        """
        assert _refusals_of(RunNotFoundError("no such run", run_id="absent")) == {}

    def test_the_two_refusal_kinds_stay_separate(self) -> None:
        """Two causes in one process are two labelled series.

        Summed together they would answer "were we busy?" — which is a real
        question, but not the one that was asked, and not the one the operator
        needs when deciding whether to raise a budget.
        """
        request = _bare_request()
        error_response(request, RateLimitedError("slow down"))
        error_response(request, ConcurrencyLimitError("too many", limit=8))

        assert _counters_of(request).snapshot() == {"rate_limited": 1, "concurrency_limit": 1}


class TestTheExportedShape:
    """A metric the pipeline cannot read is not a metric."""

    def test_a_fresh_process_exports_no_refusal_series(self) -> None:
        """No refusals means no series, not a row of zeros.

        Emitting zeros would require a static list of every possible reason, which
        is a fourth place to update when the API grows a ceiling, and a static list
        that falls behind is a list of wrong answers. Absence is also the more
        honest reading: there is nothing to see.
        """
        with _client() as client:
            assert _refusals(client) == {}

    def test_the_family_is_a_counter_and_not_a_gauge(self) -> None:
        """The exposition declares ``counter``, and stays out of the gauge family.

        Everything else on the endpoint is a gauge under ``awf_metric``, so a
        refusal series published there would be typed as an instantaneous value.
        A dashboard summing over that family would report the refusals of this
        scrape as a rate, which is the one reading that is always wrong.
        """
        with _client(api_rate_limit_per_minute=_BUDGET) as client:
            for _ in range(_BUDGET + 1):
                client.get("/v1/runs")
            body = client.get("/metrics").text

        assert f"# TYPE {FAMILY} counter" in body, body
        assert "# TYPE awf_metric gauge" in body
        # The gauge family is the only place a `name=` label appears, so this is
        # the exact shape the refusal series must not take.
        assert f'awf_metric{{name="{FAMILY}"}}' not in body, body
        assert f'{FAMILY}{{reason="rate_limited"}} 1' in body, body

    def test_the_rest_of_the_endpoint_still_reports(self) -> None:
        """Adding a family does not cost the endpoint anything.

        The counters that were already there are what a dashboard is built on; a
        regression that kept the new metric and broke ``awf_runs_registered``
        would trade a gap for a lie.
        """
        with _client() as client:
            body = client.get("/metrics").text

        assert "awf_runs_registered" in body
        assert "awf_llm_calls_total" in body


class TestTheCounterItself:
    """The object, tested apart from HTTP."""

    def test_a_snapshot_does_not_hand_out_the_counter_itself(self) -> None:
        """Callers get a copy, so rendering cannot edit the totals.

        ``/metrics`` is a read path, and a read path that can write is a bug that
        only shows up as drifting totals on a dashboard nobody is watching yet.
        """
        counters = RefusalCounters()
        counters.record("rate_limited")

        snapshot = counters.snapshot()
        snapshot["rate_limited"] = 999
        snapshot["concurrency_limit"] = 42

        assert counters.snapshot() == {"rate_limited": 1}

    def test_a_counter_starts_empty(self) -> None:
        """A fresh counter has nothing to report.

        Stated because the alternative — starting at zero for reasons nobody
        observed — is what makes a dashboard render rows that never happened.
        """
        assert RefusalCounters().snapshot() == {}

    def test_a_request_with_no_counter_still_renders_the_error(self) -> None:
        """Counting must never be the thing that breaks a refusal.

        ``error_response`` is called from unit tests and from any caller that
        renders a domain error without a full application behind it. An
        ``AttributeError`` raised while counting would replace a correct ``429``
        with a ``500``, which is strictly worse than not counting it.
        """
        bare = Request({"type": "http", "headers": [], "method": "GET", "path": "/v1/runs"})

        response = error_response(bare, RateLimitedError("slow down"))

        assert response.status_code == 429
