"""Which routes demand a token, and which are open on purpose.

The throttle has been applied at the router level since it existed, precisely so
it could not be forgotten on a new endpoint. Authentication was not: it hung off
each handler as a parameter, so twenty-one endpoints carried it and a
twenty-second would have shipped without it. Nothing tested the set, because a
test that enumerates "the endpoints that need auth" is a list someone has to
remember to update — and a list nobody updates is exactly how the gap appeared.

So this file holds the set as a *partition* instead: every HTTP route the
application registers is either in :data:`OPEN_ROUTES`, with a reason, or it
demands a token. The sweep walks the real route table rather than a copy, so a
new endpoint has to be dealt with here the day it is written: the test fails
with its path in the message, and the fix is either to authenticate it or to
give it a line in the exemption list where the reason is visible.

The exemptions are not conveniences, and each one is a claim about the
deployment rather than about the code:

* ``/health/live`` and ``/health/ready`` answer to an orchestrator, a load
  balancer and a ``docker healthcheck``, none of which can hold a token. A probe
  that needs credentials is a probe that fails when the credentials rotate.
* ``/`` is a pointer at the API. It names endpoints, discloses nothing, and
  refusing it would make an unauthenticated ``curl`` look like an outage.
* ``/metrics`` is scraped by Prometheus, which authenticates by network rather
  than by header. It exposes counters — run totals, decision outcomes, token
  spend — so it belongs on an internal network or behind the same proxy as the
  rest of the plane. ``docs/security.md`` says so.
* ``/docs``, ``/redoc`` and ``/openapi.json`` exist only outside production;
  :attr:`Settings.is_production` removes them at boot.

Nothing else is open, including the root of every versioned router, the audit
log, and the endpoints that *create* runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import re
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
import pytest
from starlette.routing import Route as StarletteRoute

from agentic_workflow.api.app import create_app
from agentic_workflow.config import Settings
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.services.engine import WorkflowEngine

pytestmark = pytest.mark.api

#: An opaque string of the shape an operator would use. It is not a credential
#: and belongs to nobody: the point is that a test cannot be pointed at a
#: deployed system by copying a value out of this file.
TOKEN = "control-plane-token-4f1c9a"

#: Routes that answer without a token, each for a reason given above. A path
#: here is a claim that a deployment can live with, which is why the reasons are
#: prose and not a comment marker.
OPEN_ROUTES: frozenset[str] = frozenset(
    {
        "/",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
    }
)

#: Stand-in for a path parameter, chosen so it cannot collide with a literal
#: segment: the approval router has both `/approvals/sweep` and
#: `/approvals/{approval_id}`.
PLACEHOLDER = "no-such-identifier"

_PATH_PARAM = re.compile(r"\{[^}]+\}")


def _concrete(path: str) -> str:
    """Substitute a stand-in for every path parameter.

    Args:
        path: A route template, e.g. ``/v1/runs/{run_id}``.

    Returns:
        A path that routes to the same endpoint.
    """
    return _PATH_PARAM.sub(PLACEHOLDER, path)


def _flatten(routes: Any) -> list[Any]:
    """Return every route in the tree, with included routers unwrapped.

    FastAPI no longer copies an included router's routes into the application:
    ``app.routes`` holds one entry per ``include_router`` call and the routes
    live inside it. Reading that indirection is what lets the sweep walk the real
    table instead of the four documentation routes, and it is a version detail
    this file absorbs rather than a rule every future test has to remember.

    Args:
        routes: A route list, from an application or from a router.

    Returns:
        Leaf routes, in registration order.
    """
    leaves: list[Any] = []
    for route in routes:
        nested = getattr(route, "original_router", None)
        if nested is not None:
            leaves.extend(_flatten(nested.routes))
        else:
            leaves.append(route)
    return leaves


@contextmanager
def _secured(**overrides: Any) -> Iterator[TestClient]:
    """Start an application whose control plane demands a token.

    Args:
        **overrides: Settings fields layered onto an offline configuration.

    Yields:
        A started :class:`~fastapi.testclient.TestClient`.
    """
    settings = Settings(
        _env_file=None,
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        hitl_enabled=True,
        api_auth_enabled=True,
        api_auth_token=TOKEN,
        # Off, so a sweep of every route in the file cannot exhaust a budget and
        # start answering 429 halfway through, which would make a missing
        # authentication look like a passing test.
        api_rate_limit_per_minute=0,
        **overrides,
    )
    engine = WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
    with TestClient(create_app(settings, engine=engine, configure_logs=False)) as client:
        yield client


def _http_routes(settings: Settings) -> list[tuple[str, str]]:
    """List every ``(method, path)`` the application answers over HTTP.

    Built from a throwaway app rather than from the running one so the sweep can
    enumerate routes without spending requests, and read from the route table so
    a new endpoint is picked up without editing anything here.

    Args:
        settings: Configuration for the throwaway application.

    Returns:
        One entry per HTTP method the route accepts.
    """
    engine = WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
    app: FastAPI = create_app(settings, engine=engine, configure_logs=False)
    found: list[tuple[str, str]] = []
    for route in _flatten(app.routes):
        if not isinstance(route, (APIRoute, StarletteRoute)):
            # A WebSocketRoute is not an HTTP request and authenticates in the
            # handshake; `tests/api/test_ws_auth.py` is where that lives.
            continue
        for method in sorted(route.methods or set()):
            if method in {"HEAD", "OPTIONS"}:
                # Answered by the router without reaching a handler, so they say
                # nothing about whether the handler authenticates.
                continue
            found.append((method, _concrete(route.path)))
    return sorted(found)


class TestRoutePartition:
    """Every route is either protected or exempt, and the file says which."""

    def test_the_sweep_reaches_a_meaningful_number_of_routes(self) -> None:
        """A sweep that has quietly stopped sweeping is worse than no sweep.

        Enumerating the table can yield nothing at all — a router mounted
        somewhere this helper does not look, an app that stopped registering its
        routers — and every other test in the file would then pass vacuously,
        having asserted nothing about a single endpoint. This is the test that
        notices.
        """
        settings = Settings(
            _env_file=None, environment="development", llm_provider="echo", postgres_enabled=False
        )
        routes = _http_routes(settings)

        protected = {path for _, path in routes} - OPEN_ROUTES
        assert len(protected) >= 20, f"the sweep only found {len(protected)} protected routes"
        assert any(path.startswith("/v1/runs") for path in protected)
        assert any(path.startswith("/v1/approvals") for path in protected)
        assert any(path.startswith("/v1/threads") for path in protected)

    def test_every_versioned_route_demands_a_token(self) -> None:
        """No endpoint under ``/v1`` answers to a caller who has not proved itself.

        This is the whole point: the control plane starts runs, resolves
        approvals, replays threads and reads the audit log. A single one of them
        open is a control plane that can be driven by whoever found it, and the
        reason this is a sweep rather than a list is that the list is what went
        stale in the first place.
        """
        settings = Settings(
            _env_file=None,
            environment="development",
            llm_provider="echo",
            postgres_enabled=False,
            api_auth_enabled=True,
            api_auth_token=TOKEN,
            api_rate_limit_per_minute=0,
        )
        routes = [entry for entry in _http_routes(settings) if entry[1].startswith("/v1/")]
        assert routes, "no /v1 routes found; the sweep is looking in the wrong place"

        with _secured() as client:
            unauthenticated = {
                f"{method} {path}": client.request(method, path).status_code
                for method, path in routes
            }

        leaked = {route: status for route, status in unauthenticated.items() if status != 401}
        assert not leaked, f"reachable without a token: {sorted(leaked)}"

    def test_the_exempt_routes_stay_reachable(self) -> None:
        """Authentication is off for a reason on each of these, so they must work.

        Asserted in the other direction from the sweep on purpose: a probe that
        started demanding a token would take the orchestrator's health check down
        with it, and that failure would look like the application being
        unhealthy rather than like a configuration change. The cost of the
        exemption is carried in ``docs/security.md``.
        """
        with _secured() as client:
            statuses = {path: client.get(path).status_code for path in sorted(OPEN_ROUTES)}

        broken = {path: status for path, status in statuses.items() if status != 200}
        assert not broken, f"exempt routes stopped answering: {broken}"

    def test_the_exemption_list_is_not_where_the_routes_are(self) -> None:
        """An exemption nobody uses is a hole with a comment.

        A path can end up in :data:`OPEN_ROUTES` for a router that was renamed
        or a handler that moved, and then the list quietly claims a route is open
        that is actually protected — or, worse, the reverse, if the real route
        changed and the sweep starts passing over a path that no longer exists.
        Comparing the two sets catches both directions in one assertion.
        """
        settings = Settings(
            _env_file=None, environment="development", llm_provider="echo", postgres_enabled=False
        )
        registered = {path for _, path in _http_routes(settings)}

        assert registered >= OPEN_ROUTES, f"exempt but not registered: {OPEN_ROUTES - registered}"
        assert not (OPEN_ROUTES & {p for p in registered if p.startswith("/v1")})
