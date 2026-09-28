"""Whose budget is it, and can a client choose a new one?

The rate limiter is the only thing standing between an unauthenticated control
plane and a caller who wants all of it. Its key decides whether it works, and
the key was derived from two things the caller controls: an ``Authorization``
header, consulted even when authentication is disabled, and
``X-Forwarded-For``, trusted unconditionally.

Measured against the code as it stood, with a budget of 5 per minute on
``GET /v1/runs``:

    the same client, 12 requests        5 allowed, 7 rejected
    rotating X-Forwarded-For, 12        0 rejected
    rotating Authorization, 12          0 rejected

The second and third rows are the defect. A limiter a client can reset by
editing a header is not a limiter; it is a counter that the caller may decline
to participate in.

The fix keeps the two properties that made keying on identity attractive —
authenticated callers get their own budget regardless of the address they share
with a load balancer, and deployments behind a proxy can distinguish clients at
all — and stops deriving either from unauthenticated input.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
import pytest

from agentic_workflow.api.app import create_app
from agentic_workflow.config import Settings
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.services.engine import WorkflowEngine

pytestmark = pytest.mark.api

#: Small enough that a handful of requests exhausts it, large enough that the
#: test is not a race against the limiter's own window.
BUDGET = 5
REQUESTS = 12


def _client(**overrides: Any) -> TestClient:
    """Build a bound application with a small request budget.

    Args:
        **overrides: Settings fields to set on an offline configuration.

    Returns:
        A started test client.
    """
    settings = Settings(
        _env_file=None,
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        api_rate_limit_per_minute=BUDGET,
        **overrides,
    )
    engine = WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
    return TestClient(create_app(settings, engine=engine, configure_logs=False))


def _throttled(client: TestClient, headers: list[dict[str, str]] | None = None) -> int:
    """Issue requests and count the ones the limiter rejected.

    Args:
        client: The bound test client.
        headers: One header set per request, cycling. ``None`` sends none.

    Returns:
        How many requests were answered ``429``.
    """
    codes: list[int] = []
    for index in range(REQUESTS):
        extra = headers[index % len(headers)] if headers else None
        codes.append(client.get("/v1/runs", headers=extra or {}).status_code)
    return codes.count(429)


class TestKeyedOnUntrustedInput:
    """The caller must not be able to pick its own budget."""

    def test_rotating_the_forwarded_header_does_not_buy_a_new_budget(self) -> None:
        """A client cannot outrun the limit by varying ``X-Forwarded-For``.

        The limiter has no way to authenticate a forwarded address. Trusting the
        header means the header decides how much traffic a client may send, so
        the honest setting of the limit stops being a property of the
        deployment and becomes a suggestion to whoever is calling.
        """
        with _client() as client:
            rejected = _throttled(
                client, [{"X-Forwarded-For": f"203.0.113.{i}"} for i in range(REQUESTS)]
            )

        assert rejected == REQUESTS - BUDGET

    def test_rotating_the_authorization_header_does_not_buy_a_new_budget(self) -> None:
        """Presenting a token nobody issued is not an identity.

        The header was read to build the key without ever checking it, so it
        worked as a free reset. The same trick amplifies a password guess: each
        candidate token landed in its own bucket, and the limiter could not see
        the pattern even when authentication was switched on.
        """
        with _client() as client:
            rejected = _throttled(
                client, [{"Authorization": f"Bearer guess-{i}"} for i in range(REQUESTS)]
            )

        assert rejected == REQUESTS - BUDGET

    def test_the_honest_client_is_still_limited(self) -> None:
        """The guarantee the limiter advertises is the one it now keeps.

        Pinning this alongside the bypass tests: a fix that simply stopped
        throttling would satisfy both of them.
        """
        with _client() as client:
            rejected = _throttled(client)

        assert rejected == REQUESTS - BUDGET


class TestFailedAuthentication:
    """Guessing a token has to be the thing the limiter bounds."""

    def test_repeated_wrong_tokens_share_the_address_budget(self) -> None:
        """Each rejected guess is counted against the caller that made it.

        With authentication on, a client guessing one wrong token per request
        used to consume a fresh budget on every attempt, so the throttle was
        inert exactly when it mattered most. Counting them together is what
        turns a slow guess into an expensive one.
        """
        with _client(api_auth_enabled=True, api_auth_token="s3cret-value") as client:
            rejected = _throttled(
                client, [{"Authorization": f"Bearer wrong-{i}"} for i in range(REQUESTS)]
            )

        assert rejected == REQUESTS - BUDGET

    def test_a_correct_token_is_still_accepted(self) -> None:
        """The limiter is not a second authentication gate.

        Counting guesses must not turn a valid credential into a rejection; the
        authentication dependency owns that decision and the limiter owns the
        count.
        """
        with _client(api_auth_enabled=True, api_auth_token="s3cret-value") as client:
            response = client.get("/v1/runs", headers={"Authorization": "Bearer s3cret-value"})

        assert response.status_code == 200


class TestProxyDeployments:
    """A deployment behind a proxy must still be able to tell clients apart."""

    def test_forwarded_clients_are_separate_once_trust_is_declared(self) -> None:
        """Opting in restores the distinction the proxy exists to provide.

        Without this the fix would be to collapse every caller onto the
        load balancer's address, which is a working limiter and a useless one:
        one busy reviewer exhausts the budget for everyone. The opt-in has to be
        explicit precisely because the header is caller-writable, and a
        deployment that trusts its proxy is the only one in a position to say so.
        """
        with _client(api_trust_forwarded_for=True) as client:
            rejected = _throttled(
                client, [{"X-Forwarded-For": f"203.0.113.{i}"} for i in range(REQUESTS)]
            )

        assert rejected == 0

    def test_trust_uses_the_address_the_proxy_observed(self) -> None:
        """The rightmost entry is the one the trusted hop wrote.

        A proxy handling a request from a client that sent its own
        ``X-Forwarded-For`` appends to it, so the leftmost entry is whatever the
        caller wrote. Reading from the right is the only ordering that survives
        a client which knows about the header.
        """
        from agentic_workflow.api.deps import client_key

        with _client(api_trust_forwarded_for=True) as client:
            app = client.app
            scope = {
                "type": "http",
                "headers": [
                    (b"x-forwarded-for", b"198.51.100.66, 203.0.113.9"),
                ],
                "client": ("10.0.0.1", 5000),
            }
            from starlette.requests import Request

            key = client_key(
                Request(scope),
                settings=app.state.settings,
            )

        assert "203.0.113.9" in key
        assert "198.51.100.66" not in key
