"""A 401 has to say what to send, or it is a dead end for anything scripted.

``require_auth`` raises its refusals with ``WWW-Authenticate: Bearer`` attached,
which is the header that tells a client what scheme and parameter the server
wants. The header never arrived: :func:`error_response` builds its own
``headers`` dict for the envelope, and the exception handler for
:class:`~starlette.exceptions.HTTPException` called it without the headers the
exception was carrying. Measured on a plane with authentication enabled, before
this file existed:

    GET /v1/runs, no Authorization header
      status   401
      headers  content-length, content-type, vary, x-request-id, x-response-time-ms

No ``WWW-Authenticate``, from a handler whose only job is to refuse.

The cost is not a leak but a dead end. A browser and a hand-written client can
guess their way to a token; a generated one reads the challenge, discovers the
plane wants a bearer token in a header, and asks for one. A 401 without the
header says "something about you is wrong" and nothing about what would fix it,
which is why the status is the one a client is most likely to retry blindly
instead of handling.

Asserted on both refusals, because the handler has two of them, and alongside the
envelope, because the body is the API's contract and a header must not change it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient
import pytest

from agentic_workflow.api.app import create_app
from agentic_workflow.config import Settings
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.services.engine import WorkflowEngine

pytestmark = pytest.mark.api

#: An opaque string of the shape an operator would use. It is not a credential
#: and belongs to nobody: the point is that a test cannot be pointed at a
#: deployed system by copying a value out of this file.
TOKEN = "control-plane-token-4f1c9a"


@contextmanager
def _secured() -> Iterator[TestClient]:
    """Start an application whose control plane demands a token.

    Yields:
        A started :class:`~fastapi.testclient.TestClient`.
    """
    settings: dict[str, Any] = {
        "environment": "development",
        "llm_provider": "echo",
        "postgres_enabled": False,
        "api_auth_enabled": True,
        "api_auth_token": TOKEN,
        "api_rate_limit_per_minute": 0,
    }
    resolved = Settings(_env_file=None, **settings)
    engine = WorkflowEngine(resolved, checkpointer=build_memory_checkpointer())
    with TestClient(create_app(resolved, engine=engine, configure_logs=False)) as client:
        yield client


class TestChallengeHeader:
    """The header is part of the contract, not a courtesy."""

    @pytest.mark.parametrize(
        ("headers", "case"),
        [
            ({}, "no Authorization header at all"),
            ({"Authorization": "Bearer not-the-token"}, "a wrong token"),
            ({"Authorization": "Basic something"}, "the wrong scheme"),
        ],
        ids=["missing", "wrong-token", "wrong-scheme"],
    )
    def test_every_refusal_carries_the_challenge(self, headers: dict[str, str], case: str) -> None:
        """All three refusals answer 401 *and* say what a correct one looks like.

        Asserted per case because each of them is a different branch in
        ``require_auth``, and a fix that only reached one of them would pass a
        weaker test: a client that authenticates correctly on the first retry and
        then loses its challenge on the second is a bug report nobody can
        reproduce.
        """
        with _secured() as client:
            refused = client.get("/v1/runs", headers=headers)

        assert refused.status_code == 401, case
        assert refused.headers.get("www-authenticate") == "Bearer", case

    def test_the_envelope_survives_the_header(self) -> None:
        """Forwarding a header must not disturb the body every client parses.

        The error envelope is the product's own contract — ``error.code`` and
        ``request_id`` are what a client branches on and what ties a user's
        screenshot to a server-side log line — so the fix is asserted to be a
        header and nothing more.
        """
        with _secured() as client:
            refused = client.get("/v1/runs")

        body = refused.json()
        assert body["error"]["code"] == "authentication_error"
        assert body["error"]["retryable"] is False
        assert body["request_id"] is not None
        assert refused.headers["x-request-id"] == body["request_id"]

    def test_a_refusal_body_says_nothing_about_the_secret(self) -> None:
        """The body is read by whoever is being turned away, so it carries no key.

        The constant-time comparison closes the timing channel; a body that
        echoed the expected token, or any fragment of it, would reopen the same
        secret with no measurement required at all.
        """
        with _secured() as client:
            bodies = [
                client.get("/v1/runs", headers={"Authorization": f"Bearer {candidate}"}).text
                for candidate in (TOKEN, TOKEN[:-1], TOKEN + "x")
            ]

        for body in bodies:
            assert TOKEN not in body
            assert TOKEN[:-4] not in body

    def test_a_guess_is_answered_the_same_however_close_it_lands(self) -> None:
        """The message must not grade the guess, only the header is the contract.

        Over HTTP the two refusals do say different things — "missing bearer
        token" against "invalid bearer token" — which is harmless, because the
        caller already knows which of the two it did. What would not be harmless
        is any signal about *how close* a guess came, so a one-character
        truncation and a token from a different deployment are asserted to be
        indistinguishable in the body.
        """
        with _secured() as client:
            near = client.get("/v1/runs", headers={"Authorization": f"Bearer {TOKEN[:-1]}"})
            far = client.get("/v1/runs", headers={"Authorization": "Bearer zzzz"})

        assert near.status_code == far.status_code == 401
        assert near.json()["error"]["message"] == far.json()["error"]["message"]
