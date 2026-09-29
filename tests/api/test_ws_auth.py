"""A refused socket must be refused, not answered with a server error.

The HTTP surface answers a bad token with 401 and a ``WWW-Authenticate``
challenge. The WebSocket surface claimed the same thing in a comment and did
neither: :func:`~agentic_workflow.api.routers.events._serve` caught the refusal
from :func:`~agentic_workflow.api.deps.websocket_auth` and returned without
sending a single ASGI message. The handshake was then nobody's job but the
server's, and what uvicorn does with an application that stops mid-handshake was
measured against the real stack (uvicorn 0.54, this app, authentication enabled,
one raw handshake per case):

    wrong token      HTTP/1.1 500 Internal Server Error
    no token         HTTP/1.1 500 Internal Server Error
    right token      HTTP/1.1 101 Switching Protocols

and, on the server side, once for each of the two refusals:

    ERROR  ASGI callable returned without completing handshake.

Three things are wrong with that. The caller is told the control plane is broken,
not that it is unwelcome, so a browser shows a generic failure and a scripted
client retries a connection that will never succeed. Every refused socket writes
an ERROR line, so a caller guessing tokens — the exact case the ``ws.auth_failed``
line exists to make visible — becomes indistinguishable from a real outage in
the log an operator is watching. And the close code the code documented, 1008, is
never delivered, because a status cannot be sent to a socket that was never
accepted.

The sibling refusal one function below, the connection limit, already completes
the handshake and answers 403. Same mechanism, same status, no ERROR: a socket
that is over its connection limit and a socket with a bad token are the same kind
of event, and treating them differently is how the auth path stayed broken for so
long without anyone noticing.

The ``websocket.http.response`` extension, which would allow a 401 with a
challenge, was measured too and the reasoning is recorded in ``_refuse``. The
short version: uvicorn marks the handshake unfinished for it, so every refusal
would still write an ERROR line, and a browser cannot read the status anyway.

These tests use the test client, which sees the close the application sends. The
500 and the ERROR line it used to produce are a property of the ASGI *server*, so
they are recorded above and in the commit rather than asserted here; the client
side of the same bug was a hang, which is what the timeout on each class is for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient
import pytest
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect
from structlog.testing import capture_logs

from agentic_workflow.api.app import create_app
from agentic_workflow.config import Settings
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.services.engine import WorkflowEngine

pytestmark = pytest.mark.api

#: An opaque string of the shape an operator would use. It is not a credential
#: and belongs to nobody: the point is that a test cannot be pointed at a
#: deployed system by copying a value out of this file.
TOKEN = "control-plane-token-4f1c9a"

#: Close code the RFC 6455 section 7.4.1 registry reserves for a policy refusal,
#: which is what a wrong token is. The ASGI server renders it as an HTTP 403 on
#: the upgrade request; the code is what a client library sees when the server
#: passes it through, and it is the answer this module documents.
POLICY_VIOLATION = 1008

#: Generous for a refusal, which happens in the handshake, and short enough that
#: the regression this file exists for fails as a failure rather than a hang. The
#: suite's global budget is 120 seconds, which turns a one-second test into a
#: two-minute one.
REFUSAL_TIMEOUT = 15


@contextmanager
def _served(**overrides: Any) -> Iterator[TestClient]:
    """Start an application with a short heartbeat and yield a bound client.

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
        api_rate_limit_per_minute=0,
        ws_heartbeat_seconds=0.05,
        **overrides,
    )
    engine = WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
    with TestClient(create_app(settings, engine=engine, configure_logs=False)) as client:
        yield client


def _refused(
    client: TestClient,
    path: str,
    inside: Callable[[WebSocketTestSession], None] | None = None,
) -> WebSocketDisconnect:
    """Open a socket that must be refused, and hand back how it was refused.

    Args:
        client: A running application.
        path: The socket path, including any query string.
        inside: Optional work to attempt on the session, for the assertion that
            a refused socket yields nothing to read.

    Returns:
        The disconnect the refusal was delivered as.

    Raises:
        AssertionError: If the socket opens at all, which is the regression.
    """
    with pytest.raises(WebSocketDisconnect) as refusal, client.websocket_connect(path) as socket:
        if inside is not None:
            inside(socket)
    return refusal.value


@pytest.mark.timeout(REFUSAL_TIMEOUT)
class TestSocketRefusal:
    """The handshake is the only moment a socket can still be refused."""

    @pytest.mark.parametrize(
        ("path", "why"),
        [
            ("/ws/runs/ws-refuse-1?token=guess", "a guessed token"),
            ("/ws/runs/ws-refuse-1", "no token at all"),
            ("/ws/events?token=guess", "the wildcard stream, same secret"),
        ],
        ids=["wrong-token", "no-token", "wildcard-stream"],
    )
    def test_a_refused_socket_is_told_why(self, path: str, why: str) -> None:
        """The refusal carries a policy code and a reason, and arrives at all.

        Asserted as a close rather than as silence, because silence is what the
        server had to invent before: an application that returns mid-handshake
        gets a 500, and an application that never answers leaves a client waiting.
        A client that knows it was refused can log a token problem; a client told
        the server is broken files a bug against the wrong component.
        """
        with _served(api_auth_enabled=True, api_auth_token=TOKEN) as client:
            refused = _refused(client, path)

        assert refused.code == POLICY_VIOLATION, why
        assert refused.reason == "unauthorised", why

    def test_a_refusal_never_reaches_the_stream(self) -> None:
        """Nothing observable may cross the socket to a caller who did not prove it.

        Asserted as a refusal to open rather than as an absence of events: a
        socket that opened and then closed would let a client believe it had
        connected, and a dashboard that reconnects on close would loop forever on
        a token it can never satisfy.
        """
        with _served(api_auth_enabled=True, api_auth_token=TOKEN) as client:
            _refused(
                client, "/ws/runs/ws-refuse-2?token=guess", lambda socket: socket.receive_json()
            )

    def test_a_refusal_is_logged_without_the_token(self) -> None:
        """The log line exists so an operator can see probing; it must not join it.

        The reason a refusal is logged at all is that a caller guessing tokens
        looks like any other storm of failures, and the path is the only useful
        field. Logging the credential that was presented would hand the whole
        secret to whoever can read logs, which is a strictly larger audience than
        the one that already has it.
        """
        with _served(api_auth_enabled=True, api_auth_token=TOKEN) as client, capture_logs() as logs:
            _refused(client, f"/ws/runs/ws-log-1?token={TOKEN}-guess")

        failures = [entry for entry in logs if entry["event"] == "ws.auth_failed"]
        assert len(failures) == 1
        assert TOKEN not in repr(failures[0])

    def test_a_correct_token_streams(self) -> None:
        """A browser cannot set headers, so the token travels in the query string.

        This is the capability the whole dashboard depends on, and it is what
        makes the refusals above refusals rather than an outage.
        """
        with (
            _served(api_auth_enabled=True, api_auth_token=TOKEN) as client,
            client.websocket_connect(f"/ws/runs/ws-ok-1?token={TOKEN}") as socket,
        ):
            assert socket.receive_json()["event"] == "stream.open"

    def test_the_default_still_admits_a_bare_socket(self) -> None:
        """Authentication off is a deliberate default, and it is asserted as one.

        ``api_auth_enabled`` defaults to false so ``docker compose up`` gives a
        working stack. That is a convenience with a cost, documented in
        ``docs/security.md``; this test exists so flipping the default is a
        deliberate act that fails here rather than an accident nobody notices.
        """
        with _served() as client, client.websocket_connect("/ws/runs/ws-open-1") as socket:
            assert socket.receive_json()["event"] == "stream.open"
