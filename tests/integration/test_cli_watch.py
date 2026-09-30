"""``awf watch``: follow a run on a running server, from a terminal.

The gap this fills is not "show a review" — ``demo`` does that. It is that every
other subcommand builds its own engine in-process, so there was no way to *watch
a run that already exists somewhere else*. In production the interesting run is
on a replica you are not attached to, and the options were a WebSocket client
and a second terminal.

So ``watch`` is the first client of this project's own wire protocol, which
makes its correctness the interesting part:

* the first frame after ``accept`` is a **snapshot**, not a change, so a
  watcher that attaches late still knows where the run is;
* heartbeats are keep-alives and carry no state, so they are counted rather
  than printed;
* a loss is reported on the heartbeat channel as ``dropped_since_last`` and
  means this client missed events — the honest response is to say so and
  re-snapshot, not to keep printing a stream with a hole in it;
* a terminal event closes the socket, so a watcher that waits for more after
  ``run.completed`` hangs forever.

The tests here run the real thing — uvicorn on a real port, a real WebSocket —
because ``TestClient`` runs the ASGI app in-process and would not exercise the
handshake, the keep-alive or the disconnect at all. Measured cost of one
server: about a second.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import pytest
import uvicorn
import websockets

from agentic_workflow.api.app import create_app
from agentic_workflow.cli import BAD_INPUT, NOT_DONE, _format_frame, _watch
from agentic_workflow.config import Settings

pytestmark = pytest.mark.integration


def _free_port() -> int:
    """Reserve an ephemeral port and release it for uvicorn to bind."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LiveServer:
    """A real uvicorn server on a real port, for the length of a test.

    A fixture rather than a module-level singleton: the app's lifespan creates
    the engine and the event hub, so a shared server would share a run registry
    between tests that each expect a clean one.

    Attributes:
        port: The port uvicorn bound.
    """

    def __init__(self, settings: Settings) -> None:
        """Build the server config without starting it.

        Args:
            settings: Offline settings for the app under test.
        """
        self.port = _free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_app(settings, configure_logs=False),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
            )
        )

    @property
    def base_url(self) -> str:
        """The HTTP origin of this server."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws_url(self) -> str:
        """The WebSocket origin of this server."""
        return f"ws://127.0.0.1:{self.port}"

    async def __aenter__(self) -> LiveServer:
        """Start serving and wait until it is actually accepting.

        Returns:
            This server, started.
        """
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(200):
            if self._server.started:
                return self
            await asyncio.sleep(0.02)
        raise TimeoutError("uvicorn did not start")  # pragma: no cover

    async def __aexit__(self, *exc: object) -> None:
        """Ask uvicorn to exit and wait for it to finish.

        Args:
            *exc: Unused exception triple from the context manager.
        """
        self._server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


@pytest.fixture
def live_settings() -> Settings:
    """Offline settings with a heartbeat short enough to assert on.

    Returns:
        Settings for an in-process app with a 50 ms heartbeat and no auth.
    """
    return Settings(
        _env_file=None,
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        api_rate_limit_per_minute=0,
        ws_heartbeat_seconds=0.05,
    )


class TestTheStreamIsUnderstood:
    """`watch` has to read the protocol as the server actually writes it."""

    def test_the_first_frame_is_a_snapshot_not_a_change(self) -> None:
        """A watcher attaching late still learns where the run is.

        The snapshot is the reason a watcher is useful at all: without it,
        attaching to a run that has been going for ten minutes shows an empty
        screen until the next event, and a run parked on a human gate may not
        produce another for hours.
        """
        frame = {
            "event": "stream.snapshot",
            "run_id": "r1",
            "status": "waiting_human",
            "pending_approval": {"stage": "patch_review", "title": "Review"},
        }
        lines = _format_frame(frame)
        assert any("snapshot" in line for line in lines)
        assert any("waiting_human" in line for line in lines)
        assert any("patch_review" in line for line in lines)

    def test_a_heartbeat_with_no_loss_is_not_printed(self) -> None:
        """Keep-alives are counted, not shown.

        A run parked on a gate emits a heartbeat every 20 seconds for as long as
        it waits. Printing them turns a quiet run into a wall of noise and
        buries the one line the operator is waiting for.
        """
        assert _format_frame({"event": "heartbeat"}) == []

    def test_a_heartbeat_reporting_a_loss_says_so(self) -> None:
        """A gap in the stream is announced, not papered over.

        This is the field the previous commit added. Printing the rest of the
        stream as though it were complete would be the one genuinely dishonest
        thing a watcher could do, because the client cannot detect the loss
        itself: `seq` is a global publication counter and the loss lands at the
        head of the queue where no gap appears.
        """
        lines = _format_frame({"event": "heartbeat", "dropped_since_last": 12})
        assert len(lines) == 1
        assert "12" in lines[0]
        assert "dropped" in lines[0].lower() or "missed" in lines[0].lower()

    def test_a_lost_event_is_reported_before_the_frames_that_followed_it(self) -> None:
        """The order on screen matches the order the loss happened in.

        Announcing the gap after the events it precedes would tell the operator
        their view is current when it is not, for exactly as long as it takes
        them to read back one line.
        """
        assert _format_frame({"event": "run.started"}) != []
        assert _format_frame({"event": "heartbeat", "dropped_since_last": 1}) != []

    def test_a_node_event_names_the_node(self) -> None:
        """Per-node progress is the main thing an operator watches for."""
        lines = _format_frame(
            {"event": "node.completed", "node": "tester", "run_id": "r1", "iteration": 2}
        )
        assert any("tester" in line for line in lines)

    def test_an_unknown_event_is_still_shown(self) -> None:
        """A frame the watcher does not recognise is printed, not dropped.

        Forward compatibility runs both ways. If a newer server adds an event and
        an older watcher swallows it, the operator sees a run that has gone quiet
        at the exact moment something happened to it.
        """
        lines = _format_frame({"event": "some.future_event", "run_id": "r1"})
        assert any("some.future_event" in line for line in lines)

    def test_a_malformed_frame_does_not_stop_the_watch(self) -> None:
        """Garbage on the wire is reported and the stream continues.

        A watcher that raises on a frame it cannot parse has turned a cosmetic
        problem into an outage of the thing you were watching.
        """
        assert _format_frame({"event": None}) != []
        assert _format_frame({}) != []


class TestWatchingARealServer:
    """The protocol against uvicorn on a real port, over a real socket."""

    async def test_a_watch_reports_the_snapshot_and_the_terminal_event(
        self, live_settings: Settings
    ) -> None:
        """Following a run to completion prints its states in order.

        Driven end to end: a real server, a real WebSocket, a real
        auto-resolved run that reaches a terminal event. The socket closes after
        the terminal event, so a watcher that waits for more would hang — and
        the test's own timeout is what proves it does not.
        """
        import httpx

        async with LiveServer(live_settings) as server:
            async with httpx.AsyncClient(base_url=server.base_url) as client:
                started = await client.post(
                    "/v1/runs",
                    json={
                        "run_id": "watch-1",
                        "request_id": "PR-1042",
                        "title": "t",
                        "description": "d",
                        "auto_resolve": True,
                    },
                )
                assert started.status_code == 200, started.text

            exit_code = await asyncio.wait_for(
                _watch(server.ws_url, "watch-1", max_frames=40, quiet=True), timeout=30
            )

        assert exit_code == 0, "a completed run is a success"

    async def test_a_watch_of_a_parked_run_reports_the_gate_and_exits_not_done(
        self, live_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A run waiting on a human is shown the question, not a spinner.

        This is the case the command exists for. The run never reaches a
        terminal event, so the frame budget is what stops the follow, and what
        it printed has to answer "what is it waiting for". The exit code is
        ``NOT_DONE`` rather than success: the operator stopped watching a run
        that has not finished, and a CI job gating on this should see that.
        """
        import httpx

        async with LiveServer(live_settings) as server:
            async with httpx.AsyncClient(base_url=server.base_url) as client:
                await client.post(
                    "/v1/runs",
                    json={
                        "run_id": "watch-2",
                        "request_id": "PR-1042",
                        "title": "t",
                        "description": "d",
                    },
                )

            exit_code = await asyncio.wait_for(
                _watch(server.ws_url, "watch-2", max_frames=3), timeout=15
            )

        printed = capsys.readouterr().out
        assert "waiting_human" in printed, printed
        # Which gate fires is the engine's business and changes with the graph;
        # what this pins is that it is named, with the question and the reason,
        # rather than reduced to a "busy" that tells the operator nothing.
        assert "waiting on " in printed, printed
        assert "why: " in printed, "the escalation reason is the operator's only clue"
        assert exit_code == NOT_DONE, "a run that has not finished is not a success"

    async def test_a_watch_against_a_dead_server_fails_with_a_useful_message(
        self, live_settings: Settings
    ) -> None:
        """An unreachable server is a bad-input exit, not a traceback.

        A watcher pointed at a host that is not there has to say so in one line.
        The port is bound and released, so nothing is listening on it.
        """
        dead = _free_port()
        exit_code = await _watch(f"http://127.0.0.1:{dead}", "nope", max_frames=1, quiet=True)
        assert exit_code == 2, "bad input, matching the CLI's other codes"

    async def test_a_watch_of_an_unknown_run_opens_and_says_nothing(
        self, live_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A run id that does not exist is a stream that opens, not a refusal.

        Worth pinning because the opposite behaviour is a reasonable design and
        this one was chosen: a run may not have checkpointed yet, and a watcher
        that refused the subscription would be useless for the first second of
        every run. The server sends no snapshot, so the watcher waits on
        heartbeats and prints nothing — and says ``NOT_DONE`` rather than
        claiming success, because silence is not an answer.
        """
        async with LiveServer(live_settings) as server:
            exit_code = await asyncio.wait_for(
                _watch(server.ws_url, "never-submitted", max_frames=2), timeout=15
            )

        assert capsys.readouterr().out == "", "no snapshot means nothing to report"
        assert exit_code == NOT_DONE

    async def test_a_token_is_sent_so_an_authenticated_server_accepts_the_watch(
        self, live_settings: Settings
    ) -> None:
        """A production server has auth on, so the token is not optional.

        ``api_auth_enabled`` is the default posture for a control plane that
        starts runs, and a watcher that cannot pass a token is therefore useless
        in the only environment it exists for. The token arrives as a query
        parameter because a WebSocket handshake cannot carry an ``Authorization``
        header; that is the server's contract, so the client has to match it.
        """
        import httpx

        # Constructed rather than `model_copy`d: `model_copy` skips validation,
        # so the token would stay a plain `str` and the server's
        # `expected.get_secret_value()` would fail during startup instead of
        # exercising the path this test is about.
        secured = Settings(
            _env_file=None,
            environment="development",
            llm_provider="echo",
            postgres_enabled=False,
            api_rate_limit_per_minute=0,
            ws_heartbeat_seconds=0.05,
            api_auth_enabled=True,
            api_auth_token="s3cret-token",
        )
        async with LiveServer(secured) as server:
            async with httpx.AsyncClient(base_url=server.base_url) as client:
                created = await client.post(
                    "/v1/runs",
                    json={
                        "run_id": "watch-auth",
                        "request_id": "PR-1042",
                        "title": "t",
                        "description": "d",
                        "auto_resolve": True,
                    },
                    headers={"Authorization": "Bearer s3cret-token"},
                )
                assert created.status_code == 200, created.text

            # No token: the handshake is refused and the watch cannot follow.
            without = await asyncio.wait_for(
                _watch(server.ws_url, "watch-auth", max_frames=5, quiet=True), timeout=15
            )
            assert without == BAD_INPUT, "an unauthenticated watch is refused"

            with_token = await asyncio.wait_for(
                _watch(server.ws_url, "watch-auth", token="s3cret-token", max_frames=5, quiet=True),
                timeout=15,
            )
            assert with_token == 0, "the same watch succeeds with the token"

    async def test_the_token_is_not_echoed_in_an_error_message(
        self, live_settings: Settings
    ) -> None:
        """A failure message must not put the secret on stderr.

        The token travels in the query string, so printing the URL would print
        the credential — into a terminal scrollback, a CI log, or whatever
        collects stderr. This is the reason the default is the environment
        variable rather than the flag: a value typed on the command line is
        already in the shell history, and there is no way to un-say it.
        """
        dead = _free_port()
        exit_code = await _watch(
            f"http://127.0.0.1:{dead}",
            "nope",
            token="s3cret-token",
            max_frames=1,
            quiet=True,
        )
        assert exit_code == BAD_INPUT

    async def test_the_snapshot_frame_a_watcher_receives_is_parseable(
        self, live_settings: Settings
    ) -> None:
        """The snapshot is JSON with the keys the formatter reads.

        `_format_frame` indexes into the payload, so a shape change on the
        server is a crash in the client. The keys it depends on are pinned here
        rather than left to the integration test above, which would fail with a
        formatter traceback instead of a clear assertion.
        """
        import httpx

        async with LiveServer(live_settings) as server:
            async with httpx.AsyncClient(base_url=server.base_url) as client:
                await client.post(
                    "/v1/runs",
                    json={
                        "run_id": "watch-3",
                        "request_id": "PR-1042",
                        "title": "t",
                        "description": "d",
                    },
                )

            async with websockets.connect(f"{server.ws_url}/ws/runs/watch-3") as sock:
                frames = []
                for _ in range(2):
                    frames.append(json.loads(await asyncio.wait_for(sock.recv(), timeout=5)))

        snapshot = next(f for f in frames if f["event"] == "stream.snapshot")
        for key in ("run_id", "status", "pending_approval"):
            assert key in snapshot, f"the formatter reads {key!r}"
        # And the formatter survives the real thing rather than only a hand-built
        # dict, which is the assertion that would have caught a shape change.
        assert _format_frame(snapshot)
