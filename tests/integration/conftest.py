"""Shared fixtures for integration tests that start a real server.

Moving the ``LiveServer`` out of the individual test modules avoids
duplicating the same start-up/shutdown code when another integration test
needs to talk to a running API.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

import pytest
import uvicorn

from agentic_workflow.api.app import create_app
from agentic_workflow.config import Settings


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
        self._task: asyncio.Task[None] | None = None

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
            if self._task is not None:
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
