"""Shared pytest fixtures.

The suite is organised in layers, and the markers are the contract between them:

* ``unit`` — pure logic: routers, reducers, the escalation policy, the error
  taxonomy. No I/O, no event loop beyond a trivial one.
* ``integration`` — the engine, the graph and the checkpointer wired together.
* ``api`` — the HTTP/WebSocket control plane via ``TestClient``.
* ``postgres`` — needs a live database. Skipped automatically when unreachable.
* ``eval`` — the automated quality suite. Off by default.

Two invariants the whole suite relies on are enforced here:

1. **No test may reach the network.** The only LLM provider used is
   ``echo``, which is deterministic and offline. A test that needed credentials
   would be a test that fails on a contributor's laptop.
2. **No test may share state.** ``load_settings`` is memoised with
   ``lru_cache``, and the engine owns a run registry. Both are reset between
   tests, or ordering would decide pass/fail.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
import os
import socket
from typing import Any

from fastapi.testclient import TestClient
import pytest

from agentic_workflow.config import Settings, load_settings, reset_settings_cache
from agentic_workflow.domain.schemas import ReviewRequest, SourceFile
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.services.engine import WorkflowEngine


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers declared in ``pyproject.toml`` with their help text.

    Declaring them in two places would let them drift; re-declaring here gives
    ``pytest --markers`` useful output while ``--strict-markers`` keeps
    ``pyproject.toml`` authoritative.

    Args:
        config: The pytest configuration object.
    """
    for name, description in (
        ("unit", "Fast, dependency-free tests of pure logic."),
        ("integration", "Tests that wire multiple components together."),
        ("postgres", "Requires a live PostgreSQL instance."),
        ("api", "Tests exercising the HTTP/WebSocket control plane."),
        ("eval", "Automated LLM-quality evaluation tests."),
        ("slow", "Long running tests."),
    ):
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect ``postgres`` tests when no database is reachable.

    Marking them as skipped up front means a contributor without Docker gets a
    clean, honest "skipped" rather than a wall of connection errors — and CI,
    which *does* run PostgreSQL, still executes them.

    Args:
        config: The pytest configuration object.
        items: The collected test items, mutated in place.
    """
    if os.environ.get("AWF_TEST_POSTGRES") == "1" or _postgres_reachable():
        return
    skip = pytest.mark.skip(reason="no PostgreSQL reachable (set AWF_TEST_POSTGRES=1)")
    for item in items:
        if "postgres" in item.keywords:
            item.add_marker(skip)


def _postgres_reachable() -> bool:
    """Return whether the configured PostgreSQL accepts TCP connections.

    Returns:
        ``True`` when a socket connects to the DSN's host and port within a
        short timeout. A TCP probe rather than a full connection: it is fast,
        needs no credentials, and is enough to decide whether to attempt the
        marker-gated tests.
    """
    settings = load_settings()
    host, port = _host_port(settings.postgres_dsn)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.75):
            return True
    except OSError:
        return False


def _host_port(dsn: str) -> tuple[str, int]:
    """Extract host and port from a libpq DSN without a database driver.

    Args:
        dsn: A ``postgresql://user:pass@host:port/db`` string.

    Returns:
        A ``(host, port)`` tuple, or ``("", 0)`` when the DSN is unparseable.
    """
    from urllib.parse import urlparse

    parsed = urlparse(dsn)
    try:
        return parsed.hostname or "", int(parsed.port or 5432)
    except ValueError:
        return "", 0


@pytest.fixture(autouse=True)
def _isolate_settings() -> Iterator[None]:
    """Reset the memoised settings around every test.

    ``load_settings`` is ``lru_cache``d with ``maxsize=1`` precisely so the whole
    process sees one configuration. Tests need to swap it, so this fixture clears
    the cache on both sides of each test. Autouse because forgetting it produces
    order-dependent failures that are miserable to debug.

    Yields:
        ``None`` — the fixture exists for its cleanup, not its value.
    """
    reset_settings_cache()
    os.environ.setdefault("AWF_LLM_PROVIDER", "echo")
    yield
    reset_settings_cache()


@pytest.fixture
def settings() -> Settings:
    """Return fully offline, deterministic settings.

    Returns:
        A :class:`~agentic_workflow.config.Settings` built without reading
        ``.env``, so a developer's local file cannot change a test's behaviour.
    """
    reset_settings_cache()
    return load_settings(
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        hitl_enabled=True,
        log_level="WARNING",
        api_rate_limit_per_minute=0,
    )


@pytest.fixture
async def checkpointer() -> AsyncIterator[Any]:
    """Yield an in-memory checkpointer, closed at the end of the test.

    Returns:
        A LangGraph ``InMemorySaver``.
    """
    saver = build_memory_checkpointer()
    setup = getattr(saver, "setup", None)
    if setup is not None:
        await setup()
    try:
        yield saver
    finally:
        close = getattr(saver, "close", None)
        if close is not None:
            await close()


@pytest.fixture
async def engine(settings: Settings, checkpointer: Any) -> AsyncIterator[WorkflowEngine]:
    """Yield a started engine backed by the in-memory checkpointer.

    Args:
        settings: Offline settings.
        checkpointer: The in-memory saver.

    Yields:
        A started :class:`~agentic_workflow.services.engine.WorkflowEngine`,
        shut down and drained at the end of the test.
    """
    instance = WorkflowEngine(settings, checkpointer=checkpointer)
    await instance.startup()
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest.fixture
def request_factory() -> Any:
    """Return a factory building realistic :class:`ReviewRequest` payloads.

    Tests need a valid request but should not repeat the same ten lines. The
    factory guarantees a unique ``run_id`` per call, which matters because run
    ids are checkpoint thread keys.

    Returns:
        A callable accepting an optional ``run_id`` and returning a
        :class:`~agentic_workflow.domain.schemas.ReviewRequest`.
    """
    from tests.helpers import make_request

    return make_request


@pytest.fixture
def app_client(settings: Settings, checkpointer: Any) -> Iterator[TestClient]:
    """Yield a ``TestClient`` bound to a fully wired in-memory application.

    Args:
        settings: Offline settings.
        checkpointer: The in-memory saver injected into the engine.

    Yields:
        A :class:`~fastapi.testclient.TestClient`. Entering the context runs the
        lifespan, so the engine is started and the hub exists.
    """
    from agentic_workflow.api.app import create_app
    from agentic_workflow.services.engine import WorkflowEngine as Engine

    instance = Engine(settings, checkpointer=checkpointer)
    with TestClient(create_app(settings, engine=instance, configure_logs=False)) as client:
        yield client


@pytest.fixture
def anyio_backend() -> str:
    """Return the asyncio backend for ``anyio``-driven tests."""
    return "asyncio"


@pytest.fixture
def timeout_seconds() -> float:
    """Return the per-test wall-clock budget.

    Returns:
        Seconds. Overridable through ``AWF_TEST_TIMEOUT`` so a loaded CI runner
        can be given more room without editing the suite.
    """
    return float(os.environ.get("AWF_TEST_TIMEOUT", "30"))


__all__ = ["ReviewRequest", "SourceFile", "asyncio"]
