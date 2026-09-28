"""Tests for the liveness and readiness probes.

These are the two handlers an orchestrator calls, and the failure mode of
getting them wrong is an outage rather than a bug report: a readiness probe
that reports 200 while the database is unreachable routes live traffic to an
instance that cannot start a run, and a liveness probe that checks a
dependency gets healthy pods killed during a database blip.

The happy path is already covered by the application's own startup tests, so
what is asserted here is the part that only runs when something is broken —
which is exactly the part that had no coverage.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from fastapi.testclient import TestClient
import pytest

from agentic_workflow import __version__
from agentic_workflow.api.app import create_app
from agentic_workflow.config import Settings, load_settings


class _Engine:
    """Minimal engine surface the probes read.

    Only ``registry`` and ``checkpointer`` are touched by the handlers, so a
    real engine would drag in a graph, a store and a lifespan for no added
    signal — and its own failures would blur what is under test here.
    """

    def __init__(self, checkpointer: Any = None) -> None:
        self.registry: dict[str, Any] = {}
        self.checkpointer = checkpointer

    async def pending_approvals(self, *args: Any, **kwargs: Any) -> list[Any]:
        """The real inbox reads this; a stub without it fails the probe with an
        ``AttributeError`` that looks like a readiness failure rather than a
        test-double gap. The inbox passes a checkpointer argument, hence ``*args``.
        """
        return []

    def parked_summary(self) -> dict[str, Any]:
        """What the approval service's probe path calls.

        Present for the same reason as ``pending_approvals``, and it is worth
        spelling out why this is such a reliable way to break a test suite: the
        readiness handler wraps each check in its own ``try``, so a missing method
        becomes ``{"ok": false}`` and a 503 rather than an ``AttributeError``. A
        stub that falls behind therefore fails every "the probe is healthy" test at
        once, with a message that blames the probe.
        """
        return {"pending": 0, "expired": 0, "resolved_pending": 0, "by_stage": {}}

    def set_event_sink(self, sink: Any) -> None:
        """The lifespan always wires one; a stub that ignored it would be fine,
        but a stub that lacked it would fail before any assertion ran."""
        self.sink = sink

    async def startup(self) -> None:
        """The lifespan awaits this. Failure here aborts the boot by design, so
        it must not raise or the probe tests never reach their assertions."""

    async def shutdown(self) -> None:
        """Symmetrically awaited on lifespan exit."""


class _Approvals:
    """Approval service whose counters can be made to fail.

    Both entry points are here because both exist: ``stats`` is what
    ``GET /v1/approvals/stats`` serves, and ``probe_stats`` is what the readiness
    probe calls. The error is raised from both, because a failure of the underlying
    inbox fails both and a double that only fails one would test a distinction
    this stub is not about.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def stats(self) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        return {"pending": 0}

    async def probe_stats(self) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        return {"pending": 0}


@contextmanager
def _client(
    engine: Any = None,
    *,
    settings: Settings | None = None,
    hub: Any = None,
    approvals: Any = None,
    drop_engine: bool = False,
) -> Iterator[TestClient]:
    """Yield a client whose app state has been shaped for one scenario.

    Args:
        engine: Engine to inject. ``None`` lets the factory build a real one.
        settings: Configuration for the app. Defaults to in-memory so no test
            depends on whether the ambient environment enables PostgreSQL.
        hub: Event hub to install on app state.
        approvals: Approval inbox to install on app state.
        drop_engine: Remove the engine from app state *after* the lifespan ran.
            Passing ``engine=None`` cannot express "never started", because the
            factory builds one when it is not supplied.
    """
    config = settings or _memory_settings()
    app = create_app(config, engine=engine, configure_logs=False)
    with TestClient(app) as client:
        if engine is not None:
            client.app.state.engine = engine
        if hub is not None:
            client.app.state.hub = hub
        if approvals is not None:
            client.app.state.approvals = approvals
        if drop_engine:
            client.app.state.engine = None
        yield client


def _memory_settings() -> Settings:
    """Settings that explicitly use the in-memory checkpointer.

    Derived from the ambient configuration with the flag forced, rather than
    relying on the environment: the suite is run with ``AWF_POSTGRES_ENABLED``
    both set and unset (the PostgreSQL job exports it), and a test that reads
    its own precondition off the environment passes in one job and fails in
    the other.
    """
    return load_settings().model_copy(update={"postgres_enabled": False})


def _durable_settings() -> Settings:
    """Settings that claim a durable store, without needing one to exist."""
    return load_settings().model_copy(update={"postgres_enabled": True})


# --------------------------------------------------------------------------- #
class TestLiveness:
    def test_live_answers_without_touching_a_dependency(self) -> None:
        """The whole point of the split: a dead database must not fail this."""
        with _client(_Engine(checkpointer=None)) as client:
            body = client.get("/health/live").json()
        assert body["status"] == "ok"
        assert body["checks"] == {"event_loop": "ok"}

    def test_live_stays_ok_with_a_broken_checkpointer(self) -> None:
        """If liveness consulted the store, a database blip would restart every
        healthy pod and turn a degradation into an outage."""

        class _Broken:
            def setup(self) -> Any:
                raise RuntimeError("database is down")

        settings = _durable_settings()
        with _client(_Engine(checkpointer=_Broken()), settings=settings) as client:
            response = client.get("/health/live")
        assert response.status_code == 200, "liveness must not depend on a dependency"
        assert response.json()["status"] == "ok"

    def test_live_reports_the_environment_it_is_running_in(self) -> None:
        with _client(_Engine()) as client:
            body = client.get("/health/live").json()
        assert body["environment"] == load_settings().environment.value
        assert body["version"] == __version__
        # Serialised as an ISO-8601 string; a naive `utcnow()` comparison
        # against the parsed JSON is a type error, not a failure of the probe.
        assert datetime.fromisoformat(body["timestamp"]).tzinfo is not None


# --------------------------------------------------------------------------- #
class TestReadiness:
    def test_ready_is_200_when_nothing_is_durable(self) -> None:
        """Without a store configured there is no store to fail, so refusing
        traffic would be wrong."""
        with _client(_Engine(checkpointer=None), settings=_memory_settings()) as client:
            response = client.get("/health/ready")
        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "ok"
        assert body["checks"]["engine"]["durable"] is False
        assert "checkpointer" not in body["checks"]

    def test_ready_is_503_when_the_engine_was_never_started(self) -> None:
        """An instance without an engine cannot start a run, whatever else is
        healthy. Answering 200 would route traffic to a process that can only
        serve 500s."""
        with _client(drop_engine=True) as client:
            response = client.get("/health/ready")
        body = response.json()
        assert response.status_code == 503
        assert body["status"] == "down"
        assert body["checks"]["engine"] == {"ok": False, "detail": "engine not initialised"}
        assert body["detail"] == "one or more readiness checks failed"

    def test_ready_is_503_when_the_durable_store_cannot_be_reached(self) -> None:
        """The check that keeps traffic away from an instance whose database is
        unreachable. A 200 here is a silent outage in production."""

        class _Broken:
            def setup(self) -> Any:
                raise RuntimeError("could not connect to server")

        settings = _durable_settings()
        with _client(_Engine(checkpointer=_Broken()), settings=settings) as client:
            response = client.get("/health/ready")
        body = response.json()
        assert response.status_code == 503
        assert body["checks"]["checkpointer"]["ok"] is False
        assert "could not connect" in body["checks"]["checkpointer"]["detail"]

    def test_ready_is_503_when_the_store_check_hangs(self) -> None:
        """A probe that hangs is indistinguishable from a dead instance, so the
        check is bounded — and a bounded failure is a recoverable 503."""

        class _Hanging:
            async def setup(self) -> Any:
                await _sleep_forever()

        settings = _durable_settings()
        with _client(_Engine(checkpointer=_Hanging()), settings=settings) as client:
            response = client.get("/health/ready")
        body = response.json()
        assert response.status_code == 503
        assert body["checks"]["checkpointer"]["ok"] is False

    def test_ready_is_200_when_the_durable_store_answers(self) -> None:
        """The other half of the bounded check: a working store must still let
        traffic through, or the instance never leaves the load balancer."""

        class _Working:
            calls = 0
            reads = 0

            async def setup(self) -> Any:
                type(self).calls += 1

            async def aget_tuple(self, *args: Any, **kwargs: Any) -> Any:
                type(self).reads += 1
                return None

        saver = _Working()
        settings = _durable_settings()
        with _client(_Engine(checkpointer=saver), settings=settings) as client:
            first = client.get("/health/ready")
            second = client.get("/health/ready")
        assert first.status_code == 200
        assert first.json()["checks"]["checkpointer"] == {"ok": True}
        assert second.status_code == 200
        # Both halves, and the distinction is the point: re-running DDL on every
        # probe is needless work against the primary, but *not asking the store
        # anything* is a readiness check that cannot fail. This assertion used to
        # be the first without the second, and the second is what was missing.
        assert saver.calls == 1, "the DDL must not be re-run on every probe"
        assert saver.reads == 2, "every probe must read from the store"

    def test_ready_reports_the_run_count(self) -> None:
        engine = _Engine()
        engine.registry = {"a": object(), "b": object()}
        with _client(engine) as client:
            assert client.get("/health/ready").json()["checks"]["engine"]["runs"] == 2

    def test_ready_is_503_when_the_approval_inbox_fails(self) -> None:
        """The inbox is what a reviewer polls, so an instance that cannot answer
        it is not serving the feature the readiness gate exists to protect."""
        with _client(_Engine(), approvals=_Approvals(RuntimeError("inbox down"))) as client:
            response = client.get("/health/ready")
        body = response.json()
        assert response.status_code == 503
        assert body["checks"]["approvals"]["ok"] is False
        assert "inbox down" in body["checks"]["approvals"]["detail"]

    def test_ready_includes_the_inbox_when_it_answers(self) -> None:
        with _client(_Engine(), approvals=_Approvals()) as client:
            response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["checks"]["approvals"] == {"pending": 0}


async def _sleep_forever() -> None:
    """Block forever, to exercise the probe's timeout."""
    import asyncio

    await asyncio.sleep(3600)


@pytest.mark.parametrize("path", ["/health/live", "/health/ready"])
def test_both_probes_are_reachable_without_a_trailing_slash(path: str) -> None:
    """An orchestrator configured with the other spelling gets a 307 redirect
    it may not follow, and a probe that "fails" forever looks like a dead pod."""
    with _client(_Engine()) as client:
        response = client.get(path, follow_redirects=False)
    assert response.status_code in (200, 307)


def test_the_application_factory_boots_and_answers() -> None:
    """Guards the job that runs the built image: the entrypoint, the factory
    and the probe have to work together in one process."""
    app = create_app(load_settings(), configure_logs=False)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
