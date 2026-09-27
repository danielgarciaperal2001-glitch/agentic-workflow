"""What the control plane tells the operator about itself at boot.

A process that quietly fails to honour its own configuration is worse than one
that refuses to start, because the operator reading the log concludes the
opposite. These tests pin the one place that class of surprise had a real
footprint: the audit trail's tamper-evidence.

``hitl_require_signature`` defaults to true, and the default deployment has
nothing to sign with — the API is unauthenticated and the provider is the
offline ``echo``, so neither the API token nor any dedicated secret is set.
Decisions were therefore written unsigned, the setting's description said they
were signed, and the only hint was an ``audit_verified: null`` on an approval
response that most clients never surface. The API already reported the gap; the
problem was that nothing said it once, where an operator would look.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient
import pytest
from structlog.testing import capture_logs

from agentic_workflow.api.app import create_app
from agentic_workflow.config import Settings
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer

pytestmark = pytest.mark.api


@contextmanager
def _booted(**overrides: Any) -> Iterator[list[dict[str, Any]]]:
    """Start the app and yield the log records its lifespan emitted.

    Args:
        **overrides: Settings fields to set on a default offline configuration.

    Yields:
        The captured log entries, in order.
    """
    from agentic_workflow.services.engine import WorkflowEngine

    settings = Settings(
        _env_file=None,
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        hitl_enabled=True,
        api_rate_limit_per_minute=0,
        **overrides,
    )
    engine = WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
    with (
        capture_logs() as logs,
        TestClient(create_app(settings, engine=engine, configure_logs=False)),
    ):
        pass
    yield list(logs)


def _notices(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the signature-disclosure records.

    Args:
        logs: Captured log entries.

    Returns:
        The matching entries.
    """
    return [entry for entry in logs if entry["event"] == "hitl.signatures_unavailable"]


class TestAuditDisclosure:
    """The control plane says out loud when it cannot sign its decisions."""

    def test_an_unsigned_audit_trail_is_announced_once(self) -> None:
        """The default deployment says so, exactly once, naming the remedy.

        The default is the case that matters: it is what almost every deployment
        starts as, and it is the one where the promise is silently broken. A
        message that only said "no secret" would be skimmed past, so it has to
        name the consequence and the setting that fixes it.
        """
        with _booted() as logs:
            pass

        notices = _notices(logs)
        assert len(notices) == 1
        detail = notices[0]["detail"].lower()
        assert "unsigned" in detail
        assert "awf_hitl_signing_secret" in detail

    def test_a_configured_secret_announces_nothing(self) -> None:
        """The disclosure is not standing noise on a correctly configured plane."""
        with _booted(hitl_signing_secret="a-signing-secret") as logs:
            pass

        assert _notices(logs) == []

    def test_opting_out_of_signatures_announces_nothing(self) -> None:
        """Declining signatures is a decision, not a misconfiguration.

        An operator who sets ``hitl_require_signature=false`` has decided they do
        not need them — for a local or single-tenant deployment that is a
        reasonable call. Warning about a choice nobody made by accident is how
        warnings get filtered out.
        """
        with _booted(hitl_require_signature=False) as logs:
            pass

        assert _notices(logs) == []

    def test_the_api_token_alone_is_enough_to_quiet_it(self) -> None:
        """A single-secret deployment is a supported shape, not a degraded one.

        Requiring a second secret would push operators toward leaving the
        requirement off entirely, which is strictly worse than reusing the token
        they already have.
        """
        with _booted(api_auth_token="an-api-token", api_auth_enabled=True) as logs:
            pass

        assert _notices(logs) == []
