"""Per-run LLM usage attribution.

``RunOutcome.usage`` answers "what did this drive spend" at the source: the
engine wraps the process-wide client for every drive and records the usage
each completion it caused actually reports, so concurrent runs and schema
repair retries are attributed exactly. The run's total across drives
accumulates on the registry record, giving a parked run a "spent so far".

The exactness contract worth pinning: however many runs complete in one
process, the sum of their attributed usage is exactly the process total —
no gaps, no double counting.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentic_workflow.config import Settings
from agentic_workflow.domain.schemas import Decision
from agentic_workflow.errors import ProviderError
from agentic_workflow.llm.echo import EchoLLM
from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
from agentic_workflow.services.engine import WorkflowEngine
from tests.helpers import make_request, unique_run_id

pytestmark = pytest.mark.integration


def _approve(pending: Any) -> dict[str, Any]:
    """Return a decision that approves every gate."""
    return {
        "approval_id": pending.approval_id,
        "decision": Decision.APPROVE.value,
        "reviewer": "alice",
        "comment": "looks right",
    }


class _BurnAfterFirst(EchoLLM):
    """Echo that completes one call, then dies on the next.

    Lets a test watch a run fail *after* it spent tokens, which is exactly the
    case where a naive attribution would drop the spend: there is no settle
    path, just an exception unwinding.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    async def _complete(
        self,
        messages: Any,
        *,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls += 1
        if self.calls >= 2:
            raise ProviderError("provider on fire", model=self.model, retryable=False)
        return await super()._complete(messages, response_format=response_format, **kwargs)


class TestRunUsage:
    """Attributing LLM usage to the run that caused it."""

    async def test_a_parked_run_reports_its_first_drive(self, engine: WorkflowEngine) -> None:
        """A start that parks has already spent tokens and must say so.

        The first drive runs every node up to the human gate, and every one of
        those model calls is this run's spend.
        """
        outcome = await engine.start(make_request(run_id=unique_run_id("usage")))
        assert outcome.is_parked
        assert outcome.usage.calls > 0
        assert outcome.usage.prompt_tokens > 0
        assert outcome.usage.completion_tokens > 0

    async def test_run_totals_reconcile_exactly_with_the_process_total(
        self, engine: WorkflowEngine
    ) -> None:
        """Attributed usage adds up to exactly the shared counter.

        If attribution double-counted concurrent work or dropped anything, this
        equality would drift. Two full runs, each driven to completion through
        several gates, must be the whole process total and nothing else.
        """
        first = await asyncio.wait_for(
            engine.run_until_done(make_request(), decide=_approve), timeout=30
        )
        assert first.usage.calls > 0

        second = await asyncio.wait_for(
            engine.run_until_done(make_request(), decide=_approve), timeout=30
        )
        assert second.usage.calls > 0

        assert first.usage.merge(second.usage) == engine.llm_usage
        assert engine.llm_usage.calls == first.usage.calls + second.usage.calls

    async def test_registry_accumulates_usage_across_drives(self, engine: WorkflowEngine) -> None:
        """A parked run's spent-so-far survives its own resume.

        Each drive reports only its own share on the outcome; the registry
        record merges every drive, so a later read answers the run's total.
        """
        run_id = unique_run_id("accum")
        parked = await engine.start(make_request(run_id=run_id))
        assert parked.is_parked and parked.pending is not None
        assert parked.usage.calls > 0

        resumed = await engine.resume(run_id, _approve(parked.pending))
        assert resumed.usage.calls > 0

        record = engine.registry.get(run_id)
        expected = parked.usage.merge(resumed.usage)
        assert record.usage == expected
        assert record.usage.calls == parked.usage.calls + resumed.usage.calls

    async def test_a_rejected_gate_still_attributes_its_drive(self, engine: WorkflowEngine) -> None:
        """A rejection is a verdict, not a drop: its drive and continuation count.

        The rejected gate flows into a report, so the drive performs real work
        after the human says no; both the gate drive and the continuation must
        land on the run's record. Whatever terminal status the run reaches, the
        accumulation must not lose a single completion.
        """
        outcome = await engine.start(make_request(run_id=unique_run_id("settle")))
        assert outcome.is_parked
        assert outcome.pending is not None

        run_id = outcome.run_id
        resumed = await engine.resume(
            run_id,
            {
                "approval_id": outcome.pending.approval_id,
                "decision": Decision.REJECT.value,
                "reviewer": "alice",
                "comment": "not good enough",
            },
        )
        # The continuation after a rejection runs the reporter, so this drive
        # spent real tokens even though the human stopped the work.
        assert resumed.usage.calls > 0
        record = engine.registry.get(run_id)
        assert record.usage == outcome.usage.merge(resumed.usage)
        assert record.usage.calls == outcome.usage.calls + resumed.usage.calls

    async def test_a_failed_run_accounts_for_the_tokens_it_spent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that dies mid-flight must not take its spend down with it.

        The failure paths raise instead of settling, so this is the riskiest
        drop: the first drive completed a model call, the provider then blew
        up, and the run record must still show the one completion that
        happened — a post-mortem on a burned run deserves the truth.
        """
        engine = WorkflowEngine(
            Settings(_env_file=None, llm_provider="echo", log_level="ERROR"),
            checkpointer=build_memory_checkpointer(),
        )
        await engine.startup()
        request = make_request(run_id=unique_run_id("burn"))
        monkeypatch.setattr(
            "agentic_workflow.services.engine.build_llm_client",
            lambda _settings: _BurnAfterFirst(model="echo-1"),
        )
        try:
            with pytest.raises(ProviderError):
                await engine.start(request)
        finally:
            await engine.shutdown()

        record = engine.registry.get(request.run_id)
        assert record.status == "failed"
        # Exactly the one completion that succeeded before the provider died:
        # the failing call spent nothing and must not be credited.
        assert record.usage.calls == 1
        assert record.usage.prompt_tokens > 0
