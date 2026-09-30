"""A run can cost whatever the caller asks for, and no budget says otherwise.

Everything that reaches an LLM is paid for twice: once in the provider's invoice
and once in the latency a client is waiting on. The control plane bounds requests
per minute (``api_rate_limit_per_minute``, 120 by default), which reads like a
spend cap and is not one — a single request is unbounded, so the ceiling on spend
per minute is 120 times whatever one request costs.

Measured through the real ``POST /v1/runs`` with the echo provider, sweeping the
payload sizes a caller can actually submit (``files`` is capped at 200 entries by
the wire schema):

    payload shape          files   KiB/file   total_tokens    ratio
    single small file           1          0           6472      1x
    one module                  1         19          20314      3x
    a few modules               5         19          95880     15x
    a real diff                20         19         328452     51x
    a large refactor           50         19        1018772    157x
    at the wire limit         200         19        4060022    627x

627x between the smallest useful submission and the largest legal one, and the
only ceiling in that path is the one the caller supplies: ``max_gates`` is theirs
to set up to 1000, and the tokens are not counted at all. The usage endpoint
reports the damage faithfully *afterwards* (``GET /v1/runs/{id}/usage``), which is
the worst moment to learn it.

So the budget is enforced where the spend happens — between completions, not
around the run. A check at the end of the drive would be a report, not a limit:
the measurement above is what one request costs, so by the time a post-hoc check
ran the tokens were already spent and the client already waiting. Refusing inside
the client means the next call is the one that does not happen.

The shipped default is set from that table rather than picked: 500k admits "a real
diff" (328k measured) and refuses "a large refactor" (1.02M measured), which is
the line between a review a person asked for and a payload that is a mistake. It
is a ceiling, not a quota: nothing is refunded, and the tokens spent before the
refusal are still charged and still reported.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentic_workflow.api.app import create_app
from agentic_workflow.config import Settings, load_settings, reset_settings_cache
from agentic_workflow.domain.schemas import Decision, ReviewRequest
from agentic_workflow.errors import TokenBudgetExceededError
from agentic_workflow.graph.context import AgentContext
from agentic_workflow.services.engine import WorkflowEngine

pytestmark = pytest.mark.api


def _settings(**overrides: Any) -> Settings:
    """Build an offline configuration.

    Args:
        **overrides: Settings fields to set.

    Returns:
        The configuration.
    """
    reset_settings_cache()
    return load_settings(  # type: ignore[no-any-return]
        **{
            "environment": "development",
            "llm_provider": "echo",
            "postgres_enabled": False,
            "hitl_enabled": True,
            "log_level": "ERROR",
            **overrides,
        }
    )


def _request(run_id: str, *, files: list[dict[str, str]] | None = None) -> ReviewRequest:
    """Build a review request for the engine-level tests.

    Args:
        run_id: Identifier for the run.
        files: Source files to submit, or ``None`` for a token-free request.

    Returns:
        The request.
    """
    return ReviewRequest(
        run_id=run_id,
        request_id="PR-1042",
        title="t",
        description="d",
        files=files or [],
    )


class TestTheCeilingStopsTheSpend:
    """The refusal has to land before the tokens are gone."""

    async def test_a_run_over_the_ceiling_is_refused_mid_drive(self) -> None:
        """The run stops at the call that would cross the budget, not after it.

        The refusal is raised from the client, so it travels through the node
        wrapper as a domain error rather than being absorbed into a generic
        ``AgentError`` — which matters because the two map to different statuses
        and a budget that answered 500 would be indistinguishable from a crash.
        """
        settings = _settings(llm_token_budget_per_run=50_000)
        engine = WorkflowEngine(settings)
        files = [{"path": f"m{i}.py", "content": "y = 2  # " + "z" * 20_000} for i in range(20)]

        with pytest.raises(TokenBudgetExceededError) as caught:
            await engine.run_until_done(_request("budget-1", files=files))

        assert caught.value.context["budget"] == 50_000
        assert caught.value.context["spent"] > 50_000, "the ceiling is reported as exceeded"

    async def test_a_run_under_the_ceiling_is_untouched(self) -> None:
        """The budget is a ceiling, not a toll: a run inside it finishes.

        Without this, a fix that simply always raised would satisfy the test above
        while making the engine refuse to review anything.
        """
        settings = _settings(llm_token_budget_per_run=500_000)
        engine = WorkflowEngine(settings)

        outcome = await engine.run_until_done(_request("budget-2"))

        assert outcome.status == "completed"
        assert outcome.usage.total_tokens > 0

    async def test_a_zero_budget_means_unbounded(self) -> None:
        """``0`` disables the ceiling, because a zero budget would refuse everything.

        Same convention as ``api_rate_limit_per_minute=0``. Stated as a test
        because the alternative reading — "0 means no tokens allowed" — would
        break every existing deployment on upgrade, which is a far worse surprise
        than the feature being off.
        """
        settings = _settings(llm_token_budget_per_run=0)
        engine = WorkflowEngine(settings)
        files = [{"path": f"m{i}.py", "content": "y = 2  # " + "z" * 20_000} for i in range(5)]

        outcome = await engine.run_until_done(_request("budget-3", files=files))

        assert outcome.status == "completed"
        assert outcome.usage.total_tokens > 50_000, "measured at ~95k, so the cap was really off"


class TestTheCeilingIsCumulative:
    """A run that is resumed is still one run, and the budget is per run."""

    async def test_the_budget_spans_every_drive_of_the_run(self) -> None:
        """Two drives that each fit still fail together once the run's total does.

        The budget belongs to the run, not the request. A run parked on a human
        decision, then answered, is two drives and one bill, and a per-drive
        ceiling would let a caller reset it by making the run park more often.

        The numbers are measured so the ceiling falls *between* the two drives:
        with 20 files the first drive spends 246,328 tokens and the second takes
        the run to 287,398, so a 260,000 ceiling is crossed by the resume alone.
        A ceiling that both drives fit inside would pass here for the wrong
        reason, and one that the first drive already breached would never test
        the accumulation at all.
        """
        settings = _settings(llm_token_budget_per_run=260_000, hitl_enabled=True)
        engine = WorkflowEngine(settings)
        files = [{"path": f"m{i}.py", "content": "y = 2  # " + "z" * 20_000} for i in range(20)]

        first = await engine.start(_request("budget-4", files=files))
        assert first.is_parked, "the first drive parks on a gate, spending part of the budget"

        approval = first.pending
        assert approval is not None
        spent_after_park = engine.registry.find("budget-4").usage.total_tokens  # type: ignore[union-attr]
        assert 0 < spent_after_park < 260_000, (
            f"one drive fits inside the ceiling, measured at {spent_after_park}"
        )

        with pytest.raises(TokenBudgetExceededError):
            await engine.resume("budget-4", _resume(approval.approval_id))

        record = engine.registry.find("budget-4")
        assert record is not None
        assert record.usage.total_tokens > 260_000, "the run's own total is what crossed it"

    async def test_the_spend_before_the_refusal_is_still_charged(self) -> None:
        """A refused run reports what it burned on the way to the ceiling.

        The tokens were spent whether or not the run finished, so dropping them
        would make ``/usage`` under-report exactly the runs an operator most needs
        to look at. The refusal stops the *next* call, it does not un-spend this one.
        """
        settings = _settings(llm_token_budget_per_run=50_000)
        engine = WorkflowEngine(settings)
        files = [{"path": f"m{i}.py", "content": "y = 2  # " + "z" * 20_000} for i in range(20)]

        with pytest.raises(TokenBudgetExceededError):
            await engine.run_until_done(_request("budget-5", files=files))

        record = engine.registry.find("budget-5")
        assert record is not None
        assert record.usage.total_tokens > 50_000, "the refused run still accounts for its spend"
        assert record.status == "failed", "and it is a failed run, not a silent success"


class TestEveryContextPathIsBounded:
    """The engine has three ways to build a run's client, and all three matter.

    A ceiling applied on one path and missed on another is worse than no ceiling:
    it looks enforced in review and is not. The path exercised here is the
    template one, because a template context that keeps an unbound wrapper is the
    one that would silently reset the budget on each resume.
    """

    async def test_a_context_template_does_not_reset_the_budget(self) -> None:
        """A context template without a client still gets a bounded wrapper.

        The template path is the one that would lose the run id, and with it the
        run's prior spend — the run would then be measured per drive, and a
        caller could reset the ceiling by making the run park more often.
        """
        settings = _settings(llm_token_budget_per_run=260_000, hitl_enabled=True)
        engine = WorkflowEngine(settings, context=AgentContext(settings=settings))
        files = [{"path": f"m{i}.py", "content": "y = 2  # " + "z" * 20_000} for i in range(20)]

        first = await engine.start(_request("budget-tmpl", files=files))
        assert first.is_parked
        approval = first.pending
        assert approval is not None

        with pytest.raises(TokenBudgetExceededError):
            await engine.resume("budget-tmpl", _resume(approval.approval_id))


class TestTheCeilingOnTheWire:
    """What the client is told has to name the budget, not just the failure."""

    def test_an_over_budget_submission_answers_429(self) -> None:
        """``429``, and the body says which budget was hit.

        ``429`` rather than ``500`` because the server is fine and the request was
        too expensive — the same reasoning that puts ``ConcurrencyLimitError`` at
        ``429``. A ``500`` would teach every client to retry, and a retry of the
        same payload spends the same tokens to reach the same answer.
        """
        settings = _settings(llm_token_budget_per_run=50_000)
        engine = WorkflowEngine(settings)
        files = [{"path": f"m{i}.py", "content": "y = 2  # " + "z" * 20_000} for i in range(20)]
        with _client(settings, engine) as client:
            response = client.post(
                "/v1/runs",
                json={
                    "run_id": "budget-wire-1",
                    "request_id": "PR-1042",
                    "title": "t",
                    "description": "d",
                    "auto_resolve": True,
                    "files": files,
                },
            )

        assert response.status_code == 429, response.text
        body = response.json()["error"]
        assert body["code"] == "token_budget_exceeded"
        assert body["context"]["budget"] == 50_000, "the client can size its own retry"

    def test_a_submission_inside_the_budget_is_unaffected(self) -> None:
        """The default ceiling does not refuse an ordinary review.

        The wire-level mirror of the engine test, because a default that is too
        tight would be a regression nobody would notice until a real review failed.
        """
        settings = _settings()
        engine = WorkflowEngine(settings)
        with _client(settings, engine) as client:
            response = client.post(
                "/v1/runs",
                json={
                    "run_id": "budget-wire-2",
                    "request_id": "PR-1042",
                    "title": "t",
                    "description": "d",
                    "auto_resolve": True,
                },
            )

        assert response.status_code == 200, response.text
        assert response.json()["usage"]["total_tokens"] > 0


def _resume(approval_id: str) -> Any:
    """Build the decision that answers one approval.

    Args:
        approval_id: The approval being answered.

    Returns:
        The decision the engine resumes the run with.
    """
    from agentic_workflow.domain.schemas import ApprovalDecision

    return ApprovalDecision(
        approval_id=approval_id,
        decision=Decision.APPROVE,
        comment="ok",
        reviewer="tester",
    )


def _client(settings: Settings, engine: WorkflowEngine) -> Any:
    """Return a test client bound to *engine*.

    Args:
        settings: The configuration under test.
        engine: The engine under test.

    Returns:
        A ``TestClient``, used as a context manager.
    """
    from fastapi.testclient import TestClient

    return TestClient(create_app(settings, engine=engine, configure_logs=False))


def test_the_budget_is_configurable() -> None:
    """The ceiling is a setting, so an operator can raise it without a code change.

    A hard-coded constant would make the feature something you either have or do
    not, and a deployment whose legitimate reviews are larger than the default
    would have no way to proceed but to disable it entirely.
    """
    assert _settings(llm_token_budget_per_run=123_456).llm_token_budget_per_run == 123_456
    assert _settings().llm_token_budget_per_run == 500_000, "the shipped default, from the sweep"
    assert _settings(llm_token_budget_per_run=0).llm_token_budget_per_run == 0, "0 disables"


def test_the_budget_is_reported_in_the_settings_disclosure() -> None:
    """A limit an operator cannot see is a limit nobody can tune.

    ``Settings.safe_summary`` is what the startup log and the CLI print, so a
    budget missing from it is invisible until a run is refused for a reason the
    operator never learned to look for.
    """
    described = _settings(llm_token_budget_per_run=777_000).safe_summary()

    assert described["llm_token_budget_per_run"] == 777_000


def test_a_refused_run_does_not_leak_its_concurrency_slot() -> None:
    """A refusal is a settlement, so the run cannot keep holding its slot.

    The engine admits a run against a concurrency budget and releases it when the
    drive settles. A budget refusal that escaped that bookkeeping would leave a
    slot consumed for the life of the process, and the plane would fill up with
    runs that finished — or rather, refused — minutes ago.
    """
    settings = _settings(llm_token_budget_per_run=50_000)
    engine = WorkflowEngine(settings)
    files = [{"path": f"m{i}.py", "content": "y = 2  # " + "z" * 20_000} for i in range(20)]

    async def drive() -> None:
        with pytest.raises(TokenBudgetExceededError):
            await engine.run_until_done(_request("budget-6", files=files))
        # A second run must be admitted, which it cannot be if the first kept
        # its slot.
        second = await engine.run_until_done(_request("budget-7"))
        assert second.status in {"completed", "parked"}

    asyncio.run(asyncio.wait_for(drive(), timeout=60))
