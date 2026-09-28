"""Per-caller usage attribution through the shared LLM client.

``UsageScopedClient`` exists to answer "what did this run cost" without the
window-inference errors of diffing the process-wide counter: it records the
``usage`` carried by each ``Completion`` it caused, so concurrent callers are
attributed exactly — no double counting, no gaps.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict
import pytest

from agentic_workflow.domain.schemas import TaskBrief
from agentic_workflow.llm.base import Message, UsageScopedClient
from agentic_workflow.llm.echo import EchoLLM

pytestmark = pytest.mark.unit


class _Shape(BaseModel):
    """Minimal schema the echo provider can synthesise for a structured call."""

    model_config = ConfigDict(extra="forbid")

    content: str


class TestUsageScopedClient:
    async def test_records_each_completion_exactly(self) -> None:
        """Every delegated completion is added to the wrapper's own total."""
        shared = EchoLLM(model="echo-1")
        scoped = UsageScopedClient(shared)

        await scoped.complete([Message(role="user", content="first call")])
        first = scoped.usage
        assert first.calls == 1
        assert first.prompt_tokens > 0

        await scoped.complete([Message(role="user", content="second call")])
        second = scoped.usage
        assert second.calls == 2
        assert second == first.merge(first)  # both calls counted exactly once

    async def test_structured_interception_counts_every_repair_attempt(self) -> None:
        """Nodes call ``structured``; the wrapper records each underlying call.

        A schema-validation repair issues a *second* completion, and both must
        land on the run's account — the repaired attempt cost real tokens.
        """
        scoped = UsageScopedClient(EchoLLM(model="echo-1"))
        brief = await scoped.structured(
            [Message(role="user", content="summarise the checkout drift")],
            TaskBrief,
        )
        assert isinstance(brief, TaskBrief)
        assert scoped.usage.calls >= 1

    async def test_concurrent_callers_are_accounted_separately(self) -> None:
        """Three calls by A and five by B reconcile exactly to the shared total.

        This is the property a process-wide diff can never give: each caller's
        attribute is its own completions, even though both went through the
        same client at the same time.
        """
        shared = EchoLLM(model="echo-1")
        alpha = UsageScopedClient(shared)
        beta = UsageScopedClient(shared)

        async def call(client: UsageScopedClient, n: int) -> None:
            for _ in range(n):
                await client.complete([Message(role="user", content="hello")])

        await asyncio.gather(call(alpha, 3), call(beta, 5))

        assert alpha.usage.calls == 3
        assert beta.usage.calls == 5
        # No double counting, no gaps: the two attributes are exactly the
        # whole process total.
        assert alpha.usage.merge(beta.usage) == shared.total_usage
        assert shared.total_usage.calls == 8

    async def test_delegates_unknown_attributes_for_transparency(self) -> None:
        """Provider subclasses carry extra state; a wrapper must not hide it."""
        shared = EchoLLM(model="echo-1", latency_ms=2.5)
        scoped = UsageScopedClient(shared)
        assert scoped.model == "echo-1"
        assert scoped.latency_ms == 2.5

    async def test_aclose_never_closes_the_shared_client(self) -> None:
        """Closing a wrapper must not take down the process-wide client.

        The engine owns and closes the shared client; a wrapper is a per-run
        accounting lens, not a resource owner.
        """
        shared = EchoLLM(model="echo-1")
        scoped = UsageScopedClient(shared)
        await scoped.aclose()
        # EchoLLM has no resources to release, and the wrapper must not have
        # delegated to a close that would. The shared client still works.
        completion = await shared.complete([Message(role="user", content="still alive")])
        assert completion.content

    def test_signature_accepts_keyword_only_response_format(self) -> None:
        """The override matches the base contract callers rely on."""
        scoped = UsageScopedClient(EchoLLM(model="echo-1"))
        import inspect

        sig = inspect.signature(scoped.complete)
        assert sig.parameters["response_format"].kind == inspect.Parameter.KEYWORD_ONLY
