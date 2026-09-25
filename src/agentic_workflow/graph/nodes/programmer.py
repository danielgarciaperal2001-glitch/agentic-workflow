"""Patch author: turn blocking findings into a minimal unified diff.

This node is the entry point of the feedback loop. After the first iteration it
is re-entered with the reviewer's findings, so it must be deterministic given
the same state — otherwise a replayed run produces a different diff and the
time-travel debugging story collapses.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

from agentic_workflow.domain.schemas import Patch, ReviewResult
from agentic_workflow.domain.state import WorkflowState, as_model
from agentic_workflow.graph.common import node
from agentic_workflow.graph.context import context_from_runtime
from agentic_workflow.graph.nodes.base import AgentNode
from agentic_workflow.logging import get_logger
from agentic_workflow.prompts import (
    PROGRAMMER_SYSTEM,
    render_findings,
    render_request,
)

log = get_logger(__name__)

#: Iteration beyond which the agent stops trying to patch and reports a blocker
#: instead. A model that keeps producing diffs without converging is worse than
#: one that escalates honestly.
STALL_ITERATION = 4


class ProgrammerAgent(AgentNode):
    """Author the minimal patch resolving the current blocking findings.

    Behaviour by iteration:

    * ``1`` — patch the raw request.
    * ``2..STALL_ITERATION`` — patch the reviewer's findings, carrying the
      previous diff forward as context so the model converges instead of
      oscillating.
    * ``> STALL_ITERATION`` — return an empty patch with a blocker note and let
      the router escalate to a human.
    """

    name: ClassVar[str] = "programmer"
    role: ClassVar[str] = "patch author"
    owns: ClassVar[frozenset[str]] = frozenset({"patch", "test_report"})
    system_prompt: ClassVar[str] = PROGRAMMER_SYSTEM

    async def __call__(self, state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
        """Produce a :class:`Patch` for the current iteration.

        Args:
            state: Workflow state.
            runtime: LangGraph runtime carrying the context.

        Returns:
            State update with ``patch``, ``findings`` consumed markers and a
            transcript entry.

        Raises:
            InvalidStateError: If the state has no request to work from.
        """
        context = context_from_runtime(runtime)
        request = self.request(state)

        iteration = int(state.get("iteration", 0)) + 1
        review = as_model(state, "review", ReviewResult)
        prior_patch = as_model(state, "patch", Patch)

        if iteration > STALL_ITERATION:
            log.warning(
                "programmer.stalled",
                run_id=request.run_id,
                iteration=iteration,
                stall_after=STALL_ITERATION,
            )
            patch = Patch(
                diff="",
                files_changed=[],
                summary=(
                    f"unable to converge after {iteration - 1} iterations; "
                    "escalating instead of emitting another speculative diff"
                ),
                strategy="escalate",
                confidence=0.1,
            )
        else:
            patch = await self.ask(
                context,
                Patch,
                render_request(request),
                f"iteration: {iteration}",
                f"previous findings:\n{render_findings(review.findings) if review else '(none)'}",
                (
                    f"previous diff:\n{prior_patch.diff[:6_000]}"
                    if prior_patch and prior_patch.diff
                    else "previous diff: (none)"
                ),
            )

        # Deterministic id: the same findings always yield the same patch id, so
        # a replayed checkpoint can be compared byte-for-byte.
        fingerprint = hashlib.sha256(
            "\0".join(f.id for f in (review.findings if review else [])).encode()
        ).hexdigest()[:16]
        log.info(
            "programmer.completed",
            run_id=request.run_id,
            iteration=iteration,
            files=len(patch.files_changed),
            diff_size=patch.diff_size,
            confidence=patch.confidence,
            patch_id=fingerprint,
        )
        await context.publish(
            "programmer.completed",
            run_id=request.run_id,
            iteration=iteration,
            diff_size=patch.diff_size,
        )

        update: dict[str, Any] = {
            "patch": patch,
            "next_action": "review",
            "context": {"last_patch_id": fingerprint},
            "transcript": [self.trace(patch.summary, files=patch.files_changed[:10])],
        }
        # A fresh patch invalidates the previous validation verdict.
        update["test_report"] = None
        self.assert_owns(update)
        return update


@node("programmer")
async def programmer_node(state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
    """LangGraph entry point for the patch author.

    Args:
        state: Current workflow state.
        runtime: LangGraph runtime.

    Returns:
        The agent's state update.
    """
    return await ProgrammerAgent()(state, runtime)


__all__ = ["STALL_ITERATION", "ProgrammerAgent", "programmer_node"]
