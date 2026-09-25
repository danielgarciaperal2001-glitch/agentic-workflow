"""Triage analyst: normalise the request into a task brief.

The first agent in the pipeline. It exists because the downstream agents need a
*stable* problem statement; feeding a raw, ambiguous request straight to the
patch author is the most common cause of wasted iterations.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agentic_workflow.domain.schemas import TaskBrief
from agentic_workflow.domain.state import WorkflowState
from agentic_workflow.graph.common import node
from agentic_workflow.graph.context import context_from_runtime
from agentic_workflow.graph.nodes.base import AgentNode
from agentic_workflow.logging import get_logger
from agentic_workflow.prompts import TRIAGE_SYSTEM, render_request

log = get_logger(__name__)


class TriageAgent(AgentNode):
    """Convert an incoming request into a :class:`TaskBrief`.

    The brief scopes the work, names the risks and states the success criteria.
    Everything downstream reads it, which keeps the agents from re-interpreting
    the original request differently on every iteration.
    """

    name: ClassVar[str] = "triage"
    role: ClassVar[str] = "triage analyst"
    owns: ClassVar[frozenset[str]] = frozenset({"task_brief"})
    system_prompt: ClassVar[str] = TRIAGE_SYSTEM

    async def __call__(self, state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
        """Run the triage step.

        Args:
            state: Workflow state containing the request.
            runtime: LangGraph runtime carrying the :class:`AgentContext`.

        Returns:
            State update with ``task_brief``, ``status`` and a transcript entry.
        """
        context = context_from_runtime(runtime)
        request = self.request(state)

        brief = await self.ask(
            context,
            TaskBrief,
            render_request(request),
            "Produce the TaskBrief for this change.",
        )

        log.info(
            "triage.completed",
            run_id=request.run_id,
            scope=len(brief.scope),
            risks=len(brief.risks),
            confidence=brief.confidence,
        )
        await context.publish(
            "triage.completed",
            run_id=request.run_id,
            confidence=brief.confidence,
        )

        update = {
            "task_brief": brief,
            "status": "triaged",
            "next_action": "program",
            "transcript": [self.trace(brief.objective, risks=brief.risks[:5])],
        }
        self.assert_owns(update)
        return update


@node("triage")
async def triage_node(state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
    """LangGraph entry point for the triage agent.

    Args:
        state: Current workflow state.
        runtime: LangGraph runtime (unused; kept for signature compatibility).

    Returns:
        The agent's state update.
    """
    return await TriageAgent()(state, runtime)


__all__ = ["TriageAgent", "triage_node"]
