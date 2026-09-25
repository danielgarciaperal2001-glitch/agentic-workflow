"""Patch application: the irreversible step, therefore always human-gated.

This is the node that would push to a branch, open a pull request or write to a
production system in a real deployment. It is isolated in its own node for two
reasons:

* it is the only node with side effects, so the blast radius of a bug is one
  node rather than the whole graph;
* it is always preceded by a human gate, which is enforced here rather than
  trusted from configuration.

The reference implementation does not mutate anything: it records the intended
application so the flow is observable end-to-end. Inject
:attr:`ApplyPatchAgent.applier` to make it real.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from agentic_workflow.domain.schemas import Patch, ReviewResult
from agentic_workflow.domain.state import WorkflowState, as_model
from agentic_workflow.errors import InvalidStateError
from agentic_workflow.graph.common import node
from agentic_workflow.graph.context import context_from_runtime
from agentic_workflow.graph.nodes.base import AgentNode
from agentic_workflow.human.policy import Stage
from agentic_workflow.logging import get_logger
from agentic_workflow.prompts import render_findings

log = get_logger(__name__)


class PatchApplier(Protocol):
    """Extension point for a real patch-application backend.

    Implementations receive the patch plus the resolved human decision and must
    be idempotent: LangGraph can re-execute this node after a resume, and a
    double-apply is a real incident.
    """

    async def __call__(  # pragma: no cover - protocol
        self,
        patch: Patch,
        state: WorkflowState,
        reviewer: str,
    ) -> dict[str, Any]:
        """Apply *patch* and return the result summary."""
        ...


class ApplyPatchAgent(AgentNode):
    """Apply the reviewed patch after an explicit human authorisation.

    Raises:
        ApprovalRejectedError: If the human rejects. The router catches this and
            terminates the run cleanly rather than looping.
    """

    name: ClassVar[str] = "apply_patch"
    role: ClassVar[str] = "release operator"
    owns: ClassVar[frozenset[str]] = frozenset({"context"})
    system_prompt: ClassVar[str] = "You authorise and apply reviewed changes."

    applier: ClassVar[PatchApplier | None] = None

    async def __call__(self, state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
        """Gate on a human, then apply the patch.

        Args:
            state: Workflow state.
            runtime: LangGraph runtime carrying the context.

        Returns:
            State update recording the authorisation in the transcript and the
            application result in ``context``.

        Raises:
            InvalidStateError: If there is no patch to apply.
            ApprovalRejectedError: If the human rejects the change.
        """
        context = context_from_runtime(runtime)
        request = self.request(state)
        patch = as_model(state, "patch", Patch)
        if patch is None:
            raise InvalidStateError("apply_patch requires a patch in state", node=self.name)
        if patch.is_empty:
            log.info("apply_patch.noop", run_id=request.run_id)
            return {
                "next_action": "route",
                "context": {"patch_applied": False, "reason": "empty patch"},
                "transcript": [self.trace("skipped: empty patch")],
            }

        review = as_model(state, "review", ReviewResult)
        iteration = int(state.get("iteration", 0))
        gate = self.maybe_gate(
            context,
            stage=Stage.PATCH_APPLY,
            review=review,
            confidence=patch.confidence,
            iteration=iteration,
        )
        # A disabled policy must not silently authorise a write. If the operator
        # turned gates off explicitly, `enabled=False` already returned here, so
        # the only way to reach this point without a gate is a bug.
        if not gate.required:
            raise InvalidStateError(
                "apply_patch reached without an authorisation gate; refusing to "
                "perform an irreversible action",
                node=self.name,
                run_id=request.run_id,
            )

        decision = await self.ask_human(
            context,
            state,
            stage=Stage.PATCH_APPLY,
            title=f"Apply patch touching {len(patch.files_changed)} file(s)?",
            gate=gate,
            payload={
                **self.stage_payload(Stage.PATCH_APPLY, patch=patch, review=review),
                "findings_addressed": render_findings(review.findings) if review else "(none)",
            },
            diff_preview=patch.diff,
            iteration=iteration,
        )

        summary: dict[str, Any]
        if self.applier is not None:
            summary = await self.applier(patch, state, decision.reviewer)
        else:
            summary = {
                "applied": True,
                "dry_run": True,
                "files_changed": patch.files_changed,
                "diff_size": patch.diff_size,
            }
            log.info(
                "apply_patch.dry_run",
                run_id=request.run_id,
                files=len(patch.files_changed),
                reviewer=decision.reviewer,
            )

        await context.publish(
            "patch.applied",
            run_id=request.run_id,
            reviewer=decision.reviewer,
            dry_run=bool(summary.get("dry_run", True)),
        )

        update: dict[str, Any] = {
            "next_action": "report",
            "pending_approval": None,
            "context": {"patch_applied": True, "application": summary},
            "human_decisions": [self.decision_log(state, decision, Stage.PATCH_APPLY.value)],
            "transcript": [self.trace(f"authorised by {decision.reviewer}")],
        }
        self.assert_owns(update)
        return update


@node("apply_patch")
async def apply_patch_node(state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
    """LangGraph entry point for the gated patch application.

    Args:
        state: Current workflow state.
        runtime: LangGraph runtime.

    Returns:
        The node's state update.
    """
    return await ApplyPatchAgent()(state, runtime)


__all__ = ["ApplyPatchAgent", "PatchApplier", "apply_patch_node"]
