"""Shared base class for agent nodes.

:class:`AgentNode` removes the boilerplate every agent needs:

* it resolves the :class:`~agentic_workflow.graph.context.AgentContext` from the
  LangGraph runtime,
* it exposes ``ask_human()`` — the single entry point to a human gate,
* it records a transcript entry for observability and for the evaluation suite,
* it declares which state keys it owns, so :func:`assert_idempotent` can catch
  bugs where a node clobbers a peer's channel.

Example:
    --------
    >>> from agentic_workflow.graph.context import AgentContext
    >>> from agentic_workflow.graph.nodes.base import AgentNode
    >>> class Demo(AgentNode):
    ...     name = "demo"
    ...     owns = frozenset({"demo"})
    >>> Demo().name
    'demo'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import hashlib
from typing import Any, ClassVar

from agentic_workflow.domain.schemas import (
    ApprovalDecision,
    Decision,
    FinalReport,
    Patch,
    ReviewRequest,
    ReviewResult,
    TaskBrief,
    TestReport,
    Verdict,
    utcnow,
)
from agentic_workflow.domain.state import WorkflowState, as_model
from agentic_workflow.errors import ApprovalRejectedError, InvalidStateError
from agentic_workflow.graph.common import transcript_entry
from agentic_workflow.graph.context import AgentContext, context_from_runtime
from agentic_workflow.human.gates import (
    build_approval_request,
    gate_payload_for_stage,
    record_decision_log,
    request_decision,
)
from agentic_workflow.human.policy import GateDecision, Stage
from agentic_workflow.llm.base import Message
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

#: Channels any node may write: lifecycle status, routing hints, observability,
#: bookkeeping and the HITL records (a node that raises a gate is, by
#: definition, the one recording the decision). Exclusive agent outputs — the
#: things ``owns`` declares — remain guarded.
SHARED_CHANNELS: frozenset[str] = frozenset(
    {
        "status",
        "next_action",
        "context",
        "error",
        "transcript",
        "node_timings",
        "updated_at",
        "human_decisions",
        "pending_approval",
    }
)


class AgentNode(ABC):
    """Base class binding a node function to its identity and context.

    Attributes:
        name: Node name, matching the key the node is registered under.
        role: Short role description used in prompts and transcripts.
        owns: State channels this node is the sole writer for. Writing any
            other channel is a bug the base class reports loudly.
        system_prompt: System prompt for the LLM call.
    """

    name: ClassVar[str]
    role: ClassVar[str] = "agent"
    owns: ClassVar[frozenset[str]] = frozenset()
    system_prompt: ClassVar[str] = "You are a precise JSON emitter."

    # ------------------------------------------------------------ context #
    @staticmethod
    def resolve(runtime: Any) -> AgentContext:
        """Return the injected runtime context."""
        return context_from_runtime(runtime)

    @property
    def logger(self) -> Any:
        """A logger bound to this node's name."""
        return get_logger(type(self).__module__).bind(node=self.name, agent=self.role)

    # -------------------------------------------------------------- LLM  #
    def messages(self, *blocks: str) -> list[Message]:
        """Build the chat history for this agent's LLM call.

        The system prompt always comes first so providers can cache the static
        prefix, and the variable blocks follow in a fixed order.
        """
        return [
            Message(role="system", content=self.system_prompt),
            *(Message(role="user", content=b) for b in blocks if b),
        ]

    async def ask(
        self,
        context: AgentContext,
        response_model: type[Any],
        *blocks: str,
    ) -> Any:
        """Request a validated, typed answer from the LLM."""
        return await context.client.structured(self.messages(*blocks), response_model)

    # ------------------------------------------------------- transcript  #
    def trace(self, summary: str, **payload: Any) -> dict[str, Any]:
        """Build a transcript entry owned by this node."""
        return transcript_entry(node=self.name, role=self.role, summary=summary, payload=payload)

    # --------------------------------------------------------- HITL gate  #
    async def ask_human(
        self,
        context: AgentContext,
        state: WorkflowState,
        *,
        stage: Stage,
        title: str,
        gate: GateDecision,
        payload: dict[str, Any] | None = None,
        diff_preview: str = "",
        iteration: int = 0,
    ) -> ApprovalDecision:
        """Pause the graph and block until a human decides.

        The caller must have already established that a gate is required (via
        :meth:`~agentic_workflow.human.policy.EscalationPolicy.evaluate`); this
        method is the mechanism, not the decision.

        Args:
            context: Runtime context (supplies the LLM and signing secret).
            state: Current state, used to build the inspection payload.
            stage: The stage being gated.
            title: One-line summary shown in the approval inbox.
            gate: The policy decision that triggered the gate.
            payload: Extra inspection data.
            diff_preview: Diff excerpt for the human.
            iteration: Feedback-loop iteration. Part of the gate's stable
                identity, so it must be the iteration this gate belongs to — not
                a re-derived value that could differ on re-execution.

        Returns:
            The human's validated decision.

        Raises:
            ApprovalRejectedError: If the human rejected the action. Callers
                decide whether that terminates the run or feeds the rejection
                back into the loop.
        """
        request = build_approval_request(
            run_id=self.request(state).run_id,
            stage=stage,
            title=title,
            gate=gate,
            policy=context.escalation,
            settings=context.settings,
            payload=payload,
            diff_preview=diff_preview,
            iteration=iteration,
        )
        await context.publish(
            "approval.requested",
            run_id=request.run_id,
            approval_id=request.approval_id,
            stage=request.stage,
            reason=gate.reason.value,
            title=request.title,
        )
        decision = request_decision(
            request,
            secret=context.signing_secret if context.sign_human_decisions else "",
            default_reviewer="api",
        )
        return decision

    def maybe_gate(
        self,
        context: AgentContext,
        *,
        stage: Stage,
        review: ReviewResult | None = None,
        confidence: float | None = None,
        iteration: int = 0,
    ) -> GateDecision:
        """Evaluate the escalation policy for *stage*."""
        return context.escalation.evaluate(
            stage,
            review=review,
            confidence=confidence,
            iteration=iteration,
            max_iterations=context.settings.max_iterations,
        )

    # ------------------------------------------------------------ state  #
    @classmethod
    def request(cls, state: WorkflowState) -> ReviewRequest:
        """Return the immutable :class:`ReviewRequest` from *state*.

        Goes through :func:`~agentic_workflow.domain.state.as_model` so a
        checkpoint store that degrades models to dicts still works.
        """
        from agentic_workflow.domain.state import as_model
        from agentic_workflow.errors import InvalidStateError

        request = as_model(state, "request", ReviewRequest)
        if request is None:
            raise InvalidStateError("workflow state has no request")
        return request

    @staticmethod
    def stage_payload(
        stage: Stage,
        *,
        patch: Patch | None = None,
        review: ReviewResult | None = None,
        tests: TestReport | None = None,
        report: FinalReport | None = None,
        brief: TaskBrief | None = None,
    ) -> dict[str, Any]:
        """Build the inspection payload shown to the human for *stage*."""
        payload = gate_payload_for_stage(
            stage, review=review, tests=tests, patch=patch, report=report
        )
        if brief is not None:
            payload["task_brief"] = brief.model_dump(mode="json")
        return payload

    @classmethod
    def decision_log(
        cls,
        state: WorkflowState,
        decision: ApprovalDecision,
        request_stage: str,
    ) -> dict[str, Any]:
        """Build the audit-log entry for a resolved approval.

        Recorded in the graph state (and therefore in the checkpoint history), so
        it is a durable, replayable record of who authorised what.
        """
        return record_decision_log(
            decision,
            run_id=cls.request(state).run_id,
            stage=request_stage,
        )

    @staticmethod
    def rejected(decision: ApprovalDecision, reason: str) -> dict[str, Any]:
        """Build the state update recorded when a human rejects an action."""
        return {
            "transcript": [
                {
                    "node": "hitl",
                    "role": "human",
                    "summary": f"rejected: {reason}",
                    "payload": {"decision": decision.decision.value},
                    "at": utcnow().isoformat(),
                }
            ],
            "next_action": "rejected",
        }

    @staticmethod
    def approved(decision: ApprovalDecision, note: str = "") -> dict[str, Any]:
        """Build the state update recorded when a human approves an action."""
        return {
            "transcript": [
                {
                    "node": "hitl",
                    "role": "human",
                    "summary": f"approved: {note}" if note else "approved",
                    "payload": {
                        "decision": decision.decision.value,
                        "reviewer": decision.reviewer,
                    },
                    "at": utcnow().isoformat(),
                }
            ]
        }

    @staticmethod
    def edited(decision: ApprovalDecision, **updates: Any) -> dict[str, Any]:
        """Build the state update recorded when a human rewrote agent output."""
        return {
            "transcript": [
                {
                    "node": "hitl",
                    "role": "human",
                    "summary": f"edited: {decision.comment or 'no comment'}",
                    "payload": dict(decision.payload),
                    "at": utcnow().isoformat(),
                }
            ],
            "context": {"human_edits": updates},
        }

    # -------------------------------------------------------- rejection #
    @classmethod
    def rejection_update(
        cls,
        state: WorkflowState,
        exc: ApprovalRejectedError,
        *,
        stage: Stage,
        review: ReviewResult | None = None,
    ) -> dict[str, Any]:
        """Build the state update recorded when a human rejects at *stage*.

        A rejection is a *verdict*, not a failure. Letting it unwind as an
        exception would end the run with no ``human_decisions`` entry, no final
        report, and a status of ``failed`` — three things that all misrepresent a
        person exercising exactly the authority the workflow exists to give
        them. So the rejection is turned back into an ordinary state update:
        the decision is written to the audit log, the review verdict becomes
        :attr:`Verdict.REJECTED`, and the router sends the run to the reporter to
        document it.

        The ``review`` verdict is written *unconditionally*, even when the gate
        that was refused had no review behind it. The router decides where to go
        from that verdict, so omitting it would strand the run: LangGraph re-runs
        the gated node from the state *before* it executed, which means a node
        that rejects on its own output — the reviewer — finds no review in
        ``state`` and would leave the router with nothing to route on, sending
        the run back to the gate that just refused it. Forever.

        Args:
            state: Current workflow state.
            exc: The rejection raised by :func:`request_decision`.
            stage: Gate the decision answered.
            review: The review in play, when the caller has just produced one.

        Returns:
            A state update ready to be returned by the gated node.

        Raises:
            ApprovalRejectedError: If the error carries no decision, because a
                rejection that cannot be recorded must not be swallowed.
        """
        decision = exc.decision
        if decision is None:
            # Recording a rejection we cannot attribute to a person and a gate
            # would be theatre. Propagate so the run fails loudly instead.
            raise exc

        run_id = cls.request(state).run_id
        current = review if review is not None else as_model(state, "review", ReviewResult)
        reason = decision.comment or f"rejected at {stage.value}"

        return {
            # The router reads this verdict to choose the reporter, and the
            # reporter derives the final decision from it, so this single write
            # is what makes the whole run read as "a human said no".
            "review": ReviewResult(
                verdict=Verdict.REJECTED,
                findings=current.findings if current else [],
                summary=(
                    current.summary
                    if current and current.summary
                    else f"rejected at {stage.value} by {decision.reviewer}"
                ),
                confidence=current.confidence if current else 0.0,
                blocking_findings=current.blocking_findings if current else [],
            ),
            "human_decisions": [record_decision_log(decision, run_id=run_id, stage=stage.value)],
            "next_action": "rejected",
            "pending_approval": None,
            "transcript": [
                {
                    "node": "hitl",
                    "role": "human",
                    "summary": f"rejected at {stage.value}: {reason}",
                    "payload": {
                        "decision": decision.decision.value,
                        "reviewer": decision.reviewer,
                    },
                    "at": utcnow().isoformat(),
                }
            ],
        }

    # --------------------------------------------------------- guards    #
    def assert_owns(self, update: dict[str, Any], *, extra: Sequence[str] = ()) -> None:
        """Validate that the returned update only touches permitted channels.

        The guard exists because LangGraph *tolerates* a node writing a peer's
        exclusive channel, and the result is a silent race that only shows up as
        a wrong answer three nodes later. Shared control channels (status,
        routing hints, timings) are exempt; exclusive agent outputs are not.

        Args:
            update: The state update returned by the node.
            *extra: Additional keys the node may write for this invocation.

        Raises:
            InvalidStateError: If the node writes a channel it does not own.
        """
        allowed = set(self.owns) | set(SHARED_CHANNELS) | {*extra}
        forbidden = set(update) - allowed
        if forbidden:
            raise InvalidStateError(
                f"node {self.name!r} wrote channels it does not own: {sorted(forbidden)}",
                node=self.name,
                owned=sorted(self.owns),
                forbidden=sorted(forbidden),
            )

    @staticmethod
    def fingerprint(*parts: Any) -> str:
        """Deterministic short hash of *parts*, used for transcript dedup."""
        basis = "\0".join(str(p) for p in parts)
        return hashlib.sha256(basis.encode()).hexdigest()[:16]

    @abstractmethod
    async def __call__(self, state: WorkflowState, runtime: Any = None) -> dict[str, Any]:
        """Execute the node and return its state update."""


def edited_decision_payload(decision: ApprovalDecision) -> dict[str, Any]:
    """Extract the edited values from a human ``EDIT`` decision."""
    return dict(decision.payload) if decision.decision is Decision.EDIT else {}


__all__ = ["AgentNode", "edited_decision_payload"]
