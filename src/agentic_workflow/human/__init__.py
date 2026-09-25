"""Human-in-the-Loop: escalation policy and LangGraph interrupt plumbing.

* :mod:`agentic_workflow.human.policy` — *whether* to ask a human.
* :mod:`agentic_workflow.human.gates` — *how* to ask and how to resume.
* :mod:`agentic_workflow.human.service` — the REST-facing orchestration of
  pending approvals (persistence, expiry, idempotency).
"""

from __future__ import annotations

from agentic_workflow.human.gates import (
    build_approval_request,
    decode_interrupt,
    diff_preview,
    encode_interrupt,
    gate_payload_for_stage,
    record_decision_log,
    request_decision,
    sign_decision,
    verify_decision,
)
from agentic_workflow.human.policy import (
    EscalationPolicy,
    EscalationReason,
    GateDecision,
    Stage,
    allowed_decisions,
    summarise_review,
)

__all__ = [
    "EscalationPolicy",
    "EscalationReason",
    "GateDecision",
    "Stage",
    "allowed_decisions",
    "build_approval_request",
    "decode_interrupt",
    "diff_preview",
    "encode_interrupt",
    "gate_payload_for_stage",
    "record_decision_log",
    "request_decision",
    "sign_decision",
    "summarise_review",
    "verify_decision",
]
