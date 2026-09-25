"""The specialist agents that make up the workflow.

Each node is a *bounded* unit of work with an explicit contract:

* it receives the whole state but returns **only the keys it owns**,
* it is wrapped by :func:`~agentic_workflow.graph.common.node`, so logging,
  timing and timeouts are uniform,
* it must be **idempotent**, because LangGraph re-executes a node from the top
  after a Human-in-the-Loop resume or a time-travel replay,
* when it needs a human it calls
  :func:`~agentic_workflow.human.gates.request_decision` and returns nothing on
  the first pass (the interrupt short-circuits execution).

The set of agents models a real review pipeline::

    triage ──▶ programmer ──▶ reviewer ──▶ tester ──┐
                 ▲                                 │
                 └──────── feedback loop ──────────┘
                              │
                              ▼
                           reporter (terminal)
"""

from __future__ import annotations

from agentic_workflow.graph.nodes.apply_patch import ApplyPatchAgent, apply_patch_node
from agentic_workflow.graph.nodes.base import AgentNode
from agentic_workflow.graph.nodes.programmer import ProgrammerAgent, programmer_node
from agentic_workflow.graph.nodes.reporter import ReporterAgent, reporter_node
from agentic_workflow.graph.nodes.reviewer import ReviewerAgent, reviewer_node
from agentic_workflow.graph.nodes.tester import TesterAgent, tester_node
from agentic_workflow.graph.nodes.triage import TriageAgent, triage_node

#: Canonical node names. Referenced by the router, the API and the diagrams, so
#: they are declared once here.
TRIAGE = "triage"
PROGRAMMER = "programmer"
REVIEWER = "reviewer"
TESTER = "tester"
REPORTER = "reporter"
ROUTER = "router"
APPLY_PATCH = "apply_patch"

ALL_NODES: tuple[str, ...] = (
    TRIAGE,
    PROGRAMMER,
    REVIEWER,
    TESTER,
    REPORTER,
    APPLY_PATCH,
    ROUTER,
)

__all__ = [
    "ALL_NODES",
    "APPLY_PATCH",
    "PROGRAMMER",
    "REPORTER",
    "REVIEWER",
    "ROUTER",
    "TESTER",
    "TRIAGE",
    "AgentNode",
    "ApplyPatchAgent",
    "ProgrammerAgent",
    "ReporterAgent",
    "ReviewerAgent",
    "TesterAgent",
    "TriageAgent",
    "apply_patch_node",
    "programmer_node",
    "reporter_node",
    "reviewer_node",
    "tester_node",
    "triage_node",
]
