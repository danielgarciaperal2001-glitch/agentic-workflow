"""Graph layer: the cyclic state machine and its runtime plumbing.

* :mod:`~agentic_workflow.graph.builder` — compiles the graph.
* :mod:`~agentic_workflow.graph.router` — the pure routing function.
* :mod:`~agentic_workflow.graph.runtime` — invocation config + interrupt decoding.
* :mod:`~agentic_workflow.graph.context` — live dependencies for nodes.
* :mod:`~agentic_workflow.graph.nodes` — the specialised agents.
"""

from __future__ import annotations

from agentic_workflow.graph.builder import build_graph, graph_topology
from agentic_workflow.graph.context import AgentContext, context_from_runtime
from agentic_workflow.graph.router import Route, RoutingDecision, decide_route
from agentic_workflow.graph.runtime import (
    create_run_config,
    extract_interrupts,
    thread_config,
)

__all__ = [
    "AgentContext",
    "Route",
    "RoutingDecision",
    "build_graph",
    "context_from_runtime",
    "create_run_config",
    "decide_route",
    "extract_interrupts",
    "graph_topology",
    "thread_config",
]
