"""Graph construction: the cyclic state machine.

Topology
--------

::

    START -> triage -> programmer -> reviewer -> router -+
                              ^          ^             |  \\
                              |          |             |   +-> apply_patch -> reporter -> END
                              |          |             |
                              |          +-> tester <--+
                              |              (validated, or repairing)
                              +--------------------------------+

The graph is *cyclic* on purpose: the value of a multi-agent system is the
feedback loop, not the single pass. Three mechanisms keep that loop safe:

* :func:`~agentic_workflow.graph.router.decide_route` is a pure function,
  so the control flow is exhaustively testable;
* ``settings.max_iterations`` bounds the number of loops;
* the programmer's stall detector stops proposing patches after a few rounds.

Note the asymmetry in the cycle: ``reviewer → router → programmer`` is the
*rejection* path, while ``router → tester → router`` is the *validation* path. An
approved patch is always validated before it is applied, so nothing irreversible
happens on a reviewer's say-so alone.

Rebuild safety
--------------
``StateGraph.add_node`` keys on the node *name*, and two distinct callables with
the same name collide. Every node therefore carries a unique name, and
:func:`build_graph` is deterministic: the same settings always produce a
structurally identical graph, which the topology tests assert.
"""

from __future__ import annotations

from typing import Any, TypeVar

from agentic_workflow.config import Settings, load_settings
from agentic_workflow.domain.state import WorkflowState
from agentic_workflow.errors import ConfigurationError
from agentic_workflow.graph.context import AgentContext
from agentic_workflow.graph.nodes import (
    APPLY_PATCH,
    PROGRAMMER,
    REPORTER,
    REVIEWER,
    ROUTER,
    TESTER,
    TRIAGE,
    apply_patch_node,
    programmer_node,
    reporter_node,
    reviewer_node,
    tester_node,
    triage_node,
)
from agentic_workflow.graph.router import Route, router_node
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

GraphT = TypeVar("GraphT")

#: Nodes the router may dispatch to. Keeping the whitelist explicit means a typo
#: in :class:`Route` fails at build time instead of producing a graph that hangs.
_ROUTABLE: frozenset[str] = frozenset({TRIAGE, PROGRAMMER, REVIEWER, TESTER, REPORTER, APPLY_PATCH})


def build_graph(
    settings: Settings | None = None,
    *,
    checkpointer: Any = None,
    context: AgentContext | None = None,
    name: str | None = None,
) -> Any:
    """Compile the multi-agent state graph.

    Args:
        settings: Application configuration. Defaults to the process settings.
        checkpointer: A LangGraph checkpointer. **Required** for
            Human-in-the-Loop: without persistence an interrupt cannot be
            resumed, so the builder refuses to compile in that case when HITL is
            enabled.
        context: Default runtime context. Optional — the context may also be
            supplied per-invocation.
        name: Graph name, defaulting to ``settings.graph_name``.

    Returns:
        A compiled ``StateGraph`` ready for ``ainvoke``/``astream``.

    Raises:
        ConfigurationError: If HITL is enabled without a checkpointer.
    """
    from langgraph.graph import END, START, StateGraph

    settings = settings or load_settings()
    graph_name = name or settings.graph_name

    if settings.hitl_enabled and checkpointer is None:
        raise ConfigurationError(
            "human-in-the-loop requires a checkpointer: an interrupt cannot be "
            "resumed without durable state. Pass checkpointer=... or set "
            "AWF_HITL_ENABLED=false for fully autonomous runs.",
            graph=graph_name,
        )

    for route in Route:
        if route is Route.END:
            continue
        if route.value not in _ROUTABLE:
            raise ConfigurationError(
                f"router target {route.value!r} is not a declared node",
                graph=graph_name,
                routable=sorted(_ROUTABLE),
            )

    # `context_schema` is only wired up when there is something to inject. The type
    # parameters are the two shapes this call can actually produce, which keeps the
    # annotation honest without a cast.
    builder: StateGraph[WorkflowState, Any] = StateGraph(
        WorkflowState,
        context_schema=AgentContext if context is not None or settings.hitl_enabled else None,
    )

    # ------------------------------------------------------------ nodes #
    # Destinations are derived from the edges below; declaring them explicitly
    # is redundant and breaks `draw_mermaid()` in LangGraph 1.x.
    builder.add_node(TRIAGE, triage_node)
    builder.add_node(PROGRAMMER, programmer_node)
    builder.add_node(REVIEWER, reviewer_node)
    builder.add_node(TESTER, tester_node)
    builder.add_node(APPLY_PATCH, apply_patch_node)
    builder.add_node(REPORTER, reporter_node)
    builder.add_node(ROUTER, router_node)

    # ------------------------------------------------------------ edges #
    builder.add_edge(START, TRIAGE)
    builder.add_edge(TRIAGE, PROGRAMMER)
    builder.add_edge(PROGRAMMER, REVIEWER)
    builder.add_edge(REVIEWER, ROUTER)
    builder.add_edge(TESTER, ROUTER)
    builder.add_edge(APPLY_PATCH, REPORTER)
    builder.add_edge(REPORTER, END)

    # The single dynamic edge: the router closes the loop. `path_map` is built
    # from the Route enum so a new destination cannot be forgotten.
    builder.add_conditional_edges(
        ROUTER,
        path=_read_next_action,
        path_map={route.value: route.value for route in Route if route is not Route.END},
    )

    compiled = builder.compile(
        checkpointer=checkpointer,
        name=graph_name,
        store=None,
    )
    log.info(
        "graph.compiled",
        graph=graph_name,
        nodes=sorted(_ROUTABLE | {ROUTER}),
        checkpointer=type(checkpointer).__name__ if checkpointer else None,
        hitl=settings.hitl_enabled,
    )
    return compiled


# --------------------------------------------------------------------------- #
# Edge resolvers (must be importable at module level for the graph to pickle)
# --------------------------------------------------------------------------- #
def _read_next_action(state: WorkflowState) -> str:
    """Resolve the router's outgoing edge from ``next_action``.

    Args:
        state: Workflow state after the router ran.

    Returns:
        A node name present in the compiled graph. An unrecognised value falls
        back to the reporter, which is always safe: it terminates the run and
        writes a report rather than looping.
    """
    target = str(state.get("next_action", ""))
    if target in _ROUTABLE:
        return target
    log.warning("router.unknown_target", target=target, falling_back=REPORTER)
    return REPORTER


def graph_topology() -> dict[str, Any]:
    """Describe the graph structure for the documentation generator.

    Returns:
        A JSON-serialisable description of nodes, edges, cycles and the routing
        table. ``tests/unit/test_graph_builder.py`` asserts this stays in sync
        with the compiled graph, so the documentation cannot silently rot.
    """
    from agentic_workflow.graph.router import ROUTING_TABLE

    return {
        "nodes": sorted(_ROUTABLE | {ROUTER}),
        "edges": [
            {"from": "START", "to": TRIAGE, "kind": "static"},
            {"from": TRIAGE, "to": PROGRAMMER, "kind": "static"},
            {"from": PROGRAMMER, "to": REVIEWER, "kind": "static"},
            {"from": REVIEWER, "to": ROUTER, "kind": "static"},
            {"from": TESTER, "to": ROUTER, "kind": "static"},
            {"from": APPLY_PATCH, "to": REPORTER, "kind": "static"},
            {"from": REPORTER, "to": "END", "kind": "static"},
            {"from": ROUTER, "to": "*", "kind": "conditional", "via": "decide_route"},
        ],
        # Closed loops, written as the visited node sequence with the entry node
        # repeated at the end. A cycle that repeats the entry node *mid*-sequence
        # (as in ``[ROUTER, TESTER, ROUTER]``) implies a self-edge that does not
        # exist, so every entry here names each node at most once per lap.
        "cycles": [
            # Rejection path: a reviewer sends the work back to the author.
            [PROGRAMMER, REVIEWER, ROUTER, PROGRAMMER],
            # Validation path: the router dispatches to the tester, which reports
            # back to the router.
            [ROUTER, TESTER, ROUTER],
        ],
        "routing_table": [list(entry) for entry in ROUTING_TABLE],
    }


__all__ = ["build_graph", "graph_topology"]
