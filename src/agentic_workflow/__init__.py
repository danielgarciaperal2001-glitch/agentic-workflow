"""Multi-Agent AI Workflow Engine with Human-in-the-Loop.

This package provides a production-grade orchestration layer on top of
LangGraph that adds the pieces a real multi-agent system needs beyond a
prototype:

* a **cyclic state graph** with bounded feedback loops between specialised
  agents (author -> reviewer -> tester -> author),
* **durable persistence** through a PostgreSQL checkpointer, enabling
  crash-safe resumption and checkpoint *time-travel* for debugging,
* **Human-in-the-Loop (HITL)** gates that pause the graph, expose the pending
  decision over REST/WebSocket and resume it with an approve / edit / reject
  verdict,
* an **automated evaluation suite** (native heuristics plus Ragas and DeepEval
  adapters) that gates regressions in answer faithfulness, relevancy and
  recall.

Example
-------
>>> import asyncio
>>> from agentic_workflow import build_graph, create_run_config
>>> from agentic_workflow.domain import ReviewRequest
>>> graph = build_graph()  # doctest: +SKIP
>>> result = await graph.ainvoke(  # doctest: +SKIP
...     ReviewRequest(title="Add rate limiting"), create_run_config("t1")
... )
"""

from __future__ import annotations

from importlib import metadata

try:  # pragma: no cover - trivial packaging shim
    __version__ = metadata.version("agentic-workflow")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.1.0"

__all__ = [
    "WorkflowEngine",
    "__version__",
    "build_graph",
    "create_run_config",
    "load_settings",
    "reset_settings_cache",
    "run_pipeline",
]


def __getattr__(name: str) -> object:
    """Lazily expose the public API.

    Importing :mod:`agentic_workflow` must stay cheap: pulling in the graph
    builder drags LangGraph and the whole settings stack, which is wasteful for
    callers that only need the version string.
    """
    if name == "build_graph":
        from agentic_workflow.graph.builder import build_graph

        return build_graph
    if name == "create_run_config":
        from agentic_workflow.graph.runtime import create_run_config

        return create_run_config
    if name == "load_settings":
        from agentic_workflow.config import load_settings

        return load_settings
    if name == "reset_settings_cache":
        from agentic_workflow.config import reset_settings_cache

        return reset_settings_cache
    if name == "run_pipeline":
        from agentic_workflow.services.runner import run_pipeline

        return run_pipeline
    if name == "WorkflowEngine":
        from agentic_workflow.services.engine import WorkflowEngine

        return WorkflowEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
