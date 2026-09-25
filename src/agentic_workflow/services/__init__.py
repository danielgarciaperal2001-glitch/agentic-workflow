"""Service layer: the engine, the batch runner and the approval inbox.

* :mod:`~agentic_workflow.services.engine` — owns the graph, the checkpointer and
  every run's lifecycle. The only component allowed to invoke the graph.
* :mod:`~agentic_workflow.services.runner` — bounded, failure-isolating batch
  execution on top of the engine.
* :mod:`~agentic_workflow.human.service` — the query/command surface over the
  approvals blocking parked runs, used by the REST and WebSocket control plane.

Layering note: nothing in :mod:`agentic_workflow.graph`,
:mod:`agentic_workflow.domain` or :mod:`agentic_workflow.human` imports from this
package, so the dependency arrow points one way — orchestration knows about
services, never the reverse.
"""

from __future__ import annotations

from agentic_workflow.services.engine import (
    CheckpointInfo,
    Decider,
    RunOutcome,
    WorkflowEngine,
)
from agentic_workflow.services.runner import BatchSummary, run_pipeline, sequential_runner

__all__ = [
    "BatchSummary",
    "CheckpointInfo",
    "Decider",
    "RunOutcome",
    "WorkflowEngine",
    "run_pipeline",
    "sequential_runner",
]
