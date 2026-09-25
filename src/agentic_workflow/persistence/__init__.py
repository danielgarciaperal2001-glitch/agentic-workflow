"""Persistence layer: checkpointers, serialisation policy and run bookkeeping.

Three concerns live here, and they are deliberately kept apart:

* :mod:`~agentic_workflow.persistence.serializer` — *what* may be written to the
  checkpoint store (an explicit allowlist of domain types, no pickle).
* :mod:`~agentic_workflow.persistence.checkpointer` — *where* checkpoints live
  (PostgreSQL in production, memory for tests and the offline demo).
* :mod:`~agentic_workflow.persistence.repository` — the fast, rebuildable
  projection of "which runs exist" used by the control plane.

Together they provide durable resume, time-travel replay and a bounded audit
surface without any of the domain or graph code importing a database driver.
"""

from __future__ import annotations

from agentic_workflow.persistence.checkpointer import (
    PostgresCheckpointer,
    build_checkpointer,
    build_memory_checkpointer,
)
from agentic_workflow.persistence.repository import (
    TERMINAL_STATUSES,
    RunRecord,
    RunRegistry,
    RunStatus,
)
from agentic_workflow.persistence.retention import CheckpointJanitor, RetentionReport
from agentic_workflow.persistence.serializer import allowed_types, build_serializer

__all__ = [
    "TERMINAL_STATUSES",
    "CheckpointJanitor",
    "PostgresCheckpointer",
    "RetentionReport",
    "RunRecord",
    "RunRegistry",
    "RunStatus",
    "allowed_types",
    "build_checkpointer",
    "build_memory_checkpointer",
    "build_serializer",
]
