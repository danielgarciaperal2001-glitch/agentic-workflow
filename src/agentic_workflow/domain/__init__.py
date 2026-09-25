"""Domain layer: typed vocabulary, graph state and run lifecycle enums.

Nothing in this package imports LangGraph, FastAPI or any provider SDK. It is
the stable core of the system: the graph, the API and the evaluation suite all
depend on it, and it depends on nothing but pydantic.
"""

from __future__ import annotations

from agentic_workflow.domain.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    Category,
    Decision,
    Finding,
    FinalReport,
    NodeTiming,
    Patch,
    ReviewRequest,
    ReviewResult,
    RunSummary,
    Severity,
    SourceFile,
    StrictModel,
    TaskBrief,
    TestReport,
    Verdict,
    utcnow,
)
from agentic_workflow.domain.state import (
    RESET,
    WorkflowState,
    append_capped,
    append_unique,
    initial_state,
    replace_or_reset,
)

# Re-exported for the API/evals layers that should not reach into submodules.
__all__ = [
    "RESET",
    "ApprovalDecision",
    "ApprovalRequest",
    "Category",
    "Decision",
    "Finding",
    "FinalReport",
    "NodeTiming",
    "Patch",
    "ReviewRequest",
    "ReviewResult",
    "RunSummary",
    "Severity",
    "SourceFile",
    "StrictModel",
    "TaskBrief",
    "TestReport",
    "Verdict",
    "WorkflowState",
    "append_capped",
    "append_unique",
    "initial_state",
    "replace_or_reset",
    "utcnow",
]
