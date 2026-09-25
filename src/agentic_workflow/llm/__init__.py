"""LLM abstraction layer.

Everything the agents need from a model is expressed by the :class:`LLMClient`
interface, so a graph never imports a vendor SDK.
"""

from __future__ import annotations

from agentic_workflow.llm.base import (
    Completion,
    LLMClient,
    Message,
    Usage,
    build_llm_client,
)
from agentic_workflow.llm.echo import EchoLLM

__all__ = [
    "Completion",
    "EchoLLM",
    "LLMClient",
    "Message",
    "Usage",
    "build_llm_client",
]
