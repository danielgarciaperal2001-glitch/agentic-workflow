"""Typed error taxonomy for the workflow engine.

Every exception the engine raises on purpose derives from
:class:`WorkflowError` and carries a machine-readable
:class:`~agentic_workflow.errors.ErrorCode`. The control plane maps codes to
HTTP statuses in a single place (:mod:`agentic_workflow.api.error_handlers`),
so domain code never has to know about HTTP.

Example
-------
>>> from agentic_workflow.errors import ApprovalRequired, ErrorCode
>>> err = ApprovalRequired(approval_id="apr_123")
>>> err.code is ErrorCode.APPROVAL_REQUIRED
True
>>> str(err)
"[approval_required] approval_id=apr_123 run_id=None thread_id=None: ..."
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self


class ErrorCode(StrEnum):
    """Stable, machine-readable error identifiers.

    These values are part of the public API contract: clients are expected to
    branch on them, so they must not be renamed without a version bump.
    """

    # --- validation / configuration ------------------------------------- #
    INVALID_REQUEST = "invalid_request"
    INVALID_STATE = "invalid_state"
    CONFIGURATION_ERROR = "configuration_error"

    # --- lifecycle ------------------------------------------------------ #
    RUN_NOT_FOUND = "run_not_found"
    RUN_ALREADY_EXISTS = "run_already_exists"
    RUN_CANCELLED = "run_cancelled"
    RUN_TIMEOUT = "run_timeout"
    ITERATION_LIMIT = "iteration_limit"
    GRAPH_ERROR = "graph_error"
    CONCURRENCY_LIMIT = "concurrency_limit"

    # --- human-in-the-loop ---------------------------------------------- #
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_NOT_FOUND = "approval_not_found"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_ALREADY_RESOLVED = "approval_already_resolved"
    APPROVAL_REJECTED = "approval_rejected"

    # --- persistence ----------------------------------------------------- #
    PERSISTENCE_ERROR = "persistence_error"
    CHECKPOINT_NOT_FOUND = "checkpoint_not_found"
    CONNECTION_ERROR = "connection_error"

    # --- provider / agent ------------------------------------------------ #
    PROVIDER_ERROR = "provider_error"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_TIMEOUT = "provider_timeout"
    AGENT_ERROR = "agent_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"

    # --- transport ------------------------------------------------------- #
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"


class WorkflowError(Exception):
    """Base class for every deliberate engine failure.

    Attributes
    ----------
    code:
        Machine-readable :class:`ErrorCode`.
    context:
        Structured, log-safe key/value pairs attached to the error. They are
        merged into the API error body and into structured log records, which
        makes production triage dramatically cheaper than parsing log lines.
    retryable:
        Advisory flag telling callers (and the run scheduler) whether a retry
        could plausibly succeed.
    """

    code: ErrorCode = ErrorCode.GRAPH_ERROR
    retryable: bool = False

    def __init__(
        self,
        message: str = "",
        /,
        *,
        code: ErrorCode | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        **context: Any,
    ) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.context: dict[str, Any] = {k: v for k, v in context.items() if v is not None}
        if code is not None:
            self.code = code

        super().__init__(self._render(message))

    # -- rendering -------------------------------------------------------- #
    def _render(self, message: str) -> str:
        parts = [f"[{self.code.value}]"]
        if self.run_id:
            parts.append(f"run_id={self.run_id}")
        if self.thread_id:
            parts.append(f"thread_id={self.thread_id}")
        for key, value in self.context.items():
            parts.append(f"{key}={value!r}")
        parts.append(message or self.__class__.__doc__ or self.__class__.__name__)
        return " ".join(parts)

    def with_context(self, **context: Any) -> Self:
        """Attach additional context in place and return ``self``.

        Lets low-level code raise a bare error and let an outer layer enrich it
        with correlation identifiers without re-raising.
        """
        self.context.update({k: v for k, v in context.items() if v is not None})
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the API error envelope."""
        return {
            "code": self.code.value,
            "message": str(self),
            "retryable": self.retryable,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "context": self.context,
        }


# --------------------------------------------------------------------------- #
# Validation & configuration
# --------------------------------------------------------------------------- #
class InvalidRequestError(WorkflowError):
    """The caller supplied a malformed or semantically invalid payload."""

    code = ErrorCode.INVALID_REQUEST


class InvalidStateError(WorkflowError):
    """The graph state is inconsistent with the requested operation."""

    code = ErrorCode.INVALID_STATE


class ConfigurationError(WorkflowError):
    """The runtime configuration is missing or internally inconsistent."""

    code = ErrorCode.CONFIGURATION_ERROR


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
class RunNotFoundError(WorkflowError):
    """No run exists for the supplied identifier."""

    code = ErrorCode.RUN_NOT_FOUND


class RunAlreadyExistsError(WorkflowError):
    """A run with the same identifier is already registered."""

    code = ErrorCode.RUN_ALREADY_EXISTS


class RunCancelledError(WorkflowError):
    """The run was cancelled by an operator."""

    code = ErrorCode.RUN_CANCELLED


class RunTimeoutError(WorkflowError):
    """The run exceeded its configured wall-clock budget."""

    code = ErrorCode.RUN_TIMEOUT
    retryable = True


class IterationLimitExceeded(WorkflowError):
    """The agent feedback loop hit its maximum number of iterations."""

    code = ErrorCode.ITERATION_LIMIT


class GraphExecutionError(WorkflowError):
    """The LangGraph execution raised an unrecoverable error."""

    code = ErrorCode.GRAPH_ERROR


class ConcurrencyLimitError(WorkflowError):
    """The engine refused the run because the concurrency budget is exhausted."""

    code = ErrorCode.CONCURRENCY_LIMIT
    retryable = True


# --------------------------------------------------------------------------- #
# Human-in-the-Loop
# --------------------------------------------------------------------------- #
class ApprovalRequiredError(WorkflowError):
    """Execution is paused and awaiting a human decision."""

    code = ErrorCode.APPROVAL_REQUIRED
    # A paused run is *not* a failure: the client resolves it and resumes.
    retryable = True


class ApprovalNotFoundError(WorkflowError):
    """The referenced approval request does not exist."""

    code = ErrorCode.APPROVAL_NOT_FOUND


class ApprovalExpiredError(WorkflowError):
    """The approval window elapsed before a human responded."""

    code = ErrorCode.APPROVAL_EXPIRED


class ApprovalAlreadyResolvedError(WorkflowError):
    """A human already decided this approval request."""

    code = ErrorCode.APPROVAL_ALREADY_RESOLVED


class ApprovalRejectedError(WorkflowError):
    """A human explicitly rejected the proposed action."""

    code = ErrorCode.APPROVAL_REJECTED


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class PersistenceError(WorkflowError):
    """The checkpoint store could not satisfy the request."""

    code = ErrorCode.PERSISTENCE_ERROR


class CheckpointNotFoundError(WorkflowError):
    """The requested checkpoint does not exist in the store."""

    code = ErrorCode.CHECKPOINT_NOT_FOUND


class ConnectionFailedError(WorkflowError):
    """The database connection pool could not be established."""

    code = ErrorCode.CONNECTION_ERROR
    retryable = True


# --------------------------------------------------------------------------- #
# Providers & agents
# --------------------------------------------------------------------------- #
class ProviderError(WorkflowError):
    """An LLM provider returned an error or malformed output."""

    code = ErrorCode.PROVIDER_ERROR
    retryable = True


class ProviderRateLimited(ProviderError):
    """The LLM provider rejected the request due to rate limiting."""

    code = ErrorCode.PROVIDER_RATE_LIMITED
    retryable = True


class ProviderTimeout(ProviderError):
    """The LLM provider did not answer within the deadline."""

    code = ErrorCode.PROVIDER_TIMEOUT
    retryable = True


class AgentError(WorkflowError):
    """A node failed for a domain reason unrelated to the provider."""

    code = ErrorCode.AGENT_ERROR


class SchemaValidationError(WorkflowError):
    """Structured agent output did not satisfy its declared schema."""

    code = ErrorCode.SCHEMA_VALIDATION_ERROR


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
class AuthenticationError(WorkflowError):
    """The caller did not present valid credentials."""

    code = ErrorCode.AUTHENTICATION_ERROR


class AuthorizationError(WorkflowError):
    """The caller is authenticated but lacks permission for this resource."""

    code = ErrorCode.AUTHORIZATION_ERROR


class RateLimitedError(WorkflowError):
    """The caller exceeded the configured request budget."""

    code = ErrorCode.RATE_LIMITED
    retryable = True


__all__ = [
    "AgentError",
    "ApprovalAlreadyResolvedError",
    "ApprovalExpiredError",
    "ApprovalNotFoundError",
    "ApprovalRejectedError",
    "ApprovalRequiredError",
    "AuthenticationError",
    "AuthorizationError",
    "CheckpointNotFoundError",
    "ConcurrencyLimitError",
    "ConfigurationError",
    "ConnectionFailedError",
    "ErrorCode",
    "GraphExecutionError",
    "InvalidRequestError",
    "InvalidStateError",
    "IterationLimitExceeded",
    "PersistenceError",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderTimeout",
    "RateLimitedError",
    "RunAlreadyExistsError",
    "RunCancelledError",
    "RunNotFoundError",
    "RunTimeoutError",
    "SchemaValidationError",
    "WorkflowError",
]
