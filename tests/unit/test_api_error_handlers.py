"""The error taxonomy, and how it becomes HTTP.

Two things are under test here, and they are easy to conflate:

* :class:`~agentic_workflow.errors.ErrorCode` values are **public API**. A client
  branches on them. Renaming one is a breaking change, so the values themselves
  are asserted.
* :func:`~agentic_workflow.api.error_handlers.status_for` is the only place the
  transport decides a status. The mapping is asserted exhaustively, because a
  silent regression there turns a 404 into a 500 and every client's error
  handling breaks at once.
"""

from __future__ import annotations

import pytest

from agentic_workflow.api.error_handlers import DEFAULT_STATUS, STATUS_MAP, status_for
from agentic_workflow.errors import (
    ApprovalAlreadyResolvedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalRejectedError,
    ConcurrencyLimitError,
    ConfigurationError,
    ConnectionFailedError,
    ErrorCode,
    InvalidRequestError,
    InvalidStateError,
    IterationLimitExceededError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    RunAlreadyExistsError,
    RunNotFoundError,
    RunTimeoutError,
    SchemaValidationError,
    WorkflowError,
)

pytestmark = pytest.mark.unit


class TestErrorCodeContract:
    """The wire values clients depend on."""

    def test_codes_are_snake_case_strings(self) -> None:
        """Every code serialises to a stable lowercase identifier."""
        for code in ErrorCode:
            assert code.value == code.value.lower()
            assert " " not in code.value
            assert code.value.replace("_", "").isalnum()

    def test_every_error_class_declares_a_code(self) -> None:
        """Each concrete error overrides the base ``graph_error`` default.

        An error that inherits the base code is a bug: a client cannot
        distinguish it, and the taxonomy is the whole contract.
        """
        subclasses = [
            InvalidRequestError,
            InvalidStateError,
            ConfigurationError,
            RunNotFoundError,
            RunAlreadyExistsError,
            RunTimeoutError,
            IterationLimitExceededError,
            ConcurrencyLimitError,
            ApprovalNotFoundError,
            ApprovalExpiredError,
            ApprovalAlreadyResolvedError,
            ApprovalRejectedError,
            ConnectionFailedError,
            ProviderError,
            ProviderRateLimitedError,
            ProviderTimeoutError,
            SchemaValidationError,
        ]
        for cls in subclasses:
            assert cls.code is not ErrorCode.GRAPH_ERROR, f"{cls.__name__} has no code"

    def test_retryable_flags_are_declared_consistently(self) -> None:
        """Transient failures are retryable; client errors are not."""
        assert RunTimeoutError("t").retryable is True
        assert ConcurrencyLimitError("t").retryable is True
        assert ProviderError("t").retryable is True
        assert RunNotFoundError("t").retryable is False
        assert InvalidRequestError("t").retryable is False


class TestStatusMapping:
    """The code-to-status translation."""

    def test_every_code_is_mapped(self) -> None:
        """No code is left unmapped, which would silently become a 500."""
        unmapped = [code for code in ErrorCode if code not in STATUS_MAP]
        assert unmapped == [], f"unmapped error codes: {unmapped}"

    def test_no_extra_codes_in_the_map(self) -> None:
        """The map carries no entry for a code the domain no longer defines."""
        extra = [code for code in STATUS_MAP if code not in set(ErrorCode)]
        assert extra == []

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (ErrorCode.RUN_NOT_FOUND, 404),
            (ErrorCode.RUN_ALREADY_EXISTS, 409),
            (ErrorCode.INVALID_STATE, 409),
            (ErrorCode.INVALID_REQUEST, 400),
            (ErrorCode.APPROVAL_NOT_FOUND, 404),
            (ErrorCode.APPROVAL_EXPIRED, 409),
            (ErrorCode.APPROVAL_ALREADY_RESOLVED, 409),
            (ErrorCode.CONCURRENCY_LIMIT, 429),
            (ErrorCode.RUN_TIMEOUT, 504),
            (ErrorCode.PERSISTENCE_ERROR, 503),
            (ErrorCode.CONNECTION_ERROR, 503),
            (ErrorCode.PROVIDER_ERROR, 502),
            (ErrorCode.PROVIDER_RATE_LIMITED, 429),
            (ErrorCode.PROVIDER_TIMEOUT, 504),
            (ErrorCode.SCHEMA_VALIDATION_ERROR, 422),
        ],
    )
    def test_known_statuses(self, code: ErrorCode, expected: int) -> None:
        """The documented mapping holds for the codes a client reacts to."""
        assert status_for(code) == expected

    def test_parked_run_is_a_success(self) -> None:
        """``waiting_human`` answers 202, not 5xx.

        A run parked on an approval is healthy and resumable. Reporting it as a
        server error would teach every client to retry a run that is waiting for
        a person, which is precisely the wrong behaviour.
        """
        assert status_for(ErrorCode.APPROVAL_REQUIRED) == 202

    def test_rejection_is_a_completed_review(self) -> None:
        """A human saying "no" is a successful outcome, not a failure."""
        assert status_for(ErrorCode.APPROVAL_REJECTED) == 200

    def test_unknown_code_fails_loudly(self) -> None:
        """A code the transport does not know about becomes a 500.

        Falling back to 200 or 404 would report a failure as a success, which is
        far worse than an unpolished status. The case is simulated by dropping an
        entry from the map, because ``ErrorCode`` is an enum and cannot hold a
        value it does not define.
        """
        assert status_for(ErrorCode.GRAPH_ERROR) == 500
        assert DEFAULT_STATUS == 500
        stripped = dict(STATUS_MAP)
        stripped.pop(ErrorCode.RUN_NOT_FOUND)
        assert stripped.get(ErrorCode.RUN_NOT_FOUND, DEFAULT_STATUS) == 500


class TestErrorRendering:
    """The message and envelope a :class:`WorkflowError` produces."""

    def test_message_includes_code_and_context(self) -> None:
        """The rendered message carries the code, ids and structured context."""
        error = RunNotFoundError("no such run", run_id="r-1", thread_id="t-1", status="running")
        rendered = str(error)
        assert "run_not_found" in rendered
        assert "run_id=r-1" in rendered
        assert "thread_id=t-1" in rendered
        assert "no such run" in rendered

    def test_none_context_is_dropped(self) -> None:
        """``None`` context is omitted rather than rendered as ``None``.

        A log line padded with ``key=None`` costs bytes and hides the values that
        actually matter.
        """
        error = RunNotFoundError("gone", run_id="r", thread_id=None, detail=None)
        assert "None" not in str(error)

    def test_to_dict_matches_the_api_envelope(self) -> None:
        """``to_dict`` is exactly what the error response body carries."""
        error = ConcurrencyLimitError("busy", run_id="r", limit=8)
        body = error.to_dict()
        assert set(body) == {"code", "message", "retryable", "run_id", "thread_id", "context"}
        assert body["code"] == "concurrency_limit"
        assert body["retryable"] is True
        assert body["context"] == {"limit": 8}

    def test_with_context_enriches_in_place(self) -> None:
        """``with_context`` adds fields without re-raising, preserving identity."""
        error = ApprovalExpiredError("late", approval_id="a")
        same = error.with_context(run_id="r-9", ignored=None)
        assert same is error
        assert error.run_id == "r-9"
        assert "ignored" not in error.context

    def test_empty_message_falls_back_to_the_class_docstring(self) -> None:
        """An error raised without a message is still readable."""
        rendered = str(RunNotFoundError())
        assert "run_not_found" in rendered
        assert "RunNotFoundError" in rendered or "No run exists" in rendered

    def test_every_error_is_catchable_as_workflow_error(self) -> None:
        """One ``except WorkflowError`` catches the whole taxonomy."""
        with pytest.raises(WorkflowError):
            raise ApprovalNotFoundError("nope", approval_id="x")
