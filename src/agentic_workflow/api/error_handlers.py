"""The single place where domain failures become HTTP responses.

Every error the API returns has the same shape — ``{"error": {...}}`` — and the
same machine-readable ``code``. That is the whole point of
:class:`~agentic_workflow.errors.ErrorCode`: a client branches on
``error.code``, never on the message and never on the status alone, because two
different failures can legitimately share an HTTP status.

The mapping lives here and only here. Domain code raises
:class:`~agentic_workflow.errors.WorkflowError` without ever importing FastAPI, so
the engine stays usable from the CLI, from tests and from a notebook.

Example:
    --------
    >>> status_for(ErrorCode.RUN_NOT_FOUND)
    404
    >>> status_for(ErrorCode.CONCURRENCY_LIMIT)
    429
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agentic_workflow.errors import ErrorCode, WorkflowError
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

#: Domain code -> HTTP status.
#:
#: Two decisions are encoded here and deserve an explanation.
#:
#: * ``waiting_human`` is a *success*, not a failure. A run parked on an approval
#:   is healthy, durable and resumable, so it maps to ``202 Accepted``. Mapping it
#:   to 5xx would teach every client to retry a run that is waiting for a person.
#: * ``ConcurrencyLimitError`` maps to ``429`` rather than ``503``. The server is
#:   fine; the caller arrived while the budget was spent, and the response says
#:   "retry", which is exactly the truth.
#: * ``TokenBudgetExceededError`` maps to ``429`` for the same reason, and the
#:   difference from the concurrency case is in the ``retryable`` flag rather than
#:   the status: the budget is per run, so the answer is to submit less work, not
#:   to come back later. A 5xx here would be worse than merely unhelpful — it
#:   tells a retrying client that the server broke, when the server is precisely
#:   the thing working as configured.
STATUS_MAP: Final[dict[ErrorCode, int]] = {
    # --- validation ------------------------------------------------------- #
    ErrorCode.INVALID_REQUEST: status.HTTP_400_BAD_REQUEST,
    ErrorCode.INVALID_STATE: status.HTTP_409_CONFLICT,
    ErrorCode.CONFIGURATION_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.SCHEMA_VALIDATION_ERROR: status.HTTP_422_UNPROCESSABLE_CONTENT,
    # --- lifecycle -------------------------------------------------------- #
    ErrorCode.RUN_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.RUN_ALREADY_EXISTS: status.HTTP_409_CONFLICT,
    ErrorCode.RUN_CANCELLED: status.HTTP_409_CONFLICT,
    ErrorCode.RUN_TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
    ErrorCode.ITERATION_LIMIT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.GRAPH_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.AGENT_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.CONCURRENCY_LIMIT: status.HTTP_429_TOO_MANY_REQUESTS,
    ErrorCode.TOKEN_BUDGET_EXCEEDED: status.HTTP_429_TOO_MANY_REQUESTS,
    # --- human-in-the-loop ------------------------------------------------ #
    ErrorCode.APPROVAL_REQUIRED: status.HTTP_202_ACCEPTED,
    ErrorCode.APPROVAL_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.APPROVAL_EXPIRED: status.HTTP_409_CONFLICT,
    ErrorCode.APPROVAL_ALREADY_RESOLVED: status.HTTP_409_CONFLICT,
    ErrorCode.APPROVAL_REJECTED: status.HTTP_200_OK,
    # --- persistence ------------------------------------------------------ #
    ErrorCode.PERSISTENCE_ERROR: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.CHECKPOINT_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.CONNECTION_ERROR: status.HTTP_503_SERVICE_UNAVAILABLE,
    # --- provider --------------------------------------------------------- #
    ErrorCode.PROVIDER_ERROR: status.HTTP_502_BAD_GATEWAY,
    ErrorCode.PROVIDER_RATE_LIMITED: status.HTTP_429_TOO_MANY_REQUESTS,
    ErrorCode.PROVIDER_TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
    # --- transport -------------------------------------------------------- #
    ErrorCode.AUTHENTICATION_ERROR: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.AUTHORIZATION_ERROR: status.HTTP_403_FORBIDDEN,
    ErrorCode.RATE_LIMITED: status.HTTP_429_TOO_MANY_REQUESTS,
    ErrorCode.NOT_FOUND: status.HTTP_404_NOT_FOUND,
}

#: Fallback for a code that is not in the map (a new code added to the domain
#: before the transport layer learns about it must not become a 500).
DEFAULT_STATUS: Final[int] = status.HTTP_500_INTERNAL_SERVER_ERROR


def status_for(code: ErrorCode) -> int:
    """Return the HTTP status for a domain error code.

    Args:
        code: The machine-readable code carried by a
            :class:`~agentic_workflow.errors.WorkflowError`.

    Returns:
        The status code. Unmapped codes fall back to ``500`` so an unmapped
        failure is loud rather than silently reported as a client error.

    Example:
        --------
        >>> status_for(ErrorCode.APPROVAL_EXPIRED)
        409
    """
    return STATUS_MAP.get(code, DEFAULT_STATUS)


def error_response(
    request: Request,
    exc: WorkflowError,
    *,
    http_status: int | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Render a :class:`WorkflowError` as the canonical error envelope.

    Args:
        request: The inbound request, used for correlation logging.
        exc: The domain error to render.
        http_status: Override the status derived from ``exc.code``. Used where the
            HTTP contract differs from the default mapping — for example a
            rejected approval, which is a 200 because the *rejection* is the
            successful outcome of a review.
        headers: Response headers the raiser attached to the exception. They are
            kept because the envelope replaces the raiser's own response: a
            ``WWW-Authenticate`` challenge attached to a 401 is the only thing
            that tells a client what to send next, and rebuilding the response
            without it turned a usable refusal into a dead end.

    Returns:
        A :class:`~fastapi.responses.JSONResponse` carrying the envelope and, for
        failures, a ``Retry-After`` hint when the code is retryable.
    """
    status_code = http_status if http_status is not None else status_for(exc.code)
    body = {
        "error": exc.to_dict(),
        "request_id": getattr(request.state, "request_id", None),
    }
    response_headers: dict[str, str] = dict(headers or {})
    if exc.retryable:
        # Advertising `Retry-After` on every retryable error would be noise, but
        # on the ones with a real backoff it lets a generic client behave. The
        # error's own value wins: the rate limiter already knows when its window
        # rolls over, and a client that honours a pessimistic "1" retries early
        # and is throttled again.
        computed = exc.context.get("retry_after_seconds")
        response_headers.setdefault("Retry-After", str(int(float(computed)) + 1 if computed else 1))
    level = log.warning if status_code < 500 else log.error
    level(
        "api.error",
        code=exc.code.value,
        status=status_code,
        run_id=exc.run_id,
        request_id=body["request_id"],
        error=str(exc),
    )
    return JSONResponse(status_code=status_code, content=body, headers=response_headers)


def install_error_handlers(app: FastAPI) -> None:
    """Register every exception handler on *app*.

    Handlers are registered for the concrete :class:`WorkflowError` subclasses as
    well as the base class: Starlette picks the *most specific* registered
    handler, so relying on the base class alone would route an
    :class:`~agentic_workflow.errors.ApprovalRejectedError` (200) through a
    handler that does not know it should answer 200.

    Args:
        app: The application to instrument.
    """
    from agentic_workflow.errors import (  # local import keeps the module graph flat
        ApprovalRejectedError,
    )

    @app.exception_handler(ApprovalRejectedError)
    async def _rejected(request: Request, exc: ApprovalRejectedError) -> JSONResponse:
        # A human saying "no" is a completed review, not a transport failure.
        # Answering 200 here is what lets a UI show "rejected" as an outcome
        # rather than an error toast.
        return error_response(request, exc, http_status=status.HTTP_200_OK)

    @app.exception_handler(WorkflowError)
    async def _workflow(request: Request, exc: WorkflowError) -> JSONResponse:
        return error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            request,
            WorkflowError(
                "request body failed validation",
                code=ErrorCode.INVALID_REQUEST,
                # `errors()` may contain non-serialisable ctx values; the projection
                # keeps the payload JSON-safe for the response body and the log.
                fields=[
                    {
                        "loc": ".".join(str(part) for part in item.get("loc", ())),
                        "msg": str(item.get("msg", "")),
                        "type": str(item.get("type", "")),
                    }
                    for item in exc.errors()
                ],
            ),
            http_status=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail: Any = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            # Already enveloped (e.g. by a router raising 404 deliberately).
            return JSONResponse(status_code=exc.status_code, content=detail, headers=exc.headers)
        code = {
            401: ErrorCode.AUTHENTICATION_ERROR,
            403: ErrorCode.AUTHORIZATION_ERROR,
            404: ErrorCode.NOT_FOUND,
            405: ErrorCode.NOT_FOUND,
            409: ErrorCode.INVALID_STATE,
            429: ErrorCode.RATE_LIMITED,
        }.get(exc.status_code, ErrorCode.INVALID_REQUEST)
        error = WorkflowError(
            str(detail),
            code=code,
            status_code=exc.status_code,
        )
        return error_response(
            request, error, http_status=exc.status_code, headers=exc.headers or {}
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the traceback, return an opaque body. An unexpected exception's
        # message can contain connection strings or source paths, so it must
        # never reach the client; the request id is the bridge between the two.
        log.error(
            "api.unhandled",
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path,
            error=f"{type(exc).__name__}: {exc}",
            exc_info=exc,
        )
        return error_response(
            request,
            WorkflowError("internal server error", code=ErrorCode.GRAPH_ERROR),
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


__all__ = ["DEFAULT_STATUS", "STATUS_MAP", "error_response", "install_error_handlers", "status_for"]
