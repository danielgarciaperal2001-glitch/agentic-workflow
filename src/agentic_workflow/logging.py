"""Structured logging built on ``structlog``.

Every log record in the system carries the same envelope (``run_id``,
``thread_id``, ``node``, ``event``) so a single index query can reconstruct a
complete execution trace. :func:`bind_context` is the canonical way to attach
that context, and the middleware / node decorators use it automatically.

Example
-------
>>> from agentic_workflow.logging import bind_context, get_logger
>>> log = get_logger(__name__).bind()
>>> with bind_context(run_id="run_1", node="reviewer"):
...     log.info("review completed", findings=0)
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import logging
import sys
from typing import Any

import structlog

from agentic_workflow.config import LogFormat, Settings, load_settings

# Context keys merged into every record emitted inside the block.
CONTEXT_KEYS = ("run_id", "thread_id", "node", "agent", "event_id", "trace_id")

_configured = False


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Install the structlog + stdlib logging configuration.

    Safe to call repeatedly; subsequent calls are no-ops unless ``force`` is
    set. This lets the CLI configure logging eagerly while FastAPI can call it
    again during lifespan startup.

    Args:
        settings: Configuration to apply. Defaults to the process-wide settings.
        force: Reconfigure even if logging was already set up.
    """
    global _configured
    if _configured and not force:
        return

    settings = settings or load_settings()
    log_level = getattr(logging, settings.log_level, logging.INFO)

    # Keep third-party chatter out of our structured stream.
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "langchain_core", "openai"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.ExceptionPrettyPrinter()
            if settings.log_format is LogFormat.CONSOLE
            else structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(sort_keys=True)
            if settings.log_format is LogFormat.JSON
            else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )

    # stderr, not stdout. The CLI's *result* goes to stdout — a JSON document
    # from `awf eval --json`, a table from `awf janitor` — and a log line
    # interleaved with it makes that output unparseable. `awf eval --json | jq`
    # has to work, so diagnostics take the other stream. This also matches the
    # colour decision above, which already asks whether *stderr* is a terminal.
    #
    # Under a container runtime both streams are captured, so this costs nothing
    # for a server deployment and is the difference between a usable and an
    # unusable CLI pipeline.
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    _configured = True


def get_logger(name: str | None = None, **initial: Any) -> structlog.stdlib.BoundLogger:
    """Return a pre-bound structlog logger.

    Args:
        name: Dotted module path, conventionally ``__name__``.
        **initial: Values bound to every record from this logger.

    Returns:
        A bound logger that renders the shared context envelope.
    """
    if not _configured:
        configure_logging()
    return structlog.stdlib.get_logger(name, **initial)


@contextmanager
def bind_context(**context: Any) -> Iterator[None]:
    """Temporarily attach correlation context to every log record in scope.

    Context is stored in ``contextvars`` so it propagates correctly across
    ``await`` boundaries and into background tasks without threading arguments
    through every function signature.

    Args:
        **context: Key/value pairs. Unknown keys are still forwarded, which
            keeps the helper usable for request-scoped metadata.

    Yields:
        ``None``; the context is popped on exit.

    Example:
        --------
        >>> with bind_context(run_id="run_42", node="tester"):
        ...     get_logger(__name__).info("tests executed")
    """
    tokens: dict[str, Any] = {}
    for key, value in context.items():
        if value is not None:
            tokens.update(structlog.contextvars.bind_contextvars(**{key: value}))
    try:
        yield
    finally:
        # `bind_contextvars` returns one token per key; resetting them in a
        # single call unwinds the whole scope, including nested re-bindings of
        # the same key.
        structlog.contextvars.reset_contextvars(**tokens)


def current_context() -> dict[str, Any]:
    """Return the correlation context bound to the current task."""
    ctx: Mapping[str, Any] = structlog.contextvars.get_contextvars()
    return dict(ctx)


def scrub(
    payload: Mapping[str, Any], *, redact_keys: frozenset[str] | None = None
) -> dict[str, Any]:
    """Redact secret-looking keys from a mapping before logging it.

    Args:
        payload: Arbitrary log payload.
        redact_keys: Extra keys to mask. Defaults to common secret names.

    Returns:
        A copy with sensitive values replaced by ``***``.
    """
    defaults = {
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "password",
        "secret",
        "token",
        "llm_api_key",
    }
    keys = frozenset(defaults | frozenset(redact_keys or frozenset()))
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in keys:
            out[key] = "***"
        elif isinstance(value, Mapping):
            out[key] = scrub(value, redact_keys=keys)
        else:
            out[key] = value
    return out


__all__ = [
    "bind_context",
    "configure_logging",
    "current_context",
    "get_logger",
    "scrub",
]
