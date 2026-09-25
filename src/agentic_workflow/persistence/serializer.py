"""Checkpoint serialisation policy.

The graph state contains Pydantic models (``ReviewRequest``, ``Finding``,
``Patch``…). LangGraph's default ``JsonPlusSerializer`` stores them as plain
dicts and warns that a future version will block unregistered types. That
matters here for a concrete reason: after a checkpoint round-trip the state
would come back as ``dict`` and every node that does ``state["request"].run_id``
would fail with an ``AttributeError`` — the kind of bug that only shows up after
a process restart, i.e. in production.

The fix is an explicit allowlist: our own domain types are registered with the
serializer, so they round-trip as the real classes. This also **tightens**
security versus the default, because the deserialiser is limited to a known set
of types instead of importing anything it finds in the database.

Security note
-------------
``JsonPlusSerializer`` is only safe when the checkpoint store is trusted. Anyone
who can write to the checkpoint table can influence what gets deserialised, so
the database must be access-controlled. We never enable ``pickle_fallback``.
"""

from __future__ import annotations

from typing import Any

from agentic_workflow.logging import get_logger

log = get_logger(__name__)

#: Domain types persisted in the graph state. Adding a model to
#: :mod:`agentic_workflow.domain.schemas` means adding it here.
_ALLOWED_TYPES: tuple[type, ...] = ()


def _collect_allowed_types() -> tuple[type, ...]:
    """Enumerate the pydantic models that may appear in the state.

    Returns:
        A tuple of model classes discovered from the domain module. Discovery is
        automatic so a newly added schema is covered without touching this file
        (and :mod:`tests.unit.test_persistence` asserts the round-trip).
    """
    from pydantic import BaseModel

    from agentic_workflow.domain import schemas

    return tuple(
        obj
        for obj in vars(schemas).values()
        if isinstance(obj, type) and issubclass(obj, BaseModel)
    )


def allowed_types() -> tuple[type, ...]:
    """Return the pydantic models allowed to round-trip through checkpoints."""
    return _ALLOWED_TYPES or _collect_allowed_types()


def build_serializer() -> Any:
    """Build a ``JsonPlusSerializer`` restricted to this project's domain types.

    Returns:
        A configured serializer. Builtins that LangGraph already handles
        (``datetime``, ``UUID``, ``Decimal``, collections) remain allowed by the
        underlying implementation.

    Raises:
        RuntimeError: If ``langgraph`` does not ship the JSON+ serializer, which
            would mean an incompatible major version.
    """
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError as exc:  # pragma: no cover - incompatible langgraph
        raise RuntimeError(
            "langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer is unavailable; "
            "the installed langgraph version is not supported by this project"
        ) from exc

    types = allowed_types()
    log.debug("serializer.built", allowed=[t.__name__ for t in types])
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=types,
    )


__all__ = ["allowed_types", "build_serializer"]
