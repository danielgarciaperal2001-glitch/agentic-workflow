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

#: Explicit allowlist. Empty means "discover from the domain module", which is the
#: normal path; a deployment can pin the exact tuple to make the set reviewable
#: and auditable in a security review.
_ALLOWED_TYPES: tuple[type, ...] = ()


def _collect_allowed_types() -> tuple[type, ...]:
    """Enumerate every type that may appear in the state.

    Models are not enough. A ``ReviewResult`` contains ``Finding`` objects whose
    ``severity`` is a ``Severity`` enum, and msgpack serialises that as a bare
    string tagged with its enum class. Deserialising it needs the *enum* in the
    allowlist too — leave it out and LangGraph blocks the value, logs
    "Blocked deserialization", and hands the node a plain string where it
    expected a ``Severity``. That is exactly the silent degradation this module
    exists to prevent, so enums are collected alongside the models.

    Returns:
        A tuple of types discovered from the domain module. Discovery is automatic
        so a newly added schema is covered without touching this file (and
        :mod:`tests.unit.test_persistence` asserts the round-trip).
    """
    from enum import Enum

    from pydantic import BaseModel

    from agentic_workflow.domain import schemas

    found: list[type] = []
    for obj in vars(schemas).values():
        if not isinstance(obj, type):
            continue
        if issubclass(obj, (BaseModel, Enum)):
            found.append(obj)
    return tuple(found)


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
