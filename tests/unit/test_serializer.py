"""Checkpoint serialisation: msgpack, and the allowlist that makes it work.

Pure in-memory logic — no database, no server — so it runs in the default suite
rather than behind the ``postgres`` marker. That placement matters: a domain
model added to ``schemas.py`` and forgotten in the allowlist fails only when a
run is *parked* on a *durable* store in *production*, and the whole point of a
test is to find it before that.

The tests are here for a second reason. ``Severity.CRITICAL == "critical"`` is
``True`` in Python, so a round-trip that degraded an enum to a plain string
passes every ordinary equality assertion. The type assertions are the ones that
carry the weight, and that is not obvious enough to leave to a reader.
"""

from __future__ import annotations

import pytest

from agentic_workflow.persistence.serializer import allowed_types, build_serializer

pytestmark = pytest.mark.unit


class TestDomainRoundTrip:
    """A resumed run must receive the same object the park produced."""

    def test_a_model_survives_the_round_trip(self) -> None:
        from agentic_workflow.domain.schemas import ReviewRequest, SourceFile

        serializer = build_serializer()
        original = ReviewRequest(
            run_id="round-trip",
            request_id="PR-1",
            title="Round trip a state object",
            description="Every field must survive.",
            files=[SourceFile(path="checkout/total.py", content="def total(): ...\n")],
            acceptance_criteria=["No field is lost."],
        )
        assert serializer.loads_typed(serializer.dumps_typed(original)) == original

    def test_an_enum_keeps_its_type(self) -> None:
        """The assertion an equality check cannot make.

        ``Severity.CRITICAL == "critical"`` is ``True``, so a serializer that
        returned the value as a bare string would pass ``restored == finding``
        and then fail silently in a node that branched on
        ``isinstance(finding.severity, Severity)``.
        """
        from agentic_workflow.domain.schemas import Finding, Severity

        serializer = build_serializer()
        original = Finding(
            id="f1",
            title="Unvalidated query",
            detail="The identifier is interpolated into the statement.",
            severity=Severity.CRITICAL,
            file="checkout/total.py",
            recommendation="Use a bound parameter.",
        )
        restored = serializer.loads_typed(serializer.dumps_typed(original))

        assert restored == original, "the round trip changed the value"
        assert type(restored.severity) is Severity, "the enum degraded to a plain string"

    def test_a_whole_state_mapping_survives(self) -> None:
        """The state is a mapping of many models, not one model.

        Serialising a single ``ReviewRequest`` proves the codec works; it does
        not prove the *state* is serialisable, which is what a checkpoint
        actually stores. A state carrying findings, decisions, timings and a
        report is the real payload.
        """
        from agentic_workflow.domain.state import WorkflowState  # noqa: F401 - import guard

        serializer = build_serializer()
        state: dict[str, object] = {
            "iteration": 2,
            "status": "waiting_human",
            "findings": [
                {"id": "f1", "title": "Float drift", "detail": "…", "severity": "high"},
                {"id": "f2", "title": "Missing test", "detail": "…", "severity": "medium"},
            ],
            "human_decisions": [
                {
                    "approval_id": "apr_x_review_01_deadbeef",
                    "decision": "approve",
                    "reviewer": "dana",
                }
            ],
        }
        assert serializer.loads_typed(serializer.dumps_typed(state)) == state


class TestForwardCompatibility:
    """An old process must be able to read a checkpoint a new one wrote."""

    def test_an_unknown_shape_is_preserved_verbatim(self) -> None:
        """Rolling deploys make this the normal case, not an edge case.

        Raising on a key this build has never seen would strand every in-flight
        run the moment a field was added, so an unrecognised structure is kept
        as-is. The cost is that a typo'd key round-trips happily; the benefit is
        that a deploy cannot lose a run, which is the trade worth making.
        """
        serializer = build_serializer()
        payload = {"a_known_field": 1, "a_field_from_the_future": {"nested": [1, 2]}}
        assert serializer.loads_typed(serializer.dumps_typed(payload)) == payload

    def test_unicode_and_control_characters_survive(self) -> None:
        """Content is arbitrary; a codec that mangles it corrupts a real diff.

        File contents arrive from whatever a user pasted in, so emoji, combining
        marks and literal control characters are all legitimate. Every one of
        them has broken a naive hand-rolled JSON serializer at some point.
        """
        serializer = build_serializer()
        awkward = "café — 🙂 \x00 \\ \" ' \n\ttab\r\nCRLF \u202e<rtl>"
        payload = {"content": awkward}
        assert serializer.loads_typed(serializer.dumps_typed(payload)) == payload


class TestAllowlistCompleteness:
    """The allowlist is what makes the state serialisable at all."""

    def test_every_domain_model_is_registered(self) -> None:
        """A model missing from the allowlist is a run that cannot be saved.

        The allowlist is built by introspection, so adding a model to
        ``domain/schemas.py`` and forgetting it here is an easy and completely
        silent omission — nothing fails until a durable store tries to park a
        run holding that model. Enumerating ``__all__`` makes the check
        automatic instead of a thing to remember.
        """
        import agentic_workflow.domain.schemas as domain

        registered = set(allowed_types())
        for name in domain.__all__:
            candidate = getattr(domain, name, None)
            if isinstance(candidate, type) and hasattr(candidate, "model_fields"):
                assert candidate in registered, f"{name} is not in the serializer allowlist"

    def test_the_allowlist_is_not_empty(self) -> None:
        """A guard on the guard.

        If the introspection ever returns nothing, the model check above would
        compare against an empty set and fail for every model — loud, so it
        cannot actually pass vacuously. Asserting the size anyway documents the
        intent and catches a *partial* failure where the tuple becomes a
        one-element list.
        """
        assert len(allowed_types()) > 5

    def test_a_non_type_is_never_registered(self) -> None:
        """Only real classes belong in a ``(type, ...)`` tuple.

        LangGraph's msgpack allowlist is type-keyed; an entry that is not a type
        would either be ignored or raise deep inside the codec, where the
        message names the encoder rather than the configuration.
        """
        for entry in allowed_types():
            assert isinstance(entry, type), f"{entry!r} is not a type"
