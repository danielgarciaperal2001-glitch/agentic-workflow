"""The gate: identity, signing, coercion and the audit record.

Human-in-the-loop is only worth the interruption it causes if the answer it
collects is trustworthy afterwards. These tests concentrate on the four things
that make an answer trustworthy — the gate can be *identified*, its decision can
be *attributed*, its content cannot be *silently altered*, and it is *recorded* —
plus the coercion layer, which is the seam where a hostile or clumsy client
payload becomes an authoritative action.
"""

from __future__ import annotations

from datetime import timedelta
import json

import pytest

from agentic_workflow.domain.schemas import ApprovalDecision, Decision
from agentic_workflow.errors import ApprovalRejectedError, InvalidStateError, WorkflowError
from agentic_workflow.human.gates import (
    PAYLOAD_LOG_LIMIT,
    _bounded_payload,
    _coerce_decision,
    approval_id_for,
    decode_interrupt,
    encode_interrupt,
    record_decision_log,
    run_id_from_approval_id,
    sign_decision,
    verify_decision,
)
from agentic_workflow.human.policy import Stage
from tests.helpers import make_approval, make_decision

pytestmark = pytest.mark.unit


class TestGateIdentity:
    """``approval_id`` is derived, not generated, and that is the point."""

    def test_the_same_coordinates_give_the_same_id(self) -> None:
        """LangGraph re-runs a node after an interrupt, building the request twice.

        A random id would differ between the two passes, so the answer the human
        was shown would no longer match the gate asking for it, and the
        stale-answer guard would refuse every legitimate decision.
        """
        first = approval_id_for("run-1", Stage.PATCH_APPLY, 2)

        assert first == approval_id_for("run-1", Stage.PATCH_APPLY, 2)

    def test_each_coordinate_moves_the_id(self) -> None:
        """A different gate must be a different id, or a stale tab applies.

        Three coordinates identify a gate, so all three have to participate: two
        runs must not collide, and neither must two stages of one run nor two
        iterations of one stage.
        """
        base = approval_id_for("run-1", Stage.PATCH_APPLY, 1)

        assert approval_id_for("run-2", Stage.PATCH_APPLY, 1) != base
        assert approval_id_for("run-1", Stage.PATCH_REVIEW, 1) != base
        assert approval_id_for("run-1", Stage.PATCH_APPLY, 2) != base

    def test_the_id_is_readable_in_a_log_line(self) -> None:
        """The run, the stage and the iteration are legible without a lookup.

        An operator reading a log at 3am should be able to tell which gate of
        which run stalled without correlating against a table.
        """
        identifier = approval_id_for("run-7", Stage.TEST_REVIEW, 3)

        assert identifier.startswith("apr_run-7_test_review_03_")

    def test_a_hostile_run_id_cannot_forge_another_gates_identity(self) -> None:
        """Run ids come from clients; the id must not be a concatenation puzzle."""
        forged = approval_id_for("run-1_patch_apply_01_deadbeef", Stage.PATCH_REVIEW, 0)

        assert forged != approval_id_for("run-1", Stage.PATCH_APPLY, 1)


class TestApprovalIdInversion:
    """The id encodes the run, which is what makes the hot path cheap.

    Locating a gate's run by scanning every run's decision log is O(runs) on the
    path every human decision takes. Parsing it out is O(1) — but only if the
    inverse is exact, so these tests pin both directions.
    """

    @pytest.mark.parametrize("stage", list(Stage))
    def test_every_stage_round_trips(self, stage: Stage) -> None:
        identifier = approval_id_for("run-77", stage, 3)

        assert run_id_from_approval_id(identifier) == "run-77"

    def test_a_dotted_or_dashed_run_id_round_trips(self) -> None:
        """The wire alphabet is ``[A-Za-z0-9._:-]``, so these are legal run ids."""
        for run_id in ("acme.checkout-42", "req:1.2.3", "a", "0"):
            identifier = approval_id_for(run_id, Stage.FINAL_REPORT, 0)

            assert run_id_from_approval_id(identifier) == run_id

    def test_a_run_id_with_a_digit_prefix_is_not_mistaken_for_a_stage(self) -> None:
        """Parsing walks in from the right, so leading digits are irrelevant."""
        identifier = approval_id_for("2024-final", Stage.PATCH_APPLY, 1)

        assert run_id_from_approval_id(identifier) == "2024-final"

    def test_a_run_id_named_after_a_stage_still_round_trips(self) -> None:
        """``test_review`` as a run id is legal and must not be truncated away."""
        identifier = approval_id_for("test_review", Stage.PATCH_APPLY, 1)

        assert run_id_from_approval_id(identifier) == "test_review"

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "approve",
            "apr_",
            "apr_no_stage",
            "apr_run-1_patch_apply_deadbeef",  # no iteration
            "apr_run-1_patch_apply_xx_deadbeef",  # iteration is not numeric
            "apr_run-1_unknown_stage_01_deadbeef",
        ],
    )
    def test_an_unparseable_id_yields_none(self, value: str) -> None:
        """``None`` means "ask the store", never "no such run".

        Guessing a run id from a malformed approval would let a client steer a
        read at an arbitrary thread, so an unrecognised value must fall through
        to the authoritative lookup.
        """
        assert run_id_from_approval_id(value) is None

    def test_a_hostile_run_id_does_not_resolve_to_another_runs_name(self) -> None:
        """The inversion must not be a way to redirect a lookup."""
        identifier = approval_id_for("victim", Stage.PATCH_APPLY, 1)

        assert run_id_from_approval_id(identifier) == "victim" != "attacker"


class TestInterruptEncoding:
    """The value carried across the interrupt boundary must survive intact."""

    def test_a_request_round_trips(self) -> None:
        """A truncated request is a gate the human cannot judge.

        The approval id, the offered options and the diff are the three things a
        reviewer needs; losing any of them turns the gate into a dialog with no
        information in it.
        """
        request = make_approval()

        payload = encode_interrupt(request)
        decoded, extra = decode_interrupt(payload)

        assert decoded.approval_id == request.approval_id
        assert decoded.stage == request.stage
        assert decoded.options == request.options
        assert decoded.diff_preview == request.diff_preview
        assert isinstance(extra, dict)

    def test_a_garbled_payload_is_refused_rather_than_guessed(self) -> None:
        """Decoding into a half-built request would gate on the wrong action."""
        with pytest.raises(WorkflowError):
            decode_interrupt({"totally": "wrong"})

    def test_a_client_supplied_id_is_preserved_for_the_stale_guard(self) -> None:
        """Coercion must not overwrite an id the client chose to send.

        The stale-answer check in :func:`request_decision` compares the id on the
        decision against the id on the gate. Filling in a default here would
        destroy that comparison and let a browser tab answering a superseded
        gate apply its answer to the new one.
        """
        request = make_approval()
        decision = _coerce_decision(
            {"approval_id": "apr_run-1_patch_apply_09_stale", "decision": "approve"},
            request,
            "api",
        )

        assert decision.approval_id == "apr_run-1_patch_apply_09_stale"


class TestSigning:
    """A signature nobody verifies is decoration; these are the properties."""

    def test_a_signed_decision_verifies(self) -> None:
        decision = make_decision("apr-1", reviewer="alice")
        signed = decision.model_copy(update={"signature": sign_decision(decision, "s3cret")})

        assert verify_decision(signed, "s3cret") is True

    def test_a_different_secret_does_not_verify(self) -> None:
        """Otherwise any holder of one secret could forge another's approvals."""
        decision = make_decision("apr-1")
        signed = decision.model_copy(update={"signature": sign_decision(decision, "theirs")})

        assert verify_decision(signed, "ours") is False

    def test_tampering_with_the_verdict_breaks_the_signature(self) -> None:
        """The signature must cover the decision, not merely exist.

        Rewriting an approval into a rejection after the fact is the exact
        forgery the audit log exists to make detectable, so a signature that
        ignored the payload would defeat it entirely.
        """
        decision = make_decision("apr-1", decision=Decision.APPROVE)
        signature = sign_decision(decision, "s3cret")
        tampered = decision.model_copy(update={"decision": Decision.REJECT, "signature": signature})

        assert verify_decision(tampered, "s3cret") is False

    def test_tampering_with_the_author_breaks_the_signature(self) -> None:
        """Re-attributing a decision to a colleague must be detectable too."""
        decision = make_decision("apr-1", reviewer="alice")
        signature = sign_decision(decision, "s3cret")
        tampered = decision.model_copy(update={"reviewer": "bob", "signature": signature})

        assert verify_decision(tampered, "s3cret") is False

    def test_an_unsigned_decision_never_verifies(self) -> None:
        """``False``, not an exception: absence of a signature is a verdict."""
        assert verify_decision(make_decision("apr-1"), "s3cret") is False

    def test_signing_without_a_secret_is_a_programming_error(self) -> None:
        """An empty secret would sign everything with a constant HMAC key."""
        with pytest.raises(ValueError, match="non-empty secret"):
            sign_decision(make_decision("apr-1"), "")


class TestDecisionCoercion:
    """Clients send whatever they send; the gate must survive that."""

    @pytest.mark.parametrize("raw", ["approve", "APPROVE", "approve "])
    def test_a_bare_string_is_accepted(self, raw: str) -> None:
        """The cheapest possible client should not be the one that fails."""
        decision = _coerce_decision(raw, make_approval(), "api")

        assert decision.decision is Decision.APPROVE

    def test_a_string_uses_the_default_reviewer(self) -> None:
        """An answer with no author still has to be attributable to something."""
        decision = _coerce_decision("approve", make_approval(), "kiosk")

        assert decision.reviewer == "kiosk"

    def test_a_bare_mapping_inherits_the_gate_id(self) -> None:
        """``{"decision": "approve"}`` from a curl call must work.

        This is what the quickstart sends. Requiring the client to echo back a
        derived identifier it had no way of knowing would make the documented
        path the broken one.
        """
        request = make_approval()
        decision = _coerce_decision({"decision": "approve"}, request, "api")

        assert decision.approval_id == request.approval_id

    def test_langgraphs_index_wrapper_is_unwrapped(self) -> None:
        """LangGraph may deliver ``{"value": ...}``; unwrap it rather than fail."""
        decision = _coerce_decision({"value": {"decision": "reject"}}, make_approval(), "api")

        assert decision.decision is Decision.REJECT

    def test_a_list_of_interrupts_uses_the_first_entry(self) -> None:
        """Several interrupts raised in one super-step arrive as a list."""
        decision = _coerce_decision(["approve", "reject"], make_approval(), "api")

        assert decision.decision is Decision.APPROVE

    def test_an_uninterpretable_value_is_refused(self) -> None:
        """A gate must fail closed: guessing "approve" is the worst option."""
        with pytest.raises(InvalidStateError):
            _coerce_decision(object(), make_approval(), "api")

    def test_an_empty_list_is_refused(self) -> None:
        """An empty resume means nobody answered, which is not an approval."""
        with pytest.raises(InvalidStateError, match="empty resume value"):
            _coerce_decision([], make_approval(), "api")

    def test_a_nonsense_decision_string_is_refused(self) -> None:
        """``"maybe"`` is not a verdict, and must not be treated as one."""
        with pytest.raises(InvalidStateError):
            _coerce_decision("maybe", make_approval(), "api")

    def test_a_malformed_mapping_is_refused(self) -> None:
        """Validation failures become typed errors, not pydantic tracebacks."""
        with pytest.raises(InvalidStateError, match="malformed decision payload"):
            _coerce_decision(
                {"decision": "approve", "reviewer": 42},
                make_approval(),
                "api",
            )


class TestAuditRecord:
    """What a decision leaves behind, which is the reason for asking."""

    def test_the_record_identifies_who_what_and_when(self) -> None:
        """A record missing any of the three cannot answer a later question."""
        decision = make_decision("apr-1", decision=Decision.APPROVE, reviewer="alice")
        entry = record_decision_log(decision, run_id="run-1", stage="patch_apply")

        assert entry["approval_id"] == "apr-1"
        assert entry["run_id"] == "run-1"
        assert entry["stage"] == "patch_apply"
        assert entry["decision"] == "approve"
        assert entry["reviewer"] == "alice"
        assert entry["decided_at"] == decision.decided_at.isoformat()

    def test_response_latency_is_measured(self) -> None:
        """A gate nobody answers is a broken deployment, and this is the signal.

        Without the interval there is no way to tell an operator the workflow
        has been sitting on an approval for six hours, which is exactly the kind
        of stall that accumulates silently.
        """
        decision = make_decision("apr-1")
        entry = record_decision_log(
            decision,
            run_id="run-1",
            stage="patch_apply",
            created_at=decision.decided_at - timedelta(seconds=90),
        )

        assert entry["latency_seconds"] == pytest.approx(90.0)

    def test_latency_is_never_negative(self) -> None:
        """Clock skew between raiser and answerer must not surface as ``-5s``."""
        decision = make_decision("apr-1")
        entry = record_decision_log(
            decision,
            run_id="run-1",
            stage="patch_apply",
            created_at=decision.decided_at + timedelta(seconds=5),
        )

        assert entry["latency_seconds"] == 0.0

    def test_an_edit_keeps_what_the_human_authored(self) -> None:
        """The payload *is* the human's contribution.

        An audit trail recording "alice edited" without recording what she
        changed answers "was a human involved?" and not "what did they author?" —
        which is the only reason to keep an audit trail at all.
        """
        decision = make_decision(
            "apr-1",
            decision=Decision.EDIT,
            payload={"instruction": "use Decimal", "touched_by": "alice"},
        )
        entry = record_decision_log(decision, run_id="run-1", stage="patch_review")

        assert entry["payload"] == {"instruction": "use Decimal", "touched_by": "alice"}

    def test_an_approval_carries_no_payload_key(self) -> None:
        """A null payload would suggest the human supplied one and it was lost."""
        entry = record_decision_log(make_decision("apr-1"), run_id="run-1", stage="patch_apply")

        assert "payload" not in entry

    def test_a_huge_payload_is_bounded(self) -> None:
        """The log lives in every checkpoint, so it cannot grow without limit.

        A run's checkpoint history is what a UI pages through; one human pasting
        a 200 KB diff would make every later page slow for every later reader.
        """
        decision = make_decision(
            "apr-1",
            decision=Decision.EDIT,
            payload={"instruction": "x" * 50_000},
        )
        entry = record_decision_log(decision, run_id="run-1", stage="patch_apply")
        stored = entry["payload"]["instruction"]

        assert len(stored) <= PAYLOAD_LOG_LIMIT + 32
        assert stored.endswith("[truncated]")
        assert stored.startswith("x" * 100)

    def test_an_oversized_structured_value_is_serialised_before_truncating(self) -> None:
        """Cutting a dict mid-key would produce something that is no longer valid."""
        decision = make_decision(
            "apr-1",
            decision=Decision.EDIT,
            payload={"findings": [{"id": i, "text": "y" * 100} for i in range(400)]},
        )
        entry = record_decision_log(decision, run_id="run-1", stage="patch_apply")

        assert len(json.dumps(entry)) < PAYLOAD_LOG_LIMIT * 3

    def test_a_bounded_payload_stays_json_serialisable(self) -> None:
        """The record is written to a checkpoint, which is JSON all the way down."""
        decision = make_decision(
            "apr-1",
            decision=Decision.EDIT,
            payload={"nested": {"k": [1, 2]}, "n": 1, "flag": True},
        )
        entry = record_decision_log(decision, run_id="run-1", stage="patch_apply")

        assert json.loads(json.dumps(entry))["payload"]["n"] == 1

    def test_bounding_preserves_scalar_types(self) -> None:
        """A count must not come back from the log as ``"3"``.

        Only strings are truncated. Coercing a number into text to make room
        would quietly break every downstream aggregation over the audit log.
        """
        bounded = _bounded_payload({"count": 3, "ratio": 0.5, "ok": False, "none": None})

        assert bounded == {"count": 3, "ratio": 0.5, "ok": False, "none": None}

    def test_a_short_structured_value_is_kept_intact(self) -> None:
        """Only oversized values are rewritten; the common case is a no-op."""
        payload = {"files": ["a.py", "b.py"], "lines": [1, 2, 3]}

        assert _bounded_payload(payload) == payload


class TestRejectionCarriesItsDecision:
    """A rejection that cannot be attributed cannot be recorded."""

    def test_the_decision_travels_with_the_error(self) -> None:
        """The node that catches the rejection still owes the log an entry.

        An exception carrying only a message would force the caller to
        reconstruct who rejected and when — and the one decision that ends a run
        is the one most likely to end up unrecorded as a result.
        """
        decision = make_decision("apr-1", decision=Decision.REJECT, reviewer="bob", comment="no")
        error = ApprovalRejectedError("no", decision=decision, stage="patch_apply")

        assert error.decision is decision
        assert error.stage == "patch_apply"
        assert error.reviewer == "bob"
        assert error.approval_id == "apr-1"

    def test_a_rejection_without_a_decision_still_renders(self) -> None:
        """Defensive: the error must carry its context even with nothing attached."""
        error = ApprovalRejectedError("human said no", reviewer="bob")

        assert error.decision is None
        assert error.reviewer == "bob"
        assert "human said no" in str(error)

    def test_an_attached_decision_is_never_in_the_audit_envelope(self) -> None:
        """The decision may carry arbitrary payloads; the error body may not.

        ``WorkflowError``'s context is merged verbatim into the HTTP error
        envelope, so leaving the decision in there would serialise a whole
        pydantic model into every rejection response.
        """
        decision: ApprovalDecision = make_decision(
            "apr-1",
            decision=Decision.REJECT,
            payload={"comment": "a very long human note " * 50},
        )
        error = ApprovalRejectedError("no", decision=decision, reviewer="bob")

        assert "decision" not in error.context
        assert "stage" not in error.context or error.stage is None
        assert len(str(error)) < 400
