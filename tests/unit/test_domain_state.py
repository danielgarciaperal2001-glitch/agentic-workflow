"""The domain layer: schemas, state reducers and initial state.

The domain layer is the one part of the project with no dependencies beyond
pydantic, which is exactly why it is worth testing exhaustively: everything else
is downstream of it, and a bug here surfaces as a mysteriously bad workflow hours
later.

The interesting parts are the **reducers**. LangGraph merges every node's return
value into the state through a per-channel reducer, and the obvious
implementations are all wrong:

* plain assignment loses the transcript, because several nodes write it per
  super-step;
* plain concatenation duplicates the approval log, because a node re-executed
  after an interrupt appends the same decision again;
* and a reducer that mutates the list it is handed corrupts the value already
  written to the checkpoint, which is not observable until a replay reads back
  state that no longer matches what ran.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

import pytest

from agentic_workflow.domain.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    Category,
    Decision,
    FinalReport,
    Finding,
    NodeTiming,
    Patch,
    ReviewRequest,
    ReviewResult,
    RunSummary,
    Severity,
    SourceFile,
    TaskBrief,
    TestReport as ReportModel,
    utcnow,
)
from agentic_workflow.domain.state import (
    MAX_TRANSCRIPT_ENTRIES,
    add_timings,
    append_capped,
    append_unique,
    as_model,
    as_models,
    initial_state,
    latest,
    overwrite,
)
from agentic_workflow.errors import SchemaValidationError
from tests.helpers import make_approval, make_request

pytestmark = pytest.mark.unit


class TestStrictModel:
    """The base model every domain object inherits."""

    def test_unknown_fields_are_rejected(self) -> None:
        """A typo in a field name fails loudly instead of being dropped.

        A silently ignored field is how ``confidnce=0.9`` becomes ``0.5`` and a
        run escalates to a human for no reason. This is a real bug that was
        caught by this test: ``StrictModel`` stripped *every* undeclared key to
        survive a checkpoint round-trip, which made ``extra="forbid"`` cosmetic.
        Now only genuinely *computed* fields are stripped.
        """
        with pytest.raises(ValueError, match="extra"):
            ReviewRequest(  # type: ignore[call-arg]
                run_id="r", request_id="p", title="t", confidnce=0.9
            )

    def test_computed_fields_survive_a_dump_round_trip(self) -> None:
        """The model's own serialised output must be accepted back.

        The checkpointer stores ``model_dump()`` output, which includes computed
        fields. Rejecting them would make a run unresumable on the second
        checkpoint — the failure would only appear after a restart.
        """
        request = ReviewRequest(run_id="r", request_id="p", title="t")
        payload = request.model_dump(mode="json")
        assert "content_hash" in payload
        assert ReviewRequest.model_validate(payload) == request

    def test_source_content_is_not_stripped(self) -> None:
        """Source files keep their exact bytes, including leading indentation.

        ``str_strip_whitespace=True`` is right for titles and wrong for code: a
        file whose first line is indented would lose that indentation, and the
        reviewer would review different code than the repository holds. This is
        the second real bug these tests caught.
        """
        indented = "    def f():\n        return 1\n"
        assert SourceFile(path="a.py", content=indented).content == indented

    def test_titles_are_stripped(self) -> None:
        """Fields where whitespace is noise still get normalised.

        The exemption is per-field, not global, so the strictness that makes
        titles readable is not lost along with the fidelity that makes code
        review correct.
        """
        assert ReviewRequest(run_id="r", request_id="p", title="  T  ").title == "T"

    def test_diffs_keep_their_leading_context_space(self) -> None:
        """A diff's leading space is the context marker and must survive.

        Stripping it would turn a context line into something a human reviewer
        reads as an addition, in the one artefact they are asked to sign off on.
        """
        diff = "--- a/x\n+++ b/x\n unchanged\n-removed\n+added\n"
        assert Patch(diff=diff, files_changed=["x"]).diff == diff

    def test_assignment_is_revalidated(self) -> None:
        """Assigning an out-of-range value raises instead of corrupting the object.

        ``StrictModel`` sets ``revalidate_instances="always"``, so a model mutated
        between two checkpoint writes cannot slip through unvalidated.
        """
        summary = RunSummary(run_id="r", status="running")
        with pytest.raises(ValueError):
            summary.iteration = -1  # type: ignore[assignment]

    def test_json_round_trip_is_lossless(self) -> None:
        """A model serialised to JSON and back equals the original.

        The checkpointer persists state as JSON, so anything that does not
        round-trip is data that silently changes on its way to disk.
        """
        request = make_request()
        assert ReviewRequest.model_validate(json.loads(request.model_dump_json())) == request

    def test_naive_timestamps_are_normalised(self) -> None:
        """Every timestamp in the system is timezone-aware.

        Mixing naive and aware datetimes only fails at comparison time, in
        production, in the code that computes an approval's remaining window.
        """
        naive = datetime(2026, 1, 1, 12, 0, 0)  # noqa: DTZ001 - deliberately naive
        timing = NodeTiming(node="reviewer", started_at=naive)
        assert timing.started_at.tzinfo is UTC


class TestSourceFile:
    """Path safety, because file paths reach both the filesystem and the prompt."""

    def test_relative_paths_are_accepted(self) -> None:
        """Normal repo-relative paths validate."""
        assert SourceFile(path="checkout/total.py", content="x").path == "checkout/total.py"

    def test_absolute_paths_are_rejected(self) -> None:
        """An absolute path would let a request read outside the repository."""
        with pytest.raises(ValueError, match="repo-relative"):
            SourceFile(path="/etc/passwd", content="x")

    def test_parent_traversal_is_rejected(self) -> None:
        """``..`` is rejected as a path *segment* anywhere, not only as a prefix."""
        with pytest.raises(ValueError, match="repo-relative"):
            SourceFile(path="src/../../etc/passwd", content="x")

    def test_a_filename_containing_dots_is_legal(self) -> None:
        """Only the ``..`` segment is dangerous, so ``a..b`` remains a valid name."""
        assert SourceFile(path="weird..name.py", content="x").path == "weird..name.py"

    def test_line_count_counts_content(self) -> None:
        """The line count feeds effort estimation, so an empty file must be zero.

        Returning ``1`` for an empty file would make an empty file look like a
        one-line file, which feeds the escalation heuristic.
        """
        assert SourceFile(path="a.py", content="").line_count == 0
        assert SourceFile(path="a.py", content="a\nb\nc").line_count == 3
        assert SourceFile(path="a.py", content="a\nb\nc\n").line_count == 4


class TestReviewRequest:
    """Identity validation and content hashing."""

    def test_identifiers_are_restricted_to_a_safe_alphabet(self) -> None:
        """Run ids become checkpoint keys and log fields, so they are constrained."""
        with pytest.raises(ValueError, match="invalid identifier"):
            ReviewRequest(run_id="a b", request_id="p", title="t")
        with pytest.raises(ValueError, match="invalid identifier"):
            ReviewRequest(run_id="a/b", request_id="p", title="t")
        with pytest.raises(ValueError, match="invalid identifier"):
            ReviewRequest(run_id="a\nb", request_id="p", title="t")

    def test_safe_identifiers_are_accepted(self) -> None:
        """Dots, dashes, colons and underscores cover real-world id formats."""
        for candidate in ("pr-1042", "run_1", "a.b:c", "PR1042"):
            assert ReviewRequest(run_id=candidate, request_id="p", title="t").run_id == candidate

    def test_content_hash_ignores_the_run_id(self) -> None:
        """The hash identifies the *content*, not the submission.

        It doubles as the REST idempotency key, so a client that resubmits the
        same review under a new run id must produce the same hash.
        """
        first = make_request()
        second = make_request(files=first.files, description=first.description)
        assert first.run_id != second.run_id
        assert first.content_hash == second.content_hash

    def test_content_hash_is_sensitive_to_content(self) -> None:
        """A single changed byte changes the hash."""
        base = make_request()
        changed = make_request(files=[SourceFile(path="checkout/total.py", content="different")])
        assert base.content_hash != changed.content_hash

    def test_content_hash_is_sensitive_to_description(self) -> None:
        """The problem statement is part of the review, so it is part of the hash."""
        base = make_request(description="fix drift")
        other = make_request(description="fix drift differently")
        assert base.content_hash != other.content_hash

    def test_content_hash_is_sensitive_to_file_order(self) -> None:
        """Two files swapped is a different submission."""
        a = SourceFile(path="a.py", content="a")
        b = SourceFile(path="b.py", content="b")
        assert make_request(files=[a, b]).content_hash != make_request(files=[b, a]).content_hash

    def test_content_hash_is_prefixed(self) -> None:
        """The prefix makes the algorithm self-describing in logs and headers."""
        assert make_request().content_hash.startswith("sha256:")


class TestFinding:
    """Deterministic identifiers, which the evaluation suite depends on."""

    def test_id_is_derived_from_content(self) -> None:
        """The same finding always gets the same id.

        The eval suite diffs runs by finding id, so a random id would make every
        run look entirely different from the last one and destroy the signal.
        """
        one = Finding(title="Use Decimal for currency", file="a.py", line=10)
        two = Finding(title="Use Decimal for currency", file="a.py", line=10)
        assert one.id == two.id != ""

    def test_id_is_case_and_whitespace_insensitive(self) -> None:
        """Re-running the same review with different casing must not fork the id.

        The fingerprint normalises the title, so a model that rephrases only the
        capitalisation still maps to the same finding.
        """
        assert (
            Finding(title="Use Decimal", file="a.py", line=1).id
            == Finding(title="  use decimal  ", file="a.py", line=1).id
        )

    def test_different_findings_get_different_ids(self) -> None:
        """File, line and title all participate in the fingerprint."""
        base = Finding(title="Use Decimal", file="a.py", line=10)
        assert base.id != Finding(title="Use Decimal", file="b.py", line=10).id
        assert base.id != Finding(title="Use Decimal", file="a.py", line=11).id
        assert base.id != Finding(title="Use float", file="a.py", line=10).id

    def test_explicit_id_is_preserved(self) -> None:
        """A caller-supplied id wins, so external systems can key on it."""
        assert Finding(id="external-1", title="t").id == "external-1"

    def test_severity_round_trips_through_json(self) -> None:
        """Severity must not degrade to a bare string on the way to a checkpoint.

        This is a real, previously-observed failure: the checkpoint serializer's
        allowlist missed enums, so ``Severity.HIGH`` came back as ``"high"`` and
        every comparison against the enum silently failed.
        """
        finding = Finding(title="t", severity=Severity.CRITICAL, category=Category.SECURITY)
        restored = Finding.model_validate(json.loads(finding.model_dump_json()))
        assert restored.severity is Severity.CRITICAL
        assert restored.category is Category.SECURITY

    def test_line_must_be_positive(self) -> None:
        """Line numbers are 1-indexed; a zero or negative line is a model error."""
        with pytest.raises(ValueError):
            Finding(title="t", line=0)
        with pytest.raises(ValueError):
            Finding(title="t", line=-3)


class TestPatch:
    """Diff arithmetic, which drives the "is there anything to apply" check."""

    def test_diff_size_counts_additions_and_removals(self) -> None:
        """Only real change lines count, not the ``---``/``+++`` headers.

        Counting the headers was the third bug these tests caught: the removal
        regex used ``^-(?!!)``, a negative lookahead for ``!`` rather than for a
        second ``-``, so every ``--- a/file`` header was counted as a removed
        line. ``diff_size`` feeds :attr:`Patch.is_empty`, so a context-only diff
        looked like a real change and the run applied a no-op.
        """
        assert Patch(diff="--- a/x\n+++ b/x\n-old\n+new", files_changed=["x"]).diff_size == 2

    def test_context_lines_are_not_changes(self) -> None:
        """A diff of pure context is zero changes."""
        assert Patch(diff="--- a/x\n+++ b/x\n context", files_changed=["x"]).diff_size == 0

    def test_a_removed_divider_is_still_a_change(self) -> None:
        """Only a *doubled* minus is a header; a single one is a removal.

        The negative lookahead is anchored on the second character, so a body
        line that itself starts with dashes is still counted.
        """
        assert Patch(diff="--- a/x\n+++ b/x\n-a", files_changed=["x"]).diff_size == 1
        assert Patch(diff="--- a/x\n+++ b/x\n-!important", files_changed=["x"]).diff_size == 1

    def test_is_empty_requires_files_and_content(self) -> None:
        """Either signal alone is enough to call a patch empty.

        The router treats an empty patch as a terminal condition, so a false
        negative here would spend a human gate on a no-op write.
        """
        assert Patch(diff="", files_changed=[]).is_empty is True
        assert Patch(diff="--- a\n+++ b\n-a\n+a", files_changed=["x"]).is_empty is False
        assert Patch(diff="--- a\n+++ b\n-a\n+a", files_changed=[]).is_empty is True
        assert Patch(diff="context only", files_changed=["x"]).is_empty is True


class TestTestReport:
    """Pass-rate arithmetic, which the feedback loop branches on."""

    def test_pass_rate_excludes_skipped_tests(self) -> None:
        """A skipped test must not be counted as a failure.

        Six of the ten tests actually ran and four of those passed, so the rate is
        4/6. Dividing by ``total`` instead would report 0.4 and send a green
        suite back to the programmer forever.
        """
        report = ReportModel(passed=False, total=10, failed=2, skipped=4)
        assert report.pass_rate == pytest.approx(4 / 6)

    def test_no_executed_tests_is_a_full_pass(self) -> None:
        """An empty run is vacuously green, not a division by zero.

        Returning ``0.0`` here would make a suite with no tests look broken and
        send the workflow into a repair loop that can never succeed.
        """
        assert ReportModel(passed=True, total=0).pass_rate == 1.0
        assert ReportModel(passed=False, total=5, skipped=5).pass_rate == 1.0

    def test_pass_rate_never_exceeds_one(self) -> None:
        """Inconsistent counts must not produce a ratio above 1."""
        assert ReportModel(passed=True, total=3, failed=0).pass_rate == 1.0
        assert ReportModel(passed=False, total=1, failed=1).pass_rate == 0.0

    def test_pass_rate_never_goes_negative(self) -> None:
        """``failed > total`` is impossible, but the clamp must hold anyway."""
        assert ReportModel(passed=False, total=2, failed=9, skipped=0).pass_rate == 0.0

    def test_coverage_is_bounded(self) -> None:
        """Coverage outside ``[0, 1]`` is a data error, not a number to clamp."""
        with pytest.raises(ValueError):
            ReportModel(passed=True, coverage=1.5)
        with pytest.raises(ValueError):
            ReportModel(passed=True, coverage=-0.1)


class TestApprovalModels:
    """The invariants the human-in-the-loop path depends on."""

    def test_edit_requires_a_payload(self) -> None:
        """An ``edit`` with nothing to apply is rejected at construction.

        Failing here rather than in the node means the mistake surfaces as a 422
        at the HTTP boundary instead of as a corrupted state.
        """
        with pytest.raises(ValueError, match="requires a non-empty"):
            ApprovalDecision(approval_id="a", decision=Decision.EDIT, reviewer="alice")

        built = ApprovalDecision(
            approval_id="a", decision=Decision.EDIT, reviewer="alice", payload={"x": 1}
        )
        assert built.payload == {"x": 1}

    def test_approve_and_reject_need_no_payload(self) -> None:
        """The other two verdicts are complete on their own."""
        for verdict in (Decision.APPROVE, Decision.REJECT):
            assert ApprovalDecision(approval_id="a", decision=verdict, reviewer="alice")

    def test_allowed_options_respects_policy(self) -> None:
        """Policy can withdraw edit or reject without changing the model.

        ``approve`` is never withdrawn: a gate that cannot be approved is not a
        gate, it is a dead end.
        """
        request = make_approval()
        assert request.allowed_options(allow_edit=False, allow_reject=False) == [Decision.APPROVE]
        assert Decision.EDIT in request.allowed_options(allow_edit=True, allow_reject=False)
        assert Decision.REJECT in request.allowed_options(allow_edit=False, allow_reject=True)

    def test_expiry_is_computed_against_the_clock(self) -> None:
        """A past deadline reads as expired; an open one does not."""
        assert make_approval(expires_in_seconds=-1.0).is_expired is True
        assert make_approval(expires_in_seconds=3600.0).is_expired is False
        assert make_approval().is_expired is False

    def test_new_builds_a_complete_request(self) -> None:
        """The factory fills the id and the expiry, so callers cannot forget them."""
        request = ApprovalRequest.new(
            run_id="r", node="apply_patch", stage="patch_apply", title="Apply?", approval_id="a-1"
        )
        assert request.approval_id == "a-1"
        assert request.options == [Decision.APPROVE, Decision.EDIT, Decision.REJECT]
        assert request.created_at.tzinfo is not None

    def test_utcnow_is_timezone_aware(self) -> None:
        """The project clock helper returns aware UTC timestamps."""
        assert utcnow().tzinfo is not None


class TestReducers:
    """The merge semantics every node's return value goes through."""

    def test_overwrite_replaces(self) -> None:
        """A single-value channel takes the newest write."""
        assert overwrite("old", "new") == "new"

    def test_overwrite_clears_a_channel(self) -> None:
        """``None`` must actually clear, not be ignored as "no update".

        This is what lets the programmer invalidate ``test_report`` by writing a
        new patch. If ``None`` were ignored, the router would keep validating the
        *previous* patch and never notice the new one was untested — the exact
        failure the validate-before-apply invariant exists to prevent.
        """
        assert overwrite("value", None) is None

    def test_latest_ignores_a_none_update(self) -> None:
        """``latest`` is the counterpart: ``None`` means "I have nothing to say"."""
        assert latest("kept", None) == "kept"
        assert latest(None, "set") == "set"
        assert latest("old", "new") == "new"

    def test_append_capped_concatenates(self) -> None:
        """Transcript entries accumulate across the nodes that write them."""
        assert append_capped(["a"], ["b", "c"]) == ["a", "b", "c"]

    def test_append_capped_handles_a_missing_left_side(self) -> None:
        """The first writer in a channel sees no previous value."""
        assert append_capped(None, ["a"]) == ["a"]
        assert append_capped([], ["a"]) == ["a"]

    def test_append_capped_drops_the_oldest_entries(self) -> None:
        """Checkpoint payloads stay bounded no matter how chatty a run gets.

        Truncation is at the tail because old transcript entries are narration,
        not evidence; the decision log and the report are what matter.
        """
        existing = [f"e{index}" for index in range(MAX_TRANSCRIPT_ENTRIES)]
        result = append_capped(existing, ["newest"])
        assert len(result) == MAX_TRANSCRIPT_ENTRIES
        assert result[-1] == "newest"
        assert "e0" not in result

    def test_append_capped_clears_on_none(self) -> None:
        """A ``None`` clear empties the channel rather than being ignored."""
        assert append_capped(["a", "b"], None) == []

    def test_append_unique_deduplicates_by_id(self) -> None:
        """A re-executed node must not double-count the findings it already raised.

        LangGraph re-runs a node from the top after an interrupt, so without
        de-duplication the counts that gate the feedback loop would inflate on
        every human decision.
        """
        first = Finding(title="A", file="a.py", line=1)
        second = Finding(title="B", file="b.py", line=2)
        merged = append_unique(append_unique([], [first]), [first, second])
        assert [item.title for item in merged] == ["A", "B"]

    def test_append_unique_replaces_in_place(self) -> None:
        """A refreshed record supersedes a stale one instead of appending beside it.

        Replacing keeps the finding's original position, so a dashboard that
        sorts by first-seen order stays stable across iterations.
        """
        stale = Finding(id="f-1", title="Old title", file="a.py", line=1)
        fresh = Finding(id="f-1", title="New title", file="a.py", line=1)
        merged = append_unique([stale], [fresh])
        assert len(merged) == 1
        assert merged[0].title == "New title"

    def test_append_unique_never_mutates_its_inputs(self) -> None:
        """Reducers receive the live state list, so mutating it corrupts the store.

        LangGraph does not promise the reducer gets a private copy. A reducer
        that appends in place would write a value that no longer matches the
        checkpoint already persisted.
        """
        left = [{"id": "a"}]
        right = [{"id": "b"}]
        result = append_unique(left, right)
        assert result == [{"id": "a"}, {"id": "b"}]
        assert left == [{"id": "a"}]
        assert right == [{"id": "b"}]
        assert result is not left

    def test_add_timings_accumulates_and_clears(self) -> None:
        """Per-node timings accumulate across the whole run."""
        first = NodeTiming(node="triage", duration_ms=1.0)
        second = NodeTiming(node="reviewer", duration_ms=2.0)
        assert len(add_timings(add_timings(None, first), [second])) == 2
        assert add_timings([first], None) == []


class TestAsModel:
    """Reading typed values back out of untyped state."""

    def test_returns_a_model_instance(self) -> None:
        """A dict payload validates into its declared model."""
        state: dict[str, Any] = {"report": {"markdown": "# ok", "decision": "approved"}}
        report = as_model(state, "report", FinalReport)
        assert isinstance(report, FinalReport)
        assert report.markdown == "# ok"

    def test_missing_channel_is_none(self) -> None:
        """An absent channel is not an error."""
        assert as_model({}, "report", FinalReport) is None

    def test_none_channel_is_none(self) -> None:
        """A cleared channel reads as ``None``, not as a validation failure."""
        assert as_model({"report": None}, "report", FinalReport) is None

    def test_invalid_payload_raises_a_typed_error(self) -> None:
        """A corrupt channel reports which channel and which model failed.

        A bare ``ValidationError`` from deep inside a node gives an operator no
        way to find the offending checkpoint.
        """
        with pytest.raises(SchemaValidationError) as info:
            as_model({"report": {"decision": "not_a_verdict"}}, "report", FinalReport)
        assert "report" in str(info.value)
        assert "FinalReport" in str(info.value)

    def test_already_modelled_value_is_rebuilt_not_shared(self) -> None:
        """A typed channel value is re-validated into an equal, distinct object.

        ``revalidate_instances="always"`` means ``as_model`` always deep-rebuilds,
        even for an instance of the right class. That is deliberate: it guarantees
        nested models are typed too, whatever shape the checkpoint stored them
        in. The test asserts *equality* rather than identity because the copy is
        the whole point — a node must not be able to mutate the object the
        checkpoint handed it.
        """
        report = FinalReport(markdown="# ok")
        rebuilt = as_model({"report": report}, "report", FinalReport)
        assert rebuilt == report
        assert rebuilt is not report

    def test_as_models_maps_element_wise(self) -> None:
        """List channels validate each element against the same model."""
        state: dict[str, Any] = {"findings": [{"title": "a"}, {"title": "b"}]}
        assert [item.title for item in as_models(state, "findings", Finding)] == ["a", "b"]

    def test_as_models_on_a_missing_channel_is_empty(self) -> None:
        """Absent list channels yield an empty list, so callers need no guard."""
        assert as_models({}, "findings", Finding) == []

    def test_as_models_on_a_none_channel_is_empty(self) -> None:
        """A cleared list channel is empty rather than a validation error."""
        assert as_models({"findings": None}, "findings", Finding) == []


class TestInitialState:
    """The seed a run starts from."""

    def test_request_is_carried_through(self) -> None:
        """The input channel is the run's only source of truth at ``t=0``."""
        request = make_request()
        assert initial_state(request)["request"] == request

    def test_artefact_channels_start_unset(self) -> None:
        """No agent has produced anything yet.

        Asserting ``is None`` rather than falsiness: an empty list would look
        identical to "the agent ran and found nothing", and the router treats
        those differently.
        """
        state = initial_state(make_request())
        for channel in ("task_brief", "patch", "review", "test_report", "report"):
            assert state.get(channel) is None, f"{channel} must start unset"

    def test_list_channels_start_empty(self) -> None:
        """Transcript and decision log are empty lists, not ``None``.

        The append reducer needs a list; seeding ``None`` would turn the first
        append into a ``TypeError`` raised from inside LangGraph, which tells the
        caller nothing about which channel was wrong.
        """
        state = initial_state(make_request())
        assert state["transcript"] == []
        assert state["human_decisions"] == []
        assert state["findings"] == []

    def test_counters_and_status_start_at_the_beginning(self) -> None:
        """A run is queued before it is running, with no loops completed."""
        state = initial_state(make_request())
        assert state["iteration"] == 0
        assert state["status"] == "pending"

    def test_next_action_seeds_the_entry_edge(self) -> None:
        """The router channel starts as ``start``, a placeholder the router overwrites.

        It is seeded rather than left empty so a checkpoint written before the
        router's first turn still has a well-formed channel. Asserting the exact
        value matters: any *other* seed would be silently read as a routing
        decision by anything that inspects the channel directly.
        """
        assert initial_state(make_request())["next_action"] == "start"

    def test_pending_approval_starts_unset(self) -> None:
        """A fresh run is not blocked on anybody."""
        assert initial_state(make_request()).get("pending_approval") is None

    def test_state_is_json_serialisable(self) -> None:
        """The seed must survive the checkpoint store's JSON round trip."""
        json.dumps(initial_state(make_request()), default=str)

    def test_state_is_accepted_by_the_declared_state_type(self) -> None:
        """The seed satisfies every required key of ``WorkflowState``.

        A missing required channel is a ``KeyError`` deep inside the first node,
        which is far harder to diagnose than an assertion here.
        """
        from agentic_workflow.domain.state import WorkflowState

        state: WorkflowState = initial_state(make_request())
        for required in ("request", "status", "iteration", "transcript", "findings"):
            assert required in state


class TestAgentOutputs:
    """The derived values the reporter and the evaluation suite consume."""

    def test_claim_units_splits_the_report(self) -> None:
        """Faithfulness scoring works on claims, so the split must be deterministic."""
        report = FinalReport(markdown="One thing. Two things!\n\nThree things?")
        assert report.claim_units == [
            "One thing.",
            "Two things!",
            "Three things?",
        ]

    def test_claim_units_of_an_empty_report_is_empty(self) -> None:
        """An empty report produces no claims rather than one empty claim.

        One empty claim would score as an unsupported assertion and fail every
        faithfulness gate for a report that simply says nothing.
        """
        assert FinalReport(markdown="").claim_units == []

    def test_highest_severity_ranks_correctly(self) -> None:
        """Severity ordering is by impact, and an empty review reports ``info``."""
        assert ReviewResult(findings=[]).highest_severity is Severity.INFO
        review = ReviewResult(
            findings=[
                Finding(title="a", severity=Severity.LOW),
                Finding(title="b", severity=Severity.CRITICAL),
                Finding(title="c", severity=Severity.MEDIUM),
            ]
        )
        assert review.highest_severity is Severity.CRITICAL

    def test_blocking_count_matches_the_list(self) -> None:
        """The router escalates on this count, so it must not be a guess."""
        review = ReviewResult(
            findings=[Finding(title="a"), Finding(title="b")],
            blocking_findings=["a"],
        )
        assert review.blocking_count == 1

    def test_task_brief_requires_an_objective(self) -> None:
        """A brief with no objective cannot steer an agent."""
        with pytest.raises(ValueError):
            TaskBrief(objective="")


__all__ = ["RunSummary", "overwrite"]
