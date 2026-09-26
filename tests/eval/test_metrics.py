"""The evaluation harness tested against reports built to break it.

A metric suite that has never been shown to *fail* has not been tested — it has
only been run. Every metric here is exercised with a report that is
deliberately wrong in one specific way, and the assertion is that the right
metric catches it and the others stay silent. A metric that fails for the wrong
reason is worse than no metric, because it sends the reader looking in the wrong
place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from evals import external, runners
from evals.datasets import (
    DEFAULT_DATASET,
    Case,
    ExpectedFinding,
    case_from_dict,
    load_cases,
)
from evals.metrics import (
    DETECTION_METRICS,
    INVARIANT_METRICS,
    NATIVE_METRICS,
    NATIVE_METRICS_BY_NAME,
    citation_coverage_score,
    claims_of,
    grounding_score,
    is_claim,
    metric_name,
    path_grounding_score,
    precision_score,
    recall_score,
    self_consistency_score,
    structure_score,
    submitted_text,
)
from evals.report import render, render_diff
from evals.runners import CaseResult, MetricScore, SuiteReport, gate_scope, smoke
import pytest

#: A case with one known defect, used as the substrate for the broken reports.
SUBSTRATE = Case(
    id="substrate",
    title="Sum prices in float",
    language="python",
    files=[
        {
            "path": "checkout/total.py",
            "content": "def total(items):\n    result = 0.0\n    for item in items:\n"
            '        result = result + float(item["price"])\n    return result\n',
        }
    ],
    expected_findings=[
        ExpectedFinding(
            path="checkout/total.py",
            token="result = result + float",
            severity="high",
            summary="Float summation accumulates representation error.",
        )
    ],
)


def _report(**overrides: Any) -> dict[str, Any]:
    """Build a plausible, passing report, then apply *overrides*.

    The defaults are a report that satisfies every metric. That matters: each
    test then breaks exactly one thing, so a failure names that thing rather
    than the first of several.

    Args:
        **overrides: Keys to replace in the report.

    Returns:
        The report mapping.
    """
    base: dict[str, Any] = {
        "report_id": "r1",
        "decision": "changes_requested",
        "markdown": (
            "## Decision\n\nChanges requested.\n\n## Findings\n\n"
            "- **Float summation** in `checkout/total.py`:\n"
            '  `result = result + float(item["price"])`\n'
        ),
        "findings": [
            {
                "id": "f1",
                "title": "Float summation accumulates error",
                "detail": "`result = result + float(...)` sums currency in binary float.",
                "severity": "high",
                "category": "correctness",
                "file": "checkout/total.py",
                "line": 4,
                "recommendation": "Use decimal.Decimal.",
                "confidence": 0.8,
            }
        ],
        "citations": ["checkout/total.py:4"],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
class TestDataset:
    def test_the_bundled_dataset_loads_and_validates(self) -> None:
        cases = load_cases()
        assert len(cases) >= 20
        assert all(case.validated() is case for case in cases)

    def test_every_expected_finding_is_present_in_the_code_it_names(self) -> None:
        """The property that keeps the set usable.

        A dataset that asserts a defect in a token the file does not contain
        produces a permanently failing recall score, and a permanently failing
        metric is a metric nobody reads.
        """
        for case in load_cases():
            for finding in case.expected_findings:
                assert finding.is_present_in(case.files), f"{case.id}: {finding.token!r}"

    def test_case_ids_are_unique(self) -> None:
        ids = [case.id for case in load_cases()]
        assert len(ids) == len(set(ids)), "duplicate case ids make a report ambiguous"

    def test_a_dataset_path_that_does_not_exist_says_so_with_the_flag(self) -> None:
        with pytest.raises(FileNotFoundError) as excinfo:
            load_cases("/nonexistent/cases.jsonl")
        assert "--dataset" in str(excinfo.value)

    def test_a_malformed_line_names_the_line(self, tmp_path: Path) -> None:
        """A parse error must say *where*, or fixing a long dataset is guesswork."""
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"id": "a", "title": "t", "language": "python", '
            '"files": [{"path": "a.py", "content": "x"}], '
            '"expected_findings": []}\nnot json\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r":2 is not valid JSON"):
            load_cases(path)

    def test_a_case_missing_a_required_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is missing"):
            Case(
                id="x",
                title="",
                language="python",
                files=[{"path": "a.py", "content": "x"}],
                expected_findings=[],
            ).validated()

    def test_a_case_with_an_ungrounded_expectation_is_rejected(self) -> None:
        """The dataset cannot assert a defect in code it does not contain."""
        raw = {
            "id": "x",
            "title": "t",
            "language": "python",
            "files": [{"path": "a.py", "content": "print(1)\n"}],
            "expected_findings": [{"path": "a.py", "token": "eval(", "summary": "s"}],
        }
        with pytest.raises(ValueError, match="absent from its code"):
            case_from_dict(raw).validated()

    def test_limit_truncates_without_skipping_validation(self) -> None:
        """A truncated run is still a valid run."""
        assert len(load_cases(limit=3)) == 3
        assert len(load_cases(DEFAULT_DATASET)) >= 20

    def test_a_limit_of_zero_is_an_error(self) -> None:
        """A typo in a CI flag must not produce a green build that measured nothing.

        Checked before the read loop, because ``len(cases) >= 0`` is true after
        the first append and a limit of 0 would otherwise return exactly one
        case while the caller asked for none.
        """
        with pytest.raises(ValueError, match="at least 1"):
            load_cases(limit=0)

    def test_a_limit_larger_than_the_dataset_is_not_an_error(self) -> None:
        """Asking for more than exists is a quiet request, not a broken one."""
        assert len(load_cases(limit=100_000)) == len(load_cases())

    def test_a_case_with_no_expected_findings_is_valid(self) -> None:
        """An empty expectation list is the assertion "this change is clean".

        It is the only case type that catches a model *inventing* findings, so
        rejecting it would leave the suite unable to see a false positive.
        """
        clean = Case(
            id="clean",
            title="No known defect",
            language="python",
            files=[{"path": "a.py", "content": "print(1)\n"}],
            expected_findings=[],
        )
        assert clean.validated() is clean

    def test_category_is_lifted_from_the_top_level(self) -> None:
        raw = {
            "id": "x",
            "title": "t",
            "category": "security",
            "files": [],
            "expected_findings": [],
        }
        assert case_from_dict(raw).category == "security"

    def test_a_case_builds_a_valid_domain_request(self) -> None:
        request = SUBSTRATE.to_request("run-1")
        assert request.run_id == "run-1"
        assert request.files[0].content == SUBSTRATE.files[0]["content"]


# --------------------------------------------------------------------------- #
# Grounding: the hallucination metrics
# --------------------------------------------------------------------------- #
class TestGrounding:
    def test_a_grounded_report_scores_one(self) -> None:
        """Every quoted span must be traceable to the submitted code.

        ``checkout/total.py`` is a real span — it is the file label in the
        fence, echoed in the report — so it has to be found in the prompt's
        framing of the case or the metric would reject a correct review for
        naming its file.
        """
        report = _report(
            markdown="## Decision\n\n## Findings\n\n"
            "- Float summation in checkout/total.py:\n"
            '  `result = result + float(item["price"])`\n'
        )
        assert grounding_score(SUBSTRATE, report).value == 1.0

    def test_a_quoted_line_nobody_wrote_fails(self) -> None:
        """The core hallucination property: no inventing quotations.

        A review that quotes a line the author never wrote is describing a
        different file, and every judgement built on it is about nothing.
        """
        report = _report(
            markdown="## Decision\n\n## Findings\n\n```python\n"
            "cursor.execute(query % user_input)\n```\n"
        )
        score = grounding_score(SUBSTRATE, report)
        assert not score.passed
        assert any("cursor.execute" in note for note in score.evidence)

    def test_a_diff_hunk_is_not_a_quotation(self) -> None:
        """A proposed *added* line is new by definition, so it is not a hallucination.

        Without this the metric would flag every correct patch suggestion, and a
        reviewer would learn to stop pasting diffs — removing exactly the
        evidence a reader needs.
        """
        report = _report(
            markdown="## Decision\n\n## Findings\n\n```diff\n"
            '-        result = result + float(item["price"])\n'
            '+        result = result + Decimal(item["price"])\n```\n'
        )
        score = grounding_score(SUBSTRATE, report)
        assert score.value == 1.0, score.evidence

    def test_a_removed_diff_line_is_still_checked(self) -> None:
        """The ``-`` side of a hunk quotes the author's code, so it must be real."""
        report = _report(
            markdown="## Decision\n\n## Findings\n\n```diff\n"
            "-        cursor.execute(user_supplied_sql)\n```\n"
        )
        score = grounding_score(SUBSTRATE, report)
        assert not score.passed
        assert any("cursor.execute" in note for note in score.evidence)

    def test_an_inline_quotation_is_checked(self) -> None:
        report = _report(
            markdown="## Decision\n\n## Findings\n\n- `import socket; socket.socket()`\n"
        )
        assert not grounding_score(SUBSTRATE, report).passed

    def test_a_short_fragment_is_not_treated_as_evidence(self) -> None:
        """A two-character span matches by accident and reports failures nobody
        could act on, which is how a metric gets turned off."""
        report = _report(markdown="## Decision\n\n## Findings\n\n- the `if` is wrong\n")
        assert grounding_score(SUBSTRATE, report).value == 1.0

    def test_a_report_that_quotes_nothing_cannot_be_unfaithful(self) -> None:
        report = _report(markdown="## Decision\n\n## Findings\n\n- none\n")
        assert grounding_score(SUBSTRATE, report).passed

    def test_an_invented_file_path_fails(self) -> None:
        report = _report(
            findings=[{**_report()["findings"][0], "file": "billing/invoices.py"}],
        )
        score = path_grounding_score(SUBSTRATE, report)
        assert not score.passed
        assert any("billing/invoices.py" in note for note in score.evidence)

    def test_a_path_inside_the_markdown_fails_too(self) -> None:
        """Not just the structured field — prose is read by humans too."""
        report = _report(
            markdown="## Decision\n\nSee `billing/invoices.py` for the same pattern.\n\n"
            "## Findings\n\n- none\n"
        )
        assert not path_grounding_score(SUBSTRATE, report).passed

    def test_a_submitted_path_passes(self) -> None:
        assert path_grounding_score(SUBSTRATE, _report()).value == 1.0

    def test_a_bare_basename_against_a_submitted_file_passes(self) -> None:
        """Citing ``total.py`` when ``checkout/total.py`` was submitted is fine.

        A reader who has one file in the diff can find it from the basename, so
        this is a shorthand, not a fabrication.
        """
        report = _report(findings=[{**_report()["findings"][0], "file": "total.py", "line": None}])
        assert path_grounding_score(SUBSTRATE, report).value == 1.0

    def test_a_different_file_with_the_same_name_is_a_hallucination(self) -> None:
        """The suffix-match rule this replaces accepted any invented path that
        ended in a submitted file's basename.

        ``billing/invoices/total.py`` and ``checkout/total.py`` are two different
        files, and the metric exists precisely to stop a review talking about one
        the author never submitted. A rule that lets it through turns the check
        into decoration that reads as a pass.
        """
        report = _report(
            findings=[{**_report()["findings"][0], "file": "billing/invoices/total.py"}]
        )
        score = path_grounding_score(SUBSTRATE, report)
        assert not score.passed
        assert any("billing/invoices/total.py" in note for note in score.evidence)


# --------------------------------------------------------------------------- #
# Recall and precision
# --------------------------------------------------------------------------- #
class TestRecall:
    def test_a_caught_defect_scores_one(self) -> None:
        assert recall_score(SUBSTRATE, _report()).value == 1.0

    def test_a_missed_defect_is_named(self) -> None:
        """Evidence must point at the token, so the reader can check it."""
        report = _report(
            markdown="## Decision\n\n## Findings\n\n- nothing found\n",
            findings=[],
        )
        score = recall_score(SUBSTRATE, report)
        assert score.value == 0.0
        assert "result = result + float" in score.evidence[0]

    def test_a_case_with_no_expectations_has_no_recall_to_be_zero_about(self) -> None:
        """Scoring it 0 would punish a case for being small, not for being wrong."""
        empty = Case(
            id="clean",
            title="t",
            language="python",
            files=[{"path": "a.py", "content": "print(1)\n"}],
            expected_findings=[],
        )
        score = recall_score(
            empty, _report(markdown="## Decision\n\n## Findings\n\n- none\n", findings=[])
        )
        assert score.value is None
        assert score.passed


class TestPrecision:
    def test_a_grounded_claim_scores_one(self) -> None:
        assert precision_score(SUBSTRATE, _report()).value == 1.0

    def test_naming_a_file_does_not_ground_a_claim_on_its_own(self) -> None:
        """The file name is the pointer, not the evidence.

        A finding that cites a submitted file always shares an identifier with
        the source — the file's own name — so matching the path against the code
        would make every cited finding pass and the metric vacuous. It is the
        prose that has to be grounded.
        """
        report = _report(
            findings=[
                {
                    "id": "f1",
                    "title": "Consider adopting hexagonal architecture",
                    "detail": "A widely advocated design pattern for large systems.",
                    "severity": "low",
                    "category": "architecture",
                    "file": "checkout/total.py",
                    "line": None,
                    "recommendation": "Refactor toward ports and adapters.",
                    "confidence": 0.4,
                }
            ]
        )
        score = precision_score(SUBSTRATE, report)
        assert score.value == 0.0
        assert score.evidence

    def test_an_invented_file_is_caught_by_path_grounding_not_precision(self) -> None:
        """Two metrics, two jobs: prose groundedness versus file existence.

        A finding can quote real code and still name the wrong file. Re-checking
        the path here would report the same defect twice under two names, and
        the reader would not know which to fix first.
        """
        report = _report(findings=[{**_report()["findings"][0], "file": "billing/totals.py"}])
        assert precision_score(SUBSTRATE, report).value == 1.0, "the prose is grounded"
        assert not path_grounding_score(SUBSTRATE, report).passed, "the file is not"

    def test_a_null_result_is_not_a_false_positive(self) -> None:
        """ "Nothing found" is not a claim, and must not be scored as a wrong one.

        Otherwise the metric would reward a model for inventing a file path in
        order to look useful, and would report an honest null result as a defect.
        """
        report = _report(
            markdown="## Decision\n\n## Findings\n\n- No obvious issue detected.\n",
            findings=[
                {
                    "id": "f1",
                    "title": "No obvious issue detected; consider adding a regression test",
                    "detail": "The rule set found nothing in this change.",
                    "severity": "medium",
                    "category": "testing",
                    "file": None,
                    "line": None,
                    "recommendation": "Add a regression test.",
                    "confidence": 0.2,
                }
            ],
        )
        score = precision_score(SUBSTRATE, report)
        assert score.value is None
        assert score.passed

    def test_is_claim_is_structural_not_wording_based(self) -> None:
        """Hard-coding one provider's phrasing would measure the provider."""
        source = submitted_text(SUBSTRATE)
        assert is_claim({"file": "a.py"}, source)
        assert is_claim({"line": 3}, source)
        assert is_claim({"title": "total is wrong", "detail": ""}, source)
        assert not is_claim(
            {"title": "No obvious issue detected", "detail": "Nothing here."}, source
        )

    def test_claims_of_splits_a_mixed_report(self) -> None:
        report = _report(
            findings=[
                _report()["findings"][0],
                {"id": "f2", "title": "Nothing to add", "detail": "No issues.", "severity": "low"},
            ]
        )
        assert len(claims_of(report, submitted_text(SUBSTRATE))) == 1


# --------------------------------------------------------------------------- #
# Citation coverage
# --------------------------------------------------------------------------- #
class TestCitationCoverage:
    def test_a_cited_claim_scores_one(self) -> None:
        assert citation_coverage_score(SUBSTRATE, _report()).value == 1.0

    def test_an_uncited_claim_is_named(self) -> None:
        report = _report(findings=[{**_report()["findings"][0], "file": None, "line": None}])
        score = citation_coverage_score(SUBSTRATE, report)
        assert score.value == 0.0
        assert score.evidence

    def test_a_report_with_no_claims_is_not_applicable(self) -> None:
        report = _report(markdown="## Decision\n\n## Findings\n\n- none\n", findings=[])
        assert citation_coverage_score(SUBSTRATE, report).value is None


# --------------------------------------------------------------------------- #
# Structure and self-consistency
# --------------------------------------------------------------------------- #
class TestStructure:
    def test_a_complete_report_scores_one(self) -> None:
        assert structure_score(SUBSTRATE, _report()).value == 1.0

    def test_a_report_with_no_verdict_is_caught(self) -> None:
        """A review's product is a verdict; prose that omits it is unusable."""
        report = _report(decision="")
        score = structure_score(SUBSTRATE, report)
        assert not score.passed
        assert "no decision" in score.evidence

    def test_a_missing_section_is_caught(self) -> None:
        report = _report(markdown="## Decision\n\nApproved.\n")
        assert any("Findings" in note for note in structure_score(SUBSTRATE, report).evidence)

    def test_a_verdict_outside_the_domain_is_caught(self) -> None:
        report = _report(decision="looks_fine_to_me")
        assert not structure_score(SUBSTRATE, report).passed


class TestSelfConsistency:
    def test_a_consistent_report_scores_one(self) -> None:
        assert self_consistency_score(SUBSTRATE, _report()).value == 1.0

    def test_approving_over_a_critical_finding_is_a_contradiction(self) -> None:
        report = _report(decision="approved")
        score = self_consistency_score(SUBSTRATE, report)
        assert score.value == 0.0
        assert "blocking" in score.evidence[0]

    def test_rejecting_with_nothing_to_reject_is_a_contradiction(self) -> None:
        report = _report(
            decision="rejected",
            findings=[],
            markdown="## Decision\n\nRejected.\n\n## Findings\n\n- none\n",
        )
        assert not self_consistency_score(SUBSTRATE, report).passed

    def test_declining_to_decide_is_exempt(self) -> None:
        """``needs_human`` is the workflow correctly declining, not contradicting.

        Holding the safe answer to the same rules as a verdict would penalise
        the behaviour the HITL design exists to produce.
        """
        for decision in ("needs_human", "blocked"):
            report = _report(decision=decision, findings=[])
            assert self_consistency_score(SUBSTRATE, report).passed, decision

    def test_approved_with_only_informational_findings_is_consistent(self) -> None:
        report = _report(
            decision="approved",
            findings=[{**_report()["findings"][0], "severity": "info"}],
        )
        assert self_consistency_score(SUBSTRATE, report).passed


# --------------------------------------------------------------------------- #
# The suite's own failure modes
# --------------------------------------------------------------------------- #
class TestSmokeReport:
    """The fabricated report is what proves the suite can say no."""

    def test_the_fabricated_report_fails(self) -> None:
        result = smoke(SUBSTRATE)
        assert not result.passed

    def test_it_fails_on_the_specific_defects_it_contains(self) -> None:
        """Not "it failed" — it failed for the reasons it was built to fail for."""
        result = smoke(SUBSTRATE)
        failed = {score.metric for score in result.failing()}
        assert "grounding" in failed, "an ungrounded quotation must be caught"
        assert "path_grounding" in failed, "an invented path must be caught"
        assert "self_consistency" in failed, "approved-over-critical must be caught"
        assert "recall" in failed, "the known defect was not reported"


class TestSuiteReport:
    def _report_for(self, passed: bool) -> CaseResult:
        return CaseResult(
            case_id="c1",
            category="security",
            status="completed",
            decision="changes_requested",
            findings=1,
            gates=2,
            scores=(
                MetricScore("recall", 1.0 if passed else 0.0, passed),
                MetricScore("precision", None, True),
            ),
            duration_seconds=0.1,
        )

    def test_an_empty_suite_does_not_pass(self) -> None:
        """ "The dataset did not load" and "the workflow is perfect" must not
        look alike.

        An empty result list is the signature of a broken run, and a suite that
        renders it as green is worse than no suite.
        """
        report = SuiteReport(
            provider="echo",
            dataset="x",
            gate_verdict="approve",
            cases=(),
            duration_seconds=0.0,
        )
        assert not report.passed

    def test_a_not_applicable_metric_is_excluded_from_the_average(self) -> None:
        report = SuiteReport(
            provider="echo",
            dataset="x",
            gate_verdict="approve",
            cases=(self._report_for(True),),
            duration_seconds=0.0,
        )
        summary = report.metric_summary()
        assert "precision" not in summary, "None must not be averaged in as zero"
        assert summary["recall"]["mean"] == 1.0

    def test_a_failing_case_fails_the_suite(self) -> None:
        report = SuiteReport(
            provider="echo",
            dataset="x",
            gate_verdict="approve",
            cases=(self._report_for(False),),
            duration_seconds=0.0,
        )
        assert not report.passed
        assert len(report.failing_cases()) == 1

    def test_an_errored_case_fails_the_suite(self) -> None:
        """A crash is not a neutral outcome."""
        broken = CaseResult(
            case_id="c1",
            category="security",
            status="error",
            decision="",
            findings=0,
            gates=0,
            scores=(),
            duration_seconds=0.0,
            error="RuntimeError: boom",
        )
        report = SuiteReport(
            provider="echo",
            dataset="x",
            gate_verdict="approve",
            cases=(broken,),
            duration_seconds=0.0,
        )
        assert not report.passed

    def test_the_dict_is_json_serialisable(self) -> None:
        report = SuiteReport(
            provider="echo",
            dataset="x",
            gate_verdict="approve",
            cases=(self._report_for(True),),
            duration_seconds=0.0,
            scorers=external.availability(),
        )
        assert json.loads(json.dumps(report.as_dict()))["passed"] is True


class TestRendering:
    def _report(self, cases: tuple[CaseResult, ...]) -> SuiteReport:
        return SuiteReport(
            provider="echo",
            dataset="x",
            gate_verdict="approve",
            cases=cases,
            duration_seconds=1.0,
        )

    def test_a_missing_scorer_is_printed_as_missing(self) -> None:
        """Silence would read as "we checked" — the exact failure to avoid."""
        result = CaseResult(
            case_id="c1",
            category="security",
            status="completed",
            decision="approved",
            findings=0,
            gates=0,
            scores=(),
            duration_seconds=0.0,
        )
        report = self._report((result,))
        report = SuiteReport(
            provider=report.provider,
            dataset=report.dataset,
            gate_verdict=report.gate_verdict,
            cases=report.cases,
            duration_seconds=report.duration_seconds,
            scorers={"ragas": external.ScorerAvailability("ragas", False, "ModuleNotFoundError")},
        )
        text = render(report)
        assert "missing    ragas" in text
        assert "did NOT run" in text

    def test_an_empty_suite_renders_as_a_failure(self) -> None:
        text = render(self._report(()))
        assert "FAIL" in text
        assert "no cases ran" in text

    def test_a_failure_renders_its_evidence(self) -> None:
        result = CaseResult(
            case_id="c1",
            category="security",
            status="completed",
            decision="approved",
            findings=0,
            gates=0,
            scores=(MetricScore("recall", 0.0, False, ["missed a.py :: eval("]),),
            duration_seconds=0.0,
        )
        text = render(self._report((result,)))
        assert "missed a.py :: eval(" in text

    def test_the_diff_view_ranks_regressions_first(self) -> None:
        before = {"metrics": {"recall": {"mean": 0.9}, "grounding": {"mean": 1.0}}}
        after = {"metrics": {"recall": {"mean": 0.5}, "grounding": {"mean": 1.0}}}
        lines = render_diff(before, after).splitlines()
        assert "recall" in lines[1]
        assert "-0.400" in lines[1]


class TestMetricRegistry:
    def test_every_metric_is_callable_and_named(self) -> None:
        for metric in NATIVE_METRICS:
            assert callable(metric)
            assert metric(SUBSTRATE, _report()).metric

    def test_a_metric_reports_the_name_it_is_registered_under(self) -> None:
        """The registered name and the reported name must be the same string.

        They are the same fact stated in two places, and nothing but this test
        keeps them in step. When they drift, a metric is invisible to ``--gate``
        and to the gate's own coverage check — the failure mode is a gate that
        quietly covers nothing while still reporting green.
        """
        for name, metric in NATIVE_METRICS_BY_NAME.items():
            assert metric_name(metric) == name
            assert metric(SUBSTRATE, _report()).metric == name

    def test_every_metric_survives_a_report_with_no_findings(self) -> None:
        """An empty report is the most common degenerate input; none may raise."""
        empty = _report(markdown="## Decision\n\n## Findings\n\n- none\n", findings=[])
        for metric in NATIVE_METRICS:
            score = metric(SUBSTRATE, empty)
            assert isinstance(score.value, float | type(None))

    def test_an_unregistered_function_has_no_name(self) -> None:
        """A loud failure beats a name derived from ``__name__``."""

        def not_a_metric(case: Any, report: dict[str, Any]) -> MetricScore:
            return MetricScore("x", 1.0, True)

        with pytest.raises(KeyError, match="is not a native metric"):
            metric_name(not_a_metric)


class TestGateScope:
    """What a gate may and may not be scoped to."""

    def test_the_two_groups_partition_the_native_metrics(self) -> None:
        """Every metric is either an invariant or a detection metric, never both.

        A metric in neither group would be computed and reported but gated by
        nothing; in both groups would be gated as if it were provider-independent
        when it is not.
        """
        assert set(INVARIANT_METRICS) | set(DETECTION_METRICS) == set(NATIVE_METRICS_BY_NAME)
        assert not set(INVARIANT_METRICS) & set(DETECTION_METRICS)

    def test_recall_is_a_detection_metric_and_nothing_else_is(self) -> None:
        """The one metric a null model cannot satisfy, named explicitly.

        Encoded rather than derived from a threshold, because the whole design
        rests on this one assignment: it is what makes an offline, free,
        reproducible gate possible without also being a gate that cannot pass.
        """
        assert DETECTION_METRICS == ("recall",)

    def test_gate_all_covers_everything(self) -> None:
        assert gate_scope("all") is None

    def test_gate_invariants_resolves_to_the_invariant_names(self) -> None:
        assert gate_scope("invariants") == frozenset(INVARIANT_METRICS)

    def test_gate_detection_resolves_to_the_detection_names(self) -> None:
        assert gate_scope("detection") == frozenset(DETECTION_METRICS)

    def test_an_unknown_gate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="gate must be one of"):
            gate_scope("everything")

    def test_a_gated_case_can_pass_while_failing_an_ungated_metric(self) -> None:
        """The point of the whole mechanism, stated as a test.

        A run that is a *correct review of nothing* satisfies every invariant and
        misses every detection metric. Under ``invariants`` that is a pass; under
        ``all`` it is a failure. Both are right, and which one you want depends
        entirely on whether you are gating a workflow or grading a model.
        """
        case = CaseResult(
            case_id="null-model",
            category="security",
            status="completed",
            decision="approved",
            findings=0,
            gates=2,
            scores=(
                MetricScore("grounding", 1.0, True),
                MetricScore("structure", 1.0, True),
                MetricScore("recall", 0.0, False, ["found none of 2 seeded defects"]),
            ),
            duration_seconds=0.1,
        )
        report = SuiteReport(
            provider="echo",
            dataset="x",
            gate_verdict="approve",
            cases=(case,),
            duration_seconds=0.0,
            gate_metrics=gate_scope("invariants"),
        )
        assert report.passed
        assert report.failing_cases() == []
        assert not case.passed, "the ungated verdict is still available"
        assert [score.metric for score in case.failing()] == ["recall"]

    def test_the_verdict_scope_never_hides_a_case_from_the_category_view(self) -> None:
        """Diagnosis is unscoped; only gating is.

        Scoping ``by_category`` to the invariants would report every category as
        perfect, erasing the one signal the view exists to give.
        """
        case = CaseResult(
            case_id="null-model",
            category="security",
            status="completed",
            decision="approved",
            findings=0,
            gates=2,
            scores=(
                MetricScore("grounding", 1.0, True),
                MetricScore("recall", 0.0, False),
            ),
            duration_seconds=0.1,
        )
        report = SuiteReport(
            provider="echo",
            dataset="x",
            gate_verdict="approve",
            cases=(case,),
            duration_seconds=0.0,
            gate_metrics=gate_scope("invariants"),
        )
        assert report.passed
        assert report.by_category()["security"] == {"cases": 1.0, "passed": 0.0}

    def test_an_errored_case_fails_under_every_gate(self) -> None:
        """A crash is not a metric miss, so scoping cannot excuse it.

        Gating on a subset of metrics must never become a way for a run that
        raised to be reported as passing.
        """
        broken = CaseResult(
            case_id="c1",
            category="security",
            status="error",
            decision="",
            findings=0,
            gates=0,
            scores=(),
            duration_seconds=0.0,
            error="RuntimeError: boom",
        )
        for gate in ("all", "invariants", "detection"):
            scoped = frozenset() if gate == "all" else gate_scope(gate)
            assert not broken.gated_passed(scoped), gate

    def test_an_empty_gate_scope_would_pass_anything(self) -> None:
        """Why :func:`gate_scope` validates its own output.

        ``all(score.passed for score in ())`` is ``True``, so a scope resolving to
        the empty frozenset passes every case — including a total fabrication.
        Pinned here as a property of the primitive, because it is a property of
        Python, and it is the reason the resolver refuses to produce one.
        """
        fabricated = CaseResult(
            case_id="fabricated",
            category="security",
            status="completed",
            decision="approved",
            findings=1,
            gates=1,
            scores=(MetricScore("grounding", 0.0, False, ["quoted a line nobody wrote"]),),
            duration_seconds=0.1,
        )
        assert fabricated.gated_passed(frozenset())
        assert not fabricated.passed
        # The resolver, by contrast, never yields an empty scope for a real gate.
        for gate in ("invariants", "detection"):
            assert gate_scope(gate), gate


class TestGateScopeValidation:
    def test_a_scope_naming_an_unregistered_metric_is_rejected(self) -> None:
        """A gate that covers nothing passes everything; catch it at resolution.

        Forced by monkeypatching rather than by a bad constant, because the
        constant is correct today and the check exists for the day someone adds
        ``DETECTION_METRICS`` entry by hand. The failure it prevents is silent:
        the suite runs, every metric is reported, and the gate is green because
        it matched no metric at all.
        """
        broken = ("recall", "typoed_metric_name")
        with (
            patch.object(runners, "DETECTION_METRICS", broken),
            pytest.raises(ValueError, match="names unregistered metrics"),
        ):
            gate_scope("detection")
