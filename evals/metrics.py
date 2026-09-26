"""Quality metrics, computed from artefacts the run already produced.

Design constraints
------------------

**No model in the loop.** A metric that asks a judge LLM to grade the output
is measuring the judge, not the workflow, and it costs money and adds
nondeterminism to a gate that is supposed to catch nondeterminism. Every
metric here is a deterministic function of (submitted source, produced
findings, produced report). That makes the score reproducible — the same commit
scores the same number twice, on any machine, offline.

**Every metric is grounded in the input.** "Faithfulness" here does not mean
"does this sound right" but "does every identifier the report names actually
appear in the code it was given". A hallucinated file path or a quoted line
that was never submitted fails the metric, and it fails for a reason a human can
check in a second. This is the property that matters most for a code review
tool and the one a text-similarity score cannot see.

**Failures are informative.** Each metric carries the evidence that made it
fail, so a score of 0.4 comes with the list of claims that were wrong rather than
a number to argue about.

The trade-off, stated plainly: these metrics cannot tell a *correct* finding
from a *plausible-sounding* one, because that judgement needs a model or a human.
What they can do is catch a workflow that started inventing file paths,
dropping the findings it produced, contradicting its own citations, or
publishing a report with no verdict in it — which are the regressions that
actually happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

#: Identifiers a report may legitimately mention that are not in the submitted
#: code: language keywords, the names of the workflow's own agents, and the
#: standard-library modules a review is allowed to suggest. Anything else that
#: looks like a path is treated as a claim about the codebase and must be
#: grounded.
_KEYWORDS = frozenset(
    {
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "const",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "let",
        "match",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "var",
        "while",
        "with",
        "yield",
        "self",
        "None",
        "True",
        "False",
    }
)

#: Names the workflow itself introduces. A report that says "the reviewer asked
#: for tests" is not making a claim about the codebase.
_WORKFLOW_VOCABULARY = frozenset(
    {
        "programmer",
        "reviewer",
        "tester",
        "triage",
        "reporter",
        "router",
        "apply_patch",
        "agentic-workflow",
        "agentic_workflow",
        "awf",
    }
)

#: A conservative identifier: a path, or a dotted module reference, or a
#: ``snake_case``/``camelCase`` symbol. Deliberately loose, because a
#: hallucination that fails to look like an identifier cannot be caught by a
#: regex and must therefore be caught by the judge we do not have.
_IDENTIFIER_RE = re.compile(
    r"""
    (?<![\w./-])
    (?:
        [\w.-]+ / [\w./-]+ (?:\.\w+)?     # a path: app/models/user.py
      | [\w.]+ \.\w+                        # a dotted reference: os.path
      | \b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b  # snake_case symbol
      | \b[a-z]+[A-Z][A-Za-z0-9]*\b         # camelCase symbol
    )
    (?![\w/-])
    """,
    re.VERBOSE,
)

#: A quoted span of code, either fenced or inline.
_CODE_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\n(?P<body>.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]{2,120})`")

#: Line numbers a review legitimately refers to.
_LINE_REF_RE = re.compile(r"\b(?:line|L|L)\s*\.?\s*(\d{1,5})\b", re.IGNORECASE)


class Metric(Protocol):
    """One scoreable property of a produced report.

    Every metric takes ``(case, report)`` whether or not it needs both halves.
    The uniform signature is what lets :data:`NATIVE_METRICS` be a flat tuple
    the runner can iterate without knowing anything about any individual
    metric, and it means a new metric never has to change how it is called. The
    metrics that ignore ``case`` say so with a ``noqa`` rather than pretending
    to use it.
    """

    name: str

    def score(self, case: Any, report: dict[str, Any]) -> MetricScore:  # pragma: no cover
        """Score one case's report.

        Args:
            case: The :class:`~evals.datasets.Case` that was run.
            report: The serialised final report.

        Returns:
            The score and its evidence.
        """
        ...


@dataclass(frozen=True, slots=True)
class MetricScore:
    """One metric's verdict on one case.

    Attributes:
        metric: Metric name.
        value: ``0.0`` to ``1.0``, or ``None`` when the metric does not apply.
            ``None`` is distinct from ``0.0``: a case with no expected findings
            has no recall to be zero about, and scoring it 0 would drag the
            average down for being small rather than wrong.
        passed: Whether the value met the metric's threshold.
        evidence: The specific claims that decided it, for a human to check.
    """

    metric: str
    value: float | None
    passed: bool
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "metric": self.metric,
            "value": self.value,
            "passed": self.passed,
            "evidence": self.evidence[:10],
        }


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def submitted_text(case: Any) -> str:
    """Concatenate the code the case submitted.

    Args:
        case: The golden case.

    Returns:
        All submitted file contents, joined.
    """
    return "\n".join(str(f.get("content", "")) for f in case.files)


def report_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the report's findings as plain dicts.

    Args:
        report: The serialised report.

    Returns:
        The findings, which may be empty.
    """
    raw = report.get("findings") or []
    return [dict(f) for f in raw if isinstance(f, dict)]


def report_text(report: dict[str, Any]) -> str:
    """Everything a reader of the report could see.

    Args:
        report: The serialised report.

    Returns:
        Markdown plus every finding field and citation, lowercased for matching.
    """
    parts: list[str] = [str(report.get("markdown", ""))]
    for finding in report_findings(report):
        for key in ("title", "detail", "file", "recommendation", "category", "severity"):
            value = finding.get(key)
            if value:
                parts.append(str(value))
    parts.extend(str(c) for c in report.get("citations", []) if c)
    return "\n".join(parts).lower()


def extract_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Pull fenced code blocks out of *text*, keeping the language tag.

    The tag is what distinguishes a *quotation* from a *proposed edit*. A block
    tagged ``python`` claims "this line is in your file"; a block tagged
    ``diff`` claims "these lines are the change", and its added lines are by
    definition new. Conflating the two makes a metric that reports every correct
    patch suggestion as a hallucination.

    Args:
        text: Markdown.

    Returns:
        ``(language, body)`` pairs, language lowercased and possibly empty.
    """
    return [
        (match.group("lang").lower(), match.group("body"))
        for match in _CODE_FENCE_RE.finditer(text)
    ]


def extract_code_spans(text: str) -> list[str]:
    """Pull quoted code out of *text*.

    Args:
        text: Markdown, or plain text.

    Returns:
        Every fenced and inline code span, stripped. Diff blocks are included
        whole; use :func:`extract_fenced_blocks` when the language tag matters.
    """
    spans = [body.strip() for _, body in extract_fenced_blocks(text)]
    spans += [span.strip() for span in _INLINE_CODE_RE.findall(text)]
    return [span for span in spans if span]


#: Fence languages whose content is an edit rather than a quotation.
_DIFF_LANGUAGES = frozenset({"diff", "patch", "unified", "udiff"})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def grounding_score(case: Any, report: dict[str, Any]) -> MetricScore:
    """Every quoted line of code must have been in the submitted source.

    This is the hallucination metric. A review that quotes a line the author
    never wrote is describing a different file, and any judgement built on it —
    the severity, the fix, the "tests pass" claim — is about nothing.

    Spans that are too short to be evidence (``x``, ``0``, ``foo()``) are
    skipped: a two-character fragment matches by accident and would make the
    metric report failures nobody could act on.

    A block tagged as a diff is graded differently from a quotation. Its
    context and removed lines claim "this is your code" and are checked
    normally; its *added* lines are the proposed change and are new by
    definition, so they are exempt. Without that exemption the metric reports
    every correct patch suggestion as a hallucination, and a reviewer learns to
    stop pasting diffs — which is the exact behaviour that removes the
    evidence a reader needs.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        The score, naming each ungrounded span.
    """
    source = submitted_text(case)
    source_lines = {line.strip() for line in source.splitlines() if line.strip()}
    markdown = str(report.get("markdown", ""))

    checked = 0
    ungrounded: list[str] = []

    def is_grounded(candidate: str) -> bool:
        """Whether a candidate line is traceable to the submitted source.

        Args:
            candidate: One stripped line of quoted code.

        Returns:
            ``True`` when the line — or the line it is part of — was submitted.
        """
        return candidate in source_lines or candidate in source

    for language, body in extract_fenced_blocks(markdown):
        is_diff = language in _DIFF_LANGUAGES
        for raw in body.splitlines():
            line = raw.strip()
            if not line:
                continue
            if is_diff and line.startswith(("+", "+++")):
                # The proposed new line. Not a quotation, so not checkable.
                continue
            candidate = line.lstrip("+-").strip() if is_diff else line
            checked += 1
            if len(candidate) >= 8 and not is_grounded(candidate):
                ungrounded.append(candidate[:120])

    for span in _INLINE_CODE_RE.findall(markdown):
        candidate = span.strip()
        checked += 1
        if len(candidate) >= 8 and not is_grounded(candidate):
            ungrounded.append(candidate[:120])

    if not checked:
        # A report that quotes nothing cannot be unfaithful, but it also is not
        # evidence of anything; the citation metric is what holds it to account.
        return MetricScore("grounding", 1.0, True)
    value = 1.0 - (len(ungrounded) / checked)
    return MetricScore("grounding", max(0.0, value), not ungrounded, ungrounded)


#: File extensions that make a path-like string a *source file* claim. A path
#: ending in `.yaml` or `.md` is as likely to be a suggestion as an assertion,
#: so those are not treated as claims about the submitted change.
_SOURCE_SUFFIXES = (".py", ".js", ".ts", ".tsx", ".rb", ".go", ".java", ".rs", ".php", ".c")


def path_grounding_score(case: Any, report: dict[str, Any]) -> MetricScore:
    """Every file path the report names must be a file that was submitted.

    The match is **exact** — full path, or a bare basename against a bare
    basename — and deliberately not a suffix match. Suffix matching accepts
    ``billing/invoices/total.py`` when the submitted change contains
    ``checkout/total.py``: two different files that happen to share a name, and
    the whole point of the metric is that a review must not talk about a file
    the author did not submit. A suffix rule here makes the check decorative,
    which is worse than not having it, because it reads as a passing check.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        The score, naming each invented path.
    """
    known = {str(f.get("path", "")) for f in case.files}
    known_lower = {path.lower() for path in known}
    known_basenames = {path.rsplit("/", 1)[-1].lower() for path in known}
    invented: list[str] = []
    for finding in report_findings(report):
        for key in ("file", "path"):
            claim = str(finding.get(key, "") or "")
            if not claim:
                continue
            normalised = claim.lstrip("./").lower()
            if normalised in known_lower or (
                "/" not in normalised and normalised in known_basenames
            ):
                continue
            invented.append(f"{key}={claim}")
    for match in _IDENTIFIER_RE.findall(report_text(report)):
        candidate = match.strip().lower()
        if "/" not in candidate:
            continue
        if candidate in known_lower or (candidate.count("/") == 0 and candidate in known_basenames):
            continue
        if candidate in _KEYWORDS or candidate in _WORKFLOW_VOCABULARY:
            continue
        # A path claim only counts as a hallucination if it *looks* like a source
        # file. Prose mentions of "os.path" or "config/settings.yaml" as
        # suggestions are not claims about the submitted change.
        if not candidate.endswith(_SOURCE_SUFFIXES):
            continue
        invented.append(f"path={candidate}")
    if not invented:
        return MetricScore("path_grounding", 1.0, True)
    return MetricScore("path_grounding", 0.0, False, sorted(set(invented)))


def is_claim(finding: dict[str, Any], source: str = "") -> bool:
    """Whether a finding asserts something a reader could check.

    A review that finds nothing still has to *say* so, and that statement is not
    a claim: it names no file, no line and no identifier from the submitted
    code, so there is nothing in it to be right or wrong about. Treating it as
    one would make a null result count as a false positive, which overstates the
    problem — and worse, it would reward a model for inventing a file path in
    order to look useful.

    The test is deliberately structural and provider-agnostic. Hard-coding one
    provider's wording ("no obvious issue") would make the metric measure that
    provider instead of the workflow. What generalises is the *reference to the
    submitted code*, so when *source* is available a claim must point at it.

    Args:
        finding: One serialised :class:`~agentic_workflow.domain.schemas.Finding`.
        source: The submitted code, concatenated. Without it, only explicit file
            and line references can be checked, so a prose-only finding counts
            as a claim — the conservative direction, because a metric should
            stay strict when it cannot be sure.

    Returns:
        ``True`` when the finding points at something in the submitted code.
    """
    if str(finding.get("file") or "").strip():
        return True
    if finding.get("line"):
        return True
    prose = " ".join(
        str(finding.get(key, "") or "") for key in ("title", "detail", "evidence", "recommendation")
    )
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", prose))
    if not source:
        # No source to check against, so the conservative answer is "yes": a
        # metric should stay strict when it cannot be sure.
        return bool(words)
    return any(word in source for word in words)


def claims_of(report: dict[str, Any], source: str = "") -> list[dict[str, Any]]:
    """Return only the findings that make checkable claims.

    Args:
        report: The serialised report.
        source: The submitted code, concatenated.

    Returns:
        The claiming findings, which may be empty even when the report is not.
    """
    return [f for f in report_findings(report) if is_claim(f, source)]


def recall_score(case: Any, report: dict[str, Any]) -> MetricScore:
    """Did the report catch the defects it was supposed to catch?

    The headline quality number, and the one that is hardest to move honestly:
    a workflow can score 1.0 here by reporting everything, so it is paired with
    :func:`precision_score`. Reporting every line in the file is not a better
    review.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        The score, naming each missed defect.
    """
    expected = list(case.expected_findings)
    if not expected:
        return MetricScore("recall", None, True, ["case declares no expected findings"])
    caught = [f for f in expected if f.mentioned_by(report)]
    missed = [f for f in expected if f not in caught]
    value = len(caught) / len(expected)
    evidence = [f"missed {f.path} :: {f.token[:60]}" for f in missed]
    return MetricScore("recall", value, not missed, evidence)


def precision_score(case: Any, report: dict[str, Any]) -> MetricScore:
    """Were the reported findings about *this* change?

    Precision here is not "is the finding correct" — that needs a judge. It is
    "is the claim about code that was submitted", which is checkable: a finding
    that names a file with no distinctive token from that file is asserting
    something about nothing.

    Only findings that make a checkable claim are scored — see
    :func:`is_claim`. A report that says "nothing found" is not a false
    positive, and scoring it as one would make an honest null result look like a
    defect. When every claim is grounded and the rest are null results, this
    metric scores ``1.0`` and the *coverage* metrics say how little was claimed.

    A claim counts as supported when it shares a real identifier with the
    submitted code, or when it cites a submitted file *and* a line. Naming a
    file and then saying nothing about what is in it is a hollow claim: it
    passes a file-existence check while telling the reader nothing, which is the
    failure mode a "review" full of real-looking file paths produces.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        The score, naming each unsupported claim; ``None`` when the report made
        no claims at all.
    """
    claims = claims_of(report, submitted_text(case))
    if not claims:
        return MetricScore(
            "precision",
            None,
            True,
            ["report made no checkable claim to be right or wrong about"],
        )
    source = submitted_text(case)
    known_paths = {str(f.get("path", "")) for f in case.files}
    # Identifiers of five or more characters, so a shared stopword or a short
    # name does not count as evidence that a claim is about this code.
    tokens = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", source)
        if token not in _KEYWORDS
    }
    supported = 0
    unsupported: list[str] = []
    for finding in claims:
        named_file = str(finding.get("file") or "").strip()
        # The file name is deliberately excluded from the word set. It is the
        # *pointer*, not the evidence: a finding that names a submitted file
        # always shares an identifier with the source (the file's own name), so
        # including it makes every cited finding pass and the metric vacuous.
        # What has to be grounded is the prose — the claim about what is in that
        # file — so only the prose is matched.
        claim = " ".join(
            str(finding.get(key, "") or "") for key in ("title", "detail", "recommendation")
        )
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", claim))
        # A claim is supported when it shares a real identifier with the source —
        # a function name, a field, a literal, a call — or when it cites a
        # submitted file at a specific line, which is a precise pointer. A
        # file-only citation with no shared identifier is hollow and fails.
        precise_citation = (
            bool(named_file) and named_file in known_paths and bool(finding.get("line"))
        )
        if words & tokens or precise_citation:
            supported += 1
        else:
            unsupported.append(claim[:100] or "(empty finding)")
    value = supported / len(claims)
    return MetricScore("precision", value, not unsupported, unsupported)


def citation_coverage_score(case: Any, report: dict[str, Any]) -> MetricScore:
    """Does the report point at the file it is talking about?

    A finding with no file is a claim about the codebase in the abstract. It
    costs the reader a search and it cannot be acted on without one, so a
    reviewer who cannot find the line will not check the line.

    Only claims are counted — see :func:`is_claim`. Requiring a null result to
    cite a file would be requiring the model to be wrong in order to score well,
    which is the opposite of what a citation metric is for. A report that makes
    no claims scores ``None`` here and lets :func:`recall_score` carry the
    information that nothing was found.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        The score, naming each unattributed claim; ``None`` when the report made
        no claims to attribute.
    """
    claims = claims_of(report, submitted_text(case))
    if not claims:
        return MetricScore(
            "citation_coverage",
            None,
            True,
            ["report made no checkable claim to attribute"],
        )
    cited = [f for f in claims if str(f.get("file") or "")]
    value = len(cited) / len(claims)
    missing = [str(f.get("title", "unnamed"))[:80] for f in claims if not str(f.get("file") or "")]
    return MetricScore("citation_coverage", value, not missing, missing)


def structure_score(case: Any, report: dict[str, Any]) -> MetricScore:  # noqa: ARG001 - uniform metric signature
    """Does the report actually answer the question that was asked?

    A review workflow's product is a verdict. A report that never states one
    forces the reader to infer it, and inference from prose is where "approved"
    and "rejected" get confused. This also checks that every section promised
    by the schema is populated, because an empty section is a silent failure.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        The score, naming each missing element.
    """
    problems: list[str] = []
    decision = str(report.get("decision", "") or "").strip()
    if not decision:
        problems.append("no decision")
    elif decision not in _VERDICTS:
        problems.append(f"decision {decision!r} is not a verdict")
    markdown = str(report.get("markdown", "") or "")
    if len(markdown.strip()) < 40:
        problems.append("markdown is effectively empty")
    for heading in ("Decision", "Findings"):
        if heading.lower() not in markdown.lower():
            problems.append(f"no '{heading}' section")
    for finding in report_findings(report):
        if not str(finding.get("severity", "") or "").strip():
            problems.append(f"finding without severity: {str(finding.get('title', ''))[:60]}")
            break
    value = 1.0 - (len(problems) / 4.0)
    return MetricScore("structure", max(0.0, value), not problems, problems)


#: Verdicts the reporter may legitimately publish. Anything else means the
#: report claimed a decision the domain does not have.
_VERDICTS = frozenset({"approved", "changes_requested", "rejected", "needs_human", "blocked"})

#: Verdicts that assert the change is or is not acceptable. ``needs_human`` and
#: ``blocked`` are neither — they decline to decide, so no consistency rule
#: applies to their findings.
_DECISIVE_VERDICTS = frozenset({"approved", "changes_requested", "rejected"})

#: Severities that should stop a merge on their own.
_BLOCKING_SEVERITIES = frozenset({"high", "critical"})


def self_consistency_score(case: Any, report: dict[str, Any]) -> MetricScore:  # noqa: ARG001 - uniform metric signature
    """Must the verdict agree with the findings?

    A report that says ``approved`` while listing a critical finding is
    self-contradictory, and a reader who notices loses trust in both. This is
    the one consistency check that needs no model, because the contradiction is
    definitional: the verdict is a claim about the findings, so it can be
    checked against them.

    ``needs_human`` and ``blocked`` are exempt. They are the workflow correctly
    declining to decide, which is the opposite of a contradiction, and holding
    them to the same rules would penalise the safe answer.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        The score; ``1.0`` when the report is not contradictory.
    """
    decision = str(report.get("decision", "") or "").strip().lower()
    severities = [str(f.get("severity", "") or "").strip().lower() for f in report_findings(report)]
    blocking = [s for s in severities if s in _BLOCKING_SEVERITIES]
    problems: list[str] = []
    if not decision:
        problems.append("no decision to be consistent with")
    elif decision in _DECISIVE_VERDICTS:
        if decision == "approved" and blocking:
            problems.append(
                f"approved with {len(blocking)} blocking finding(s): "
                + ", ".join(sorted(set(blocking)))
            )
        if decision in ("rejected", "changes_requested") and not severities:
            problems.append(f"{decision} with no findings to justify it")
    return MetricScore("self_consistency", 1.0 if not problems else 0.0, not problems, problems)


#: Every metric the native suite computes, in reporting order, keyed by the name
#: the runner, the report and ``--gate`` all refer to it by.
#:
#: A dict rather than a tuple because the *name* is part of the interface: it is
#: what appears in the JSON, what a gate matches on, and what a reader quotes
#: back to you. Holding the name in a parallel list invites the two to drift, and
#: a metric whose name is misspelled in one place and not another is
#: un-gateable in a way nothing would report.
NATIVE_METRICS_BY_NAME: dict[str, Any] = {
    "grounding": grounding_score,
    "path_grounding": path_grounding_score,
    "recall": recall_score,
    "precision": precision_score,
    "citation_coverage": citation_coverage_score,
    "structure": structure_score,
    "self_consistency": self_consistency_score,
}

#: The same metrics in reporting order, for iteration.
NATIVE_METRICS: tuple[Any, ...] = tuple(NATIVE_METRICS_BY_NAME.values())


def metric_name(scoring_function: Any) -> str:
    """Return the registered name of a scoring function.

    Args:
        scoring_function: One of the functions in :data:`NATIVE_METRICS`.

    Returns:
        The name it is registered under.

    Raises:
        KeyError: If the function is not a native metric. Better a loud failure
            than a name derived from ``__name__``, which would differ from the
            registered one by a rename and quietly pass a gate meant for
            another metric.
    """
    for name, function in NATIVE_METRICS_BY_NAME.items():
        if function is scoring_function:
            return name
    raise KeyError(
        f"{getattr(scoring_function, '__name__', scoring_function)} is not a native metric"
    )


#: Metrics that must hold **whatever model produced the report**.
#:
#: These are correctness properties of the workflow, not measures of the model's
#: ability. A report that quotes a line nobody wrote, names a file that was
#: never submitted, states no verdict, or approves over its own critical finding
#: is broken *as a review* regardless of which model produced it — and a
#: deterministic stub is just as capable of demonstrating the violation as a
#: frontier model is. That makes these the metrics a CI job can gate on without
#: spending a token, and the only ones it should.
INVARIANT_METRICS: tuple[str, ...] = (
    "grounding",
    "path_grounding",
    "precision",
    "citation_coverage",
    "structure",
    "self_consistency",
)

#: Metrics that measure whether the model **found** the seeded defects.
#:
#: Held separately because they are the only numbers in the suite that depend on
#: the provider's capability, and therefore the only ones a deterministic null
#: model cannot be expected to satisfy. Gating a build on them against a stub
#: produces a permanently red pipeline, which trains everyone to ignore the
#: eval job. Against a real model they become the numbers that matter — and the
#: stub's score on them is the *floor* a real model has to clear, which is what
#: makes it a useful baseline rather than a failed test.
DETECTION_METRICS: tuple[str, ...] = ("recall",)

#: A metric must reach this to count as passed for a case.
THRESHOLD = 1.0

__all__ = [
    "DETECTION_METRICS",
    "INVARIANT_METRICS",
    "NATIVE_METRICS",
    "NATIVE_METRICS_BY_NAME",
    "THRESHOLD",
    "Metric",
    "MetricScore",
    "citation_coverage_score",
    "claims_of",
    "extract_code_spans",
    "grounding_score",
    "is_claim",
    "metric_name",
    "path_grounding_score",
    "precision_score",
    "recall_score",
    "report_findings",
    "report_text",
    "self_consistency_score",
    "structure_score",
    "submitted_text",
]
