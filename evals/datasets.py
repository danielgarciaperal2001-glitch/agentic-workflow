"""The golden dataset: reviewed pull requests with a known-correct answer.

Why a golden set at all
-----------------------

An LLM-backed workflow can be made to *look* right. The only way to know a
review report is any good is to compare it against an answer a human already
vouched for, and the comparison has to be mechanical or it becomes a matter of
opinion. Every metric in :mod:`evals.metrics` is therefore computed from
artifacts that already exist in the run: the source the agents were given, the
findings they reported, and the report they published.

What a case contains
--------------------

Each line is one reviewed change. ``expected_findings`` is the part that
matters: the defects a competent reviewer was known to catch, stated as
substrings of a real file path plus a distinctive token. Substrings rather than
regex because a metric that fails on a legitimate paraphrase is a metric people
turn off, and substring matching is the most forgiving honest form.

The dataset is small on purpose. Twenty cases with human-verified expectations
is enough to catch a regression that makes the workflow systematically worse;
it is not enough to claim statistical significance, and nothing here pretends
otherwise. ``README.md`` quotes these numbers as what they are: a regression
gate, not a leaderboard.

Format
------

One JSON object per line (JSONL), so a case can be read with ``head -1`` and
appended with ``>>`` without a parser. See :func:`evals.datasets.load_cases`
for the schema and its validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Self

#: The dataset shipped with the repository.
DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "pr_review_golden.jsonl"

#: The fields a case must declare. Anything else is rejected: a dataset that
#: silently accepts a typo'd key is a dataset that quietly stops testing
#: something.
REQUIRED_FIELDS = ("id", "title", "language", "files", "expected_findings")


@dataclass(frozen=True, slots=True)
class ExpectedFinding:
    """One defect a competent reviewer was known to catch.

    Attributes:
        path: File the defect lives in. Matched as a substring so a case can
            name a directory when that is enough to be specific.
        token: A distinctive fragment of the offending code. The metric checks
            that the token really is present in that file, so a case cannot
            assert a finding about code that does not contain it.
        severity: Why it matters, for the report's readability.
        summary: One line describing the defect, in a reviewer's words.
    """

    path: str
    token: str
    severity: str = "medium"
    summary: str = ""

    def is_present_in(self, files: list[dict[str, str]]) -> bool:
        """Whether this finding is grounded in the submitted code.

        Args:
            files: The case's source files.

        Returns:
            ``True`` when a file whose path contains :attr:`path` also contains
            :attr:`token`.
        """
        for source in files:
            if self.path in source.get("path", "") and self.token in source.get("content", ""):
                return True
        return False

    def mentioned_by(self, report: dict[str, Any]) -> bool:
        """Whether the produced report reported this defect.

        A finding counts as recalled when the report names the file *or* quotes
        the token. Demanding both would fail a correct review that described
        the bug in prose without repeating the identifier.

        Args:
            report: The serialised :class:`~agentic_workflow.domain.schemas.FinalReport`.

        Returns:
            ``True`` when the report refers to the finding.
        """
        haystack = " ".join(
            str(finding.get(key, ""))
            for finding in report.get("findings", [])
            for key in ("title", "detail", "file", "evidence")
        )
        haystack = f"{haystack}\n{report.get('markdown', '')}"
        return self.path in haystack or self.token in haystack


@dataclass(frozen=True, slots=True)
class Case:
    """One golden review case.

    Attributes:
        id: Stable identifier, used in reports so a failure names its case.
        title: What the change was, in one line.
        language: Primary language of the submitted files.
        files: The submitted sources, as ``{"path", "content"}`` mappings.
        expected_findings: Defects a reviewer was known to catch.
        description: The change request as a client would phrase it.
        acceptance_criteria: What an acceptable fix must satisfy.
        constraints: Limits the fix must respect.
        metadata: Free-form annotations; the ``category`` key is used to group
            results in the report.
    """

    id: str
    title: str
    language: str
    files: list[dict[str, str]] = field(default_factory=list)
    expected_findings: list[ExpectedFinding] = field(default_factory=list)
    description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str:
        """The case's defect category, or ``uncategorised``."""
        return str(self.metadata.get("category", "uncategorised"))

    def to_request(self, run_id: str) -> Any:
        """Build the domain request this case describes.

        Args:
            run_id: Identifier for the run that will process it.

        Returns:
            A validated :class:`~agentic_workflow.domain.schemas.ReviewRequest`.
        """
        from agentic_workflow.domain.schemas import ReviewRequest, SourceFile

        return ReviewRequest(
            run_id=run_id,
            request_id=f"golden-{self.id}",
            title=self.title,
            description=self.description or self.title,
            language=self.language,
            files=[SourceFile(path=f["path"], content=f["content"]) for f in self.files],
            acceptance_criteria=self.acceptance_criteria,
            constraints=self.constraints,
            metadata={"golden_case": self.id, **self.metadata},
        )

    def validated(self) -> Self:
        """Check the case is internally consistent.

        The important part is that every expected finding is actually grounded
        in the submitted code. A dataset that asserts a defect in a token the
        file does not contain produces a permanently failing recall score, and
        a permanently failing metric is a metric nobody reads.

        Returns:
            This case, unchanged.

        Raises:
            ValueError: If a required field is missing or an expected finding
                is not present in the code it names.
        """
        # A string field must be non-empty; a list field only has to be *present*.
        # The distinction matters: an empty ``expected_findings`` is not a defect
        # in the case, it is the assertion that this change is known to be clean
        # — the case type that catches a model inventing findings. A truthiness
        # test would reject it, and the dataset would lose its only false-positive
        # probe while claiming to check for hallucinations.
        missing = [
            name
            for name in REQUIRED_FIELDS
            if getattr(self, name, None) is None
            or (isinstance(getattr(self, name, None), str) and not getattr(self, name))
        ]
        if missing:
            raise ValueError(f"case {self.id!r} is missing: {', '.join(missing)}")
        if not self.files:
            # Distinct from "files is missing": there is nothing to review, so
            # the case could never produce a grounded claim either way.
            raise ValueError(f"case {self.id!r} submits no files")
        for source in self.files:
            if not source.get("path") or "content" not in source:
                raise ValueError(f"case {self.id!r} has a malformed file entry: {source!r}")
        ungrounded = [f for f in self.expected_findings if not f.is_present_in(self.files)]
        if ungrounded:
            detail = ", ".join(f"{f.path}:{f.token!r}" for f in ungrounded)
            raise ValueError(f"case {self.id!r} expects findings absent from its code: {detail}")
        return self


def case_from_dict(raw: dict[str, Any]) -> Case:
    """Build a :class:`Case` from one JSONL record.

    Args:
        raw: The decoded object.

    Returns:
        The parsed, unvalidated case.

    Raises:
        ValueError: If ``expected_findings`` is not a list of mappings.
    """
    findings_raw = raw.get("expected_findings", [])
    if not isinstance(findings_raw, list):
        raise ValueError(f"expected_findings must be a list, got {type(findings_raw).__name__}")
    findings = [
        ExpectedFinding(
            path=str(item.get("path", "")),
            token=str(item.get("token", "")),
            severity=str(item.get("severity", "medium")),
            summary=str(item.get("summary", "")),
        )
        for item in findings_raw
        if isinstance(item, dict)
    ]
    # `category` is a top-level key in the JSONL so `head -1` shows it, but
    # grouping happens through `metadata`, so it is lifted rather than duplicated
    # in the record itself.
    metadata = dict(raw.get("metadata", {}))
    if raw.get("category") and "category" not in metadata:
        metadata["category"] = str(raw["category"])
    return Case(
        id=str(raw.get("id", "")),
        title=str(raw.get("title", "")),
        language=str(raw.get("language", "python")),
        files=[dict(f) for f in raw.get("files", []) if isinstance(f, dict)],
        expected_findings=findings,
        description=str(raw.get("description", "")),
        acceptance_criteria=[str(c) for c in raw.get("acceptance_criteria", [])],
        constraints=[str(c) for c in raw.get("constraints", [])],
        metadata=metadata,
    )


def load_cases(path: Path | str | None = None, *, limit: int | None = None) -> list[Case]:
    """Read and validate the golden dataset.

    Args:
        path: Dataset to read; defaults to the bundled one.
        limit: Only the first *limit* cases. Applied after validation, so a
            truncated run is still a valid one.

    Returns:
        The validated cases, in file order.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If a record is malformed or internally inconsistent.
    """
    target = Path(path) if path is not None else DEFAULT_DATASET
    if not target.is_file():
        raise FileNotFoundError(
            f"golden dataset not found at {target}. Point --dataset at a JSONL file "
            f"with one case per line; see the module docstring of evals.datasets."
        )
    if limit is not None and limit < 1:
        # Checked before the loop as well as after, because `len(cases) >= 0` is
        # true after the first append, so a limit of 0 would otherwise sail
        # through and silently return one case.
        raise ValueError(f"--limit must be at least 1, got {limit}")
    cases: list[Case] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{number} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{target}:{number} is not a JSON object")
        cases.append(case_from_dict(raw).validated())
        if limit is not None and len(cases) >= limit:
            break

    if limit is not None and not cases:
        # A limit is a request for the first N cases, so a limit that excludes
        # everything is a bad request — silently returning nothing turns a typo
        # in a CI flag into a green build that measured nothing.
        raise ValueError(f"--limit {limit} selected no cases from {target}")

    return cases


__all__ = [
    "DEFAULT_DATASET",
    "REQUIRED_FIELDS",
    "Case",
    "ExpectedFinding",
    "case_from_dict",
    "load_cases",
]
