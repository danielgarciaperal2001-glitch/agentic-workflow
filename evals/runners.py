"""Run the review workflow over the golden dataset and score what comes out.

The shape of a run
------------------

Each case is driven through the *real* graph — the same
:class:`~agentic_workflow.services.engine.WorkflowEngine` the API uses, the same
agents, the same gates — against an in-memory checkpointer. Nothing is stubbed
except persistence, because a stubbed graph measures the stub. The gate verdict
is chosen by the caller (``--gates approve``) so the same dataset can be scored
on the happy path and on a hostile one.

A case that raises is **recorded and counted, not swallowed**. A suite that
drops its failures is worse than no suite, because the absence of failures is
indistinguishable from the absence of runs.

What "passing" means
--------------------

Native metrics must reach :data:`~evals.metrics.THRESHOLD` (1.0) on a case to
count. A partial score is a failure with a number attached, and the evidence
list says which claims were wrong. Judge metrics are gated at 0.5 because they
are graded rather than computed, and holding a judgement to bit-exactness
would make the gate flap.

The suite passes when *every* case passes every native metric. That is strict on
purpose: a golden set of twenty hand-checked cases is small enough that a
regression is visible in the diff, and a mean would hide exactly the case that
broke.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any

from evals import external
from evals.datasets import Case, load_cases
from evals.metrics import (
    DETECTION_METRICS,
    INVARIANT_METRICS,
    NATIVE_METRICS,
    NATIVE_METRICS_BY_NAME,
    THRESHOLD,
    MetricScore,
    metric_name,
)

#: How many cases run at once. The engine serialises graph execution per run,
#: but each run is independent and spends most of its wall-clock inside the
#: provider, so a small pool is a real saving. Kept modest: the point is not
#: throughput, it is a score that does not depend on machine load.
DEFAULT_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class CaseResult:
    """What one golden case produced, and how it scored.

    Attributes:
        case_id: The case's identifier, so a failure names itself.
        category: The case's defect category, for grouping.
        status: Terminal run status, or ``error`` if the run itself failed.
        decision: The verdict the report published.
        findings: Findings in the produced report.
        gates: Gates the run presented, and how many were answered.
        scores: One entry per metric evaluated.
        duration_seconds: Wall-clock for this case.
        error: The failure, if the run or a metric raised.
        report: The serialised report, kept so a failure can be diffed.
    """

    case_id: str
    category: str
    status: str
    decision: str
    findings: int
    gates: int
    scores: tuple[MetricScore, ...]
    duration_seconds: float
    error: str | None = None
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether every metric for this case met its threshold."""
        return self.error is None and all(score.passed for score in self.scores)

    def gated_passed(self, gate_names: frozenset[str] | None) -> bool:
        """Whether this case clears the metrics a gate actually covers.

        A run can be a *good review of nothing* and still be correct in every way
        the invariants measure. Deciding whether that is a pass depends entirely
        on what the caller is gating on, so it is a parameter rather than a
        property: gating on every metric turns the null-model baseline into a
        permanently failing job, and a permanently failing job gets ignored.

        Args:
            gate_names: Metric names the gate covers, or ``None`` for all of
                them.

        Returns:
            ``True`` when the case errored never and every gated metric passed.
        """
        if self.error is not None:
            return False
        if gate_names is None:
            return all(score.passed for score in self.scores)
        return all(score.passed for score in self.scores if score.metric in gate_names)

    def failing(self, gate_names: frozenset[str] | None = None) -> list[MetricScore]:
        """Return the metrics that did not pass.

        Args:
            gate_names: Restrict to these metric names, or ``None`` for all.

        Returns:
            The failing scores, in reporting order.
        """
        return [
            score
            for score in self.scores
            if not score.passed and (gate_names is None or score.metric in gate_names)
        ]

    def as_dict(self, *, include_report: bool = False) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Args:
            include_report: Include the full report body. Off by default, since
                twenty reports make a CI log unreadable and the interesting part
                is the score.

        Returns:
            The mapping, with the report attached only when asked for.
        """
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "category": self.category,
            "status": self.status,
            "decision": self.decision,
            "findings": self.findings,
            "gates": self.gates,
            "passed": self.passed,
            "duration_seconds": round(self.duration_seconds, 4),
            "error": self.error,
            "scores": [score.as_dict() for score in self.scores],
        }
        if include_report:
            payload["report"] = self.report
        return payload


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """The outcome of scoring the whole dataset.

    Attributes:
        provider: The LLM provider the runs used.
        dataset: Where the cases came from.
        gate_verdict: What the automated reviewer answered at each gate.
        cases: Per-case results, in dataset order.
        duration_seconds: Wall-clock for the whole suite.
        scorers: Availability of the optional third-party scorers, so a report
            states what it did not measure.
        threshold: The native threshold applied.
        gate_metrics: Metric names the pass/fail verdict covers, or ``None`` for
            all of them. Scoped by :func:`~evals.runners.gate_scope` so a build
            can gate on the provider-independent invariants while still
            *reporting* the detection metrics.
    """

    provider: str
    dataset: str
    gate_verdict: str
    cases: tuple[CaseResult, ...]
    duration_seconds: float
    scorers: dict[str, external.ScorerAvailability] = field(default_factory=dict)
    threshold: float = THRESHOLD
    gate_metrics: frozenset[str] | None = None

    @property
    def passed(self) -> bool:
        """Whether every case passed every gated metric.

        A dataset that could not be read produces an empty ``cases`` tuple, and
        an empty suite does **not** pass: "no cases" must never be green.
        """
        return bool(self.cases) and all(case.gated_passed(self.gate_metrics) for case in self.cases)

    def metric_summary(self) -> dict[str, dict[str, float]]:
        """Aggregate every metric across the suite.

        Averages are computed over cases where the metric applied, so a case
        with no expected findings is excluded from recall instead of scoring it
        zero. ``failed`` counts the cases that missed the threshold.

        Returns:
            Metric name to ``{"mean", "min", "max", "failed", "counted"}``.
        """
        collected: dict[str, list[float]] = {}
        failures: dict[str, int] = {}
        for case in self.cases:
            for score in case.scores:
                if not score.passed:
                    failures[score.metric] = failures.get(score.metric, 0) + 1
                if score.value is not None:
                    collected.setdefault(score.metric, []).append(score.value)
        summary: dict[str, dict[str, float]] = {}
        for metric, values in collected.items():
            summary[metric] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "counted": len(values),
                "failed": failures.get(metric, 0),
            }
        return summary

    def by_category(self) -> dict[str, dict[str, float]]:
        """Aggregate pass rate per defect category, over **every** metric.

        Deliberately not scoped by ``gate_metrics``. This view exists to show
        where the next prompt change is needed, and a view that only counted
        the invariants would report every category as perfect — which is exactly
        the signal it is supposed to be sensitive to. Gating is
        :attr:`passed`; diagnosis is this.

        A single overall number hides that the workflow is excellent on SQL
        injection and useless on numeric correctness.

        Returns:
            Category name to ``{"cases", "passed"}``. The counts are returned as
            floats so the mapping has one value type throughout, which is what
            makes it safe to embed in JSON alongside the metric statistics.
        """
        tally: dict[str, dict[str, int]] = {}
        for case in self.cases:
            entry = tally.setdefault(case.category, {"cases": 0, "passed": 0})
            entry["cases"] += 1
            entry["passed"] += 1 if case.passed else 0
        return {
            name: {key: float(value) for key, value in stats.items()}
            for name, stats in tally.items()
        }

    def failing_cases(self) -> list[CaseResult]:
        """Return the cases that did not pass the gate.

        Scoped by ``gate_metrics``, because a case that misses only a
        non-gated metric is not something a reader of this report should go and
        fix. Its scores are still attached, so the evidence is there for anyone
        who wants the wider picture.

        Returns:
            The failing cases, in dataset order.
        """
        return [case for case in self.cases if not case.gated_passed(self.gate_metrics)]

    def as_dict(self, *, include_reports: bool = False) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Args:
            include_reports: Attach each case's full report.

        Returns:
            The mapping, safe to ``json.dumps`` and to diff between commits.
        """
        return {
            "provider": self.provider,
            "dataset": self.dataset,
            "gate_verdict": self.gate_verdict,
            "threshold": self.threshold,
            "gate_metrics": sorted(self.gate_metrics) if self.gate_metrics else None,
            "passed": self.passed,
            "cases_run": len(self.cases),
            "cases_passed": sum(1 for case in self.cases if case.gated_passed(self.gate_metrics)),
            "duration_seconds": round(self.duration_seconds, 3),
            "metrics": self.metric_summary(),
            "by_category": self.by_category(),
            "scorers": {name: probe.as_dict() for name, probe in sorted(self.scorers.items())},
            "cases": [case.as_dict(include_report=include_reports) for case in self.cases],
        }


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _gate_decider(verdict: str, operator: str) -> Any:
    """Build the callable that answers every gate with *verdict*.

    The resume value is a plain mapping, built here rather than through
    :class:`~agentic_workflow.domain.schemas.ApprovalDecision`, because the
    engine's ``run_until_done`` takes the mapping directly and constructing the
    full model would add a validation step that the CLI path does not have.
    Keeping the harness on the *same* shape the CLI uses means a bug in gate
    resumption shows up in the demo too, not only in the eval suite.

    Args:
        verdict: ``approve``, ``edit`` or ``reject``.
        operator: Name recorded as the decider, so an audit shows the automation
            rather than a person who never looked.

    Returns:
        A ``Decider``-shaped callable.
    """

    def decide(pending: Any) -> dict[str, Any]:
        """Answer one gate.

        Args:
            pending: The gate the run is parked on.

        Returns:
            The resume value for *verdict*.
        """
        return {
            "approval_id": pending.approval_id,
            "decision": verdict,
            "reviewer": operator,
            "comment": f"answered by the eval harness with `{verdict}`",
        }

    return decide


async def _run_case(
    case: Case,
    *,
    gate_verdict: str,
    engine: Any,
    scorers: tuple[str, ...],
) -> CaseResult:
    """Run one case through the workflow and score its report.

    Args:
        case: The golden case.
        gate_verdict: What to answer at each gate.
        engine: A started engine to run through.
        scorers: Optional third-party scorers to apply.

    Returns:
        The case's result, including a populated ``error`` if anything raised.
    """
    started = time.perf_counter()
    run_id = f"eval-{case.id}"
    try:
        outcome = await engine.run_until_done(
            case.to_request(run_id),
            decide=_gate_decider(gate_verdict, "eval-harness"),
            max_gates=12,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            category=case.category,
            status="error",
            decision="",
            findings=0,
            gates=0,
            scores=(),
            duration_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    if outcome.report is None:
        return CaseResult(
            case_id=case.id,
            category=case.category,
            status=str(outcome.status),
            decision="",
            findings=0,
            gates=len(outcome.decisions),
            scores=(),
            duration_seconds=time.perf_counter() - started,
            error=outcome.error or f"run produced no report (status={outcome.status})",
        )

    report = outcome.report.model_dump(mode="json")
    scores: list[MetricScore] = []
    for metric in NATIVE_METRICS:
        try:
            scores.append(metric(case, report))
        except Exception as exc:
            # The registered name, not the function's `__name__`: a failure has
            # to be attributable to a metric a `--gate` can select, and the two
            # spellings differ the moment a function is renamed.
            scores.append(
                MetricScore(
                    metric=metric_name(metric),
                    value=None,
                    passed=False,
                    evidence=[f"metric raised {type(exc).__name__}: {exc}"],
                )
            )
    for name in scorers:
        adapter = external.ragas_scores if name == "ragas" else external.deepeval_scores
        try:
            scores.extend(adapter(case, report))
        except external.ScorerUnavailableError as exc:
            # A judge that cannot run is not a zero; it is absent, and the
            # suite's `scorers` block says so at the top level too.
            scores.append(MetricScore(metric=name, value=None, passed=False, evidence=[str(exc)]))

    return CaseResult(
        case_id=case.id,
        category=case.category,
        status=str(outcome.status),
        decision=str(report.get("decision", "")),
        findings=len(report.get("findings", []) or []),
        gates=len(outcome.decisions),
        scores=tuple(scores),
        duration_seconds=time.perf_counter() - started,
        report=report,
    )


#: Accepted ``gate=`` values, and what each one covers.
GATE_SCOPES: dict[str, str] = {
    "all": "every metric, including detection",
    "invariants": "provider-independent correctness properties only",
    "detection": "whether the model found the seeded defects",
}


def gate_scope(gate: str) -> frozenset[str] | None:
    """Resolve a gate name to the metric names it covers.

    The distinction this function exists to make: the invariants
    (:data:`~evals.metrics.INVARIANT_METRICS`) are properties of the
    *workflow* — never quote code that was not submitted, never invent a path,
    never approve over your own critical finding — and the detection metrics
    (:data:`~evals.metrics.DETECTION_METRICS`) are properties of the *model*.

    That split is what makes an offline gate possible. A CI job that gated on
    every metric would have to spend a token to run, and gating the null
    provider on recall would fail forever. A CI job that gated on nothing would
    prove nothing at all. Gating the invariants keeps the job free,
    reproducible, and still able to catch the class of regression that actually
    breaks this system.

    Args:
        gate: ``all``, ``invariants`` or ``detection``.

    Returns:
        A frozenset of metric names, or ``None`` for every metric.

    Raises:
        ValueError: If *gate* is not one of :data:`GATE_SCOPES`.
    """
    if gate not in GATE_SCOPES:
        raise ValueError(f"gate must be one of {', '.join(sorted(GATE_SCOPES))}, got {gate!r}")
    if gate == "all":
        return None
    source = INVARIANT_METRICS if gate == "invariants" else DETECTION_METRICS
    unknown = [name for name in source if name not in NATIVE_METRICS_BY_NAME]
    if unknown:
        # A gate that covers nothing passes everything, which is the exact
        # failure the gate exists to prevent. Caught here rather than at scoring
        # time, where it would have to be inferred from an all-green run.
        raise ValueError(f"gate {gate!r} names unregistered metrics: {', '.join(unknown)}")
    return frozenset(source)


async def run_suite(
    *,
    dataset: str | Any | None = None,
    provider: str = "echo",
    limit: int | None = None,
    memory: bool = True,
    gates: str = "approve",
    scorers: tuple[str, ...] = (),
    concurrency: int = DEFAULT_CONCURRENCY,
    gate: str = "all",
) -> SuiteReport:
    """Score the workflow's output quality over a golden dataset.

    Args:
        dataset: JSONL path, or a list of already-parsed cases. ``None`` uses
            the bundled golden set.
        provider: LLM provider to run the workflow with. The default ``echo``
            provider is deterministic and offline, so the score is
            reproducible; a real provider measures the real workflow and costs
            money.
        limit: Only the first *limit* cases.
        memory: Must be ``True``. The suite grades against an in-memory
            checkpointer; passing ``False`` is refused rather than ignored, so a
            caller cannot believe its golden runs were persisted.
        gates: Verdict applied at every gate — ``approve``, ``edit`` or
            ``reject``.
        scorers: Optional third-party scorers to add (``ragas``,
            ``deepeval``). Availability is reported whether or not they work.
        gate: Which metrics the pass/fail verdict covers — see
            :func:`gate_scope`. Every metric is always *computed* and reported;
            this only decides which of them the verdict reads.
        concurrency: Cases in flight at once.

    Returns:
        A :class:`SuiteReport`.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset is malformed, or *gates* is not a verdict.
    """
    if gates not in ("approve", "edit", "reject"):
        raise ValueError(f"gates must be approve, edit or reject, got {gates!r}")
    # Resolved before the cases are read, so a typo'd gate is an error about the
    # flag rather than a suite that runs for a minute and then fails.
    gated = gate_scope(gate)
    if not memory:
        # Refused rather than ignored. Honouring the flag would write twenty
        # golden cases into whatever checkpoint store the environment points at
        # — usually the operator's real one — and then hold them to the retention
        # sweep. Ignoring it silently would leave the caller believing the runs
        # were durable, which is the kind of belief that turns into an incident
        # when someone goes looking for a score they thought was saved.
        raise ValueError(
            "run_suite always uses an in-memory checkpointer: grading a dataset "
            "must not write into the configured store. Drop the memory=False, or "
            "call the engine directly if you need a durable run."
        )

    cases: list[Case]
    dataset_name: str
    if isinstance(dataset, list):
        cases, dataset_name = list(dataset), "<in-memory>"
    else:
        cases = load_cases(dataset, limit=limit)
        dataset_name = (
            str(dataset) if dataset is not None else "evals/datasets/pr_review_golden.jsonl"
        )

    availability = external.availability()
    started = time.perf_counter()
    results = await _run_all(
        cases,
        provider=provider,
        gate_verdict=gates,
        scorers=tuple(scorers),
        concurrency=max(1, concurrency),
    )
    return SuiteReport(
        provider=provider,
        dataset=dataset_name,
        gate_verdict=gates,
        cases=tuple(results),
        duration_seconds=time.perf_counter() - started,
        scorers=availability,
        gate_metrics=gated,
    )


async def _run_all(
    cases: list[Case],
    *,
    provider: str,
    gate_verdict: str,
    scorers: tuple[str, ...],
    concurrency: int,
) -> list[CaseResult]:
    """Run every case, bounded in flight, against one engine.

    A single engine is shared deliberately: it owns the compiled graph and the
    checkpointer, and building one per case would measure graph compilation
    rather than review quality. The checkpointer is always in-memory — golden
    cases must not evict each other's history, and grading a dataset must not
    require a database to be up.

    Args:
        cases: The cases to run.
        provider: LLM provider name.
        gate_verdict: Verdict applied at each gate.
        scorers: Optional third-party scorers.
        concurrency: Maximum cases in flight.

    Returns:
        Results in dataset order.
    """
    from agentic_workflow.config import load_settings
    from agentic_workflow.persistence.checkpointer import build_memory_checkpointer
    from agentic_workflow.services.engine import WorkflowEngine

    if not cases:
        return []
    settings = load_settings(
        environment="development",
        llm_provider=provider,
        postgres_enabled=False,
        hitl_enabled=True,
        log_level="ERROR",
        max_iterations=3,
    )
    checkpointer = build_memory_checkpointer()
    engine = WorkflowEngine(settings, checkpointer=checkpointer)
    await engine.startup()
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(case: Case) -> CaseResult:
        """Run one case under the concurrency bound.

        Args:
            case: The case to run.

        Returns:
            The case's result.
        """
        async with semaphore:
            return await _run_case(
                case,
                gate_verdict=gate_verdict,
                engine=engine,
                scorers=scorers,
            )

    try:
        return list(await asyncio.gather(*(guarded(case) for case in cases)))
    finally:
        await engine.shutdown()
        close = getattr(checkpointer, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result


def smoke(case: Case | None = None) -> CaseResult:
    """Score one case with a hand-written report, to prove the metrics work.

    Exists because a metric that has never been shown to fail is a metric that
    has never been tested. This builds a report that is *deliberately* broken —
    an invented file path, a quoted line nobody wrote, an ``approved`` verdict
    next to a critical finding — and asserts the suite rejects it. Called from
    ``tests/eval/``.

    Args:
        case: The case to score against; defaults to the first golden one.

    Returns:
        The :class:`CaseResult` for the fabricated report.
    """
    target = case or (load_cases(limit=1) or [])[0]
    broken = {
        "decision": "approved",
        "markdown": (
            "## Decision\n\nApproved.\n\n## Findings\n\n"
            "- Nothing of note in this change.\n"
            "```python\ndef total(items):\n    return sum(i for i in items)\n```\n"
        ),
        "findings": [
            {
                "id": "f1",
                "title": "Totals should use integer cents",
                "detail": "Summing floats accumulates representation error.",
                "severity": "critical",
                "category": "correctness",
                "file": "billing/invoices/total.py",
                "line": 12,
                "recommendation": "Use decimal.Decimal.",
                "confidence": 0.9,
            }
        ],
        "citations": ["billing/invoices/total.py:12"],
    }
    scores = tuple(metric(target, broken) for metric in NATIVE_METRICS)
    return CaseResult(
        case_id=f"{target.id}-fabricated",
        category=target.category,
        status="fabricated",
        decision=str(broken["decision"]),
        findings=1,
        gates=0,
        scores=scores,
        duration_seconds=0.0,
        report=broken,
    )


__all__ = [
    "DEFAULT_CONCURRENCY",
    "GATE_SCOPES",
    "CaseResult",
    "SuiteReport",
    "gate_scope",
    "run_suite",
    "smoke",
]
