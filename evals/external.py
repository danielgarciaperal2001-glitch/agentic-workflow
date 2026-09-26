"""Optional third-party scorers, imported only when asked for.

Why this file exists
--------------------

Ragas and DeepEval are excellent scorers and terrible *defaults*. Both pull in
openai, numpy, pandas, datasets and a substantial model download; between them
they add hundreds of megabytes and about ninety seconds to a cold install. The
native metrics in :mod:`evals.metrics` need nothing at all, so a dependency
that every ``pip install`` would pay for to be unused most of the time is the
wrong trade.

So the dependency is declared as an optional extra
(``pip install 'agentic-workflow[eval]'``), imported inside the function that
needs it, and reported as unavailable — loudly — rather than silently skipped.
The failure mode to avoid is a green checkmark that means "the judge was not
installed", because that is indistinguishable from a green checkmark that means
"the output was good".

What they add
-------------

Native metrics can check that a report is *grounded* and *complete*. Neither
can tell whether the finding that was produced is any **right** — that needs a
judge. Ragas and DeepEval supply exactly that, and are worth installing before
trusting a change to the agents:

* **Ragas** scores context precision, context recall and answer relevancy
  against a reference, which is what the golden dataset's ``expected_findings``
  are shaped for.
* **DeepEval** scores answer relevancy, faithfulness and contextual precision
  with per-metric explanations, which is more useful in CI because a failure
  explains itself.

Both are run in *judge* mode: the workflow under test produces the report, and
the third-party library produces the score. They are never in the generation
path, so adding them cannot change what the workflow does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from evals.metrics import MetricScore

#: Extra names, and the pip extra that provides them.
SCORERS: dict[str, str] = {
    "ragas": "agentic-workflow[eval]",
    "deepeval": "agentic-workflow[eval-deepeval]",
}


class ScorerUnavailableError(RuntimeError):
    """A scorer was requested but its dependency is not installed.

    Raised instead of returning a zero or a ``None``, because a missing judge
    and a failing score must never look alike in a report.
    """


@dataclass(frozen=True, slots=True)
class ScorerAvailability:
    """Whether a scorer can run here, and why not if it cannot.

    Attributes:
        name: The scorer's name.
        available: Whether the import succeeded.
        detail: The import error, or the version when it succeeded.
    """

    name: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {"name": self.name, "available": self.available, "detail": self.detail}


def _probe(module: str, version_attr: str) -> ScorerAvailability:
    """Try to import *module* and report the outcome.

    Args:
        module: Importable module name.
        version_attr: Attribute holding the version, tried in turn.

    Returns:
        The availability record. Never raises: probing is a diagnostic, and a
        diagnostic that fails is not a diagnostic.
    """
    from importlib import import_module

    try:
        imported = import_module(module)
    except Exception as exc:
        return ScorerAvailability(module, False, f"{type(exc).__name__}: {exc}")
    version = next(
        (
            str(getattr(imported, attr, ""))
            for attr in (version_attr, "__version__")
            if getattr(imported, attr, None)
        ),
        "unknown",
    )
    return ScorerAvailability(module, True, version)


def availability() -> dict[str, ScorerAvailability]:
    """Report which optional scorers are importable right now.

    Returns:
        One record per scorer, including the ones that are missing, so a report
        can state what it did *not* measure.
    """
    return {name: _probe(name, "__version__") for name in sorted(SCORERS)}


def _require(name: str) -> Any:
    """Import a scorer module or explain how to install it.

    Args:
        name: Scorer name, a key of :data:`SCORERS`.

    Returns:
        The imported module.

    Raises:
        ScorerUnavailableError: If the dependency is not installed.
    """
    probe = _probe(name, "__version__")
    if not probe.available:
        raise ScorerUnavailableError(
            f"the {name} scorer is not installed ({probe.detail}). "
            f"Install it with: pip install '{SCORERS[name]}'"
        )
    from importlib import import_module

    return import_module(name)


def _reference(case: Any) -> str:
    """Render a case's expected findings as the reference a judge compares to.

    Args:
        case: The golden case.

    Returns:
        A short natural-language statement of the known defects.
    """
    if not case.expected_findings:
        return "No defects are known in this change; a correct review reports none."
    return "\n".join(
        f"- {finding.path} ({finding.severity}): {finding.summary or finding.token}"
        for finding in case.expected_findings
    )


def ragas_scores(case: Any, report: dict[str, Any]) -> list[MetricScore]:
    """Score one case with Ragas' context-precision and recall metrics.

    The submitted source is the *retrieved context*, the report is the
    *answer*, and the golden findings are the *reference*. Context precision
    then asks how much of what the report cites is real, and context recall
    asks how much of the real defect it found — the same two questions the
    native metrics ask, but graded by a model rather than by string matching.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        One score per Ragas metric that produced a value.

    Raises:
        ScorerUnavailableError: If ``ragas`` is not installed.
        ScorerUnavailableError: If Ragas produced no usable score.
    """
    from evals.metrics import submitted_text

    _require("ragas")
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import context_precision, context_recall
    except Exception as exc:
        raise ScorerUnavailableError(
            f"ragas is installed but its API does not match what this adapter "
            f"expects ({type(exc).__name__}: {exc}). Install a version with "
            f"`from ragas.metrics import context_precision, context_recall`."
        ) from exc

    import json

    dataset = {
        "user_inputs": [case.description or case.title],
        "retrieved_contexts": [submitted_text(case)],
        "reference": [_reference(case)],
        "response": [str(report.get("markdown", ""))],
    }
    try:
        outcome = ragas_evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall],
            show_progress=False,
        )
        frame = outcome.to_pandas()
    except Exception as exc:
        raise ScorerUnavailableError(f"ragas could not score {case.id}: {exc}") from exc

    scores: list[MetricScore] = []
    for metric in ("context_precision", "context_recall"):
        raw = _column(frame, metric)
        if raw is None:
            continue
        value = _as_unit(raw)
        scores.append(
            MetricScore(
                metric=f"ragas.{metric}",
                value=value,
                # Ragas is a graded judgement with its own noise; holding it to
                # the native threshold of 1.0 would make it an oracle rather
                # than a signal, so it is reported and gated at 0.5.
                passed=value is not None and value >= 0.5,
                evidence=json.loads(str(raw)) if isinstance(raw, str) else [],
            )
        )
    if not scores:
        raise ScorerUnavailableError(f"ragas returned no scores for {case.id}")
    return scores


def deepeval_scores(case: Any, report: dict[str, Any]) -> list[MetricScore]:
    """Score one case with DeepEval's answer-relevancy and faithfulness.

    Args:
        case: The golden case.
        report: The serialised report.

    Returns:
        One score per DeepEval metric that ran.

    Raises:
        ScorerUnavailableError: If ``deepeval`` is not installed or produced no
            usable score.
    """
    from evals.metrics import submitted_text

    _require("deepeval")
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    except Exception as exc:
        raise ScorerUnavailableError(
            f"deepeval is installed but its API does not match what this adapter "
            f"expects ({type(exc).__name__}: {exc})."
        ) from exc

    try:
        outcome = deepeval_evaluate(
            metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()],
            input=case.description or case.title,
            actual_output=str(report.get("markdown", "")),
            retrieval_context=[submitted_text(case)],
        )
    except Exception as exc:
        raise ScorerUnavailableError(f"deepeval could not score {case.id}: {exc}") from exc

    scores: list[MetricScore] = []
    for item in outcome:
        name = str(getattr(item, "metric", None) or type(item).__name__).lower()
        raw = getattr(item, "score", None)
        value = None if raw is None else max(0.0, min(1.0, float(raw)))
        reason = str(getattr(item, "reason", "") or "")
        scores.append(
            MetricScore(
                metric=f"deepeval.{name}",
                value=value,
                passed=value is not None and value >= 0.5,
                evidence=[reason] if reason else [],
            )
        )
    if not scores:
        raise ScorerUnavailableError(f"deepeval returned no scores for {case.id}")
    return scores


def _column(frame: Any, name: str) -> Any:
    """Read one column from a Ragas result frame.

    Args:
        frame: A pandas ``DataFrame`` or a mapping.
        name: Column name.

    Returns:
        The column's single value, or ``None`` when absent or empty.
    """
    try:
        if hasattr(frame, "columns"):
            if name not in frame.columns:
                return None
            series = frame[name]
            return series.iloc[0] if len(series) else None
        value = frame.get(name)
    except Exception:
        return None
    return value


def _as_unit(raw: Any) -> float | None:
    """Coerce a judge score to ``0.0``-``1.0``.

    Ragas has historically returned ``0``-``1``, ``NaN`` for a failed metric,
    and occasionally a string. All three are handled, because a judge that
    cannot be read must not silently become a perfect score.

    Args:
        raw: The value Ragas produced.

    Returns:
        A unit-interval float, or ``None`` when the value is unusable.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return max(0.0, min(1.0, value))


__all__ = [
    "SCORERS",
    "ScorerAvailability",
    "ScorerUnavailableError",
    "availability",
    "deepeval_scores",
    "ragas_scores",
]
