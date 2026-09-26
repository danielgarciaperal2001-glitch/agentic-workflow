"""Render a :class:`~evals.runners.SuiteReport` for a terminal.

Written for the person who has to act on it
-------------------------------------------

A CI log is read by someone whose change just failed. So the rendering answers
three questions in this order: did it pass, which cases failed, and what exactly
was wrong with each. Everything else — per-metric means, category breakdowns,
scorer availability — is printed after, because it is context for the failure
and noise when there is none.

Three rules the renderer holds to:

* **A missing judge is printed as missing.** When Ragas or DeepEval is not
  installed, the suite says so in a line of its own. A report that silently
  omits an unrun metric is how a team concludes their hallucination check is
  passing when nothing checked anything.
* **Failures show evidence, not just a number.** ``grounding 0.5`` tells nobody
  anything; ``ungrounded: `def total(items): return sum(...)` `` does.
* **An empty run is not a pass.** Zero cases prints a failure, because "the
  dataset did not load" and "the workflow is perfect" must not look alike.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from evals.runners import CaseResult, SuiteReport

#: Width of the rule lines. Wide enough for a report line, short enough not to
#: wrap in a default 80-column terminal.
WIDTH = 72


def _rule(title: str = "") -> str:
    """Return a horizontal rule, optionally titled.

    Args:
        title: Text to centre in the rule.

    Returns:
        The rendered rule, with a leading and trailing newline so it can be
        printed on its own line.
    """
    if not title:
        return "\n" + "─" * WIDTH + "\n"
    label = f" {title} "
    pad = max(0, WIDTH - len(label) - 2)
    left = pad // 2
    return "\n" + "─" * left + label + "─" * (pad - left) + "\n"


def _verdict(passed: bool) -> str:
    """Return a compact pass/fail marker.

    Args:
        passed: Whether the thing passed.

    Returns:
        ``PASS`` or ``FAIL``.
    """
    return "PASS" if passed else "FAIL"


def _score_line(score: Any) -> str:
    """Render one metric score.

    Args:
        score: The :class:`~evals.metrics.MetricScore`.

    Returns:
        A single line: the metric, its value, and whether it passed.
    """
    value = "  n/a" if score.value is None else f"{score.value:6.3f}"
    return f"    {_verdict(score.passed)}  {score.metric:<22} {value}"


def render_case(case: CaseResult) -> str:
    """Render one case's result.

    Args:
        case: The case result.

    Returns:
        A block naming the case, its verdict, and every failing metric with its
        evidence.
    """
    lines = [
        (
            f"  {_verdict(case.passed)}  {case.case_id}  "
            f"[{case.category}]  decision={case.decision or '-'}  "
            f"findings={case.findings}  gates={case.gates}  "
            f"{case.duration_seconds:.2f}s"
        )
    ]
    if case.error:
        lines.append(f"        error: {case.error}")
    for score in case.scores:
        if score.passed and not score.evidence:
            continue
        lines.append(_score_line(score))
        # Only a failure's evidence is worth printing: a passing metric's notes
        # ("case declares no expected findings") are noise in a CI log.
        for note in (score.evidence if not score.passed else [])[:5]:
            lines.append(f"            - {note}")
    return "\n".join(lines)


def render(report: SuiteReport) -> str:
    """Render a full suite report.

    Args:
        report: The suite report.

    Returns:
        Multi-line text, ready to print.
    """
    parts: list[str] = [_rule("evaluation")]
    failing = report.failing_cases()
    total = len(report.cases)
    gated = total - len(failing)
    parts.append(
        f"  {_verdict(report.passed)}  "
        f"{gated}/{total} cases passed  "
        f"provider={report.provider}  gates={report.gate_verdict}  "
        f"{report.duration_seconds:.1f}s"
    )
    parts.append(f"        dataset={report.dataset}")
    if report.gate_metrics is not None:
        # Stated on the report rather than left to the invocation, because a
        # "20/20 passed" that is really "20/20 passed the six things that do not
        # depend on the model" is a claim the reader cannot evaluate from the
        # number alone.
        parts.append(f"        gated on: {', '.join(sorted(report.gate_metrics))}")
        parts.append(
            "        the metrics below are all computed; the verdict reads only the gated ones."
        )

    if not total:
        # The failure that matters most is the one that produced no output at
        # all, and it is the one an empty list renders as success.
        parts.append(_rule("no cases ran"))
        parts.append(
            "  The dataset produced zero cases, so nothing was measured. This is\n"
            "  reported as a failure: an unrun suite is not a passing suite."
        )
        return "\n".join(parts)

    if failing:
        parts.append(_rule(f"failing cases ({len(failing)})"))
        parts.extend(render_case(case) for case in failing)

    parts.append(_rule("metrics"))
    summary = report.metric_summary()
    if summary:
        parts.append(f"  {'metric':<24} {'mean':>7} {'min':>7} {'max':>7} {'failed':>7}  cases")
        for metric, stats in summary.items():
            parts.append(
                f"  {metric:<24} {stats['mean']:>7.3f} {stats['min']:>7.3f} "
                f"{stats['max']:>7.3f} {int(stats['failed']):>7}  "
                f"{int(stats['counted'])}"
            )
    else:
        parts.append("  (no metric produced a value)")

    # A metric that is installed but was not asked for, or asked for and could
    # not run, has to be visible. Silence here reads as "we checked".
    unavailable = [p for p in report.scorers.values() if not p.available]
    parts.append(_rule("optional scorers"))
    for name, probe in sorted(report.scorers.items()):
        if probe.available:
            parts.append(f"  available  {name} {probe.detail}")
        else:
            parts.append(f"  missing    {name} — {probe.detail}")
    if unavailable:
        parts.append(
            "\n  These judge-based metrics did NOT run. The native metrics above\n"
            "  check that the report is grounded and complete; they cannot tell\n"
            "  whether a finding is any RIGHT. Install the extras for that:\n"
            "      pip install 'agentic-workflow[eval,eval-deepeval]'"
        )

    by_category = report.by_category()
    if len(by_category) > 1:
        parts.append(_rule("by category"))
        for category, tally in sorted(by_category.items()):
            rate = tally["passed"] / tally["cases"] if tally["cases"] else 0.0
            filled = round(rate * 10)
            bar = "█" * filled + "·" * (10 - filled)
            parts.append(f"  {category:<20} {bar} {tally['passed']}/{tally['cases']}")

    parts.append(_rule(""))
    parts.append(
        f"  {report.threshold:.2f} threshold on every gated metric, "
        f"per case. A mean would hide the case that broke."
    )
    return "\n".join(parts)


def render_diff(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Render the metric deltas between two suite reports.

    Useful in a pull request: the numbers matter less than which way they moved.
    A metric that improved while the suite still fails is not progress.

    Args:
        before: The baseline report's :meth:`~evals.runners.SuiteReport.as_dict`.
        after: The current report's mapping.

    Returns:
        A table of changes, largest regression first.
    """
    old = dict((before or {}).get("metrics", {}))
    new = dict((after or {}).get("metrics", {}))
    if not old and not new:
        return "  no metrics in either report"
    rows: list[tuple[str, float, float, float]] = []
    for metric in sorted(set(old) | set(new)):
        was = float(old.get(metric, {}).get("mean", 0.0))
        now = float(new.get(metric, {}).get("mean", 0.0))
        rows.append((metric, was, now, now - was))
    rows.sort(key=lambda row: row[3])
    lines = [f"  {'metric':<24} {'before':>8} {'after':>8} {'delta':>8}"]
    for metric, was, now, delta in rows:
        arrow = "" if abs(delta) < 1e-9 else ("▼" if delta < 0 else "▲")
        lines.append(f"  {metric:<24} {was:>8.3f} {now:>8.3f} {delta:>+8.3f} {arrow}")
    return "\n".join(lines)


__all__ = ["WIDTH", "render", "render_case", "render_diff"]
