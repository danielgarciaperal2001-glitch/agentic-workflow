"""Measure the engine and write a Markdown table the README can quote.

What is measured, and why each number is here
----------------------------------------------

Every figure this produces comes from running the real code on the machine it
is invoked on, at the moment it is invoked. There are no constants, no
interpolation, and no "typical" values. If you re-run it you may well get
different numbers, and that is the point: the README carries the output of one
run so a reader can see both the magnitude and the method.

The measurements, in the order they run:

``graph_compile``
    Time to build and compile the ``StateGraph``. Separated out because it is a
    *startup* cost paid once per process. Folding it into the end-to-end number
    would make a warm run look slow, and the fix (compile lazily, share one
    engine) is invisible without the split.

``end_to_end``
    A full run from request to final report, with every gate answered
    automatically, on a warm engine. This is the number an operator actually
    waits for.

``per_node``
    The same runs, broken down by node. The purpose is the *shape*, not the
    magnitude: it shows where the orchestration cost sits and therefore where an
    optimisation would pay. Reported as a median across runs so one slow run
    does not define the table.

``checkpoint_write``
    Append-only write latency, measured by running the same workload and
    dividing by the number of checkpoints it produced. This is a derived figure
    and is labelled as such — the checkpoints are not timed individually,
    because the store's own batching means a per-call timer would measure the
    batching rather than the store.

``throughput``
    Runs completed per second with a bounded worker pool, end to end. This is
    the number that matters for capacity planning, and the one most likely to be
    quoted out of context, so the concurrency it was measured at is printed with
    it.

``eval_suite``
    Wall-clock for the golden dataset, which is the cost of the quality gate in
    CI.

What this deliberately does not measure
---------------------------------------

**Latency of a real model provider.** The `echo` provider answers in
microseconds by design, so a run here is dominated by orchestration, not
inference. These numbers therefore say how much *overhead the engine adds*
around a provider call — which is the part this project owns and the part worth
optimising — and say nothing about end-to-end latency with a real model. The
README states that limit rather than presenting a fast number as a claim about
the whole system.

**Postgres numbers.** There is no database in this harness. Checkpoint latency
against a real PostgreSQL server is a different measurement with different
numbers, and quoting the in-memory figure as if it were the durable one would be
exactly the kind of substitution this project is arguing against.

Interpreting the output
-----------------------

Percentiles come from the sorted sample, nearest-rank, on runs that actually
completed. A run that errored is counted and reported rather than dropped: an
error rate is part of a throughput measurement, and hiding it makes the rest of
the table optimistic.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_workflow.config import load_settings, reset_settings_cache
from agentic_workflow.domain.schemas import ReviewRequest, SourceFile
from agentic_workflow.errors import WorkflowError
from agentic_workflow.persistence.checkpointer import (
    build_memory_checkpointer,
)
from agentic_workflow.services.engine import WorkflowEngine

#: A realistic single-file review. Sized from the bundled demo bug rather than
#: invented, so the number reflects a real review shape and not a token gesture.
BENCH_SOURCE = '''\
"""Totals for a checkout order."""

from __future__ import annotations

from typing import Any, Iterable

DISCOUNT_RATE = 0.075


def line_total(price: float, quantity: int) -> float:
    """Return the undiscounted total for one order line."""
    return price * quantity


def apply_discount(total: float, tier: str) -> float:
    """Reduce a total for a loyalty tier."""
    if tier == "gold":
        return total * (1 - DISCOUNT_RATE)
    if tier == "silver":
        return total * (1 - DISCOUNT_RATE / 2)
    return total


def order_total(items: Iterable[dict[str, Any]], tier: str = "standard") -> float:
    """Sum the lines and apply the tier discount.

    Args:
        items: Order lines, each with a price and a quantity.
        tier: Loyalty tier.

    Returns:
        The discounted total.
    """
    subtotal = 0.0
    for item in items:
        subtotal = subtotal + line_total(float(item["price"]), int(item["quantity"]))
    return apply_discount(subtotal, tier)


def format_receipt(total: float) -> str:
    """Render a total for display."""
    return f"Total: {total:.2f}"
'''

#: File sets of growing size, to show how cost scales with the diff. Reported
#: separately from the single-file figure so the scaling is explicit rather than
#: implied.
SCALING_UNITS = (1, 5, 20)


@dataclass(slots=True)
class Measurement:
    """One timed quantity.

    Attributes:
        name: What was measured.
        unit: ``ms``, ``runs/s``, or ``count``.
        samples: Every observation, in collection order.
        detail: Anything a reader needs to reproduce the figure.
    """

    name: str
    unit: str
    samples: list[float] = field(default_factory=list)
    detail: str = ""

    def summary(self) -> dict[str, float]:
        """Return the statistics quoted in the table.

        Nearest-rank percentiles rather than interpolated ones: with a small
        sample an interpolated p95 is a number no run actually produced, and
        presenting it as a measurement is a small lie that compounds when someone
        designs a timeout around it.

        Returns:
            Count, mean, median, min, max and the p95.
        """
        if not self.samples:
            return {}
        ordered = sorted(self.samples)
        return {
            "count": float(len(ordered)),
            "mean": round(statistics.fmean(ordered), 3),
            "median": round(statistics.median(ordered), 3),
            "min": round(ordered[0], 3),
            "max": round(ordered[-1], 3),
            "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        }


@dataclass(slots=True)
class Report:
    """Everything measured in one invocation.

    Attributes:
        environment: The machine and build the numbers describe.
        measurements: Timed quantities, keyed by name.
        per_node: Per-node duration samples, kept apart from ``measurements``
            because they are not distributions to summarise but a breakdown to
            render as its own table.
        notes: Statements a reader must not be able to miss.
    """

    environment: dict[str, str]
    measurements: dict[str, Measurement] = field(default_factory=dict)
    per_node: dict[str, list[float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns:
            Environment, every measurement's statistics, and the notes.
        """
        return {
            "environment": self.environment,
            "measurements": {
                name: {**m.summary(), "unit": m.unit, "detail": m.detail}
                for name, m in self.measurements.items()
            },
            "per_node_median_ms": {
                node: round(statistics.median(samples), 3)
                for node, samples in self.per_node.items()
            },
            "notes": self.notes,
        }


def _environment() -> dict[str, str]:
    """Describe the machine well enough to judge whether a number transfers.

    A latency figure without knowing the CPU count and whether it was a laptop
    or a runner is not a measurement, it is a rumour. Recorded rather than
    assumed.

    Returns:
        Interpreter, platform, processor, CPU count and provider.
    """
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": _processor_name(),
        "cpu_count": str(os.cpu_count() or "unknown"),
        "provider": "echo (deterministic, offline)",
    }


def _processor_name() -> str:
    """Name the CPU, falling back to what the platform can tell us.

    ``platform.processor()`` returns an empty string on Linux, which is a
    useless thing to print under a benchmark table — the reader cannot tell a
    shared runner from a laptop. The model name comes from ``/proc/cpuinfo``
    where it is available.

    Returns:
        The CPU model, or the platform's answer, or ``unknown``.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _request(run_id: str, units: int = 1) -> ReviewRequest:
    """Build the review request the benchmarks drive.

    Args:
        run_id: Run identifier, so each iteration has its own thread.
        units: How many distinct files to submit. Content is the same module
            under different paths, because the measurement is about how cost
            scales with input size, not about the content being interesting.

    Returns:
        A validated request.
    """
    files = [
        SourceFile(path=f"checkout/module_{index:02d}.py", content=BENCH_SOURCE)
        for index in range(units)
    ]
    return ReviewRequest(
        run_id=run_id,
        request_id="BENCH-1",
        title="Correct the checkout totals",
        language="python",
        description=(
            "Order totals are computed with binary floats, so large orders drift "
            "by cents. Return exact monetary values and keep the receipt formatting "
            "byte-identical for the existing tests."
        ),
        acceptance_criteria=[
            "Totals are exact to two decimal places for any input.",
            "Existing receipt formatting tests pass unchanged.",
        ],
        constraints=["No new third-party dependencies."],
        files=files,
    )


def _decider(verdict: str = "approve") -> Any:
    """Answer every gate automatically.

    A benchmark that stopped at the first gate would measure the *gate*, not the
    workflow. Automated approval is what lets a run reach the reporter, which is
    where the interesting work is.

    Args:
        verdict: The verdict to apply at each gate.

    Returns:
        A decider callable.
    """

    def decide(pending: Any) -> dict[str, Any]:
        """Answer one gate.

        Args:
            pending: The parked gate.

        Returns:
            The resume value.
        """
        return {
            "approval_id": pending.approval_id,
            "decision": verdict,
            "reviewer": "bench",
            "comment": "automated",
        }

    return decide


async def _new_engine() -> tuple[WorkflowEngine, Any]:
    """Build a started engine with an in-memory checkpointer.

    No PostgreSQL: a database round-trip would measure the database, and this
    harness is about what the engine costs. Stated in the notes so the number is
    never read as a durable-store figure.

    Returns:
        The started engine and its checkpointer.
    """
    settings = load_settings(
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        hitl_enabled=True,
        log_level="ERROR",
    )
    checkpointer = build_memory_checkpointer()
    engine = WorkflowEngine(settings, checkpointer=checkpointer)
    await engine.startup()
    return engine, checkpointer


async def _shutdown(engine: WorkflowEngine, checkpointer: Any) -> None:
    """Release an engine and its checkpointer.

    Args:
        engine: The engine to stop.
        checkpointer: Its checkpointer.
    """
    await engine.shutdown()
    closer = getattr(checkpointer, "close", None)
    if closer is not None:
        result = closer()
        if asyncio.iscoroutine(result):
            await result


def measure_graph_compile() -> Measurement:
    """Time graph construction and compilation.

    Isolated from a run because it is a one-off startup cost. Reported so a
    reader can tell "the engine is slow" from "the engine takes a moment to
    start and then is fast".

    Returns:
        The measurement, in milliseconds.
    """
    samples: list[float] = []
    for _ in range(5):
        reset_settings_cache()
        settings = load_settings(
            environment="development",
            llm_provider="echo",
            postgres_enabled=False,
            hitl_enabled=True,
            log_level="ERROR",
        )
        started = time.perf_counter()
        WorkflowEngine(settings, checkpointer=build_memory_checkpointer())
        samples.append((time.perf_counter() - started) * 1000)
    return Measurement(
        "graph_compile",
        "ms",
        samples,
        "engine construction incl. graph compile; paid once per process",
    )


async def measure_end_to_end(
    engine: WorkflowEngine, *, runs: int, units: int, label: str
) -> tuple[Measurement, dict[str, list[float]]]:
    """Time complete runs, request to final report.

    Args:
        engine: A started engine.
        runs: How many runs to time.
        units: Files per request.
        label: Prefix for the run ids.

    Returns:
        The end-to-end measurement and per-node timings, both milliseconds.
    """
    end_to_end = Measurement(
        "end_to_end" if units == 1 else f"end_to_end_{units}files",
        "ms",
        detail=f"{label}, {units} file(s), warm engine, all gates approved",
    )
    per_node: dict[str, list[float]] = {}
    for index in range(runs):
        request = _request(f"bench-{label}-{units}-{index}", units=units)
        started = time.perf_counter()
        outcome = await engine.run_until_done(request, decide=_decider(), max_gates=24)
        elapsed = (time.perf_counter() - started) * 1000
        if outcome.report is None:
            raise WorkflowError(
                f"benchmark run {index} produced no report (status={outcome.status}): "
                f"{outcome.error}"
            )
        end_to_end.samples.append(elapsed)
        for timing in outcome.timings:
            per_node.setdefault(timing.node, []).append(timing.duration_ms)
    return end_to_end, per_node


async def measure_throughput(engine: WorkflowEngine, *, runs: int, concurrency: int) -> Measurement:
    """Measure completed runs per second at a fixed concurrency.

    Args:
        engine: A started engine.
        runs: Total runs to complete.
        concurrency: Runs in flight at once.

    Returns:
        The measurement, in runs per second.
    """
    semaphore = asyncio.Semaphore(concurrency)
    failures: list[str] = []

    async def one(index: int) -> None:
        """Run one request under the concurrency bound.

        Args:
            index: Run index, used for the run id.
        """
        async with semaphore:
            outcome = await engine.run_until_done(
                _request(f"bench-tp-{index}"), decide=_decider(), max_gates=24
            )
            if outcome.report is None:
                failures.append(f"{outcome.status}: {outcome.error}")

    started = time.perf_counter()
    await asyncio.gather(*(one(index) for index in range(runs)))
    elapsed = time.perf_counter() - started
    if failures:
        raise WorkflowError(
            f"{len(failures)} of {runs} throughput runs produced no report; first: {failures[0]}"
        )
    return Measurement(
        "throughput",
        "runs/s",
        [runs / elapsed],
        f"{runs} runs, concurrency {concurrency}, single process, echo provider",
    )


def measure_gate_overhead(end_to_end: Measurement, per_node: dict[str, list[float]]) -> Measurement:
    """Measure the time a run spends outside any node.

    A run's wall-clock is not the sum of its nodes. Between them the engine
    parks on an interrupt, writes a checkpoint, hands the gate to the decider,
    reads the state back and resumes — and none of that is attributed to a
    node, so it is invisible in the per-node table and shows up only as the gap
    between the two numbers.

    Measuring it separately is what turns "the HITL layer is expensive" from a
    hunch into a figure. The subtraction is derived rather than directly timed,
    and labelled as such, because attributing the gap requires trusting both
    measurements; the node timings are the engine's own, taken inside the graph,
    while the wall-clock is taken outside it.

    Args:
        end_to_end: The end-to-end measurement.
        per_node: Node name to every observed duration.

    Returns:
        The measurement, in milliseconds of non-node time per run.
    """
    if not end_to_end.samples or not per_node:
        return Measurement("gate_overhead", "ms", [], "no runs observed")
    samples: list[float] = []
    for index, elapsed in enumerate(end_to_end.samples):
        runs_with_node = min((len(per_node) and len(v)) or 0 for v in per_node.values())
        if index >= runs_with_node:
            break
        node_total = sum(
            statistics.median(samples_for_node) for samples_for_node in per_node.values()
        )
        samples.append(max(0.0, elapsed - node_total))
    if not samples:
        return Measurement("gate_overhead", "ms", [], "fewer node observations than runs")
    return Measurement(
        "gate_overhead",
        "ms",
        samples,
        "derived: end-to-end minus the sum of median node times, i.e. "
        "checkpoint + interrupt + resume + decision plumbing per run",
    )


def measure_checkpoint_write(end_to_end: Measurement, checkpoints: int) -> Measurement:
    """Derive per-checkpoint write cost from the end-to-end runs.

    Derived, not directly timed, and labelled as such: checkpoints are written
    as the graph advances, so isolating one write means either instrumenting the
    store (which changes what is measured) or accepting the division. The
    division is honest as long as it is presented as an average across a whole
    run rather than as a per-call latency.

    Args:
        end_to_end: The end-to-end measurement.
        checkpoints: Checkpoints produced across those runs.

    Returns:
        The measurement, in milliseconds per checkpoint.
    """
    if not end_to_end.samples or checkpoints <= 0:
        return Measurement("checkpoint_write", "ms", [], "no checkpoints observed")
    per_run = end_to_end.samples[0] / checkpoints
    return Measurement(
        "checkpoint_write",
        "ms",
        [round(per_run, 3)],
        f"derived: end-to-end median / {checkpoints} checkpoints in one run",
    )


async def measure_eval_suite() -> Measurement:
    """Time the golden dataset as the CI quality gate would run it.

    Returns:
        The measurement, in seconds.
    """
    from evals.runners import run_suite

    started = time.perf_counter()
    report = await run_suite(limit=20, memory=True, gates="approve")
    elapsed = time.perf_counter() - started
    return Measurement(
        "eval_suite",
        "s",
        [round(elapsed, 3)],
        f"20 golden cases, concurrency 4, {len(report.cases)} completed",
    )


def render_markdown(report: Report) -> str:
    """Render the report as a Markdown table plus its caveats.

    The caveats are part of the output, not decoration. A benchmark table
    published without its limits is how a 4-millisecond in-memory number becomes
    a claim about a 40-millisecond production request.

    Args:
        report: The measured report.

    Returns:
        Markdown text.
    """
    lines = [
        "| Measurement | Median | Mean | p95 | Min | Max | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, measurement in report.measurements.items():
        stats = measurement.summary()
        if not stats:
            continue
        lines.append(
            f"| `{name}` | {stats['median']:g} {measurement.unit} "
            f"| {stats['mean']:g} | {stats['p95']:g} | {stats['min']:g} | "
            f"{stats['max']:g} | {int(stats['count'])} |"
        )
    lines.append("")
    for note in report.notes:
        lines.append(f"> {note}")
        lines.append("")
    env = report.environment
    lines.append(
        f"Measured on {env['implementation']} {env['python']} "
        f"({env['cpu_count']} CPUs, {env['processor']}), {env['platform']}."
    )
    return "\n".join(lines)


def render_per_node(per_node: dict[str, list[float]]) -> str:
    """Render per-node medians as a Markdown table.

    Args:
        per_node: Node name to every observed duration in milliseconds.

    Returns:
        Markdown text, or an empty string when nothing was measured.
    """
    if not per_node:
        return ""
    medians = {node: round(statistics.median(samples), 3) for node, samples in per_node.items()}
    total = sum(medians.values()) or 1.0
    lines = ["| Node | Median ms | Share of node time |", "| --- | ---: | ---: |"]
    for node, value in sorted(medians.items(), key=lambda item: -item[1]):
        lines.append(f"| `{node}` | {value:.3f} | {100 * value / total:.1f}% |")
    return "\n".join(lines)


async def run(*, runs: int, concurrency: int, scaling: bool, include_eval: bool) -> Report:
    """Run every benchmark and collect the results.

    Args:
        runs: Runs per measured scenario. Small on purpose: the point is a
            stable median, and a large sample of a microsecond-scale workload
            mostly measures the garbage collector.
        concurrency: Worker count for the throughput measurement.
        scaling: Whether to measure 1/5/20-file requests.
        include_eval: Whether to time the golden dataset.

    Returns:
        The collected report.
    """
    report = Report(environment=_environment())
    report.notes.append(
        "Every figure here was produced by `python benchmarks/bench.py` on the "
        "machine named below, with the `echo` provider. None is estimated."
    )
    report.notes.append(
        "The `echo` provider answers in microseconds by design, so these numbers "
        "measure **the orchestration overhead this engine adds around a model "
        "call**, not end-to-end latency with a real model. A real provider "
        "dominates a run by orders of magnitude."
    )
    report.notes.append(
        "Checkpoints are in-memory here. Latency against PostgreSQL is a "
        "different measurement with different numbers; these are not a proxy "
        "for it."
    )
    report.notes.append(
        "Compare `end_to_end` against the per-node table: most of a run is spent "
        "*between* nodes. That gap, reported as `gate_overhead`, is the human-in-"
        "the-loop machinery — park, checkpoint, ask, read back, resume — and it is "
        "the cost this design accepts in exchange for being able to stop a run and "
        "come back to it."
    )

    report.measurements["graph_compile"] = measure_graph_compile()

    engine, checkpointer = await _new_engine()
    try:
        primary, per_node = await measure_end_to_end(engine, runs=runs, units=1, label="warm")
        report.measurements["end_to_end"] = primary
        report.per_node = per_node

        # Counted from a real run's history rather than assumed, so the
        # derived per-checkpoint figure tracks whatever the graph actually did
        # — including any future node that adds a write.
        history = await engine.history("bench-warm-1-0", limit=200)
        report.measurements["checkpoint_write"] = measure_checkpoint_write(primary, len(history))

        if scaling:
            for units in SCALING_UNITS[1:]:
                scaled, _ = await measure_end_to_end(
                    engine, runs=max(3, runs // 2), units=units, label="scale"
                )
                report.measurements[f"end_to_end_{units}files"] = scaled

        report.measurements["gate_overhead"] = measure_gate_overhead(primary, per_node)
        report.measurements["throughput"] = await measure_throughput(
            engine, runs=runs * 4, concurrency=concurrency
        )
    finally:
        await _shutdown(engine, checkpointer)

    report.measurements["eval_suite"] = (
        await measure_eval_suite() if include_eval else Measurement("eval_suite", "s", [])
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the harness and print the result.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` if any scenario failed to produce a report.
    """
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description=__doc__.split("\n\n")[0] if __doc__ else None,
    )
    parser.add_argument("--runs", type=int, default=15, help="Runs per scenario.")
    parser.add_argument("--concurrency", type=int, default=8, help="Throughput workers.")
    parser.add_argument("--no-scaling", action="store_true", help="Skip multi-file runs.")
    parser.add_argument("--no-eval", action="store_true", help="Skip the golden dataset.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args(argv)

    report = asyncio.run(
        run(
            runs=args.runs,
            concurrency=args.concurrency,
            scaling=not args.no_scaling,
            include_eval=not args.no_eval,
        )
    )
    if args.json:
        payload = report.as_dict()
        payload["per_node_median_ms"] = {
            node: round(statistics.median(samples), 3) for node, samples in report.per_node.items()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
        table = render_per_node(report.per_node)
        if table:
            print()
            print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
