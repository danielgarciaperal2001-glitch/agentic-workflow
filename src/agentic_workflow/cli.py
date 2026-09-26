"""Command-line entry point: ``awf`` / ``agentic-workflow``.

Five subcommands, chosen so that every claim the README makes can be checked from
a terminal without starting the API or writing a test:

* ``demo`` — run a real review end to end, pausing at each human gate.
* ``replay`` — list a run's checkpoints and re-execute from any of them.
* ``topology`` — print the graph's nodes, edges, cycles and routing table.
* ``eval`` — score the workflow's output quality against a golden dataset.
* ``janitor`` — one checkpoint-retention pass.

Design notes:

* **No network, no credentials required.** The default provider is ``echo``, so
  ``awf demo`` works on a fresh clone and in CI. ``--provider`` switches it.
* **Exit codes mean something.** ``0`` success, ``1`` the run did not reach a
  successful terminal state, ``2`` bad input. A CI job can gate on this.
* **Failures print the error class, not a traceback.** A traceback from a CLI
  buries the one line that matters; ``--trace`` restores it.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os as _os
from pathlib import Path
import sys
from typing import Any

from agentic_workflow import __version__

#: Exit code used when a run reached a terminal state that is not a success.
NOT_DONE = 1
#: Exit code used for bad input.
BAD_INPUT = 2

#: The defect the ``demo`` command reviews. Chosen because it is small enough to
#: read on a terminal, obviously wrong to a reviewer, and has an exact answer:
#: binary floating point cannot represent 0.1 + 0.2, so a checkout total that adds
#: prices as floats is a real bug rather than a contrived one.
DEMO_BUGGY_SOURCE = '''\
"""Order total calculation."""


def total(items: list[dict]) -> float:
    """Sum the price of every item.

    >>> total([{"price": 10.0}, {"price": 5.5}])
    15.5
    """
    result = 0.0
    for item in items:
        result = result + float(item["price"])
    return result
'''


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _echo(message: str = "", *, file: Any = None) -> None:
    """Write one line to stdout, or to *file* when given.

    Args:
        message: Text to print. A blank argument prints a blank line.
        file: Destination stream; ``None`` means stdout. Diagnostics go to stderr
            so that ``awf eval --json | jq`` still works.
    """
    print(message, file=file if file is not None else sys.stdout, flush=True)


def _rule(title: str = "") -> None:
    """Print a section separator.

    Args:
        title: Section name; omitted for a plain rule.
    """
    _echo()
    _echo(f"── {title} " + "─" * max(0, 58 - len(title)) if title else "─" * 62)


def _out(payload: Any) -> None:
    """Print a payload as indented JSON, the format a machine can consume.

    Args:
        payload: Any JSON-serialisable object.
    """
    _echo(json.dumps(payload, indent=2, default=str, sort_keys=False))


# --------------------------------------------------------------------------- #
# Shared wiring
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Runtime:
    """An engine plus everything needed to shut it down again.

    Attributes:
        engine: The workflow engine.
        checkpointer: The checkpointer the engine owns, if any.
    """

    engine: Any
    checkpointer: Any


async def _runtime(args: argparse.Namespace) -> _Runtime:
    """Build an engine from parsed arguments.

    Args:
        args: Parsed arguments carrying ``provider``, ``postgres`` and ``memory``.

    Returns:
        A started runtime. The caller is responsible for shutting it down.
    """
    from agentic_workflow.config import load_settings
    from agentic_workflow.persistence.checkpointer import (
        build_checkpointer,
        build_memory_checkpointer,
    )
    from agentic_workflow.services.engine import WorkflowEngine

    settings = load_settings(
        llm_provider=args.provider,
        postgres_enabled=not args.memory,
        log_level=args.log_level,
        graph_name=f"awf-cli-{args.command}",
    )
    checkpointer = build_memory_checkpointer() if args.memory else build_checkpointer(settings)
    engine = WorkflowEngine(settings, checkpointer=checkpointer)
    await engine.startup()
    return _Runtime(engine=engine, checkpointer=checkpointer)


async def _shutdown(runtime: _Runtime | None) -> None:
    """Release a runtime's resources, reporting rather than raising on failure.

    Args:
        runtime: What :func:`_runtime` returned, or ``None`` if it never started.
    """
    if runtime is None:
        return
    try:
        await runtime.engine.shutdown()
    except Exception as exc:  # a shutdown failure must not mask the real result
        _echo(f"warning: shutdown failed: {exc}", file=sys.stderr)
    closer = getattr(runtime.checkpointer, "close", None)
    if closer is not None:
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            _echo(f"warning: checkpointer close failed: {exc}", file=sys.stderr)


def _demo_request(run_id: str) -> Any:
    """Build the review request the ``demo`` command runs.

    Args:
        run_id: Identifier for the run.

    Returns:
        A validated :class:`~agentic_workflow.domain.schemas.ReviewRequest`.
    """
    from agentic_workflow.domain.schemas import ReviewRequest, SourceFile

    return ReviewRequest(
        run_id=run_id,
        request_id="PR-1042",
        title="Fix rounding drift in the order total",
        description=(
            "checkout/total.py sums prices as binary floats. For two-decimal "
            "currency the result drifts: 0.1 + 0.2 is 0.30000000000000004, and "
            "support sees totals that are off by a cent."
        ),
        language="python",
        files=[SourceFile(path="checkout/total.py", content=DEMO_BUGGY_SOURCE)],
        acceptance_criteria=[
            "The result must be exact for two-decimal currency values.",
            "Existing callers of `total` must keep working.",
            "No third-party dependencies may be added.",
        ],
        constraints=["Keep the public signature of `total` unchanged."],
        metadata={"repo": "acme/checkout", "author": "team-payments"},
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
async def _cmd_demo(args: argparse.Namespace) -> int:
    """Run a review end to end, pausing at every human gate.

    Args:
        args: Parsed arguments.

    Returns:
        The process exit code.
    """

    runtime = await _runtime(args)
    try:
        request = _demo_request(args.run_id)
        _rule("request")
        _echo(f"run        {request.run_id}")
        _echo(f"title      {request.title}")
        _echo(f"files      {', '.join(f.path for f in request.files)}")
        _echo(f"criteria   {len(request.acceptance_criteria)} stated")

        gates: list[Any] = []
        decide = _interactive_decider() if args.interactive else _auto_decider(args.decision)
        _rule("workflow")
        outcome = await runtime.engine.run_until_done(
            request,
            decide=decide,
            max_gates=args.max_gates,
        )

        for entry in outcome.state.get("human_decisions", []) if outcome.state else []:
            gates.append(entry)
        for entry in outcome.decisions:
            if entry not in gates:
                gates.append(entry)

        for index, entry in enumerate(gates, start=1):
            _echo(
                f"gate {index}  {entry.get('stage', '?'):<13}"
                f" {entry.get('decision', '?')!s:<8}"
                f" by {entry.get('reviewer', '?')}"
            )

        _rule("outcome")
        _echo(f"status     {outcome.status}")
        _echo(f"iteration  {outcome.iteration}")
        _echo(f"gates      {len(gates)} answered")

        if outcome.error:
            _echo(f"error      {outcome.error}")

        timings = {t.node: t.duration_ms for t in outcome.timings}
        if timings:
            _rule("per-node timing (ms)")
            for node, ms in sorted(timings.items(), key=lambda kv: -kv[1]):
                _echo(f"  {node:<12} {ms:>8.1f}")

        report = outcome.report
        if report is not None:
            _rule("final report")
            _echo(report.markdown)
        else:
            _rule("final report")
            _echo("(none: the run did not reach the reporter)")

        if args.history:
            _rule("checkpoint history")
            for info in await runtime.engine.history(request.run_id, limit=args.history):
                _echo(
                    f"  step {info.step:>3}  {info.source:<10} {info.checkpoint_id}"
                    + (f"  -> {info.next_nodes}" if info.next_nodes else "")
                )

        return 0 if outcome.status in ("completed", "rejected") else NOT_DONE
    finally:
        await _shutdown(runtime)


def _auto_decider(verdict: str) -> Any:
    """Build a synchronous decider that always answers *verdict*.

    Args:
        verdict: The decision to apply at every gate.

    Returns:
        A callable suitable for
        :meth:`~agentic_workflow.services.engine.WorkflowEngine.run_until_done`.
    """

    def decide(pending: Any) -> dict[str, Any]:
        """Answer one gate.

        Args:
            pending: The gate to answer.

        Returns:
            The resume value.
        """
        return {
            "approval_id": pending.approval_id,
            "decision": verdict,
            "reviewer": f"cli-{verdict}",
            "comment": f"answered by `awf demo --decision {verdict}`",
        }

    return decide


def _interactive_decider() -> Any:
    """Build a decider that asks a human what to do at each gate.

    The prompt shows the same three things the API shows — the diff, the
    rationale and the options — because a gate answered without the diff is a
    rubber stamp, and a CLI that hides it teaches the operator to approve blindly.

    An unrecognised answer re-asks rather than defaulting to approval. A
    mistyped ``r`` must never be read as consent to change a file.

    Returns:
        A callable suitable for
        :meth:`~agentic_workflow.services.engine.WorkflowEngine.run_until_done`.
    """
    aliases = {
        "a": "approve",
        "e": "edit",
        "r": "reject",
        "approve": "approve",
        "edit": "edit",
        "reject": "reject",
    }

    def decide(pending: Any) -> dict[str, Any] | None:
        """Show a gate and read the verdict.

        Args:
            pending: The gate to answer.

        Returns:
            The resume value, or ``None`` to leave the run parked — which is what
            a non-interactive stdin does, since the alternative would be
            approving changes nobody looked at.
        """
        _rule(f"gate: {pending.stage}")
        _echo(f"  title     {pending.title}")
        _echo(f"  rationale {pending.rationale}")
        _echo(f"  confidence {pending.confidence}")
        if pending.diff_preview:
            _echo("  diff")
            for line in pending.diff_preview.splitlines():
                _echo(f"    {line}")
        options = ", ".join(d.value for d in pending.options)
        while True:
            _echo()
            try:
                answer = input(f"  decision [{options}] (a/e/r): ").strip().lower()
            except EOFError:
                # No terminal to read: refusing to decide is the only safe exit.
                _echo("  no input available; leaving the run parked.")
                return None
            verdict = aliases.get(answer)
            if verdict is not None:
                break
            _echo(f"  '{answer}' is not one of {options}; ask again.")
        comment = ""
        if not sys.stdin.isatty():  # pragma: no cover - piped input
            comment = ""
        try:
            comment = input("  comment (optional): ").strip()
        except EOFError:  # pragma: no cover - piped input
            comment = ""
        return {
            "approval_id": pending.approval_id,
            "decision": verdict,
            "reviewer": _operator_name(),
            "comment": comment,
        }

    return decide


def _operator_name() -> str:
    """Ask who is deciding, falling back to something attributable.

    Returns:
        A reviewer name for the audit log.
    """
    default = _os.environ.get("USER") or _os.environ.get("USERNAME") or "operator"
    if not sys.stdin.isatty():  # pragma: no cover - piped input
        return default
    try:
        return input(f"  your name [{default}]: ").strip() or default
    except EOFError:  # pragma: no cover - piped input
        return default


async def _cmd_replay(args: argparse.Namespace) -> int:
    """Inspect a run's checkpoint history and optionally re-execute from one.

    Args:
        args: Parsed arguments.

    Returns:
        The process exit code.
    """
    runtime = await _runtime(args)
    try:
        history = await runtime.engine.history(args.run_id, limit=args.limit)
        _rule(f"history of {args.run_id} ({len(history)} checkpoints)")
        for index, info in enumerate(history):
            marker = "  <- replay target" if index == args.index else ""
            _echo(
                f"  [{index:>2}] step {info.step:>3}  {info.source:<10}"
                f" {info.checkpoint_id}"
                + (f"  -> {info.next_nodes}" if info.next_nodes else "")
                + marker
            )

        if args.index is None:
            _echo()
            _echo("Pass --index N to replay from that checkpoint.")
            return 0

        if not 0 <= args.index < len(history):
            _echo(f"error: --index must be between 0 and {len(history) - 1}", file=sys.stderr)
            return BAD_INPUT

        target = history[args.index]
        _rule(f"replaying from {target.checkpoint_id} (step {target.step})")
        before = await runtime.engine.state_at(args.run_id, target.checkpoint_id)
        _echo(f"  before   status={before.status} iteration={before.iteration}")

        outcome = await runtime.engine.replay_from(args.run_id, target.checkpoint_id)

        _echo(f"  after    status={outcome.status} iteration={outcome.iteration}")
        _echo(
            f"  history  {len(history)} -> {len(await runtime.engine.history(args.run_id, limit=99))} checkpoints"
        )
        if outcome.report is not None:
            _rule("replayed report")
            _echo(outcome.report.markdown)
        return 0
    finally:
        await _shutdown(runtime)


async def _cmd_topology(args: argparse.Namespace) -> int:
    """Print the graph's structure.

    Args:
        args: Parsed arguments.

    Returns:
        The process exit code.
    """
    from agentic_workflow.graph.builder import graph_topology

    topology = graph_topology()
    if args.json:
        _out(topology)
        return 0

    _rule("agents")
    for node in topology["nodes"]:
        _echo(f"  {node}")

    _rule("edges")
    for edge in topology["edges"]:
        via = edge.get("via")
        suffix = f"   (via {via})" if via else ""
        arrow = "-->" if edge["kind"] == "static" else "-?->"
        _echo(f"  {edge['from']:<12} {arrow} {edge['to']}{suffix}")

    cycles = topology.get("cycles") or []
    if cycles:
        _rule("feedback loops")
        for loop in cycles:
            # Each entry repeats its entry node at the end, so joining the list as
            # written prints the seam twice. Collapsing it keeps the arrow honest.
            path = [str(node) for node in loop]
            if path and path[0] == path[-1]:
                path[-1] = ""
            _echo(f"  {' -> '.join(node for node in path if node)}")

    _rule("routing table")
    for condition, destination in topology["routing_table"]:
        _echo(f"  when {condition}")
        _echo(f"    -> {destination}")
    return 0


async def _cmd_eval(args: argparse.Namespace) -> int:
    """Score workflow output quality against the golden dataset.

    Args:
        args: Parsed arguments.

    Returns:
        The process exit code.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from evals.runners import run_suite
    except ImportError as exc:  # pragma: no cover - depends on the checkout
        _echo(f"error: the evals package is not importable: {exc}", file=sys.stderr)
        return BAD_INPUT

    report = await run_suite(
        dataset=args.dataset,
        provider=args.provider,
        limit=args.limit,
        memory=True,
        gates=args.gates,
    )
    if args.json:
        _out(report.as_dict())
    else:
        from evals.report import render

        _echo(render(report))
    return 0 if report.passed else NOT_DONE


async def _cmd_janitor(args: argparse.Namespace) -> int:
    """Run one checkpoint-retention pass.

    Args:
        args: Parsed arguments.

    Returns:
        The process exit code.
    """
    from agentic_workflow.config import load_settings
    from agentic_workflow.persistence.checkpointer import (
        build_checkpointer,
        build_memory_checkpointer,
    )
    from agentic_workflow.persistence.retention import CheckpointJanitor

    settings = load_settings(
        llm_provider=args.provider,
        postgres_enabled=not args.memory,
        log_level=args.log_level,
    )
    checkpointer = build_memory_checkpointer() if args.memory else build_checkpointer(settings)
    _rule("checkpoint janitor" + ("  (dry run)" if args.dry_run else ""))
    report = await CheckpointJanitor(checkpointer, settings).run(
        retention_days=args.retention_days,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    if args.json:
        _out(report.as_dict())
    else:
        _echo(f"  examined threads   {report.examined}")
        _echo(f"  stale threads      {report.stale}")
        _echo(f"  deleted threads    {report.deleted}")
        _echo(f"  freed checkpoints  {report.freed_checkpoints}")
        if report.undated:
            _echo(f"  undated (kept)     {report.undated}")
        _echo(f"  duration           {report.duration_seconds:.3f}s")
        if report.examined == 0:
            _echo()
            _echo(
                "  warning: the sweep examined no threads. That means it could not read the store,",
                file=sys.stderr,
            )
            _echo("  not that there was nothing to clean.", file=sys.stderr)
    closer = getattr(checkpointer, "close", None)
    if closer is not None:
        result = closer()
        if asyncio.iscoroutine(result):
            await result
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        A parser with every subcommand attached.
    """
    parser = argparse.ArgumentParser(
        prog="awf",
        description="Multi-agent workflow engine with human-in-the-loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  awf demo                              run the showcase review\n"
            "  awf demo --interactive                pause at each gate for input\n"
            "  awf demo --decision reject --history   see the rejection path\n"
            "  awf topology                          print the agent graph\n"
            "  awf replay <run-id> --index 3         re-execute from a checkpoint\n"
            "  awf eval --limit 5                    score against the golden set\n"
            "  awf janitor --dry-run                 preview checkpoint cleanup\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"agentic-workflow {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_engine_arguments(target: argparse.ArgumentParser) -> None:
        """Attach the engine-wide flags to a subcommand.

        Args:
            target: The subcommand parser to extend.
        """
        target.add_argument(
            "--provider",
            default="echo",
            choices=("echo", "openai", "openai_compatible"),
            help="LLM provider (default: echo, deterministic and offline).",
        )
        target.add_argument(
            "--memory",
            action="store_true",
            help="Use the in-memory checkpointer instead of PostgreSQL.",
        )
        target.add_argument(
            "--log-level",
            default="WARNING",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            help="Logging verbosity (default: WARNING).",
        )

    demo = subparsers.add_parser(
        "demo",
        help="Run a review end to end, pausing at every human gate.",
        description="Run a review end to end. The default answers every gate itself.",
    )
    add_engine_arguments(demo)
    demo.add_argument("--run-id", default="demo-run-1", help="Run identifier.")
    demo.add_argument(
        "--decision",
        default="approve",
        choices=("approve", "edit", "reject"),
        help="Verdict applied at every gate (default: approve).",
    )
    demo.add_argument(
        "--interactive",
        action="store_true",
        help="Print each gate and read the verdict from stdin instead.",
    )
    demo.add_argument(
        "--max-gates",
        type=int,
        default=8,
        help="Safety bound on gates answered (default: 8).",
    )
    demo.add_argument(
        "--history",
        type=int,
        nargs="?",
        const=20,
        default=0,
        help="Print this many checkpoints of the run's history.",
    )
    demo.set_defaults(handler=_cmd_demo)

    replay = subparsers.add_parser(
        "replay",
        help="List a run's checkpoints and re-execute from one.",
        description=(
            "Time travel. Lists every super-step LangGraph checkpointed and, with "
            "--index, branches the run from that point. The original history is "
            "never modified."
        ),
    )
    add_engine_arguments(replay)
    replay.add_argument("run_id", help="Run to inspect.")
    replay.add_argument(
        "--index",
        type=int,
        default=None,
        help="Checkpoint index to replay from (0 is the newest).",
    )
    replay.add_argument("--limit", type=int, default=25, help="Checkpoints to list.")
    replay.set_defaults(handler=_cmd_replay)

    topology = subparsers.add_parser(
        "topology",
        help="Print the agent graph, its cycles and its routing table.",
        description="Print the compiled graph's structure.",
    )
    topology.add_argument("--json", action="store_true", help="Emit raw JSON.")
    topology.set_defaults(handler=_cmd_topology)

    evaluate = subparsers.add_parser(
        "eval",
        help="Score output quality against the golden dataset.",
        description=(
            "Runs the review workflow over a golden dataset and scores the "
            "reports for faithfulness, citation coverage and structural "
            "completeness. Exits non-zero when a threshold is missed, so it can "
            "gate a pull request."
        ),
    )
    evaluate.add_argument("--provider", default="echo", help="LLM provider.")
    evaluate.add_argument(
        "--dataset",
        default=None,
        help="Path to a JSONL dataset (default: the bundled golden set).",
    )
    evaluate.add_argument("--limit", type=int, default=None, help="Only the first N cases.")
    evaluate.add_argument(
        "--gates",
        default="approve",
        choices=("approve", "edit", "reject"),
        help="Verdict applied at every gate (default: approve).",
    )
    evaluate.add_argument("--json", action="store_true", help="Emit raw JSON.")
    evaluate.set_defaults(handler=_cmd_eval)

    janitor = subparsers.add_parser(
        "janitor",
        help="Delete checkpoint histories older than the retention window.",
        description="One retention pass over the checkpoint store.",
    )
    add_engine_arguments(janitor)
    janitor.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Override the configured window.",
    )
    janitor.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without deleting it.",
    )
    janitor.add_argument("--limit", type=int, default=1_000, help="Threads per pass.")
    janitor.add_argument("--json", action="store_true", help="Emit raw JSON.")
    janitor.set_defaults(handler=_cmd_janitor)

    return parser


async def _run(args: argparse.Namespace) -> int:
    """Dispatch a parsed command.

    Args:
        args: Parsed arguments.

    Returns:
        The process exit code.
    """
    handler = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - argparse enforces the choice
        return BAD_INPUT
    return int(await handler(args))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        _echo("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if _os.environ.get("AWF_CLI_TRACE"):
            raise
        # A traceback from a CLI buries the one line that matters. The class name
        # is kept because "WorkflowError: [concurrency_limit] ..." and
        # "RuntimeError: ..." call for very different responses from the reader.
        _echo(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        _echo("       set AWF_CLI_TRACE=1 for the full traceback.", file=sys.stderr)
        return NOT_DONE


if __name__ == "__main__":  # pragma: no cover - module execution
    raise SystemExit(main())
