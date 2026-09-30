"""The CLI, invoked as a process.

Everything here runs ``awf`` in a subprocess. That is deliberate and it is the
only honest way to test this layer: two of the bugs these tests exist to pin
down were invisible from inside the interpreter.

* ``--log-level`` was accepted, listed in ``--help``, and then ignored, because
  ``get_logger`` lazily configures with the *default* settings rather than the
  ones the command built. A unit test that called ``configure_logging`` first
  would have passed, because it would have done the thing the production path
  does not.
* ``--memory`` inverted the environment — ``postgres_enabled=not args.memory`` —
  so omitting the flag *enabled* PostgreSQL. ``awf demo`` on a fresh clone then
  sat for thirty seconds retrying a connection nobody asked for. Only a real
  invocation, with a real environment, shows that.

Both were found by running the README's quickstart. That is the argument for
these tests: the quickstart is a claim, and these are the tests that keep it
true.
"""

from __future__ import annotations

from functools import cache
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from agentic_workflow.config import load_settings

pytestmark = pytest.mark.integration

#: Every invocation is capped. A test that hangs is a test that stalls CI, and
#: the previous bug's most recognisable symptom was a 30-second connect retry.
TIMEOUT = 90

REPO_ROOT = Path(__file__).resolve().parents[2]


@cache
def run_cli(*args: str, env: tuple[tuple[str, str], ...] = ()) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a subprocess with a clean environment.

    Memoised on ``(args, env)``. A subprocess test cannot share the interpreter
    state it is testing — that is the whole reason these exist — but it does not
    have to re-run the process either. Three of the assertions below all run the
    same default demo, and paying three seconds each for an identical answer is
    a slow suite for no extra coverage.

    Args:
        *args: Arguments after the module name.
        env: Extra environment entries, as a sorted tuple so it is hashable.

    Returns:
        The completed process, with text output captured.
    """
    package = str(REPO_ROOT / "src")
    if not (REPO_ROOT / "src" / "agentic_workflow" / "cli.py").is_file():
        pytest.fail("the source tree is not where this test expects it")

    child_env = {
        "PATH": os.environ.get("PATH", ""),
        # S108: HOME is passed to the child so pydantic-settings can find a
        # `.env` if one exists, not as a scratch directory. `/tmp` is the
        # conventional fallback when the parent has no HOME at all, which is
        # the case in some minimal CI runners.
        "HOME": os.environ.get("HOME") or "/tmp",  # noqa: S108
        # A deliberately minimal PYTHONPATH: `evals` is a top-level package that
        # does not live under `src/`, and a CLI that can only find it through a
        # hand-set path variable is not shippable.
        "PYTHONPATH": os.pathsep.join((package, str(REPO_ROOT))),
        "AWF_LOG_FORMAT": "json",
    }
    child_env.update(dict(env))

    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "agentic_workflow.cli", *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env=child_env,
        cwd=REPO_ROOT,
        check=False,
    )


class TestLogLevel:
    """``--log-level`` must reach the logger that actually emits.

    The renderer key is ``level``, not ``log_level``. That is worth stating
    because getting it wrong produces tests that pass for the wrong reason: a
    substring that never occurs in the output is trivially absent, so every
    ``assert "x" not in stderr`` in a first draft of these tests was green while
    asserting nothing at all.
    """

    def test_the_default_is_quiet(self) -> None:
        """The default has to be usable as a command, not just as a library call.

        A demo whose output is 90 lines of structured log drowns the six lines
        that say what the workflow did. That is the whole reason the flag
        defaults to WARNING.
        """
        result = run_cli("demo")
        assert result.returncode == 0, result.stderr[-2000:]
        assert '"event": "node.start"' not in result.stderr
        assert '"level": "info"' not in result.stderr

    def test_the_default_emits_nothing_at_all(self) -> None:
        """Belt and braces on the assertion above.

        A single parsed log line would be enough to break ``--json | jq`` for
        any future command that starts logging during its result. Asserting
        stderr is *empty* is the property that matters, and it is strictly
        stronger than the "no info lines" check that came first.
        """
        result = run_cli("demo")
        assert result.stderr == "", result.stderr[:2000]

    def test_debug_is_actually_debug(self) -> None:
        """The flag is only real if turning it up produces more."""
        result = run_cli("demo", "--log-level", "DEBUG")
        assert result.returncode == 0, result.stderr[-2000:]
        assert '"level": "debug"' in result.stderr

    def test_error_silences_everything_below_it(self) -> None:
        result = run_cli("demo", "--log-level", "ERROR")
        assert result.returncode == 0, result.stderr[-2000:]
        assert '"level": "info"' not in result.stderr
        assert '"level": "debug"' not in result.stderr

    def test_the_environment_overrides_the_default(self) -> None:
        """``AWF_LOG_LEVEL`` is the deployment's call, not the flag's."""
        result = run_cli("demo", env=(("AWF_LOG_LEVEL", "ERROR"),))
        assert result.returncode == 0, result.stderr[-2000:]
        assert '"level": "info"' not in result.stderr


class TestMemoryCheckpointer:
    """A fresh clone has no database, and the default must not look for one."""

    def test_a_bare_demo_needs_no_postgres(self) -> None:
        """The README's headline command must work on a machine with no server.

        Regression test for the inversion: with no ``--memory`` the CLI passed
        ``postgres_enabled=True``, so this exact command spent thirty seconds
        retrying a connection and then failed with a persistence error. A reader
        following the quickstart got a stack trace.
        """
        result = run_cli("demo")
        assert result.returncode == 0, result.stderr[-2000:]
        assert "PersistenceError" not in result.stderr
        assert "pool" not in result.stderr

    def test_the_flag_still_works_explicitly(self) -> None:
        result = run_cli("demo", "--memory")
        assert result.returncode == 0, result.stderr[-2000:]

    def test_a_configured_database_is_honoured_without_the_flag(self) -> None:
        """The flag is an override, not the only way to *choose*.

        The correction to the inversion: an omitted ``--memory`` defers to
        ``AWF_POSTGRES_ENABLED`` rather than forcing either answer. This asserts
        the deferral goes the other way too — asking for PostgreSQL explicitly
        must actually attempt it, and the attempt must name the checkpointer.

        ``connect_timeout=1`` is not an optimisation. Port 1 refuses instantly,
        but the pool *retries* its backoff schedule, and the unwaited version of
        this test spent 31 of the file's 46 seconds proving that. A test whose
        cost is dominated by a connection timeout nobody is reading is a test
        that will eventually be skipped for being annoying.
        """
        result = run_cli(
            "demo",
            env=(
                ("AWF_POSTGRES_ENABLED", "true"),
                (
                    "AWF_POSTGRES_DSN",
                    "postgresql://nobody:nothing@127.0.0.1:1/absent?connect_timeout=1",
                ),
                # Bound the pool's own wait, which is what actually dominates.
                ("AWF_POSTGRES_POOL_TIMEOUT_SECONDS", "1"),
            ),
        )
        # The connection cannot succeed, and that is the point: the command tried
        # to reach the database the operator asked for rather than silently
        # falling back to memory and pretending to succeed.
        assert "pool" in result.stderr.lower() or "postgres" in result.stderr.lower()
        assert result.returncode != 0


class TestJsonOutputIsMachineReadable:
    """stdout carries the result, so a log line there is a bug."""

    def test_eval_json_parses_with_nothing_above_it(self) -> None:
        """``awf eval --json | jq`` is the documented pipeline, so it must hold.

        Two separate bugs broke this and both are invisible in a human-facing
        test: the log handler wrote to stdout alongside the document, and the
        janitor printed a banner rule above its JSON. Both produced output that
        *looked* like it had worked and failed at the `jq` stage instead.
        """
        result = run_cli("eval", "--json", "--limit", "2")
        payload = json.loads(result.stdout)  # raises if anything else is there
        assert payload["cases_run"] == 2
        assert "metrics" in payload

    def test_janitor_json_parses_with_nothing_above_it(self) -> None:
        result = run_cli("janitor", "--memory", "--dry-run", "--json")
        payload = json.loads(result.stdout)
        assert "examined" in payload

    @pytest.mark.postgres
    def test_janitor_over_postgres_opens_the_pool(self) -> None:
        """A durable retention sweep must list the store, not crash on entry.

        The janitor built its checkpointer the way every other command does —
        through ``build_checkpointer``, whose docstring is explicit that the
        PostgreSQL variant is a *wrapper* that must be entered or have its
        ``setup`` awaited before use — and then never set it up. The engine's
        ``startup`` awaits exactly that; the janitor skipped the whole lifecycle,
        so the first list hit a pool that was never opened, and the runbook's
        readiness probe ``awf janitor --dry-run`` could not run against a real
        database at all. The memory path never surfaced the bug, because
        ``InMemorySaver`` has no lifecycle to skip.

        Pre-fix, against the suite's own DSN, the command exited 1:

            PersistenceError: janitor could not list checkpoints:
            the pool 'pool-1' is not open yet
        """
        result = run_cli(
            "janitor",
            "--dry-run",
            "--json",
            env=(
                ("AWF_POSTGRES_ENABLED", "true"),
                ("AWF_POSTGRES_DSN", load_settings().postgres_dsn),
            ),
        )
        assert result.returncode == 0, result.stderr[-2000:]
        payload = json.loads(result.stdout)
        assert payload["dry_run"] is True
        assert "examined" in payload


class TestExitCodes:
    """Exit codes are the CLI's only contract with a shell or a CI job."""

    def test_a_complete_run_is_zero(self) -> None:
        assert run_cli("demo").returncode == 0

    def test_an_unknown_command_is_bad_input(self) -> None:
        assert run_cli("nonsense").returncode == 2

    def test_a_failed_gate_on_the_echo_provider_is_not_done(self) -> None:
        """Recall is a detection metric and the null model cannot pass it.

        Exit 1 rather than 0: the suite's purpose is to fail a build. Silently
        returning 0 here would make ``awf eval`` useless as a gate, and the
        reason it is 1 is not a crash — it is a metric that did not reach its
        threshold.
        """
        assert run_cli("eval", "--limit", "5").returncode == 1

    def test_the_invariant_gate_passes_offline(self) -> None:
        """The gate CI depends on, and it must be runnable with no credentials."""
        assert run_cli("eval", "--gate", "invariants", "--limit", "5").returncode == 0

    def test_tracebacks_are_opt_in(self) -> None:
        """A stack trace by default buries the one line that says what went wrong."""
        quiet = run_cli("eval", "--provider", "nonexistent")
        assert "Traceback (most recent call last)" not in quiet.stderr

        loud = run_cli("eval", "--provider", "nonexistent", env=(("AWF_CLI_TRACE", "1"),))
        assert "Traceback (most recent call last)" in loud.stderr


class TestRun:
    """``awf run`` as a process: the argument wiring nothing else can see.

    ``tests/integration/test_cli_run.py`` builds ``argparse.Namespace`` objects by
    hand, which is deliberate — it lets one test vary a single field — but it
    means every one of those tests would pass unchanged if ``--file`` were wired
    to ``--path`` or the handler were never attached. Those are exactly the bugs
    a CLI has and a library does not, so the flags are exercised as a command.
    """

    def test_the_command_is_listed_in_help(self) -> None:
        """A feature nobody can find is a feature nobody has."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "run" in result.stdout

    def test_the_title_is_required(self) -> None:
        """A review with no subject is not a review, and argparse says so.

        Making it optional would only move the failure: the id is derived from the
        title, so without one there would be no id to report back and nothing for
        ``awf watch`` to follow.
        """
        result = run_cli("run")
        assert result.returncode == 2, result.stderr
        assert "--title" in result.stderr

    def test_an_unreachable_server_is_bad_input_and_not_a_traceback(self, tmp_path: Any) -> None:
        """A server that is not there is a configuration problem, reported as one.

        This is the first failure a user meets, and the exit code has to
        distinguish it from a run that is merely unfinished, or a script that
        polls ``awf run`` would retry a typo'd port forever.
        """
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")

        result = run_cli(
            "run",
            "--title",
            "unreachable",
            "--file",
            str(source),
            "--url",
            "http://127.0.0.1:9",
            "--timeout",
            "5",
        )
        assert result.returncode == 2, result.stderr
        assert "cannot reach" in result.stderr
        assert "Traceback (most recent call last)" not in result.stderr

    def test_a_missing_file_names_the_path_and_sends_nothing(self, tmp_path: Any) -> None:
        """A typo in ``--file`` is caught locally, and the path is in the message.

        The port is one nothing listens on, so if the command tried to submit
        before reading, the message would be "cannot reach" instead of naming the
        file. Both outcomes exit 2, which is why the assertion is on the text.
        """
        absent = tmp_path / "not-here.py"

        result = run_cli(
            "run",
            "--title",
            "missing file",
            "--file",
            str(absent),
            "--url",
            "http://127.0.0.1:9",
        )
        assert result.returncode == 2, result.stderr
        assert "not-here.py" in result.stderr
        assert "cannot reach" not in result.stderr

    def test_a_malformed_metadata_pair_is_rejected_locally(self) -> None:
        """``--metadata`` without an ``=`` is a typo, not a server error.

        Forwarding it would produce a validation message about the ``metadata``
        field of the body — a name the user never typed — instead of about the
        argument they actually got wrong.
        """
        result = run_cli(
            "run",
            "--title",
            "bad metadata",
            "--metadata",
            "justakey",
            "--url",
            "http://127.0.0.1:9",
        )
        assert result.returncode == 2, result.stderr
        assert "--metadata" in result.stderr
        assert "cannot reach" not in result.stderr


class TestWatch:
    """``awf watch`` as a process: the argument wiring nothing else can see.

    ``tests/integration/test_cli_watch.py`` drives the follower functions against
    a real server and so proves they work. What it cannot prove is that argparse
    accepts the flags, that the handler is attached, or that the exit code
    survives ``main()`` — and this file already documents two bugs that were
    invisible from inside the interpreter and would have been invisible here too
    if the wiring had never been exercised as a command.
    """

    def test_the_command_is_listed_in_help(self) -> None:
        """A feature nobody can find is a feature nobody has."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "watch" in result.stdout

    def test_it_reports_an_unreachable_server_as_bad_input(self) -> None:
        """One line naming the address, and exit 2 rather than a traceback.

        The port is bound and released so nothing is listening on it. ``--trace``
        is the opt-in that restores the stack trace, and this asserts only on the
        default.
        """
        result = run_cli("watch", "no-such-run", "--url", "http://127.0.0.1:9", "--max-frames", "1")
        assert result.returncode == 2, result.stderr
        assert "cannot reach" in result.stderr
        assert "Traceback (most recent call last)" not in result.stderr

    def test_the_url_comes_from_the_environment(self) -> None:
        """``AWF_API_URL`` is the documented default and has to be honoured.

        Hard-coding the default instead would leave the flag documented and
        ignored — the same shape as the ``--log-level`` bug at the top of this
        file, where a setting was accepted, listed in ``--help``, and never read.
        """
        result = run_cli(
            "watch",
            "no-such-run",
            "--max-frames",
            "1",
            env=(("AWF_API_URL", "http://127.0.0.1:9"),),
        )
        assert result.returncode == 2, result.stderr
        assert "127.0.0.1:9" in result.stderr, result.stderr

    def test_a_token_from_the_environment_is_not_echoed(self) -> None:
        """The credential must not reach stderr on the failure path."""
        result = run_cli(
            "watch",
            "no-such-run",
            "--max-frames",
            "1",
            env=(
                ("AWF_API_URL", "http://127.0.0.1:9"),
                ("AWF_API_TOKEN", "s3cret-token"),
            ),
        )
        assert result.returncode == 2, result.stderr
        assert "s3cret-token" not in result.stderr
        assert "s3cret-token" not in result.stdout


class TestTopology:
    """``awf topology`` is the documentation the docs cannot go stale on."""

    def test_it_names_every_node(self) -> None:
        result = run_cli("topology")
        assert result.returncode == 0, result.stderr[-2000:]
        for node in (
            "triage",
            "programmer",
            "reviewer",
            "tester",
            "router",
            "apply_patch",
            "reporter",
        ):
            assert node in result.stdout, node

    def test_it_emits_json(self) -> None:
        payload = json.loads(run_cli("topology", "--json").stdout)
        assert payload["nodes"]
        assert payload["edges"] or payload["routes"]
