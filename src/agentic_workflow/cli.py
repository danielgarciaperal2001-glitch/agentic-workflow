"""Command-line entry point: ``awf`` / ``agentic-workflow``.

Six subcommands, chosen so that every claim the README makes can be checked from
a terminal without starting the API or writing a test:

* ``demo`` — run a real review end to end, pausing at each human gate.
* ``replay`` — list a run's checkpoints and re-execute from any of them.
* ``topology`` — print the graph's nodes, edges, cycles and routing table.
* ``eval`` — score the workflow's output quality against a golden dataset.
* ``janitor`` — one checkpoint-retention pass.
* ``watch`` — follow a run on a *running server*, as it happens.

Design notes:

* **No network, no credentials required** for the first five. The default
  provider is ``echo``, so ``awf demo`` works on a fresh clone and in CI.
  ``--provider`` switches it. ``watch`` is the exception and is a client: it
  talks to a server over a WebSocket, and needs one to talk to.
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
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote

from agentic_workflow import __version__

#: Exit code used when a run reached a terminal state that is not a success.
NOT_DONE = 1
#: Exit code used for bad input.
BAD_INPUT = 2
#: Exit code used when the server refused the request because a budget was
#: exhausted — the per-client request budget, the concurrency budget, or a run's
#: own token budget. Distinct from :data:`NOT_DONE` because the run never
#: started: nothing is pending and nothing is progressing, so a script that
#: treats ``NOT_DONE`` as "waiting, check again later" would wait forever for a
#: run that does not exist. The remedy differs too — wait, versus submit less or
#: raise the ceiling.
THROTTLED = 3

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
    from agentic_workflow.logging import configure_logging
    from agentic_workflow.persistence.checkpointer import (
        build_checkpointer,
        build_memory_checkpointer,
    )
    from agentic_workflow.services.engine import WorkflowEngine

    settings = load_settings(
        llm_provider=args.provider,
        log_level=args.log_level,
        graph_name=f"awf-cli-{args.command}",
        # `--memory` is an override, not an inversion. Passing
        # `postgres_enabled=not args.memory` unconditionally meant that omitting
        # the flag *enabled* PostgreSQL — so `awf demo` on a fresh clone, with
        # no database and no DSN, sat for 30 seconds retrying a connection
        # nobody asked for. Omitting the key entirely lets the environment
        # decide, which is the same choice `docker compose` already makes.
        **({"postgres_enabled": False} if args.memory else {}),
    )
    # Installed here, with these settings, and forced. `get_logger` lazily
    # configures on first use with the *default* settings rather than the ones
    # this command built, so without this line `--log-level` was accepted,
    # printed in `--help`, and then ignored — every command logged at INFO.
    # `force` because a module-level import may already have triggered the lazy
    # path with the wrong settings, and a no-op reconfigure would leave them.
    configure_logging(settings, force=True)
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
# Watching a run on a server
# --------------------------------------------------------------------------- #
#: The event that opens a stream, before any run state.
STREAM_OPEN = "stream.open"
#: The event carrying the run's state at the moment the client attached.
STREAM_SNAPSHOT = "stream.snapshot"
#: Keep-alive frame, and the only channel a loss is reported on.
HEARTBEAT = "heartbeat"

#: The event after which the server closes the socket.
#:
#: Kept here rather than imported from the API package: `watch` is a *client*,
#: and a client that imports the server's constants stops noticing when the
#: server changes them. The names are the wire contract, and the contract is
#: duplicated on purpose so a rename shows up as a test failure instead of
#: silently agreeing with itself.
TERMINAL_EVENT_NAMES = frozenset(
    {"run.completed", "run.failed", "run.cancelled", "run.interrupted", "run.rejected"}
)


def _format_frame(frame: dict[str, Any]) -> list[str]:
    """Render one wire frame as the lines a terminal should show.

    Returns lines rather than printing them, because the decision of *whether*
    to print is part of the protocol: a keep-alive is not information, and a
    formatter that always returned a line would have the caller deciding that
    instead.

    Args:
        frame: One decoded JSON frame from the stream.

    Returns:
        The lines to print, or an empty list for a frame that carries nothing
        worth showing.
    """
    name = str(frame.get("event") or "")

    if name == HEARTBEAT:
        # The only frame that can report a loss, and the reason a client cannot
        # detect one itself: `seq` counts publications across every run, so the
        # gaps a subscriber sees belong to other runs, and because the loss is
        # always the *oldest* event it lands at the head of the queue where no
        # gap appears at all. A watcher that printed the rest of the stream as
        # though it were complete would be the one dishonest thing this command
        # could do.
        lost = frame.get("dropped_since_last")
        if isinstance(lost, int) and lost > 0:
            return [
                (
                    f"  ⚠ dropped {lost} event(s) — this view has a hole in it; "
                    f"the run itself is unaffected"
                )
            ]
        return []

    if name == STREAM_SNAPSHOT:
        lines = [f"snapshot  {frame.get('status', '?')}"]
        pending = frame.get("pending_approval")
        if isinstance(pending, dict):
            stage = pending.get("stage", "?")
            lines.append(f"  waiting on {stage}: {pending.get('title', '?')}")
            rationale = pending.get("rationale")
            if rationale:
                lines.append(f"  why: {rationale}")
        report = frame.get("report")
        if isinstance(report, dict) and report.get("verdict"):
            lines.append(f"  verdict: {report['verdict']}")
        return lines

    if name == STREAM_OPEN:
        # Proof of life, and the only place the heartbeat interval is visible.
        # Not printed: the caller has already said it is connecting, and a
        # "connected" line is noise before anything has happened.
        return []

    if name in TERMINAL_EVENT_NAMES:
        return [f"── {name}"]

    if not name:
        # A frame with no event name is malformed. Reported rather than dropped:
        # silently skipping frames is how a watcher ends up showing a run that
        # has gone quiet at the exact moment something happened to it.
        return [f"  ? unrecognised frame: {json.dumps(frame, default=str)[:120]}"]

    detail = frame.get("node") or frame.get("stage") or ""
    suffix = f"  {detail}" if detail else ""
    return [f"{name}{suffix}"]


async def _watch(
    base_url: str,
    run_id: str,
    *,
    token: str | None = None,
    max_frames: int = 0,
    quiet: bool = False,
) -> int:
    """Follow one run's event stream and print it until the run ends.

    The client half of the WebSocket protocol. Three properties of the server's
    stream shape what this can do, and each is a decision rather than an
    accident:

    * The first content frame is a **snapshot**, so attaching late still shows
      where the run is. Without it, watching a run that has been parked on a
      human gate for an hour shows an empty screen until the next event, which
      may never come.
    * A **terminal event closes the socket**, so waiting for more after
      ``run.completed`` would hang forever. The wait ends on that event. A run
      that had already finished is replayed onto this subscription from the
      snapshot's status, so attaching late ends the stream too.
    * A **loss is reported on the heartbeat channel**, and a client that ignored
      it would print a stream with a hole in it and say nothing.

    Args:
        base_url: HTTP or WebSocket origin of the server, with or without a
            trailing slash.
        run_id: The run to follow.
        token: Bearer token for a server with ``api_auth_enabled``. It travels in
            the query string because a WebSocket handshake cannot carry an
            ``Authorization`` header from a browser, and the server reads it from
            there. That is why the default is the environment: a token typed on
            the command line is in the shell history and in ``ps`` output.
        max_frames: Stop after this many frames. ``0`` means never, which is what
            an interactive watch wants; a bounded value is how a test stops a
            follow of a run that is not going to finish.
        quiet: Suppress per-frame output, keeping only the summary. Used by the
            tests, which assert on the return value rather than on the text.

    Returns:
        ``0`` when the run reached a successful terminal state, ``NOT_DONE`` for
        any other ending, ``BAD_INPUT`` when the server could not be reached or
        refused the connection.
    """
    import websockets

    origin = base_url.rstrip("/")
    if origin.startswith("http://"):
        origin = f"ws://{origin[len('http://') :]}"
    elif origin.startswith("https://"):
        origin = f"wss://{origin[len('https://') :]}"
    url = f"{origin}/ws/runs/{run_id}"
    if token:
        url = f"{url}?token={quote(token, safe='')}"

    terminal: str | None = None
    seen = 0
    try:
        async with websockets.connect(url) as socket:
            while True:
                raw = await socket.recv()
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    lines = [f"  ? unparseable frame: {str(raw)[:120]}"]
                else:
                    lines = _format_frame(frame)

                name = str(frame.get("event") or "") if isinstance(frame, dict) else ""
                seen += 1
                if not quiet:
                    for line in lines:
                        _echo(line)
                if name in TERMINAL_EVENT_NAMES:
                    terminal = name
                    break
                if max_frames and seen >= max_frames:
                    break
    except OSError as exc:
        # One line naming the address, not a traceback: a watcher pointed at a
        # host that is not there has to be able to tell which host. The query
        # string is dropped for the same reason the token went in it — this line
        # goes to stderr, which is exactly where a pasted secret gets collected.
        _echo(f"cannot reach {url.split('?')[0]}: {exc}", file=sys.stderr)
        return BAD_INPUT
    except Exception as exc:  # websockets raises its own hierarchy
        _echo(f"watch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return BAD_INPUT

    if terminal is None:
        return NOT_DONE
    return 0 if terminal == "run.completed" else NOT_DONE


async def _cmd_watch(args: argparse.Namespace) -> int:
    """Follow a run on a running server.

    Args:
        args: Parsed arguments carrying ``url``, ``run_id``, ``token`` and
            ``max_frames``.

    Returns:
        The process exit code.
    """
    return await _watch(args.url, args.run_id, token=args.token, max_frames=args.max_frames)


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #
def _content_hash(files: list[dict[str, str]], description: str) -> str:
    """Compute the server's ``X-Content-Hash`` for a submission.

    A byte-for-byte restatement of :attr:`ReviewRequest.content_hash`, and that
    duplication is the point: the CLI cannot import the domain model and ask it,
    because the whole point of this command is that the engine lives in another
    process. So the formula is restated here, and
    ``test_the_client_hash_matches_the_domains_own`` asserts the two agree. A
    hash computed any other way would not be "a slightly different key", it
    would be a *different* key, and the server would answer a mismatch with
    ``400`` — a failure that reads like the tool is broken.

    Args:
        files: The submitted files, each with ``path`` and ``content``, in the
            order they will be sent. Order matters: the digest is a stream.
        description: The run's description.

    Returns:
        The header value, ``sha256:`` plus 32 hex characters.
    """
    import hashlib

    hasher = hashlib.sha256()
    for entry in files:
        hasher.update(entry["path"].encode())
        hasher.update(b"\0")
        hasher.update(entry["content"].encode())
        hasher.update(b"\0")
    hasher.update(description.encode())
    return f"sha256:{hasher.hexdigest()[:32]}"


def _run_id_for(title: str) -> str:
    """Derive a wire-legal run id from *title*.

    ``run_id`` is required on the wire and is also the handle ``awf watch``
    takes, so a default that a user cannot predict would make the run
    unfollowable without first reading the output. The title is the one thing
    the user always supplied, so it is the natural id.

    The wire constrains ids to ``^[A-Za-z0-9._:-]+$``, which a perfectly
    ordinary title can violate — ``Fix: checkout/total`` is a normal thing to
    type. Only the characters the pattern forbids are replaced; case is kept,
    because folding ``Fix the total`` to ``fix-the-total`` would be a
    transformation the user never asked for and could not predict, and a run id
    is meant to be guessable from the title they typed. Runs of dashes collapse
    and the result is truncated to the 128-character limit. A title that
    sanitises to nothing (``"///"``, whitespace) falls back to ``run`` rather
    than sending an empty id, which the server would reject as a validation
    error with no explanation.

    Args:
        title: The run's title.

    Returns:
        A non-empty id matching the wire's pattern.
    """
    import re

    slug = re.sub(r"[^A-Za-z0-9._:-]+", "-", title.strip()).strip("-.")
    slug = re.sub(r"-{2,}", "-", slug)[:128].strip("-")
    return slug or "run"


def _wire_path(raw: str) -> str:
    """Turn a path the user typed into one the API will accept.

    ``SourceFile.path`` rejects anything absolute and anything containing ``..``,
    because the name is fed to the agents as repository-relative context. That is
    a good rule for a client assembling a body by hand and a bad surprise for a
    command line, where ``--file /home/me/project/src/checkout.py`` and
    ``--file ../shared/util.py`` are the most natural things to type. Rejecting
    them would make the command fail on ordinary use with a validation error
    about a field the user never named.

    So the path is made relative to the working directory, which is the
    repository root as far as a command line is concerned. A file outside it
    cannot be expressed relatively without a ``..`` the API forbids, so it is
    sent by name alone: the name is a label the agents read, and the content —
    which is what actually gets reviewed — is unaffected.

    Args:
        raw: The path as typed.

    Returns:
        A relative path with no ``..``, falling back to the file's name.
    """
    path = Path(raw)
    posix = path.as_posix()
    if not posix.startswith("/") and ".." not in posix.split("/"):
        return posix
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


def _read_files(paths: list[str]) -> list[dict[str, str]]:
    """Read the submitted files, or report the first one that cannot be read.

    Reading happens before the request is built, and therefore before anything
    is sent. A submission whose body cannot be assembled should not cost a
    request, and the message has to carry the offending path: "no such file"
    without it is a wild goose chase through a ``--file`` list.

    Args:
        paths: Paths to read, in the order the user gave them.

    Returns:
        One ``{"path", "content"}`` mapping per file, with the path rewritten
        by :func:`_wire_path` so the API accepts it.

    Raises:
        OSError: If a path cannot be read. The message names the path.
    """
    files: list[dict[str, str]] = []
    for raw in paths:
        path = Path(raw)
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError(f"cannot read {raw}: {exc.strerror or exc}") from exc
        files.append({"path": _wire_path(raw), "content": content})
    return files


def _read_metadata(pairs: list[str]) -> dict[str, str]:
    """Parse repeated ``key=value`` metadata arguments.

    A malformed pair is a typo the user can see and fix, so it is rejected here
    rather than forwarded. Forwarding it would produce a server-side validation
    error whose message describes the field, not the ``--metadata`` argument
    that was actually typed — the same "silent until it fails" shape as any
    other argument mistake.

    Args:
        pairs: The raw ``key=value`` strings.

    Returns:
        The metadata mapping.

    Raises:
        ValueError: If a pair has no ``=`` or an empty key.
    """
    metadata: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"--metadata expects key=value, got {pair!r}")
        metadata[key] = value
    return metadata


def _error_of(payload: Any, status: int) -> tuple[str, str, dict[str, Any]]:
    """Pull ``(code, message, context)`` out of an error envelope.

    The server answers errors with a stable envelope, so the CLI reads that
    rather than pattern-matching on prose. When the body is not an envelope — a
    proxy's HTML error page, say — the status is reported instead of a
    ``KeyError`` escaping out of a formatter.

    Args:
        payload: The decoded response body.
        status: The HTTP status, used when the body is not an envelope.

    Returns:
        The error code, its message, and any context the server attached.
    """
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        error = payload["error"]
        context = error.get("context")
        return (
            str(error.get("code") or f"http_{status}"),
            str(error.get("message") or f"HTTP {status}"),
            context if isinstance(context, dict) else {},
        )
    return f"http_{status}", f"HTTP {status}", {}


class SubmissionError(OSError):
    """A submission that never produced an HTTP response.

    Two things can go wrong that are not answers from the server: the connection
    never opened, or it opened and then went quiet. They need different advice,
    so this carries a message already written for the case rather than making the
    caller re-derive one from an exception class.

    Deriving it as an :class:`OSError` is also what keeps ``httpx`` from leaking
    out of this module. ``httpx`` is imported inside the functions that use it so
    that a plain ``awf demo`` on a machine without the ``api`` extra still works,
    and the natural place for that isolation to leak is a caller writing
    ``except httpx.HTTPError`` — which would then be an ``ImportError`` on
    exactly the installations the lazy import exists to protect.

    Note what is *not* here: ``httpx``'s own exceptions do not inherit from
    ``OSError``, so a bare ``except OSError`` around the request catches none of
    them. An unreachable server therefore arrives as an unhandled
    ``ConnectError`` — a traceback-free one-line message and exit code 1, which
    reads as "the run did not finish" rather than "nothing was listening".
    """


async def _submit(
    base_url: str,
    body: dict[str, Any],
    *,
    content_hash: str,
    token: str | None = None,
    timeout: float = 900.0,
) -> tuple[int, Any]:
    """POST a run body to a server and return its status and decoded body.

    The token travels as an ``Authorization`` header, not in the URL. ``watch``
    cannot do that — a WebSocket handshake from a browser carries no
    ``Authorization`` header, which is why its token rides in the query string
    and why its error messages strip the query string before printing. HTTP has
    no such constraint, and a credential in a URL is written down by every
    proxy and load balancer on the path.

    Args:
        base_url: The server's HTTP origin.
        body: The wire body.
        content_hash: The ``X-Content-Hash`` idempotency key.
        token: Bearer token for a server with ``api_auth_enabled``.
        timeout: Seconds to wait for the response. Generous by default because
            a synchronous ``POST /v1/runs`` drives the whole graph — up to
            ``run_timeout_seconds`` — and a client deadline shorter than the
            server's own would abandon a run that is still working.

    Returns:
        The HTTP status and the decoded body.

    Raises:
        SubmissionError: If the server could not be reached, or the response
            never arrived. A timeout says so explicitly, because the run may
            still be going and re-submitting it would duplicate the work.
    """
    import httpx

    headers = {"X-Content-Hash": content_hash}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/v1/runs", json=body, headers=headers
            )
    except httpx.TimeoutException as exc:
        # The run may well exist: the request reached the server, and the server
        # is still driving the graph. Saying so is the difference between an
        # operator watching the run and an operator submitting it again.
        raise SubmissionError(
            f"no response from {base_url} within {timeout:g}s. The run may still "
            f"be running on the server — look for it with "
            f"GET /v1/runs/{body.get('run_id', '?')} rather than submitting again"
        ) from exc
    except httpx.HTTPError as exc:
        raise SubmissionError(f"cannot reach {base_url}: {exc}") from exc

    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _report_refusal(code: str, message: str, context: dict[str, Any]) -> int:
    """Print one refusal the way a user can act on it, and pick an exit code.

    The refusals a submission can hit mean different things, so they are told
    apart rather than collapsed into "request failed":

    * a **budget** refusal is not transient, so the message says so and says
      which budget. Retrying the same payload reaches the same answer;
    * a **401** is a missing or wrong token, so it names the env var that
      supplies it. A bare status code teaches nothing about the fix;
    * a **409** on a run id is usually a real duplicate, so it names the two
      ways out: the content hash turns a genuine resubmission into a replay, and
      ``--run-id`` starts a different thread.

    Args:
        code: The error code from the envelope.
        message: Its message.
        context: Whatever context the server attached.

    Returns:
        The process exit code.
    """
    if code == "token_budget_exceeded":
        budget = context.get("budget")
        spent = context.get("spent")
        detail = f" (budget {budget}, spent {spent})" if budget is not None else ""
        _echo(
            f"refused: this run reached its token budget{detail}, which is more "
            f"than one submission should be allowed to cost. Submit fewer or "
            f"smaller files, or raise llm_token_budget_per_run. Retrying the same "
            f"payload spends the same tokens to reach the same answer.",
            file=sys.stderr,
        )
        return THROTTLED

    if code in {"rate_limited", "concurrency_limit"}:
        _echo(
            f"refused: {code}. The server is at a limit; wait and submit again.",
            file=sys.stderr,
        )
        return THROTTLED

    if code == "authentication_error":
        _echo(
            "refused: 401. The server has api_auth_enabled on. Set AWF_API_TOKEN or pass --token.",
            file=sys.stderr,
        )
        return BAD_INPUT

    if code == "run_already_exists":
        _echo(
            f"refused: {message}. A run with that id already exists. Resubmitting "
            f"the same content replays it instead; pass --run-id to start a "
            f"different one.",
            file=sys.stderr,
        )
        return BAD_INPUT

    _echo(f"refused: {message}", file=sys.stderr)
    return BAD_INPUT


def _report_run(detail: dict[str, Any], *, as_json: bool = False) -> int:
    """Print a submitted run, or its verbatim payload when ``--json`` is set.

    ``--json`` prints the server's payload unchanged. Reformatting it here would
    mean this command had to learn about every field the API adds, and would
    leave ``--json`` disagreeing with ``GET /v1/runs/{id}`` — which defeats the
    purpose of a flag whose entire job is to make ``jq`` work.

    Args:
        detail: The ``RunDetail`` the server returned.
        as_json: Print the payload verbatim instead of the summary.

    Returns:
        ``0`` for a finished run, :data:`NOT_DONE` for one waiting on a human.
    """
    if as_json:
        _echo(json.dumps(detail, indent=2, sort_keys=True))
        return 0 if detail.get("status") in {"completed", "rejected"} else NOT_DONE

    run_id = str(detail.get("run_id", ""))
    status = str(detail.get("status", ""))
    usage = detail.get("usage") or {}

    _rule(f"run {run_id} — {status}")
    pending = detail.get("pending_approval")
    if isinstance(pending, dict):
        stage = pending.get("stage", "a human decision")
        _echo(f"  waiting on: {stage} — {pending.get('title', '')}")
    if detail.get("error"):
        _echo(f"  error: {detail['error']}")
    if usage.get("total_tokens") is not None:
        _echo(
            f"  usage: {usage['total_tokens']} tokens over {usage.get('calls', 0)} calls"
            f" ({usage.get('prompt_tokens', 0)} in,"
            f" {usage.get('completion_tokens', 0)} out)"
        )
    if detail.get("request_id"):
        _echo(f"  request: {detail['request_id']}")

    if status in {"completed", "rejected"}:
        return 0

    _echo(f"  follow it:  awf watch {run_id}")
    _echo(f"  read it:    GET /v1/runs/{run_id}")
    return NOT_DONE


async def _cmd_run(args: argparse.Namespace) -> int:
    """Submit a review to a running server.

    The body is assembled entirely from local state before a socket is opened,
    so a bad ``--file`` or ``--metadata`` costs nothing and names itself. The
    ``X-Content-Hash`` header goes out with every submission, which is what
    makes pressing enter twice return the original run instead of creating a
    second one.

    Args:
        args: Parsed arguments for the ``run`` subcommand.

    Returns:
        The process exit code: ``0`` finished, :data:`NOT_DONE` parked on a
        human, :data:`THROTTLED` refused by a budget, :data:`BAD_INPUT` for
        anything the caller can fix. With ``--watch`` the code comes from
        following the run instead, because how the run ended is the
        longer-lived fact than how its submission was answered.
    """
    try:
        files = _read_files(args.files)
        metadata = _read_metadata(args.metadata)
    except (OSError, ValueError) as exc:
        _echo(str(exc), file=sys.stderr)
        return BAD_INPUT

    run_id = args.run_id or _run_id_for(args.title)
    body: dict[str, Any] = {
        "run_id": run_id,
        "request_id": args.request_id or run_id,
        "title": args.title,
        "description": args.description,
        "language": args.language,
        "files": files,
        "acceptance_criteria": args.acceptance_criteria,
        "constraints": args.constraints,
        "metadata": metadata,
        "auto_resolve": args.auto_resolve,
        "auto_decision": args.auto_decide,
        "max_gates": args.max_gates,
    }

    try:
        status, payload = await _submit(
            args.url,
            body,
            content_hash=_content_hash(files, args.description),
            token=args.token,
            timeout=args.timeout,
        )
    except SubmissionError as exc:
        _echo(str(exc), file=sys.stderr)
        return BAD_INPUT

    if status >= 400 or not isinstance(payload, dict):
        code, message, context = _error_of(payload, status)
        return _report_refusal(code, message, context)

    if args.watch:
        # The events that follow would be buried under a summary of the run
        # they are events *of*, so with --watch only the stream is printed.
        return await _watch(args.url, run_id, token=args.token)

    return _report_run(payload, as_json=args.json)


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
    default = os.environ.get("USER") or os.environ.get("USERNAME") or "operator"
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

    from agentic_workflow.config import load_settings

    # The flag wins so a one-off run needs no config edit; the setting is the
    # default so a deployment that grades its own dataset is configured once.
    # Passing ``None`` would silently fall back to the bundled golden set.
    dataset = args.dataset or load_settings().eval_dataset_path

    report = await run_suite(
        dataset=dataset,
        provider=args.provider,
        limit=args.limit,
        memory=True,
        gates=args.gates,
        gate=args.gate,
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
    from agentic_workflow.logging import configure_logging
    from agentic_workflow.persistence.checkpointer import (
        build_checkpointer,
        build_memory_checkpointer,
    )
    from agentic_workflow.persistence.retention import CheckpointJanitor

    settings = load_settings(
        llm_provider=args.provider,
        log_level=args.log_level,
        # Same override-not-inversion rule as the engine-building commands: an
        # omitted `--memory` honours AWF_POSTGRES_ENABLED rather than forcing a
        # connection to a database the operator may not have.
        **({"postgres_enabled": False} if args.memory else {}),
    )
    # Installed with these settings, for the same reason as the other commands:
    # otherwise --log-level is accepted and then ignored.
    configure_logging(settings, force=True)
    checkpointer = build_memory_checkpointer() if args.memory else build_checkpointer(settings)
    if not args.json:
        # The rule is suppressed under --json so stdout carries the document and
        # nothing else. `awf janitor --json | jq` has to work, and a banner above
        # the JSON makes it unparseable — the failure is silent, because the
        # output still looks like it printed something.
        _rule("checkpoint janitor" + ("  (dry run)" if args.dry_run else ""))
    # The engine's `startup` awaits this for its own checkpointer; the janitor
    # builds one directly and had to do the same, or the first list hit a pool
    # that was never opened. Without it the durable command could not run at all:
    # for the `postgres` extra, `build_checkpointer` returns a
    # `PostgresCheckpointer` wrapper whose docstring is explicit that `setup`
    # must be awaited before use. `InMemorySaver` has no `setup`, so the guard
    # is the branch.
    setup = getattr(checkpointer, "setup", None)
    if setup is not None:
        await setup()
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
            "  awf watch pr-1042                      follow a run on a running server\n"
            "  awf run --title 'fix totals' -f a.py  submit work to a running server\n"
            "  awf run --title 'fix totals' --watch  submit it and follow the events\n"
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
        help=(
            "Path to a JSONL dataset. Defaults to AWF_EVAL_DATASET_PATH, so a "
            "deployment that grades against its own set needs no flag."
        ),
    )
    evaluate.add_argument("--limit", type=int, default=None, help="Only the first N cases.")
    evaluate.add_argument(
        "--gates",
        default="approve",
        choices=("approve", "edit", "reject"),
        help="Verdict applied at every gate (default: approve).",
    )
    evaluate.add_argument(
        "--gate",
        default="all",
        choices=("all", "invariants", "detection"),
        help=(
            "Which metrics decide pass/fail. 'invariants' gates the "
            "provider-independent correctness properties (no fabricated quotes, "
            "no invented paths, no self-contradiction) and leaves recall as a "
            "reported floor — use it to gate a build on the offline provider, "
            "which cannot detect defects and would fail every run."
        ),
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

    watch = subparsers.add_parser(
        "watch",
        help="Follow a run's event stream on a running server.",
        description=(
            "Follow a run that is happening somewhere else. Prints the run's "
            "current state, then every event, and exits when the run reaches a "
            "terminal state. The stream drops events for a client that cannot "
            "keep up and says so, so treat a 'dropped N events' line as a cue "
            "to re-read the run over REST rather than as noise."
        ),
    )
    watch.add_argument("run_id", help="Run to follow.")
    watch.add_argument(
        "--url",
        default=os.environ.get("AWF_API_URL", "http://localhost:8000"),
        help=(
            "Server to watch (default: $AWF_API_URL, else "
            "http://localhost:8000). http:// is upgraded to ws://."
        ),
    )
    watch.add_argument(
        "--token",
        default=os.environ.get("AWF_API_TOKEN"),
        help=(
            "Bearer token for a server with api_auth_enabled (default: "
            "$AWF_API_TOKEN). Prefer the environment: this value travels in the "
            "URL query string and would otherwise be in your shell history."
        ),
    )
    watch.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames (default: 0, follow until the run ends).",
    )
    watch.set_defaults(handler=_cmd_watch)

    _add_run_parser(subparsers)

    return parser


def _add_run_parser(subparsers: Any) -> None:
    """Register the ``run`` subcommand.

    Split out of the parser builder because this one carries a dozen arguments,
    and inlining them would bury the other subcommands.

    Args:
        subparsers: The parser's subparser action.
    """
    run = subparsers.add_parser(
        "run",
        help="Submit a review to a running server.",
        description=(
            "Submit work to a server and print what it did with it. The files "
            "are read locally and their contents sent, so the reviewed content "
            "and the content on the wire are the same bytes. Every submission "
            "carries an X-Content-Hash, so submitting the same work twice returns "
            "the original run instead of starting a second one. A run parked on a "
            "human decision exits 1; a run refused by a budget exits 3, because "
            "that run never started and waiting for it would wait forever."
        ),
    )
    run.add_argument("--title", required=True, help="One-line subject of the review.")
    run.add_argument(
        "--file",
        dest="files",
        action="append",
        default=[],
        metavar="PATH",
        help="File to review; repeat for several. The path is sent as its name.",
    )
    run.add_argument("--description", default="", help="The problem statement.")
    run.add_argument("--language", default="python", help="Primary language (default: python).")
    run.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Run id (default: the title, slugged). Also the handle `awf watch` takes.",
    )
    run.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help="Business id such as a PR number (default: the run id).",
    )
    run.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Correlation data echoed back on every read of the run; repeatable.",
    )
    run.add_argument(
        "--criteria",
        dest="acceptance_criteria",
        action="append",
        default=[],
        metavar="TEXT",
        help="Condition the report must satisfy; repeatable.",
    )
    run.add_argument(
        "--constraint",
        dest="constraints",
        action="append",
        default=[],
        metavar="TEXT",
        help="Hard limit the agents must respect; repeatable.",
    )
    run.add_argument(
        "--auto-resolve",
        dest="auto_resolve",
        action="store_true",
        help="Answer every human gate automatically and run to completion in one call.",
    )
    run.add_argument(
        "--auto-decide",
        dest="auto_decide",
        choices=("approve", "edit", "reject"),
        default="approve",
        help="Verdict used with --auto-resolve (default: approve).",
    )
    run.add_argument(
        "--max-gates",
        dest="max_gates",
        type=int,
        default=32,
        help="Safety bound on auto-resolved gates (default: 32).",
    )
    run.add_argument(
        "--watch",
        action="store_true",
        help="Follow the run's events after submitting it.",
    )
    run.add_argument(
        "--json",
        dest="json",
        action="store_true",
        help="Print the server's response verbatim, for jq.",
    )
    run.add_argument(
        "--url",
        default=os.environ.get("AWF_API_URL", "http://localhost:8000"),
        help="Server to submit to (default: $AWF_API_URL, else http://localhost:8000).",
    )
    run.add_argument(
        "--token",
        default=os.environ.get("AWF_API_TOKEN"),
        help=(
            "Bearer token for a server with api_auth_enabled (default: "
            "$AWF_API_TOKEN). Sent as an Authorization header, never in the URL."
        ),
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help=(
            "Seconds to wait for the response (default: 900). A submission drives "
            "the whole graph, so keep this at or above the server's "
            "run_timeout_seconds or you will abandon a run that is still working."
        ),
    )
    run.set_defaults(handler=_cmd_run)


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
        if os.environ.get("AWF_CLI_TRACE"):
            raise
        # A traceback from a CLI buries the one line that matters. The class name
        # is kept because "WorkflowError: [concurrency_limit] ..." and
        # "RuntimeError: ..." call for very different responses from the reader.
        _echo(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        _echo("       set AWF_CLI_TRACE=1 for the full traceback.", file=sys.stderr)
        return NOT_DONE


if __name__ == "__main__":  # pragma: no cover - module execution
    raise SystemExit(main())
