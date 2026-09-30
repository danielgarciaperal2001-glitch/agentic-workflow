"""``awf run``: submit a review to a running server, from a terminal.

Every other subcommand builds its own engine in-process — ``demo``, ``replay``,
``topology``, ``eval`` and ``janitor`` all run the workflow locally. ``watch``
went the other way and reads a server. That left a gap in the middle: the one
command you want against a real deployment, *start a run*, did not exist, so
submitting work to a server meant ``curl`` and hand-assembling a body with the
file contents inlined. The gap is not cosmetic. ``curl`` means the ``X-Content-
Hash`` idempotency key nobody types by hand, so a double submit creates two
runs, and it means no read of the answer, so the run's status and its token
spend have to be fetched by a second command that did not exist either.

So the interesting properties here are not "can it POST" — ``httpx`` does that
in one line — but the three that make a submission safe to retry and readable
without ``jq``:

* the **content hash is computed by the same formula the server uses**, so
  ``X-Content-Hash`` is a real idempotency key and a retried command returns the
  original run instead of a duplicate. A hash computed any other way would be a
  *different* hash, and the server answers a mismatch with ``400``, which reads
  like the tool is broken;
* a **refusal is distinguished from a wait**. A run parked on a human gate is a
  ``202`` and a success by the server's own definition, but a script needs to
  tell it apart from a completed run, so the two get different exit codes. A
  budget refusal is a third thing again: the run never started;
* the **token is an ``Authorization`` header, not a query string**. The
  WebSocket handshake in ``watch`` cannot carry a header, which is why the token
  travels in the URL there and why that code has to strip the query string out
  of its error messages. HTTP can carry the header, so ``run`` does not put a
  credential in a URL that ends up in a proxy log, a shell history or a
  ``ps`` listing.

The server-side tests run against a real uvicorn on a real port, because
``TestClient`` runs the ASGI app in-process and would exercise neither the
socket nor the timeouts. Measured cost of one server: about a second.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentic_workflow.cli import (
    BAD_INPUT,
    NOT_DONE,
    THROTTLED,
    _content_hash,
    _run_id_for,
    _wire_path,
)
from agentic_workflow.config import Settings
from agentic_workflow.domain.schemas import ReviewRequest, SourceFile
from tests.integration.conftest import LiveServer

pytestmark = pytest.mark.integration


def _settings(**overrides: Any) -> Settings:
    """Build a real, validated :class:`Settings` for a live server.

    Constructed rather than derived with ``model_copy``. That was the first
    version of these tests and it failed with ``SystemExit: 3``: ``model_copy``
    skips validation, so ``api_auth_token`` stayed a plain ``str`` instead of a
    ``SecretStr``, and the app refused to boot over a field the test thought it
    had set. A test that cannot start its server is a confusing way to learn
    about ``model_copy``.

    Args:
        **overrides: Fields to set over the offline defaults.

    Returns:
        Validated settings.
    """
    return Settings(
        _env_file=None,
        **{
            "environment": "development",
            "llm_provider": "echo",
            "postgres_enabled": False,
            "api_rate_limit_per_minute": 0,
            "ws_heartbeat_seconds": 0.05,
            **overrides,
        },
    )


class TestTheContentHashIsTheServersHash:
    """The idempotency key has to be the server's hash, not a lookalike."""

    def test_the_client_hash_matches_the_domains_own(self) -> None:
        """The CLI's hash equals what the server will compute for the same body.

        The formula is ``sha256`` over ``path NUL content NUL`` per file, then
        the description, truncated to 32 hex characters. A client that got this
        subtly wrong would not fail loudly: the server would compare two
        different digests and answer ``400`` with a message about
        ``X-Content-Hash``, which is indistinguishable from the client being
        broken. Asserting the two implementations agree is the only way to know
        the key means something.
        """
        files = [
            {"path": "src/checkout.py", "content": "def total():\n    return 0.1\n"},
            {"path": "tests/test_checkout.py", "content": "def test_total():\n    pass\n"},
        ]
        description = "the total is wrong"

        from_client = _content_hash(files, description)
        from_domain = ReviewRequest(
            run_id="r1",
            request_id="PR-1",
            title="t",
            description=description,
            files=[SourceFile(path=f["path"], content=f["content"]) for f in files],
        ).content_hash

        assert from_client == from_domain
        assert from_client.startswith("sha256:")

    def test_the_hash_changes_when_a_file_changes(self) -> None:
        """Editing one byte of one file is a different submission.

        Without this the hash could be stable across edits — e.g. hashing the
        path but not the content — and the idempotency key would then treat an
        edited review as a replay of the old one, returning the previous run's
        verdict for work that had changed.
        """
        before = [{"path": "a.py", "content": "x = 1\n"}]
        after = [{"path": "a.py", "content": "x = 2\n"}]

        assert _content_hash(before, "d") != _content_hash(after, "d")

    def test_the_hash_covers_the_description(self) -> None:
        """The description is part of what was asked, so it is part of the hash.

        The domain model hashes it last, and a client that skipped it would
        consider two differently-worded requests to be the same submission.
        """
        files = [{"path": "a.py", "content": "x = 1\n"}]

        assert _content_hash(files, "one") != _content_hash(files, "another")


class TestTheRunIdIsDerivable:
    """``run_id`` is required on the wire, so a default has to be a good one."""

    def test_a_title_becomes_a_readable_run_id(self) -> None:
        """The default id is the title, slugged — so ``watch`` can be predicted.

        A generated id would make the run unfollowable without reading the
        output, which defeats the point of having a ``watch`` command at all.
        Case is preserved rather than folded: lowercasing would be a
        transformation the user did not ask for and could not guess, and the id
        is only useful if it is predictable from the title that was typed.
        """
        assert _run_id_for("Fix checkout total") == "Fix-checkout-total"

    def test_a_title_the_id_pattern_would_reject_still_yields_a_legal_id(self) -> None:
        """Slugs are sanitised to the wire's ``SafeId`` pattern.

        The wire constrains ids to ``^[A-Za-z0-9._:-]+$``. A title with a slash,
        a space run or a non-ASCII character is perfectly ordinary, and passing
        it through unsanitised would be a ``422`` from the server — the tool
        refusing a legitimate request for no reason the user can see.
        """
        for title in ("Fix: checkout/total", "Añadir prueba", "a" * 300, "///", "  "):
            derived = _run_id_for(title)
            assert derived, f"no id derived from {title!r}"
            assert all(c.isalnum() or c in "._:-" for c in derived), title
            assert len(derived) <= 128, title


class TestAServerThatAnswersNothing:
    """A submission that stalls is not the same failure as one that is refused."""

    async def test_a_timeout_says_the_run_may_still_be_running(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A stalled response is reported as "it may still be working".

        The request did arrive — only the reply did not — so the run most likely
        exists on the server and is still driving the graph. Telling the user to
        submit again is the actively harmful advice: the content hash makes the
        second submission a replay, so the cost is a confused operator watching
        two ids for one piece of work. The message names the run id for the same
        reason, so the follow-up is a read rather than a retry.
        """
        import argparse
        import asyncio

        from agentic_workflow.cli import _cmd_run

        held: list[asyncio.StreamWriter] = []

        async def stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Accept the connection, read the request, never answer."""
            held.append(writer)
            await reader.read(1024)
            await asyncio.sleep(10)
            writer.close()

        server = await asyncio.start_server(stall, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            args = argparse.Namespace(
                run_id=None,
                request_id=None,
                title="will stall",
                description="",
                language="python",
                files=[],
                acceptance_criteria=[],
                constraints=[],
                metadata=[],
                auto_resolve=True,
                auto_decide="approve",
                max_gates=32,
                url=f"http://127.0.0.1:{port}",
                token=None,
                timeout=0.5,
                watch=False,
                json=False,
            )
            code = await _cmd_run(args)
        finally:
            # `Server.wait_closed()` waits for every handler to return, so the
            # sleeping connections have to be shut rather than awaited: without
            # this the test pays the handler's whole sleep on the way out.
            for writer in held:
                writer.close()
            server.close()
            await server.wait_closed()

        assert code == BAD_INPUT
        err = capsys.readouterr().err
        assert "still" in err.lower()
        assert "will-stall" in err, "the message must say which run to go and look for"


class TestRefusalsAreDistinguished:
    """A refusal is a decision the caller has to make, not a failure to report.

    Each of these is reachable by ordinary use, and collapsing them into one
    "request failed" line would leave the operator unable to tell a transient
    limit from a duplicate they should not retry. So the mapping is asserted per
    code rather than once for the shape.
    """

    @pytest.mark.parametrize(
        ("code", "expected", "mentions"),
        [
            ("token_budget_exceeded", THROTTLED, "token budget"),
            ("rate_limited", THROTTLED, "rate_limited"),
            ("concurrency_limit", THROTTLED, "concurrency_limit"),
            ("authentication_error", BAD_INPUT, "AWF_API_TOKEN"),
            ("run_already_exists", BAD_INPUT, "--run-id"),
            ("something_new_from_a_newer_server", BAD_INPUT, "the server said so"),
        ],
    )
    def test_each_refusal_code_maps_to_one_exit_code(
        self, code: str, expected: int, mentions: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every code produces advice and one of exactly three exit codes.

        The wording is this command's, not the server's, because the server
        cannot know that retrying the same payload costs the same tokens again —
        so a code the CLI has advice for is reported in prose rather than echoed
        verbatim, while an unrecognised one falls through with the server's own
        message. The catch-all matters as much as the specific cases: the CLI
        must not crash on a code added after this release, because a formatter
        raising on an unfamiliar code is how a refusal becomes an unexplained
        non-zero exit.
        """
        from agentic_workflow.cli import _report_refusal

        assert _report_refusal(code, "the server said so", {}) == expected
        assert mentions in capsys.readouterr().err

    def test_an_unrecognised_refusal_falls_back_to_the_servers_own_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """For a code with no advice of its own, the server's words are used.

        Discarding them would lose the only information available: an unfamiliar
        code with the server's explanation attached is still an explanation,
        while the code alone explains nothing.
        """
        from agentic_workflow.cli import _report_refusal

        assert _report_refusal("some_future_code", "disk is full", {}) == BAD_INPUT
        assert "disk is full" in capsys.readouterr().err

    def test_a_budget_refusal_names_both_numbers(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The two figures a user needs to decide what to change are both shown.

        "Raise the budget" is useless advice without knowing how far over it is,
        and "submit fewer files" is useless advice without knowing the ceiling.
        """
        from agentic_workflow.cli import _report_refusal

        _report_refusal(
            "token_budget_exceeded",
            "budget exceeded",
            {"budget": 500_000, "spent": 512_340, "calls": 12},
        )

        err = capsys.readouterr().err
        assert "500000" in err and "512340" in err
        assert "llm_token_budget_per_run" in err, "the message must name the knob to turn"

    def test_a_malformed_body_is_reported_not_formatted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A body that is not an error envelope still produces a message.

        Something in front of the server — a proxy, a gateway — answers with its
        own document. Reading ``payload["error"]["code"]`` from that raises, and
        the traceback replaces the one line that said what went wrong.
        """
        from agentic_workflow.cli import _error_of

        code, message, context = _error_of({"detail": "nginx"}, 502)

        assert code == "http_502"
        assert "502" in message
        assert context == {}


class TestAProxyThatAnswersWithItsOwnDocument:
    """The realistic non-envelope response, end to end over a real socket."""

    async def test_an_html_error_page_is_reported_as_a_refusal(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A gateway's HTML reaches no formatter and costs one clean line.

        Everything between the CLI and the app can answer with something that is
        not this API's error envelope, so this path is not hypothetical. What it
        must not do is raise, and what it must say is the status — the only thing
        in an HTML page that is actionable.
        """
        import argparse
        import asyncio

        from agentic_workflow.cli import _cmd_run

        page = b"<html><head><title>502 Bad Gateway</title></head></html>"

        async def gateway(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Answer every request with an HTML error page."""
            await reader.read(65536)
            writer.write(
                b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/html\r\n"
                b"Content-Length: " + str(len(page)).encode() + b"\r\n\r\n" + page
            )
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(gateway, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            args = argparse.Namespace(
                run_id=None,
                request_id=None,
                title="behind a gateway",
                description="",
                language="python",
                files=[],
                acceptance_criteria=[],
                constraints=[],
                metadata=[],
                auto_resolve=True,
                auto_decide="approve",
                max_gates=32,
                url=f"http://127.0.0.1:{port}",
                token=None,
                timeout=10.0,
                watch=False,
                json=False,
            )
            code = await _cmd_run(args)
        finally:
            server.close()
            await server.wait_closed()

        assert code == BAD_INPUT
        err = capsys.readouterr().err
        assert "502" in err
        assert "Traceback" not in err


class TestThePathTheApiWillAccept:
    """The API rejects absolute paths, and a command line is full of them."""

    def test_an_already_relative_path_is_sent_unchanged(self, tmp_path: Any) -> None:
        """The common case is passed through untouched.

        Rewriting a path that already satisfies the API's rule would make the
        name in the report differ from the name the user typed, for no gain.
        """
        assert _wire_path("src/checkout.py") == "src/checkout.py"

    def test_an_absolute_path_inside_the_working_directory_becomes_relative(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--file /cwd/src/a.py`` is the most natural thing to type.

        ``SourceFile.path`` rejects a leading ``/``, so sending it verbatim
        fails with a validation error about a field the user never named. The
        working directory is the repository root as far as a command line goes,
        so relativising against it produces exactly the name the user meant.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")

        assert _wire_path(str(tmp_path / "src" / "a.py")) == "src/a.py"

    def test_a_path_that_escapes_the_working_directory_is_sent_by_name(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file outside the repository is labelled by name, not by location.

        No relative form of that path avoids the ``..`` the API forbids. The
        name is a label the agents read as context; the content, which is what
        actually gets reviewed, is sent either way. Failing the submission over
        a label would cost the user the review they asked for.
        """
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "elsewhere" / "util.py"
        outside.parent.mkdir(exist_ok=True)
        outside.write_text("y = 2\n", encoding="utf-8")

        assert _wire_path(str(outside)) == "util.py"

    def test_parent_traversal_in_a_relative_path_is_resolved(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``a/../b.py`` is not sent with the ``..`` still in it.

        The API rejects a ``..`` path part, and a path written that way is a
        perfectly ordinary way to name a file, so it is normalised first.
        """
        monkeypatch.chdir(tmp_path)
        inner = tmp_path / "pkg" / "inner"
        inner.mkdir(parents=True)

        resolved = _wire_path("pkg/inner/../a.py")

        assert ".." not in resolved.split("/")
        assert resolved == "pkg/a.py"


class TestSubmittingToARealServer:
    """The submission itself, end to end over a socket."""

    async def test_an_auto_resolved_run_comes_back_finished(self, live_settings: Settings) -> None:
        """The happy path: one call, a completed run, exit 0.

        ``auto_resolve`` is the flag that makes this worth having, because the
        server drives every gate itself and the caller gets the whole run in a
        single round trip. Without it a CLI user would have to script gate
        resolution to get an answer at all.
        """
        import argparse

        from agentic_workflow.cli import _cmd_run

        args = argparse.Namespace(
            run_id=None,
            request_id=None,
            title="fix the total",
            description="the total is wrong",
            language="python",
            files=[],
            acceptance_criteria=[],
            constraints=[],
            metadata=[],
            auto_resolve=True,
            auto_decide="approve",
            max_gates=32,
            url="",
            token=None,
            timeout=90.0,
            watch=False,
            json=False,
        )
        async with LiveServer(live_settings) as server:
            args.url = server.base_url
            assert await _cmd_run(args) == 0

    async def test_a_run_that_parks_exits_distinctly_from_a_finished_one(
        self, live_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Parking is a wait, not a completion, and the exit code says so.

        The server answers ``202`` and treats a parked run as a success — it is
        healthy and resumable. For a shell script that is not enough: ``0`` on
        both a finished run and a run waiting for a person means a pipeline
        cannot tell whether there is work left to do.
        """
        import argparse

        from agentic_workflow.cli import _cmd_run

        args = argparse.Namespace(
            run_id=None,
            request_id=None,
            title="park me",
            description="",
            language="python",
            files=[],
            acceptance_criteria=[],
            constraints=[],
            metadata=[],
            auto_resolve=False,
            auto_decide="approve",
            max_gates=32,
            url="",
            token=None,
            timeout=90.0,
            watch=False,
            json=False,
        )
        async with LiveServer(live_settings) as server:
            args.url = server.base_url
            code = await _cmd_run(args)

        assert code == NOT_DONE, "a parked run is not a finished run"
        out = capsys.readouterr().out
        assert "waiting" in out.lower() or "parked" in out.lower()
        assert "watch" in out, "the output must say how to follow the run"

    async def test_a_budget_refusal_is_neither_a_wait_nor_a_finished_run(
        self, live_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``429`` gets its own exit code and names the budget.

        This is the newest failure mode in the API, and the one a CLI user is
        most likely to hit by accident: submit a big enough diff and the run is
        refused. It is not a transient error — retrying the same payload spends
        the same tokens to reach the same answer — so it must not share a code
        with "try again in a moment", and the message has to say which budget
        was hit and by how much.
        """
        import argparse

        from agentic_workflow.cli import _cmd_run

        args = argparse.Namespace(
            run_id=None,
            request_id=None,
            title="too expensive",
            description="",
            language="python",
            files=[],
            acceptance_criteria=[],
            constraints=[],
            metadata=[],
            auto_resolve=True,
            auto_decide="approve",
            max_gates=32,
            url="",
            token=None,
            timeout=90.0,
            watch=False,
            json=False,
        )
        tight = _settings(llm_token_budget_per_run=1)
        async with LiveServer(tight) as server:
            args.url = server.base_url
            code = await _cmd_run(args)

        assert code == THROTTLED
        err = capsys.readouterr().err
        assert "token" in err.lower()
        assert "budget" in err.lower()

    async def test_a_missing_token_is_bad_input_not_a_crash(
        self, live_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``401`` is a configuration mistake, reported as one.

        This is the failure a user hits first on a secured server, and the
        message has to name the fix. A traceback, or a bare "401", teaches
        nothing; the env var is the answer and it should be in the text.
        """
        import argparse

        from agentic_workflow.cli import _cmd_run

        secured = _settings(api_auth_enabled=True, api_auth_token="s3cret-value")
        args = argparse.Namespace(
            run_id=None,
            request_id=None,
            title="unauthorised",
            description="",
            language="python",
            files=[],
            acceptance_criteria=[],
            constraints=[],
            metadata=[],
            auto_resolve=True,
            auto_decide="approve",
            max_gates=32,
            url="",
            token=None,
            timeout=90.0,
            watch=False,
            json=False,
        )
        async with LiveServer(secured) as server:
            args.url = server.base_url
            code = await _cmd_run(args)

        assert code == BAD_INPUT
        err = capsys.readouterr().err
        assert "AWF_API_TOKEN" in err

    async def test_the_token_is_sent_as_a_header_never_in_the_url(
        self, live_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No credential ends up in a URL on this path.

        ``watch`` has to put its token in the query string — a WebSocket
        handshake from a browser cannot carry an ``Authorization`` header — and
        that is why its error messages strip the query string before printing.
        ``run`` is plain HTTP and has no such constraint, so a token in the URL
        here would be a credential in a proxy log for no reason at all. The
        assertion is on the printed output, since that is what ends up pasted
        into a bug report.
        """
        import argparse

        from agentic_workflow.cli import _cmd_run

        secured = _settings(api_auth_enabled=True, api_auth_token="s3cret-value")
        args = argparse.Namespace(
            run_id=None,
            request_id=None,
            title="authenticated",
            description="",
            language="python",
            files=[],
            acceptance_criteria=[],
            constraints=[],
            metadata=[],
            auto_resolve=True,
            auto_decide="approve",
            max_gates=32,
            url="",
            token="s3cret-value",
            timeout=90.0,
            watch=False,
            json=False,
        )
        async with LiveServer(secured) as server:
            args.url = server.base_url
            assert await _cmd_run(args) == 0

        captured = capsys.readouterr()
        assert "s3cret-value" not in captured.out
        assert "s3cret-value" not in captured.err


class TestTheSubmissionIsRetryable:
    """The reason this command exists instead of a ``curl`` incantation."""

    async def test_a_resubmission_returns_the_original_run(self, live_settings: Settings) -> None:
        """Submitting the same work twice is idempotent, not two runs.

        The classic way to create a duplicate is to press enter twice because
        the terminal did not echo anything back. The API already answers this
        with ``X-Content-Hash``, and the CLI sends it, so the second submission
        returns the first run. Without the header the second call is a
        ``409 run_already_exists`` — correct, but it means the retry path a user
        actually takes fails, and the fix is not discoverable.
        """
        import argparse

        from agentic_workflow.cli import _cmd_run

        def _args(url: str) -> argparse.Namespace:
            return argparse.Namespace(
                run_id="retry-me",
                request_id=None,
                title="same work",
                description="identical",
                language="python",
                files=[],
                acceptance_criteria=[],
                constraints=[],
                metadata=[],
                auto_resolve=True,
                auto_decide="approve",
                max_gates=32,
                url=url,
                token=None,
                timeout=90.0,
                watch=False,
                json=False,
            )

        async with LiveServer(live_settings) as server:
            assert await _cmd_run(_args(server.base_url)) == 0
            assert await _cmd_run(_args(server.base_url)) == 0

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(f"{server.base_url}/v1/runs/retry-me")

            assert response.status_code == 200
            detail = response.json()
            assert detail["status"] == "completed", "the replay must be the same run, not a new one"


def test_json_output_is_the_servers_payload_verbatim(
    live_settings: Settings, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """``--json`` prints the run detail unchanged, for a script to parse.

    Reformatting the server's answer would mean the CLI had to be updated every
    time a field is added, and would leave ``--json`` disagreeing with
    ``GET /v1/runs/{id}``. So the assertion is that the printed document and the
    document the server hands to any other client carry the *same fields* —
    nothing renamed, nothing dropped, nothing invented — which is the property
    that lets a pipeline treat the two as one answer.

    The values are deliberately not compared field by field. That is the
    server's contract to keep, not the CLI's to assert: this test is about
    whether the bytes were forwarded untouched, and a test that also pinned
    every value would fail for reasons that have nothing to do with forwarding.
    """
    import argparse
    import asyncio

    from agentic_workflow.cli import _cmd_run

    source = tmp_path / "checkout.py"
    source.write_text("def total():\n    return 0.1 + 0.2\n", encoding="utf-8")

    async def drive() -> int:
        async with LiveServer(live_settings) as server:
            args = argparse.Namespace(
                run_id=None,
                request_id=None,
                title="json please",
                description="",
                language="python",
                files=[str(source)],
                acceptance_criteria=[],
                constraints=[],
                metadata=[],
                auto_resolve=True,
                auto_decide="approve",
                max_gates=32,
                url=server.base_url,
                token=None,
                timeout=90.0,
                watch=False,
                json=True,
            )
            code = await _cmd_run(args)
            printed = json.loads(capsys.readouterr().out)

            import httpx

            async with httpx.AsyncClient() as client:
                fetched = await client.get(f"{server.base_url}/v1/runs/{printed['run_id']}")
            return code, printed, fetched.json()

    code, printed, fetched = asyncio.run(drive())

    assert code == 0
    assert set(printed) == set(fetched), "--json must carry the server's own fields"
    assert printed["run_id"] == "json-please", "the id has to come back to fetch or watch it"
    assert printed["usage"]["total_tokens"] > 0, "a completed run reports what it spent"
    assert printed["is_finished"] is True


def test_a_file_that_does_not_exist_is_reported_before_any_request(
    capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """A typo in ``--file`` is caught locally, with the path, and sends nothing.

    Reading the files happens before the request on purpose. A submission whose
    body cannot be built should not cost a request, and the message has to name
    the file: "no such file" with no path is a wild goose chase through a
    ``--file`` list.
    """
    import argparse

    from agentic_workflow.cli import _cmd_run

    args = argparse.Namespace(
        run_id=None,
        request_id=None,
        title="missing file",
        description="",
        language="python",
        files=[str(tmp_path / "not-here.py")],
        acceptance_criteria=[],
        constraints=[],
        metadata=[],
        auto_resolve=True,
        auto_decide="approve",
        max_gates=32,
        url="http://127.0.0.1:1",  # nothing is listening; nothing must be tried
        token=None,
        timeout=5.0,
        watch=False,
        json=False,
    )

    import asyncio

    assert asyncio.run(_cmd_run(args)) == BAD_INPUT
    err = capsys.readouterr().err
    assert "not-here.py" in err
