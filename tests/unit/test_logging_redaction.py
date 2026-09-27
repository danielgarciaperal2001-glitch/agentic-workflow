"""Tests for the log scrubber and its place in the processor chain.

``scrub`` is the package's last line of defence against a credential reaching
a log file. It shipped with no caller and no test, so two things were true at
once: nothing in the codebase relied on it, and nothing checked it. The first
is the risk — a redaction nobody calls redacts nothing.

These tests cover both halves, because either alone is insufficient. Fixing
only the walk-through bug would leave the function dead; wiring it in without
testing the walk would leave a redaction that fails on the one payload shape
most likely to carry a secret, a list of per-item results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
import structlog

from agentic_workflow.config import LogFormat, load_settings
from agentic_workflow.logging import (
    _redacting_processor,
    bind_context,
    configure_logging,
    current_context,
    get_logger,
    scrub,
)


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    """Put structlog back the way it was; ``configure_logging`` is global."""
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)
    logging.getLogger().handlers.clear()


class TestScrub:
    @pytest.mark.parametrize(
        "key",
        ["api_key", "apikey", "authorization", "auth_token", "password", "secret", "token"],
    )
    def test_every_default_key_is_redacted(self, key: str) -> None:
        assert scrub({key: "hunter2"})[key] == "***"

    def test_matching_ignores_case(self) -> None:
        """``API_KEY`` is how an environment-sourced payload spells it."""
        assert scrub({"API_KEY": "sk-real"})["API_KEY"] == "***"

    def test_the_llm_key_is_redacted_by_its_configured_name(self) -> None:
        assert scrub({"llm_api_key": "sk-real"})["llm_api_key"] == "***"

    def test_extra_keys_can_be_added(self) -> None:
        assert scrub({"session": "abc"}, redact_keys=frozenset({"session"}))["session"] == "***"

    def test_diagnostic_fields_survive_untouched(self) -> None:
        """Redaction that eats operational fields is its own outage. Only names
        that declare themselves are masked."""
        payload = {
            "run_id": "run_42",
            "node": "reviewer",
            "tokens": 1500,
            "latency_ms": 4.9,
            "attempt": 2,
        }
        assert scrub(payload) == payload

    def test_a_nested_mapping_is_walked(self) -> None:
        out = scrub({"config": {"db": {"password": "hunter2", "host": "localhost"}}})
        assert out["config"]["db"]["password"] == "***"
        assert out["config"]["db"]["host"] == "localhost"

    def test_a_secret_inside_a_list_is_redacted(self) -> None:
        """The bug this file exists for.

        Lists of files, findings or per-node results are an ordinary log
        payload, and the original walk only descended into mappings — so every
        credential inside a list reached the log verbatim while redaction
        appeared to be working.
        """
        out = scrub({"files": [{"path": "a.py", "api_key": "sk-in-a-list"}]})
        assert out["files"][0]["api_key"] == "***"
        assert out["files"][0]["path"] == "a.py"

    def test_a_list_of_lists_is_walked_to_the_bottom(self) -> None:
        out = scrub({"matrix": [[{"secret": "s"}], [{"secret": "t"}]]})
        assert out["matrix"][0][0]["secret"] == "***"
        assert out["matrix"][1][0]["secret"] == "***"

    def test_tuples_keep_their_type(self) -> None:
        """Coercing to a list would change the shape a caller logs and later
        re-reads, which is a subtle break rather than a visible one."""
        out = scrub({"tags": ("api_key",)})
        assert isinstance(out["tags"], tuple)

    def test_the_input_is_not_mutated(self) -> None:
        """A scrubber that consumed its argument would blank the caller's own
        data on the way past, and the caller never sees why."""
        payload = {"password": "hunter2"}
        scrub(payload)
        assert payload["password"] == "hunter2"

    def test_a_non_mapping_value_is_left_alone(self) -> None:
        assert scrub({"count": 3, "flag": True, "none": None}) == {
            "count": 3,
            "flag": True,
            "none": None,
        }

    def test_the_event_field_is_never_redacted(self) -> None:
        """structlog's own keys. Masking one would corrupt the record, and
        ``event`` is not a credential no matter what it contains."""
        out = _redacting_processor(None, "logger", {"event": "api.ready", "token": "t"})
        assert out["event"] == "api.ready"
        assert out["token"] == "***"


class TestProcessorWiring:
    """The scrubber has to be in the chain, not merely available."""

    def test_the_processor_redacts_an_event_dictionary(self) -> None:
        assert _redacting_processor(None, "n", {"api_key": "sk", "x": 1}) == {
            "api_key": "***",
            "x": 1,
        }

    def test_a_real_log_line_carries_no_secret(self, capfd: pytest.CaptureFixture[str]) -> None:
        """The end-to-end assertion.

        Every other test here calls ``scrub`` directly and would still pass if
        the processor were never installed — which is exactly the state the
        package was in for its whole life. This renders an actual log record
        through the configured pipeline and reads the bytes the process would
        write.
        """
        configure_logging(load_settings(), force=True)
        get_logger("test").info(
            "llm.configured",
            llm_api_key="sk-must-not-appear",
            nested={"password": "hunter2"},
            run_id="run_42",
        )
        rendered = capfd.readouterr().err
        assert "sk-must-not-appear" not in rendered
        assert "hunter2" not in rendered
        assert "run_42" in rendered, "diagnostics must survive redaction"

    def test_json_output_also_carries_no_secret(self, capfd: pytest.CaptureFixture[str]) -> None:
        """The JSON renderer is what ships to a log aggregator, and it is a
        different renderer from the console one — passing the console test is
        not evidence about this one."""
        settings = load_settings().model_copy(update={"log_format": LogFormat.JSON})
        configure_logging(settings, force=True)
        get_logger("test").info("llm.configured", api_key="sk-must-not-appear")
        rendered = capfd.readouterr().err
        assert "sk-must-not-appear" not in rendered
        assert json.loads(rendered.strip())["api_key"] == "***"

    def test_logs_never_reach_stdout(self, capfd: pytest.CaptureFixture[str]) -> None:
        """The CLI's *result* goes to stdout so `awf eval --json | jq` works;
        a log line interleaved with it makes that output unparseable. Worth
        pinning, because the fix for any stdout problem is always "log it
        too", and the cost lands on the caller.
        """
        configure_logging(load_settings(), force=True)
        get_logger("test").info("cli.invocado")
        captured = capfd.readouterr()
        assert captured.out == ""
        assert "cli.invocado" in captured.err


class TestContextBinding:
    def test_the_bound_context_is_visible_and_unwound(self) -> None:
        """Context is how a log line is traced back to a run, and it leaks into
        the same event dictionary the scrubber walks."""
        with bind_context(run_id="run_42", node="tester"):
            assert current_context()["run_id"] == "run_42"
        assert "run_id" not in current_context()

    def test_a_none_value_is_not_bound(self) -> None:
        """Binding ``None`` would render ``run_id=None`` on every event, which
        reads as a real identifier."""
        with bind_context(run_id=None, node="tester"):
            assert "run_id" not in current_context()

    def test_the_context_is_unwound_even_when_the_body_raises(self) -> None:
        """Without the finally, one exception would leave every later log line
        in the suite carrying that run's identifiers."""
        with pytest.raises(RuntimeError), bind_context(run_id="run_42"):
            raise RuntimeError("boom")
        assert current_context() == {}

    def test_nested_contexts_unwind_in_order(self) -> None:
        with bind_context(run_id="outer"):
            with bind_context(run_id="inner"):
                assert current_context()["run_id"] == "inner"
            assert current_context()["run_id"] == "outer"
        assert current_context() == {}
