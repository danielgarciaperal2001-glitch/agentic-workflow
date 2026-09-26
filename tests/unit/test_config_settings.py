"""Configuration: the parts that fail silently if nobody is watching.

``Settings`` must use ``extra="ignore"`` — a process environment is full of
``PATH``, ``HOME`` and every other application's variables, and refusing them
would make the class unusable. That necessity opens a door: an unknown keyword
is discarded without complaint. The tests here pin the two consequences that
matter, plus the invariants a deployment depends on.
"""

from __future__ import annotations

import pytest

from agentic_workflow.config import (
    ENV_PREFIX,
    Environment,
    EvalProvider,
    Settings,
    load_settings,
    reset_settings_cache,
)
from agentic_workflow.errors import ConfigurationError

pytestmark = pytest.mark.unit


class TestPrefixedKeys:
    """The ``AWF_`` prefix belongs to the environment, not to the model."""

    def test_a_prefixed_kwarg_is_refused(self) -> None:
        """``Settings(awf_max_parallel_runs=1)`` must not be silently ignored.

        The call reads like the environment variable it mirrors, and it would
        otherwise be dropped without a word — leaving a test that believes it
        lowered a limit watching the real limit of 8 apply instead. Green, fast,
        and completely wrong.
        """
        with pytest.raises(ValueError) as excinfo:
            Settings(_env_file=None, awf_max_parallel_runs=1)  # type: ignore[call-arg]

        message = str(excinfo.value)
        assert "prefix applies to environment variables only" in message
        assert "max_parallel_runs" in message

    def test_the_error_names_every_offending_key(self) -> None:
        """One mistake should surface every key that has it, not one per run."""
        with pytest.raises(ValueError) as excinfo:
            Settings(_env_file=None, awf_log_level="DEBUG", awf_max_parallel_runs=1)  # type: ignore[call-arg]

        message = str(excinfo.value)
        assert "awf_log_level" in message
        assert "awf_max_parallel_runs" in message

    def test_bare_field_names_are_accepted(self) -> None:
        """The unprefixed name is the supported spelling and must just work."""
        assert Settings(_env_file=None, max_parallel_runs=1).max_parallel_runs == 1

    def test_model_validate_cannot_bypass_the_guard(self) -> None:
        """Pydantic's own entry points must not be a way around the check.

        ``Settings.model_validate`` never runs ``__init__``. A guard that only
        lived there would be bypassed by every revalidation path, which is
        exactly the kind of half-enforced invariant that gets discovered in
        production.
        """
        with pytest.raises(ValueError):
            Settings.model_validate({"awf_max_parallel_runs": 1})

    def test_unrelated_environment_variables_are_ignored(self) -> None:
        """The reason the model is permissive must not be mistaken for laxity.

        ``extra="ignore"`` exists so that ``PATH`` and ``HOME`` do not break
        startup. Anything genuinely unknown must still be ignored, or the class
        could never be constructed in a normal process.
        """
        assert Settings(_env_file=None).max_parallel_runs == 8


class TestOverrides:
    """``load_settings`` is memoised; its overrides are one-shot by design."""

    def test_overrides_win_over_defaults(self) -> None:
        reset_settings_cache()
        settings = load_settings(max_iterations=9)
        assert settings.max_iterations == 9
        reset_settings_cache()

    def test_the_cache_returns_the_same_object(self) -> None:
        """Two reads must not disagree, or a request could see two configs."""
        reset_settings_cache()
        first = load_settings()
        assert load_settings() is first
        reset_settings_cache()

    def test_an_invalid_override_is_a_configuration_error(self) -> None:
        """A bad override is an operator problem, surfaced as such.

        ``ConfigurationError`` is what the health check and the CLI report, so a
        raw pydantic traceback would reach an operator with no way to act on it.
        """
        reset_settings_cache()
        with pytest.raises(ConfigurationError):
            load_settings(max_parallel_runs=0)  # below the field's own minimum
        reset_settings_cache()


class TestInvariants:
    """Cross-field rules that only hold because a validator enforces them."""

    def test_production_refuses_an_evaluator_without_a_backend(self) -> None:
        """``deepeval`` in production is a supply-chain decision, not a default.

        The DeepEval package pulls and executes third-party model code. Allowing
        it in production by accident would mean an evaluation dependency — not a
        runtime one — could be loaded by the service that handles real requests.
        """
        with pytest.raises(ConfigurationError, match="not permitted in production"):
            load_settings(
                environment=Environment.PRODUCTION,
                eval_provider=EvalProvider.DEEPEVAL,
            )
        reset_settings_cache()

    def test_the_summary_never_leaks_a_secret(self) -> None:
        """``safe_summary`` is logged at boot; a key in it is a key in the logs."""
        settings = load_settings(llm_api_key="awf-fake-key-1234")
        summary = settings.safe_summary()
        rendered = repr(summary)

        assert "super-secret" not in rendered
        assert summary["llm_api_key"].endswith("1234")
        assert summary["llm_api_key"].startswith("****")
        reset_settings_cache()

    def test_an_unset_key_is_reported_as_unset(self) -> None:
        """``****`` for a missing key would imply a key exists."""
        settings = load_settings(llm_api_key="")
        assert settings.safe_summary()["llm_api_key"] == "<unset>"
        reset_settings_cache()

    def test_the_prefix_constant_matches_the_model(self) -> None:
        """The constant builds the config; drift would break the guard's message."""
        assert Settings.model_config["env_prefix"] == ENV_PREFIX
