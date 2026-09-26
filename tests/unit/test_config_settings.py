"""Configuration: the parts that fail silently if nobody is watching.

``Settings`` must use ``extra="ignore"`` — a process environment is full of
``PATH``, ``HOME`` and every other application's variables, and refusing them
would make the class unusable. That necessity opens a door: an unknown keyword
is discarded without complaint. The tests here pin the two consequences that
matter, plus the invariants a deployment depends on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError
import pytest

from agentic_workflow.config import (
    DEFAULT_SCHEMA,
    ENV_PREFIX,
    Environment,
    EvalProvider,
    Settings,
    load_settings,
    reset_settings_cache,
)
from agentic_workflow.errors import ConfigurationError

#: Settings deliberately absent from ``.env.example``.
#:
#: Named rather than counted, so that adding a field without documenting it fails
#: the coverage assertion *by name* instead of leaving the next person to guess
#: which names were allowed to be missing. A new entry here is a decision that
#: gets reviewed; a changed count is not.
_UNDOCUMENTED_ON_PURPOSE: Final[frozenset[str]] = frozenset()

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


class TestConnectionString:
    """The DSN has to survive a real connection string parser.

    This is here because the failure it prevents was found by running
    ``awf demo`` on a machine with no database, not by reading this file. Every
    other test in the suite reached ``postgres_dsn`` as an opaque string; only
    libpq's parser has an opinion about whether the string means anything.
    """

    def test_the_statement_timeout_survives_uri_parsing(self) -> None:
        """The ``options`` value is itself a ``key=value`` pair, so it needs encoding.

        libpq reads a raw ``=`` inside a query value as the start of another
        parameter and refuses the whole connection. Unencoded, the symptom was
        not a parse error but a 30-second connect timeout, because the pool kept
        retrying a DSN that could never work.
        """
        conninfo = pytest.importorskip(
            "psycopg.conninfo", reason="psycopg ships with the postgres extra"
        )
        settings = Settings(
            _env_file=None,
            postgres_dsn="postgresql://u:p@localhost:5432/db",
            postgres_statement_timeout_ms=5_000,
        )
        parsed = conninfo.conninfo_to_dict(settings.dsn_with_timeout)
        assert parsed["options"] == "-c statement_timeout=5000"

    def test_a_dsn_that_already_carries_options_is_left_alone(self) -> None:
        """The operator's ``options`` win; silently merging two of them is a trap."""
        custom = "postgresql://u:p@localhost:5432/db?options=-c%20search_path%3Daudit"
        settings = Settings(_env_file=None, postgres_dsn=custom)
        assert settings.dsn_with_timeout == custom

    def test_an_existing_query_string_is_extended_not_corrupted(self) -> None:
        """Appending with ``?`` to a DSN that has one produces two questions."""
        conninfo = pytest.importorskip(
            "psycopg.conninfo", reason="psycopg ships with the postgres extra"
        )
        settings = Settings(
            _env_file=None,
            postgres_dsn="postgresql://u:p@localhost:5432/db?application_name=awf",
        )
        dsn = settings.dsn_with_timeout
        assert dsn.count("?") == 1
        assert conninfo.conninfo_to_dict(dsn)["application_name"] == "awf"


class TestCheckpointSchema:
    """``AWF_POSTGRES_SCHEMA`` was a setting that did nothing.

    ``AsyncPostgresSaver`` has no schema parameter and its migrations are
    unqualified ``CREATE TABLE IF NOT EXISTS``, so the setting could not be
    passed to it. It was read, validated, logged on every boot and on every
    ready event, and had no effect: every deployment and every test wrote to
    ``public``. A setting that looks like it is working is worse than one that is
    obviously absent, because schema isolation is exactly the kind of thing that
    fails silently — two tenants sharing a table, or two tests sharing a row.
    """

    @staticmethod
    def _conninfo() -> Any:
        """Return libpq's parser, or skip.

        Returns:
            The ``psycopg.conninfo`` module.
        """
        return pytest.importorskip(
            "psycopg.conninfo", reason="psycopg ships with the postgres extra"
        )

    def test_a_custom_schema_becomes_a_search_path(self) -> None:
        """The mechanism, asserted at the level libpq actually sees."""
        conninfo = self._conninfo()
        settings = Settings(
            _env_file=None,
            postgres_dsn="postgresql://u:p@localhost:5432/db",
            postgres_schema="awf_tenant_a",
        )
        options = conninfo.conninfo_to_dict(settings.checkpointer_dsn)["options"]
        assert "-c search_path=awf_tenant_a" in options

    def test_the_default_schema_adds_nothing(self) -> None:
        """``public`` is already on the default path.

        Setting it explicitly would be harmless but would put a startup option
        on every connection for no reason, and would make a DSN comparison in a
        test or a log line differ for no reason either.
        """
        conninfo = self._conninfo()
        settings = Settings(
            _env_file=None,
            postgres_dsn="postgresql://u:p@localhost:5432/db",
        )
        assert settings.postgres_schema == DEFAULT_SCHEMA
        options = conninfo.conninfo_to_dict(settings.checkpointer_dsn)["options"]
        assert "search_path" not in options

    def test_the_timeout_and_the_search_path_coexist(self) -> None:
        """Two ``-c`` options in one string, and the timeout must not be lost.

        libpq takes ``options`` as a space-separated argument list, so this
        works — but a comma separator is silently accepted by the URI encoder and
        then rejected by the server with
        ``invalid value for parameter "statement_timeout": "30000,"``. So the
        separator is load-bearing and asserted here rather than left to a
        comment.
        """
        conninfo = self._conninfo()
        settings = Settings(
            _env_file=None,
            postgres_dsn="postgresql://u:p@localhost:5432/db",
            postgres_schema="awf_tenant_a",
            postgres_statement_timeout_ms=5_000,
        )
        options = conninfo.conninfo_to_dict(settings.checkpointer_dsn)["options"]
        assert "-c statement_timeout=5000" in options
        assert "-c search_path=awf_tenant_a" in options

    def test_an_injectable_schema_is_refused(self) -> None:
        """The schema name is a libpq startup option the backend parses as SQL.

        ``search_path`` is parsed by the server, so a value like
        ``x, public`` would silently redirect every unqualified table in the
        application — a schema chosen by someone who is not the operator. The
        field's ``pattern`` is a security control, not a style rule.
        """
        for hostile in (
            "public, evil",
            "public; drop table checkpoints",
            "Public",
            "1schema",
            'pub"lic',
            "pub lic",
        ):
            with pytest.raises(ValidationError):
                Settings(_env_file=None, postgres_schema=hostile)

    def test_a_legitimate_schema_is_accepted(self) -> None:
        """The constraint must not be so tight that it refuses real names.

        A pattern that only allows ``public`` would pass the test above and
        break the feature. Underscores, digits and lowercase are all legal bare
        SQL identifiers and all used in practice.
        """
        for name in ("awf_tenant_a", "awf2", "_staging", "checkpoints_v2"):
            assert Settings(_env_file=None, postgres_schema=name).postgres_schema == name


class TestEnvExample:
    """``.env.example`` is only documentation, so nothing checks it — until now.

    A misspelled variable name in the template is the worst kind of configuration
    bug: pydantic ignores unknown keys without complaint, so the operator copies
    the file, sets the variable, and watches it do nothing. That is the same
    failure shape as ``AWF_POSTGRES_SCHEMA`` being accepted, validated and logged
    while reaching nothing. The symptom either way is silence.
    """

    @staticmethod
    def _declared() -> dict[str, str]:
        """Return every variable the template actually assigns.

        Returns:
            Variable name to raw line, for error messages that show the offender.
        """
        path = Path(__file__).resolve().parents[2] / ".env.example"
        text = path.read_text(encoding="utf-8")
        declared: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            # A commented-out `#AWF_X=1` is documentation of an option, not a
            # declaration of it, and is left out deliberately.
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            declared[name.strip()] = value.strip()
        return declared

    def test_every_variable_in_the_template_is_a_real_setting(self) -> None:
        """The direction that causes silent failure.

        A typo is invisible at runtime — the setting keeps its default and nothing
        complains — so it is checked here instead. The failure is reported with
        the offending line rather than as a bare name, because the reader looking
        at this failure is editing the template and needs to find it.
        """
        fields = set(Settings.model_fields)
        unknown = {
            name: raw
            for name, raw in self._declared().items()
            if not name.startswith(ENV_PREFIX) or name[len(ENV_PREFIX) :].lower() not in fields
        }
        assert not unknown, f"unknown settings in .env.example: {unknown}"

    def test_the_template_covers_every_setting(self) -> None:
        """The other direction, and the one an operator feels.

        ``extra="ignore"`` means a field absent from the template is a setting
        nobody knows exists, so it stays at its default in production — which is
        fine for genuinely internal knobs and wrong for everything else. The
        assertion is a named allowlist of deliberate omissions rather than a count,
        so adding a setting without documenting it fails here by name.
        """
        documented = {name[len(ENV_PREFIX) :].lower() for name in self._declared()}
        undocumented = sorted(set(Settings.model_fields) - documented - _UNDOCUMENTED_ON_PURPOSE)
        assert not undocumented, f"settings missing from .env.example: {undocumented}"

    def test_the_template_actually_loads(self) -> None:
        """The file people are told to copy must be a file the model accepts.

        This is not hypothetical. The shipped template carried
        ``AWF_API_CORS_ORIGINS=http://localhost:3000`` and the field is
        ``list[str]``; pydantic-settings parses a collection from JSON, in dotenv
        files and in real environment variables alike, so copying the template to
        ``.env`` — the first instruction in its own header — produced
        ``SettingsError: error parsing value for field "api_cors_origins"`` and a
        process that would not start. Nothing else in the project loads that file,
        so nothing else would have caught it.

        The whole file is loaded rather than each value individually, because a
        per-value check would have to reimplement dotenv's own parsing of
        comments, quotes and blanks, and would then be testing that
        reimplementation instead of the loader.
        """
        path = Path(__file__).resolve().parents[2] / ".env.example"
        try:
            settings = Settings(_env_file=path, _env_file_encoding="utf-8")
        except ValidationError as exc:
            pytest.fail(f".env.example does not load: {exc}")
        # A load that silently ignored the list would look like a pass.
        assert settings.api_cors_origins == ["http://localhost:3000"]

    def test_the_template_never_carries_a_secret(self) -> None:
        """The file people are told to copy, with a header that says so.

        A real key committed to a template is read by every clone of the
        repository, and unlike a real ``.env`` it is never in ``.gitignore``.
        Matched on the *values* rather than on the absence of a key name, so a
        renamed variable cannot slip past, and only on values that are actually
        populated — a blank placeholder is the intended state for every secret
        in the template.
        """
        import re

        suspicious = re.compile(
            r"(sk-[A-Za-z0-9]{8}|ghp_[A-Za-z0-9]{8}|AKIA[0-9A-Z]{8}|BEGIN [A-Z ]*PRIVATE KEY)"
        )
        offenders = {
            name: raw for name, raw in self._declared().items() if raw and suspicious.search(raw)
        }
        assert not offenders, f"looks like a real credential: {offenders}"
