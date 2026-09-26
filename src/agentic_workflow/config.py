"""Typed, layered application configuration.

The whole engine is configured through a single frozen
:class:`Settings` object. It is built once per process via
:func:`load_settings` (which caches aggressively) and threaded explicitly
through the graph, the persistence layer and the API.

Design rules
------------
* **No global mutable state.** Everything is passed by reference; the object is
  frozen so a node cannot mutate the budget of its peers.
* **Fail fast.** Validators reject impossible combinations at import time
  instead of letting them surface as a broken run an hour later.
* **Twelve-factor.** Every field is overridable with an ``AWF_``-prefixed
  environment variable, and ``.env`` is loaded for local development.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any, Literal, Self
from urllib.parse import quote, urlencode

from pydantic import Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_workflow.errors import ConfigurationError

_UNSET = object()


class Environment(StrEnum):
    """Deployment environment. Drives log format, CORS strictness and metrics."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    """Structured-log renderer."""

    CONSOLE = "console"
    JSON = "json"


class LLMProvider(StrEnum):
    """Supported LLM backends.

    ``ECHO`` is a first-class, dependency-free provider that returns
    deterministic, schema-shaped answers derived from the prompt. It keeps the
    entire system (graph, HITL, persistence, evals) runnable in CI without
    credentials or network access.
    """

    ECHO = "echo"
    OPENAI = "openai"


class EvalProvider(StrEnum):
    """Evaluation backend for the automated quality suite."""

    NATIVE = "native"
    RAGAS = "ragas"
    DEEPEVAL = "deepeval"


def _split_origins(value: str) -> list[str]:
    """Parse a comma-separated CORS origin list, tolerating whitespace."""
    return [origin.strip() for origin in value.split(",") if origin.strip()]


#: Environment-variable prefix. Applies to variables only — never to field names.
ENV_PREFIX = "AWF_"

#: The schema used when none is configured, and the one PostgreSQL already
#: searches by default. Naming it means "unset" is not spelled ``public`` in
#: three separate comparisons that could drift.
DEFAULT_SCHEMA = "public"


def _reject_prefixed_kwargs(data: dict[str, Any]) -> None:
    """Fail loudly on ``awf_``-prefixed configuration keys.

    ``env_prefix="AWF_"`` applies to *environment variables* only. It does not
    create aliases, so ``Settings(awf_max_parallel_runs=1)`` looks like a
    perfectly good call — and is then discarded without a word, because the model
    has to use ``extra="ignore"`` to survive an environment full of unrelated
    variables such as ``PATH`` and ``HOME``.

    That silence is the worst kind of configuration bug. A test that writes
    ``awf_max_parallel_runs=1`` and watches the real limit of 8 apply instead is
    green, fast, and completely wrong; the mistake only surfaces under load.
    Turning it into an error that names the right argument costs one comparison.

    Args:
        data: Candidate field names and values.

    Raises:
        ValueError: If an ``awf_``-prefixed name is used as a field name.
    """
    # No field is itself named `awf_*` — the prefix belongs to the environment,
    # not to the model — so any prefixed key is a mistake, full stop. Checking
    # that the *unprefixed* form is a real field would be exactly backwards: it
    # would reject the typo and accept the mistake.
    prefixed = {
        key
        for key in data
        if isinstance(key, str)
        and key.lower().startswith(ENV_PREFIX.lower())
        and key not in Settings.model_fields
    }
    if not prefixed:
        return
    suggestions = ", ".join(sorted(key[len(ENV_PREFIX) :] for key in prefixed))
    raise ValueError(
        f"Settings does not accept {ENV_PREFIX}-prefixed keyword arguments: "
        f"{sorted(prefixed)}. The prefix applies to environment variables only; "
        f"use the field name(s) directly: {suggestions}."
    )


class Settings(BaseSettings):
    """Immutable runtime configuration.

    Two ways in, and they do not use the same names. Environment variables carry
    the ``AWF_`` prefix (``AWF_LOG_LEVEL=DEBUG``); constructor arguments and
    :func:`load_settings` overrides use the bare field name
    (``log_level="DEBUG"``). Passing a prefixed name to the constructor is
    rejected rather than ignored — see :func:`_reject_prefixed_kwargs`.

    Example
    -------
    >>> settings = Settings(_env_file=None, log_level="DEBUG")
    >>> settings.log_level
    'DEBUG'
    >>> settings.is_production
    False
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    # ----------------------------------------------------------------- app #
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment name.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Minimum severity emitted by the structured logger.",
    )
    log_format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        description="Renderer for structured logs (`json` for aggregators).",
    )
    log_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of DEBUG records sampled in non-development environments.",
    )
    service_name: str = Field(
        default="agentic-workflow",
        min_length=1,
        max_length=64,
        description="Logical service name attached to every log record.",
    )

    # --------------------------------------------------------------- graph #
    graph_name: str = Field(
        default="pr_review_workflow",
        min_length=1,
        max_length=128,
        description="Checkpoint namespace. Bump this to invalidate old checkpoints.",
    )
    max_iterations: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Hard cap on the agent feedback loop (prevents infinite cycles).",
    )
    node_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        le=3600.0,
        description="Per-node wall-clock budget enforced by the run manager.",
    )
    run_timeout_seconds: float = Field(
        default=900.0,
        ge=0.0,
        description="Per-run wall-clock budget. `0` disables the deadline.",
    )
    max_parallel_runs: int = Field(
        default=8,
        ge=1,
        le=1024,
        description="In-process concurrency ceiling for the embedded executor.",
    )
    state_retention_days: int = Field(
        default=30,
        ge=0,
        description="TTL applied to checkpoints and audit rows by the janitor.",
    )

    # ----------------------------------------------------------------- llm #
    llm_provider: LLMProvider = Field(
        default=LLMProvider.ECHO,
        description="LLM backend. `echo` is deterministic and offline.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="Model identifier passed to the provider.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        min_length=1,
        description="Base URL of any OpenAI-compatible endpoint.",
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="Provider credential. Prefer a secret manager in production.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Keep at 0 for reproducible runs.",
    )
    llm_max_tokens: int = Field(
        default=2048,
        ge=64,
        le=131072,
        description="Upper bound on tokens generated per call.",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=600.0,
        description="Per-request provider deadline.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry attempts with exponential backoff on transient errors.",
    )
    llm_max_concurrency: int = Field(
        default=8,
        ge=1,
        le=512,
        description="Client-side semaphore size guarding provider rate limits.",
    )

    # ---------------------------------------------------------- postgres #
    postgres_enabled: bool = Field(
        default=False,
        description="Use the durable PostgreSQL checkpointer instead of memory.",
    )
    postgres_dsn: str = Field(
        default="postgresql://agentic:agentic@localhost:5432/agentic",
        min_length=1,
        description="libpq connection string for the checkpoint store.",
    )
    postgres_pool_min_size: int = Field(default=1, ge=0, le=256)
    postgres_pool_max_size: int = Field(default=10, ge=1, le=512)
    postgres_pool_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    postgres_statement_timeout_ms: int = Field(
        default=30_000,
        ge=100,
        le=600_000,
        description="Server-side statement timeout guarding runaway queries.",
    )
    postgres_schema: str = Field(
        default=DEFAULT_SCHEMA,
        min_length=1,
        max_length=63,
        pattern=r"^[a-z_][a-z0-9_]*$",
        description=(
            "Schema hosting the LangGraph checkpoint tables. Applied as a "
            "`search_path` on every pooled connection, so the checkpointer's "
            "unqualified DDL lands here. Must be a bare SQL identifier: the "
            "backend parses `search_path` as SQL."
        ),
    )
    recovery_max_runs: int = Field(
        default=1_000,
        ge=1,
        le=100_000,
        description=(
            "Upper bound on the runs rehydrated into the registry on boot. A "
            "restart must not stall on a large checkpoint table, and the "
            "checkpointer remains the authoritative read either way."
        ),
    )
    postgres_auto_setup: bool = Field(
        default=True,
        description="Create checkpoint tables on boot (idempotent). Disable in prod.",
    )
    postgres_sslmode: str = Field(default="prefer", min_length=1)

    # ---------------------------------------------------------------- api #
    # Binding to every interface is the correct default for a container: the
    # service is not reachable from outside the compose network otherwise.
    api_host: str = Field(default="0.0.0.0", min_length=1)  # noqa: S104
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_root_path: str = Field(
        default="",
        description="ASGI root path when served behind a reverse proxy.",
    )
    api_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed browser origins for the control-plane UI.",
    )
    api_auth_enabled: bool = Field(
        default=False,
        description="Require a bearer token on every control-plane route.",
    )
    api_auth_token: SecretStr | None = Field(
        default=None,
        description="Bearer token accepted when `api_auth_enabled` is true.",
    )
    api_rate_limit_per_minute: int = Field(default=120, ge=0)
    api_embedded_worker: bool = Field(
        default=True,
        description="Execute runs in-process. Disable to scale the API separately.",
    )
    api_docs_enabled: bool = Field(
        default=True,
        description="Expose OpenAPI docs. Force off in production.",
    )

    # ---------------------------------------------------------------- ws  #
    ws_heartbeat_seconds: float = Field(default=20.0, gt=0.0, le=300.0)
    ws_send_timeout_seconds: float = Field(default=10.0, gt=0.0, le=120.0)
    ws_max_connections_per_run: int = Field(default=16, ge=1, le=1024)

    # -------------------------------------------------------------- hitl  #
    hitl_enabled: bool = Field(
        default=True,
        description="Master switch for every human gate. Disable for batch runs.",
    )
    hitl_default_timeout_seconds: float = Field(
        default=86_400.0,
        gt=0.0,
        description="How long a pending approval stays actionable before expiring.",
    )
    hitl_require_approval_before_apply: bool = Field(
        default=True,
        description="Gate the `apply_patch` node behind a human decision.",
    )
    hitl_escalation_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Agent confidence below this forces escalation to a human.",
    )
    hitl_allow_edit: bool = Field(
        default=True,
        description="Let humans rewrite agent output before resuming.",
    )
    hitl_allow_reject: bool = Field(default=True)
    hitl_require_signature: bool = Field(
        default=True,
        description="Sign decisions and persist them in the immutable audit log.",
    )

    # -------------------------------------------------------------- eval  #
    eval_provider: EvalProvider = Field(default=EvalProvider.NATIVE)
    eval_judge_model: str = Field(default="gpt-4o-mini", min_length=1)
    eval_dataset_path: str = Field(
        default="evals/datasets/pr_review_golden.jsonl",
        min_length=1,
    )
    eval_faithfulness_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_relevancy_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    eval_answer_recall_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    eval_context_precision_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    eval_fail_under_threshold: bool = Field(
        default=True,
        description="Fail the evaluation suite when a metric drops below its floor.",
    )

    # ------------------------------------------------------ integrations  #
    langsmith_tracing: bool = Field(default=False)
    langsmith_project: str = Field(default="agentic-workflow", min_length=1)
    langsmith_api_key: SecretStr | None = Field(default=None)

    # ---------------------------------------------------------- validators #
    @model_validator(mode="after")
    def _validate_cross_field(self) -> Self:
        """Reject configurations that cannot work."""
        if self.postgres_pool_max_size < self.postgres_pool_min_size:
            raise ValueError(
                "postgres_pool_max_size must be >= postgres_pool_min_size "
                f"(got {self.postgres_pool_max_size} < {self.postgres_pool_min_size})"
            )
        if self.llm_provider is LLMProvider.OPENAI and not self.llm_api_key:
            raise ValueError(
                "llm_api_key is required when llm_provider='openai'. "
                "Set AWF_LLM_API_KEY or switch to the offline `echo` provider."
            )
        if self.api_auth_enabled and not self.api_auth_token:
            raise ValueError(
                "api_auth_token is required when api_auth_enabled=true. "
                "Refusing to start an unauthenticated control plane."
            )
        if self.is_production and self.api_docs_enabled:
            # Documentation is a reconnaissance surface; keep it off by default.
            object.__setattr__(self, "api_docs_enabled", False)
        if (
            self.eval_provider is EvalProvider.DEEPEVAL
            and self.environment is Environment.PRODUCTION
        ):
            raise ValueError("eval_provider='deepeval' is not permitted in production")
        return self

    @model_validator(mode="before")
    @classmethod
    def _parse_cors_origins(cls, data: Any) -> Any:
        """Accept a comma-separated string for ``api_cors_origins``."""
        if isinstance(data, dict):
            raw = data.get("api_cors_origins", _UNSET)
            if isinstance(raw, str):
                data = {**data, "api_cors_origins": _split_origins(raw)}
        return data

    # ------------------------------------------------------ computed props #
    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        """``True`` when running with production hardening applied."""
        return self.environment is Environment.PRODUCTION

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_development(self) -> bool:
        """``True`` for local development."""
        return self.environment is Environment.DEVELOPMENT

    @computed_field  # type: ignore[prop-decorator]
    @property
    def use_durable_checkpointer(self) -> bool:
        """Whether a durable (PostgreSQL) checkpointer must be used."""
        return self.postgres_enabled

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn_with_timeout(self) -> str:
        """Connection string with the statement timeout applied.

        The ``options`` parameter is percent-encoded, and it has to be. Its
        value is itself a ``key=value`` pair (``-c statement_timeout=5000``), and
        a raw ``=`` inside a URI query value is not a legal query value —
        libpq reads the second ``=`` as the start of another parameter and
        refuses the whole connection with ``extra key/value separator "=" in URI
        query parameter: "options"``. Quote and the space becomes ``%20``, which
        libpq percent-decodes back to exactly the string the server expects.

        The failure this fixes was invisible until something actually tried to
        connect: the DSN looked correct, the pool retried, and the symptom was
        a 30-second connect timeout rather than a parse error naming the cause.
        """
        return self._dsn_with_options(f"-c statement_timeout={self.postgres_statement_timeout_ms}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checkpointer_dsn(self) -> str:
        """Connection string for the checkpointer's pool.

        Adds a ``search_path`` when the schema is not ``public``, which is how
        :attr:`postgres_schema` is actually honoured.

        ``AsyncPostgresSaver`` has no schema parameter, and its migrations are
        unqualified ``CREATE TABLE IF NOT EXISTS``. So the setting *was* read,
        validated, logged on every boot and on every ready event — and had no
        effect whatsoever: every deployment and every test wrote to ``public``.
        A setting that looks like it is working is worse than one that is
        obviously absent, because multi-tenant isolation and test isolation
        both depend on it and neither would show a failure.

        The schema name reaches PostgreSQL as a libpq startup option, whose
        value the backend parses as SQL. The field's ``pattern`` is therefore a
        security control and not a style rule: it admits only what a bare
        unquoted SQL identifier may contain, so a value like ``x, public`` —
        which would silently redirect the search path to somewhere else — is
        rejected at configuration time.
        """
        options = [f"-c statement_timeout={self.postgres_statement_timeout_ms}"]
        if self.postgres_schema != DEFAULT_SCHEMA:
            options.append(f"-c search_path={self.postgres_schema}")
        return self._dsn_with_options(" ".join(options))

    def _dsn_with_options(self, options: str) -> str:
        """Append libpq startup *options* to the configured DSN.

        Args:
            options: The ``-c key=value`` string libpq should apply.

        Returns:
            The DSN, or the caller's own unchanged when they already set
            ``options``. A caller who supplied their own is overriding the
            defaults on purpose, and silently merging a second ``options=``
            would be a DSN libpq itself rejects.
        """
        dsn = self.postgres_dsn.rstrip(" ")
        if "options=" in dsn:
            return dsn
        encoded = urlencode({"options": options}, quote_via=quote)
        sep = "&" if "?" in dsn else "?"
        return f"{dsn}{sep}{encoded}"

    # ------------------------------------------------------------- guards #
    def __init__(self, **kwargs: Any) -> None:
        """Validate keyword arguments before pydantic-settings discards them.

        ``BaseSettings.__init__`` merges its sources and drops keys that match no
        field *before* the model sees them, so a ``before`` validator alone would
        never observe the mistake this guard exists to catch.
        """
        _reject_prefixed_kwargs(kwargs)
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _reject_prefixed_field_names(cls, data: Any) -> Any:
        """Same guard as :meth:`__init__`, for the non-``__init__`` entry points.

        ``model_validate`` and pydantic's own revalidation paths bypass the
        constructor, and those must not be a way around the check.
        """
        if isinstance(data, dict):
            _reject_prefixed_kwargs(data)
        return data

    # ----------------------------------------------------------- utilities #
    def masked_api_key(self) -> str:
        """Return the provider key with all but the last four chars redacted."""
        if not self.llm_api_key:
            return "<unset>"
        raw = self.llm_api_key.get_secret_value()
        if len(raw) <= 4:
            return "****"
        return f"****{raw[-4:]}"

    def safe_summary(self) -> dict[str, Any]:
        """Redacted, log-safe view of the configuration.

        Secrets are never emitted; this mapping is safe to log at boot.
        """
        return {
            "environment": self.environment.value,
            "service_name": self.service_name,
            "log_level": self.log_level,
            "log_format": self.log_format.value,
            "graph_name": self.graph_name,
            "max_iterations": self.max_iterations,
            "llm_provider": self.llm_provider.value,
            "llm_model": self.llm_model,
            "llm_api_key": self.masked_api_key(),
            "postgres_enabled": self.postgres_enabled,
            "api_auth_enabled": self.api_auth_enabled,
            "hitl_enabled": self.hitl_enabled,
            "eval_provider": self.eval_provider.value,
        }


@lru_cache(maxsize=1)
def load_settings(**overrides: Any) -> Settings:
    """Build (and memoise) the process-wide :class:`Settings` instance.

    Args:
        **overrides: Field values that win over environment variables. Used by
            tests and by the CLI. Because the result is cached, passing overrides
            with a single call and then reading it again returns the same object.

    Returns:
        The frozen settings object.

    Raises:
        ConfigurationError: If the resulting configuration is invalid.

    Example:
        --------
        >>> s = load_settings(awf_environment="production")
        >>> s.is_production
        True
    """
    try:
        return Settings(**overrides) if overrides else Settings()
    except Exception as exc:
        context: dict[str, Any] = {"cause": type(exc).__name__} if overrides else {}
        raise ConfigurationError(f"invalid configuration: {exc}", **context) from exc


def reset_settings_cache() -> None:
    """Clear the memoised settings. Intended for tests and the CLI."""
    load_settings.cache_clear()


def get_settings() -> Settings:
    """FastAPI dependency returning the process-wide settings."""
    return load_settings()


# Aliases used by validators/tests that prefer the annotated form.
PositiveInt = Annotated[int, Field(gt=0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]

__all__ = [
    "Environment",
    "EvalProvider",
    "LLMProvider",
    "LogFormat",
    "Settings",
    "get_settings",
    "load_settings",
    "reset_settings_cache",
]
