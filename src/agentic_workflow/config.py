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


class Settings(BaseSettings):
    """Immutable runtime configuration.

    Example
    -------
    >>> settings = Settings(_env_file=None, awf_log_level="DEBUG")
    >>> settings.log_level
    'DEBUG'
    >>> settings.is_production
    False
    """

    model_config = SettingsConfigDict(
        env_prefix="AWF_",
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
        default="public",
        min_length=1,
        max_length=63,
        description="Schema hosting the LangGraph checkpoint tables.",
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
        """Connection string with the statement timeout applied."""
        parts = [self.postgres_dsn.rstrip(" ")]
        existing = f"options='-c statement_timeout={self.postgres_statement_timeout_ms}'"
        if "options=" in parts[0]:
            return parts[0]
        sep = "&" if "?" in parts[0] else "?"
        return f"{parts[0]}{sep}{existing}"

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
