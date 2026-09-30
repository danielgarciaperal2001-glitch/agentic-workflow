"""Provider-agnostic LLM interface.

Agents never talk to a vendor SDK directly. They receive an
:class:`LLMClient`, ask for a *structured* object via
:meth:`LLMClient.structured`, and the implementation is responsible for:

* applying a bounded retry policy with exponential backoff and jitter,
* enforcing a concurrency semaphore so a burst of parallel agents cannot
  blow through the provider's rate limit,
* rendering a Pydantic schema into the provider's response-format request and
  then **validating** the reply (a provider that returns prose instead of JSON
  becomes a normal, attributable error),
* recording token usage and latency for the observability layer.

Two implementations ship with the project:

* :class:`~agentic_workflow.llm.echo.EchoLLM` — deterministic, offline, used by
  tests, CI and the ``make demo`` target.
* :class:`~agentic_workflow.llm.openai_compat.OpenAICompatibleLLM` — any
  OpenAI-compatible endpoint, loaded lazily so the dependency stays optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from agentic_workflow.errors import (
    ProviderError,
    SchemaValidationError,
)
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat message.

    Attributes:
        role: ``system``, ``user``, ``assistant`` or ``tool``.
        content: Plain-text body.
        name: Optional author tag.
    """

    role: str
    content: str
    name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialise to the OpenAI chat-completions wire format."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one or more provider calls.

    Prompt caching and cost attribution both need these numbers, and a run
    without usage data cannot be billed, so every response carries them.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: Usage) -> Usage:
        """Return the element-wise sum of two usage records."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            calls=self.calls + other.calls,
        )

    def as_dict(self) -> dict[str, float | int]:
        """Flat mapping for structured logs and metrics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "calls": self.calls,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class Completion:
    """A raw (unstructured) provider response.

    Attributes:
        content: Text body.
        model: Model that actually served the request.
        usage: Token accounting.
        latency_ms: Round-trip latency.
        raw: Provider-specific payload, kept for debugging.
    """

    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the response for transport over the API."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage.as_dict(),
            "latency_ms": self.latency_ms,
        }


class LLMClient(ABC):
    """Abstract base class every provider implementation extends.

    Subclasses only need to implement :meth:`_complete`; the retry, timeout,
    concurrency and validation behaviour is inherited.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        max_concurrency: int = 8,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.base_url = base_url
        self._semaphore: Any = None
        self._total_usage = Usage()

    # ------------------------------------------------------------ hooks #
    @abstractmethod
    async def _complete(self, messages: Sequence[Message], **kwargs: Any) -> Completion:
        """Perform a single provider round-trip. Must not retry internally."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release provider resources. Safe to call more than once.

        Deliberately *not* abstract: most providers hold no resources, and making
        every subclass implement a no-op would be noise. Providers that do open
        sockets or file handles override this.
        """

    # ---------------------------------------------------------- public  #
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Call the provider, retrying transient failures.

        Args:
            messages: Chat history to send.
            response_format: Optional JSON-schema hint forwarded to the provider.
            **kwargs: Provider-specific overrides.

        Returns:
            The provider response.

        Raises:
            ProviderError: After exhausting the retry budget.
        """
        self._ensure_semaphore()

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._semaphore:
                    started = time.perf_counter()
                    completion = await self._complete(
                        messages,
                        response_format=response_format,
                        timeout=self.timeout_seconds,
                        **kwargs,
                    )
                completion = Completion(
                    content=completion.content,
                    model=completion.model or self.model,
                    usage=completion.usage,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    raw=completion.raw,
                )
                self._total_usage = self._total_usage.merge(completion.usage)
                log.debug(
                    "llm.completed",
                    model=completion.model,
                    latency_ms=round(completion.latency_ms, 2),
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                    attempt=attempt,
                )
                return completion
            except ProviderError as exc:
                # Honour the flag rather than a hardcoded tuple of exception
                # types. Every error class declares `retryable` precisely so an
                # adapter can say "a 503 is worth another attempt, a 401 is
                # not"; keying off the class name instead silently discarded
                # both, so a transient 5xx failed the whole run on the first
                # try and a caller could not make a permanent failure cheap.
                if not exc.retryable:
                    raise
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self._backoff(attempt, exc)
                log.warning(
                    "llm.retryable_failure",
                    error=str(exc),
                    attempt=attempt + 1,
                    sleep_s=round(delay, 2),
                    model=self.model,
                )
                await _sleep(delay)
            except Exception as exc:
                last_error = ProviderError(
                    f"provider call failed: {exc}",
                    model=self.model,
                    cause=type(exc).__name__,
                )
                if attempt >= self.max_retries:
                    break
                delay = self._backoff(attempt, last_error)
                log.warning(
                    "llm.unexpected_failure",
                    error=str(exc),
                    attempt=attempt + 1,
                    sleep_s=round(delay, 2),
                )
                await _sleep(delay)

        raise last_error or ProviderError("provider call failed", model=self.model)

    async def structured(
        self,
        messages: Sequence[Message],
        response_model: type[T],
        *,
        max_attempts: int = 2,
    ) -> T:
        """Ask the provider for a JSON object and validate it into *response_model*.

        Provider JSON support varies wildly, so the request asks for JSON in the
        system prompt *and* passes a JSON schema hint. The reply is then parsed
        and validated. If validation fails, the error message is appended to the
        conversation and the call is retried once — a cheap, very effective
        repair loop.

        Args:
            messages: Chat history to send.
            response_model: Pydantic model describing the required shape.
            max_attempts: Total attempts including the first one.

        Returns:
            A validated instance of *response_model*.

        Raises:
            SchemaValidationError: If the provider never returns a valid object.
        """
        schema_json = response_model.model_json_schema()
        hint = _schema_hint(response_model, schema_json)
        base_system = Message(role="system", content=hint)
        history: list[Message] = [base_system, *messages]

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            completion = await self.complete(
                history,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema_json,
                        "strict": False,
                    },
                },
            )
            try:
                return _parse_structured(completion.content, response_model)
            except (SchemaValidationError, ValueError) as exc:
                last_error = exc
                log.warning(
                    "llm.schema_repair",
                    attempt=attempt + 1,
                    model=response_model.__name__,
                    error=str(exc),
                )
                if attempt + 1 >= max_attempts:
                    break
                history = [
                    *history,
                    Message(role="assistant", content=completion.content[:4_000]),
                    Message(
                        role="user",
                        content=(
                            f"Your previous reply was rejected: {exc}\n"
                            "Return ONLY a valid JSON object matching the schema. "
                            "No prose, no markdown fences."
                        ),
                    ),
                ]

        raise SchemaValidationError(
            f"provider failed to return valid {response_model.__name__} after "
            f"{max_attempts} attempts: {last_error}",
            model=response_model.__name__,
        )

    # -------------------------------------------------------- internals #
    @property
    def total_usage(self) -> Usage:
        """Cumulative token usage for this client instance."""
        return self._total_usage

    def _ensure_semaphore(self) -> None:
        """Lazily bind the concurrency semaphore to the running event loop."""
        if self._semaphore is None:
            import asyncio

            self._semaphore = asyncio.Semaphore(self.max_concurrency)

    @staticmethod
    def _backoff(attempt: int, exc: Exception) -> float:
        """Exponential backoff with full jitter, honouring ``retry_after``.

        Jitter is essential: without it, every agent that failed on the same
        rate-limit window retries in lockstep and reproduces the throttling.

        ``random`` is used deliberately and is *not* a security problem: the value
        only has to decorrelate concurrent retries, it is never a token or a key.
        """
        import random

        base = min(0.5 * (2**attempt), 8.0)
        retry_after = getattr(exc, "retry_after", None)
        if isinstance(retry_after, int | float) and retry_after > 0:
            return min(float(retry_after), 30.0)
        jitter = random.random()  # noqa: S311 - decorrelating retries, not crypto
        return float(base * (0.5 + jitter / 2))


#: Provider-identity fields copied from the delegate so callers see the same
#: configuration through a wrapper. The concurrency ceiling, retries and
#: timeouts stay where they belong: on the shared client.
_PROVIDER_FIELDS: tuple[str, ...] = (
    "model",
    "temperature",
    "max_tokens",
    "timeout_seconds",
    "max_retries",
    "max_concurrency",
    "base_url",
)


class UsageScopedClient(LLMClient):
    """Account the completions one caller causes through a shared client.

    Not a provider: it delegates every call to a shared :class:`LLMClient`
    (the process-wide one), so the provider semaphore, retries and timeouts
    all keep working exactly once. What it adds is attribution: the ``usage``
    carried by each returned :class:`Completion` is merged into this wrapper's
    own :class:`Usage`, read off the response rather than inferred from
    counters.

    Because attribution happens per completed call instead of by diffing a
    shared counter over a time window, concurrent callers are charged exactly
    — a sibling run's tokens are never credited to this one, and this run's
    schema-repair retries always are. The engine binds one of these per drive
    and accumulates the run's total across drives in the registry.

    This is also where a run's token ceiling is enforced, and the reason is that
    it is the one place that knows the spend as it happens. A check after the
    drive would be a report rather than a limit: measured through the real
    endpoint, one submission of 200 files costs 4.06M tokens, so by the time a
    post-hoc check could run every one of them was already paid for. Refusing
    here means the *next* call is the one that does not happen.

    The check is after the merge rather than before the call, because a
    completion's cost is only knowable once the provider has answered. So
    ``spent`` overshoots ``budget`` by at most one call, and the overshoot is
    charged and reported rather than discarded: the tokens were spent either way.

    Args:
        delegate: The client that actually talks to the provider. Ownership
            stays with the caller: closing a wrapper never closes the delegate.
        budget: Ceiling on this run's total tokens, or ``0`` for unbounded. The
            run's *whole* total, not this drive's share, so a run that is resumed
            cannot reset its ceiling by parking more often.
        spent_before: What the run had already spent in earlier drives. Read
            from the registry when the wrapper is built, so a multi-drive run is
            measured as one bill.
        run_id: The run these tokens are charged to, for the error's context.
    """

    def __init__(
        self,
        delegate: LLMClient,
        *,
        budget: int = 0,
        spent_before: Usage | None = None,
        run_id: str = "",
    ) -> None:
        super().__init__(**{name: getattr(delegate, name) for name in _PROVIDER_FIELDS})
        self.delegate = delegate
        self._usage = Usage()
        self._budget = budget
        self._spent_before = spent_before or Usage()
        self._run_id = run_id

    @property
    def usage(self) -> Usage:
        """The usage attributed to this wrapper so far."""
        return self._usage

    @property
    def run_total(self) -> Usage:
        """The whole run's usage: earlier drives plus this one.

        The budget is per run, so this is the number the ceiling is compared
        against. A drive-local total would let a caller reset the budget by
        making the run park, which is the opposite of what a ceiling is for.

        Returns:
            The merged usage across every drive of this run.
        """
        return self._spent_before.merge(self._usage)

    @property
    def total_usage(self) -> Usage:
        """The attributed usage; identical to :attr:`usage`.

        Overridden so nothing reading the ``LLMClient`` contract sees the
        process-wide total through this wrapper instead of its own share.
        """
        return self._usage

    async def _complete(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> Completion:
        """Unreachable: the wrapper delegates and never calls this hook.

        Raised rather than implemented so a future refactor cannot silently
        drop the accounting by reverting to ``super().complete``.
        """
        raise NotImplementedError("UsageScopedClient delegates instead of completing")

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Delegate to the shared client, record the call, then hold the ceiling.

        The delegate handles the concurrency semaphore, retries and timeouts;
        this method only observes the response, so the process-wide ceiling is
        untouched and attribution is exact per completed call.

        The budget check sits after the merge and before returning, so the run
        that crosses its ceiling still gets the completion it paid for and the
        *next* call is what raises. Checking before the call would be tidier on
        paper and wrong in practice: the cost of a call is unknown until the
        provider answers, so a pre-check would either refuse calls that would
        have fit or admit calls that would not.

        Raises:
            TokenBudgetExceededError: If this completion took the run past its
                configured ceiling.
        """
        completion = await self.delegate.complete(
            messages, response_format=response_format, **kwargs
        )
        self._usage = self._usage.merge(completion.usage)
        self._enforce_budget()
        return completion

    def _enforce_budget(self) -> None:
        """Raise if the run has spent more than its ceiling allows.

        Raises:
            TokenBudgetExceededError: If a budget is configured and the run's
                total now exceeds it.
        """
        if self._budget <= 0:
            return
        total = self.run_total
        if total.total_tokens <= self._budget:
            return
        from agentic_workflow.errors import TokenBudgetExceededError

        raise TokenBudgetExceededError(
            f"run exceeded its {self._budget} token budget",
            run_id=self._run_id,
            budget=self._budget,
            spent=total.total_tokens,
            calls=total.calls,
        )

    async def aclose(self) -> None:
        """Release nothing: the delegate is the engine's, not this wrapper's.

        Closing a per-run wrapper must not take down the process-wide client
        while other runs are mid-flight on it.
        """

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the shared client.

        Provider subclasses carry extra state (``EchoLLM.latency_ms``) that
        callers may legitimately read; the wrapper must not hide it.
        """
        return getattr(self.delegate, name)


async def _sleep(seconds: float) -> None:
    """Await *seconds* without importing asyncio at module import time."""
    import asyncio

    await asyncio.sleep(seconds)


def _schema_hint(model: type[BaseModel], schema: dict[str, Any]) -> str:
    """Build the system prompt describing the required JSON shape.

    Kept compact on purpose: a short, explicit schema instruction measurably
    improves structured-output reliability across providers, while a full JSON
    Schema dump wastes context and confuses smaller models.
    """
    fields = ", ".join(
        f'"{name}": {prop.get("type", "any")!s}'
        + (" (required)" if name in schema.get("required", []) else " (optional)")
        for name, prop in schema.get("properties", {}).items()
    )
    return (
        f"You are a precise JSON emitter. Respond with a single JSON object matching "
        f"the `{model.__name__}` schema. Fields: {fields}. "
        "Respond with JSON only: no prose, no markdown fences, no trailing commas."
    )


def _parse_structured(content: str, model: type[T]) -> T:
    """Parse and validate a provider reply into *model*.

    Tolerates markdown fences and leading/trailing noise, which is the most
    common deviation from a strict JSON contract in practice.

    Raises:
        SchemaValidationError: If no valid JSON object can be recovered.
    """
    payload = _extract_json_object(content)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise SchemaValidationError(
            f"response did not satisfy {model.__name__}: {exc.error_count()} error(s); "
            f"first: {exc.errors()[0].get('loc')} {exc.errors()[0].get('msg')}",
            model=model.__name__,
        ) from exc


def _extract_json_object(content: str) -> dict[str, Any]:
    """Return the first balanced JSON object found in *content*.

    A brace-counting scan (string-aware) beats a regex because it survives nested
    objects, escaped quotes and braces inside string literals.

    Raises:
        SchemaValidationError: If no balanced object is present.
    """
    import json

    text = content.strip()
    if text.startswith("```"):
        # Strip a ```json ... ``` fence if present.
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)

    raise SchemaValidationError(
        f"no JSON object found in response ({len(content)} chars): {content[:200]!r}"
    )


def build_llm_client(settings: Any, **overrides: Any) -> LLMClient:
    """Instantiate the provider selected by configuration.

    Args:
        settings: A :class:`~agentic_workflow.config.Settings` instance.
        **overrides: Field overrides applied on top of *settings*.

    Returns:
        A ready-to-use client.

    Raises:
        ConfigurationError: If the provider is unknown or its dependency is absent.
    """
    from agentic_workflow.config import LLMProvider
    from agentic_workflow.errors import ConfigurationError

    params: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
        "max_concurrency": settings.llm_max_concurrency,
        "base_url": settings.llm_base_url,
    }
    params.update(overrides)

    if settings.llm_provider is LLMProvider.ECHO:
        from agentic_workflow.llm.echo import EchoLLM

        return EchoLLM(**params)
    if settings.llm_provider is LLMProvider.OPENAI:
        from agentic_workflow.llm.openai_compat import OpenAICompatibleLLM

        return OpenAICompatibleLLM(api_key=settings.llm_api_key, **params)

    raise ConfigurationError(f"unsupported llm provider: {settings.llm_provider!r}")


__all__ = [
    "Completion",
    "LLMClient",
    "Message",
    "Usage",
    "build_llm_client",
]
