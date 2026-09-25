"""OpenAI-compatible provider (OpenAI, Azure, vLLM, Ollama, LiteLLM, ...).

The ``openai`` SDK is imported lazily inside :meth:`OpenAICompatibleLLM._complete`
so the package remains importable — and the whole test suite runnable — without
the optional ``llm`` extra installed.

Reliability features worth calling out:

* **Transport errors are classified.** Connection resets and 5xx responses
  become retryable :class:`~agentic_workflow.errors.ProviderError` subclasses;
  4xx responses other than 429 fail fast because retrying them just burns
  budget.
* **``retry_after`` is honoured** so the client backs off exactly as long as the
  provider asked instead of guessing.
* **Usage is always reported**, including cached prompt tokens, so cost
  attribution works on every provider that exposes them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import SecretStr

from agentic_workflow.errors import (
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from agentic_workflow.llm.base import Completion, LLMClient, Message, Usage
from agentic_workflow.logging import get_logger

log = get_logger(__name__)


class ProviderHTTPError(ProviderError):
    """HTTP error carrying the provider's ``Retry-After`` hint.

    Attributes:
        status_code: HTTP status returned by the provider.
        retry_after: Parsed ``Retry-After`` header in seconds, if present.
    """

    retryable = True

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class OpenAICompatibleLLM(LLMClient):
    """Client for any endpoint implementing the OpenAI chat-completions API.

    Example:
        --------
        >>> client = OpenAICompatibleLLM(  # doctest: +SKIP
        ...     api_key=SecretStr("sk-..."), model="gpt-4o-mini"
        ... )
        >>> await client.complete([Message(role="user", content="hi")])  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None,
        model: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, base_url=base_url, **kwargs)
        if api_key is None or (isinstance(api_key, SecretStr) and not api_key.get_secret_value()):
            raise ProviderError(
                "an API key is required for the openai provider",
                model=model,
                hint="set AWF_LLM_API_KEY",
            )
        self._api_key = (
            api_key.get_secret_value() if isinstance(api_key, SecretStr) else str(api_key)
        )
        self._client: Any = None

    # ------------------------------------------------------------ setup  #
    def _get_client(self) -> Any:
        """Build the async SDK client on first use (bound to the current loop)."""
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderError(
                "the 'openai' package is not installed; install the 'llm' extra",
                model=self.model,
            ) from exc

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,  # retries are handled by the base class
        )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # --------------------------------------------------------- request  #
    async def _complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        timeout: float | None = None,  # noqa: ARG002 - client-wide timeout instead
        **kwargs: Any,
    ) -> Completion:
        """Issue one chat-completions request and normalise the response."""
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Normalise the JSON-schema hint: some gateways only accept
        # {"type": "json_object"} and reject the richer json_schema variant.
        payload["response_format"] = _normalise_response_format(response_format)

        try:
            response = await client.chat.completions.create(**payload, **kwargs)
        except Exception as exc:
            raise _classify(exc, model=self.model) from exc

        return _to_completion(response, fallback_model=self.model)

    # ------------------------------------------------------------ usage #
    def mask_credentials(self) -> str:
        """Return a redacted form of the API key, safe for diagnostics."""
        return f"sk-...{self._api_key[-4:]}" if len(self._api_key) >= 4 else "sk-***"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalise_response_format(hint: dict[str, Any] | None) -> dict[str, Any] | None:
    """Downgrade the strict JSON-schema hint when the endpoint cannot honour it.

    ``json_object`` is supported by strictly more gateways than
    ``json_schema``; the full schema still travels in the system prompt, so the
    downgrade costs a little accuracy but avoids hard failures.
    """
    if not hint:
        return None
    kind = hint.get("type")
    if kind == "json_object":
        return hint
    if kind == "json_schema":
        return {"type": "json_object"}
    return hint


def _to_completion(response: Any, *, fallback_model: str) -> Completion:
    """Map an SDK response object onto our provider-agnostic :class:`Completion`."""
    choices = getattr(response, "choices", None) or []
    content = ""
    finish_reason = "stop"
    if choices:
        message = getattr(choices[0], "message", None)
        content = (getattr(message, "content", None) or "") if message else ""
        finish_reason = getattr(choices[0], "finish_reason", None) or "stop"

    raw_usage = getattr(response, "usage", None)
    usage = Usage()
    if raw_usage is not None:
        details = getattr(raw_usage, "prompt_tokens_details", None)
        cached = 0
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        usage = Usage(
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
            cached_tokens=cached,
            calls=1,
        )

    model = getattr(response, "model", None) or fallback_model
    raw: dict[str, Any] = {"finish_reason": finish_reason}
    if finish_reason == "length":
        # Truncation is a silent quality bug: surface it loudly in the log.
        log.warning("llm.truncated", model=model, finish_reason=finish_reason)

    return Completion(content=content, model=model, usage=usage, raw=raw)


def _classify(exc: Exception, *, model: str) -> ProviderError:
    """Translate an SDK exception into our retry-aware error hierarchy.

    The mapping is intentionally conservative: anything we cannot positively
    identify as transient is raised as a non-retryable
    :class:`~agentic_workflow.errors.ProviderError` so a bad API key fails fast
    instead of burning the whole retry budget.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)

    if name in {"APITimeoutError", "Timeout"} or "timeout" in name.lower():
        return ProviderTimeoutError(f"provider timed out: {exc}", model=model)
    if name in {"RateLimitError", "TooManyRequests"} or status == 429:
        return ProviderRateLimitedError(
            f"provider rate limited: {exc}",
            model=model,
            retry_after=_parse_retry_after(getattr(exc, "headers", None)),
        )
    if name in {"APIConnectionError", "APIError", "InternalServerError"} or (
        isinstance(status, int) and status >= 500
    ):
        return ProviderHTTPError(
            f"transient provider failure: {exc}",
            status_code=int(status or 503),
            model=model,
            retry_after=_parse_retry_after(getattr(exc, "headers", None)),
        )
    if isinstance(status, int) and 400 <= status < 500:
        # 4xx other than 429: bad request, bad key, bad model. Retrying is futile.
        return ProviderHTTPError(
            f"non-retryable provider error ({status}): {exc}",
            status_code=status,
            model=model,
            retryable=False,
        )
    return ProviderError(f"provider error: {exc}", model=model)


def _parse_retry_after(headers: Any) -> float | None:
    """Read ``Retry-After`` from an SDK header mapping."""
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = ["OpenAICompatibleLLM", "ProviderHTTPError"]
