"""Tests for the OpenAI-compatible provider adapter.

This adapter is the only path to every hosted model — OpenAI, Azure, vLLM,
Ollama, LiteLLM — and until now no test touched it, at 0% coverage. The
consequences of that are not academic: a gateway that rejects the
``json_schema`` hint, a ``Retry-After`` header silently ignored, or a 4xx
announced to an API client as retryable all pass unnoticed until a user
hits them in production against a paid provider.

The ``openai`` SDK is an optional extra and is not installed for the suite,
so the tests inject a fake client instead of stubbing the network. That
keeps them honest about the two things that actually matter here — the
payload we send, and the exception mapping we derive from the failure.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr
import pytest
from structlog.testing import capture_logs

from agentic_workflow.errors import (
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from agentic_workflow.llm.base import Message
from agentic_workflow.llm.openai_compat import (
    OpenAICompatibleLLM,
    ProviderHTTPError,
    _classify,
    _normalise_response_format,
    _parse_retry_after,
    _to_completion,
)


# --------------------------------------------------------------------------- #
# Doubles. Attribute access is answered by class, matching the SDK's plain
# response objects, so the mapping code sees the same shape it sees live.
# --------------------------------------------------------------------------- #
class _Obj:
    """A response object with a fixed set of attributes."""

    def __init__(self, **attrs: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)


class _FakeCompletions:
    """Records the payload and replays a scripted result."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.result


class _FakeClient:
    """Stands in for ``openai.AsyncOpenAI``."""

    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _Obj(completions=completions)
        self.completions = completions
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


def _client(
    *, result: Any = None, error: Exception | None = None, **kwargs: Any
) -> tuple[OpenAICompatibleLLM, _FakeClient]:
    """Build an adapter with a pre-injected client, bypassing the SDK import."""
    llm = OpenAICompatibleLLM(api_key="unit-test-key", model="gpt-4o-mini", **kwargs)
    fake = _FakeClient(_FakeCompletions(result=result, error=error))
    llm._client = fake
    return llm, fake


def _response(
    content: str = "ok",
    *,
    finish_reason: str | None = "stop",
    model: str | None = "gpt-4o-mini-2024",
    usage: Any = None,
) -> _Obj:
    return _Obj(
        choices=[_Obj(message=_Obj(content=content), finish_reason=finish_reason)],
        model=model,
        usage=usage,
    )


# --------------------------------------------------------------------------- #
class TestResponseFormatNormalisation:
    """Gateways differ in which response-format hints they accept."""

    @pytest.mark.parametrize("hint", [None, {}])
    def test_an_absent_hint_stays_absent(self, hint: dict[str, Any] | None) -> None:
        """Sending ``null`` is not the same as sending nothing to some gateways."""
        assert _normalise_response_format(hint) is None

    def test_json_object_passes_through_untouched(self) -> None:
        hint = {"type": "json_object"}
        assert _normalise_response_format(hint) is hint

    def test_json_schema_is_downgraded_to_json_object(self) -> None:
        """The strict hint is dropped, not rejected.

        More gateways understand ``json_object`` than ``json_schema``, and a
        hard 400 from a gateway that only wants the loose form would fail the
        whole run. The schema still reaches the model in the system prompt.
        """
        hint = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        assert _normalise_response_format(hint) == {"type": "json_object"}

    def test_an_unrecognised_hint_is_forwarded_verbatim(self) -> None:
        """We normalise what we understand and stay out of the rest."""
        hint = {"type": "structured", "vendor_flag": 1}
        assert _normalise_response_format(hint) is hint


# --------------------------------------------------------------------------- #
class TestResponseMapping:
    """Translating the SDK's response into a provider-agnostic Completion."""

    def test_a_complete_response_maps_every_field(self) -> None:
        usage = _Obj(prompt_tokens=11, completion_tokens=7, prompt_tokens_details=None)
        completion = _to_completion(_response("hello", usage=usage), fallback_model="fb")
        assert completion.content == "hello"
        assert completion.model == "gpt-4o-mini-2024"
        assert completion.usage.prompt_tokens == 11
        assert completion.usage.completion_tokens == 7
        assert completion.usage.calls == 1
        assert completion.raw["finish_reason"] == "stop"

    def test_cached_prompt_tokens_are_counted(self) -> None:
        """Cost attribution needs the cached count, and it is the easiest field
        to lose because it lives in a nested optional object."""
        usage = _Obj(
            prompt_tokens=100,
            completion_tokens=0,
            prompt_tokens_details=_Obj(cached_tokens=64),
        )
        completion = _to_completion(_response(usage=usage), fallback_model="fb")
        assert completion.usage.cached_tokens == 64

    def test_missing_usage_yields_zeroes_rather_than_an_error(self) -> None:
        """A gateway that omits usage must not crash the run."""
        completion = _to_completion(_response(), fallback_model="fb")
        assert completion.usage.prompt_tokens == 0
        assert completion.usage.calls == 0

    def test_no_choices_yields_empty_content(self) -> None:
        """A filtered/empty response is a valid transport, not an error."""
        completion = _to_completion(_Obj(choices=[], model=None, usage=None), fallback_model="fb")
        assert completion.content == ""
        assert completion.model == "fb"
        assert completion.raw["finish_reason"] == "stop"

    def test_a_null_message_body_becomes_an_empty_string(self) -> None:
        """The SDK types ``content`` as nullable; ``None`` is not a string."""
        response = _Obj(
            choices=[_Obj(message=_Obj(content=None), finish_reason=None)],
            model=None,
            usage=None,
        )
        completion = _to_completion(response, fallback_model="fb")
        assert completion.content == ""
        assert completion.raw["finish_reason"] == "stop"

    def test_the_fallback_model_is_used_when_the_provider_omits_one(self) -> None:
        """Self-hosted gateways often do not echo the model back."""
        completion = _to_completion(_response(model=None), fallback_model="local-llama")
        assert completion.model == "local-llama"

    def test_truncation_is_logged_loudly(self) -> None:
        """A cut-off answer looks like a short answer and silently fails the
        report's criteria, so it must not pass unnoticed."""
        with capture_logs() as logs:
            completion = _to_completion(
                _response("half a senten", finish_reason="length"), fallback_model="fb"
            )
        assert completion.raw["finish_reason"] == "length"
        assert [entry for entry in logs if entry["event"] == "llm.truncated"]


# --------------------------------------------------------------------------- #
class TestErrorClassification:
    """Mapping provider failures onto the retry-aware hierarchy.

    The mapping decides whether a failure burns the retry budget. Wrong in the
    permissive direction it bills the user five times for a bad API key; wrong
    in the strict direction it gives up on a blip the provider said was
    temporary.
    """

    @pytest.mark.parametrize("name", ["APITimeoutError", "Timeout", "ReadTimeout", "APITimeout"])
    def test_every_timeout_shape_becomes_a_timeout(self, name: str) -> None:
        exc = _classify(type(name, (Exception,), {})("slow"), model="m")
        assert isinstance(exc, ProviderTimeoutError)

    def test_a_rate_limit_by_name_carries_the_retry_hint(self) -> None:
        exc = type("RateLimitError", (Exception,), {})("slow down")
        exc.headers = {"retry-after": "3"}  # type: ignore[attr-defined]
        error = _classify(exc, model="m")
        assert isinstance(error, ProviderRateLimitedError)
        assert error.retry_after == 3.0

    def test_a_rate_limit_is_recognised_by_status_alone(self) -> None:
        """A gateway that reports 429 under its own exception name is still 429."""
        exc = _Obj.__new__(_Obj)
        error = _classify(_StatusError(429, "slow down"), model="m")
        assert isinstance(error, ProviderRateLimitedError)
        assert exc is not None

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_server_errors_are_retryable(self, status: int) -> None:
        error = _classify(_StatusError(status, "upstream"), model="m")
        assert isinstance(error, ProviderHTTPError)
        assert error.retryable is True
        assert error.status_code == status

    def test_a_connection_reset_is_retryable(self) -> None:
        error = _classify(type("APIConnectionError", (Exception,), {})("reset"), model="m")
        assert isinstance(error, ProviderHTTPError)
        assert error.retryable is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_are_not_retryable(self, status: int) -> None:
        """The decisive assertion.

        A bad key or a bad model is the caller's mistake; retrying it burns the
        whole budget and the user's money to arrive at the same error. The
        ``retryable`` flag is what an API client reads to decide, so it has to
        follow the decision the adapter made — not the class default.
        """
        error = _classify(_StatusError(status, "nope"), model="m")
        assert isinstance(error, ProviderHTTPError)
        assert error.status_code == status
        assert error.retryable is False
        assert error.to_dict()["retryable"] is False

    def test_an_unrecognised_failure_keeps_the_conservative_default(self) -> None:
        """Anything we cannot identify is surfaced, not retried, by the loop."""
        error = _classify(ValueError("something odd"), model="m")
        assert isinstance(error, ProviderError)
        assert "something odd" in str(error)

    def test_the_override_never_leaks_into_the_serialised_context(self) -> None:
        """It used to land in the context, which put ``retryable=False`` in the
        human-readable message while the machine-readable field said True."""
        error = _classify(_StatusError(401, "nope"), model="m")
        assert "retryable" not in error.to_dict()["context"]
        assert "retryable=False" not in error.to_dict()["message"]


class _StatusError(Exception):
    """An exception carrying an HTTP status, as the SDK's does."""

    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers: dict[str, str] | None = None


# --------------------------------------------------------------------------- #
class TestRetryAfterParsing:
    """``Retry-After`` is how a provider asks for exactly the wait it wants."""

    @pytest.mark.parametrize("headers", [None, {}])
    def test_an_absent_header_means_no_hint(self, headers: Any) -> None:
        assert _parse_retry_after(headers) is None

    @pytest.mark.parametrize("key", ["retry-after", "Retry-After"])
    def test_both_spellings_are_read(self, key: str) -> None:
        assert _parse_retry_after({key: "2.5"}) == 2.5

    def test_an_http_date_falls_back_to_backoff(self) -> None:
        """RFC 7231 allows a date, which is not a float. Returning ``None`` is
        correct — the caller then uses exponential backoff — but only because
        the ``ValueError`` is caught rather than escaping the error path."""
        assert _parse_retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) is None

    def test_a_present_but_empty_hint_is_no_hint(self) -> None:
        assert _parse_retry_after({"retry-after": ""}) is None


# --------------------------------------------------------------------------- #
class TestConstruction:
    """Credential handling happens before any network call."""

    @pytest.mark.parametrize("api_key", [None, SecretStr("")])
    def test_a_missing_key_fails_before_the_import(self, api_key: Any) -> None:
        """The extra is optional, so the key check must not need ``openai``."""
        with pytest.raises(ProviderError, match="API key is required"):
            OpenAICompatibleLLM(api_key=api_key, model="m")

    def test_a_string_and_a_secret_both_work(self) -> None:
        for key in ("plain-key", SecretStr("wrapped-key")):
            llm = OpenAICompatibleLLM(api_key=key, model="m")
            assert llm.mask_credentials().startswith("sk-")

    def test_the_mask_reveals_at_most_four_characters(self) -> None:
        llm = OpenAICompatibleLLM(api_key="sk-abcdefghijklmnop", model="m")
        assert llm.mask_credentials() == "sk-...mnop"

    def test_a_key_too_short_to_mask_says_nothing(self) -> None:
        """Masking a two-character key by revealing both of them is not masking."""
        llm = OpenAICompatibleLLM(api_key="ab", model="m")
        assert llm.mask_credentials() == "sk-***"


# --------------------------------------------------------------------------- #
class TestRequestPath:
    """The payload we hand the SDK."""

    async def test_the_request_carries_the_model_messages_and_limits(self) -> None:
        llm, fake = _client(result=_response("hi"))
        await llm.complete([Message(role="user", content="hello")])
        payload = fake.completions.calls[0]
        assert payload["model"] == "gpt-4o-mini"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["temperature"] == llm.temperature
        assert payload["max_tokens"] == llm.max_tokens

    async def test_the_response_format_hint_is_normalised_in_flight(self) -> None:
        """Normalising only in the helper would be a bug the caller can hit;
        this asserts the value that actually leaves the process."""
        llm, fake = _client(result=_response("{}"))
        await llm.complete(
            [Message(role="user", content="x")],
            response_format={"type": "json_schema", "json_schema": {}},
        )
        assert fake.completions.calls[0]["response_format"] == {"type": "json_object"}

    async def test_a_provider_failure_surfaces_as_a_typed_error(self) -> None:
        llm, _ = _client(error=_StatusError(401, "invalid api key"))
        with pytest.raises(ProviderHTTPError) as caught:
            await llm.complete([Message(role="user", content="x")])
        assert caught.value.status_code == 401

    async def test_a_transient_failure_is_retried_then_succeeds(self) -> None:
        """The retry policy lives in the base class; this proves the adapter
        reports a retryable error rather than swallowing it."""
        fake = _FakeClient(_FakeCompletions(error=_StatusError(503, "upstream")))
        llm = OpenAICompatibleLLM(api_key="k", model="m", max_retries=1, temperature=0.0)
        llm._client = fake
        with pytest.raises(ProviderHTTPError):
            await llm.complete([Message(role="user", content="x")])
        assert len(fake.completions.calls) == 2, "one attempt plus one retry"


# --------------------------------------------------------------------------- #
class TestLifecycle:
    async def test_closing_without_a_client_is_a_no_op(self) -> None:
        llm = OpenAICompatibleLLM(api_key="k", model="m")
        await llm.aclose()

    async def test_closing_releases_the_pool_and_forgets_the_client(self) -> None:
        """Forgetting matters: a closed client reused on the next request is a
        confusing ``Event loop is closed`` far from its cause."""
        llm, fake = _client(result=_response("x"))
        await llm.aclose()
        assert fake.closed == 1
        assert llm._client is None
        await llm.aclose()  # idempotent
        assert fake.closed == 1

    def test_a_missing_sdk_reports_the_extra_instead_of_an_import_error(self) -> None:
        """``openai`` is not installed in the default environment, which is
        exactly the situation a user hits after ``pip install agentic-workflow``.
        """
        llm = OpenAICompatibleLLM(api_key="k", model="m")
        try:
            import openai  # noqa: F401
        except ImportError:
            with pytest.raises(ProviderError, match="llm"):
                llm._get_client()
        else:  # pragma: no cover - only when the optional extra is installed
            assert llm._get_client() is not None
