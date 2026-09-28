"""The process-wide LLM usage counters on the metrics endpoint.

``LLMClient.total_usage`` has existed since the client gained its measurement
seam, and nothing ever read it: the engine merges every completion into one
per-process record and nobody published the result. The aggregate is
process-wide by construction — the engine owns a single client and injects it
into every run — so the surface that can honestly report it is the long-lived
process itself, and ``/metrics`` is the endpoint that already carries
operational counters.

Echo reports ``calls=1`` plus deterministic token approximations per
completion, so a run completed through the test client must move every
counter.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.api

_COUNTERS: dict[str, str] = {
    "calls": "awf_llm_calls_total",
    "prompt_tokens": "awf_llm_prompt_tokens_total",
    "completion_tokens": "awf_llm_completion_tokens_total",
    "cached_tokens": "awf_llm_cached_tokens_total",
}


def _body(request_factory: Any, run_id: str, **overrides: Any) -> dict[str, Any]:
    """Build a ``POST /v1/runs`` body from a domain request fixture.

    Args:
        request_factory: The ``ReviewRequest`` factory fixture.
        run_id: Identifier for the run under test.
        **overrides: Fields to override on the serialised body.

    Returns:
        A JSON-serialisable request body.
    """
    body = request_factory(run_id=run_id).model_dump(mode="json")
    # The hash is derived, not supplied: a client that sends it back would be
    # asserting something the server is responsible for computing.
    body.pop("content_hash", None)
    body.update(overrides)
    return body


def _usage_counts(client: Any) -> dict[str, int]:
    """Parse the four LLM usage counters out of the metrics text.

    Args:
        client: The bound test client.

    Returns:
        Mapping from counter name to its current value.

    Raises:
        AssertionError: If any counter line is absent from the exposition.
    """
    body = client.get("/metrics").text
    counts: dict[str, int] = {}
    for raw in body.splitlines():
        for key, name in _COUNTERS.items():
            if raw.startswith(f"{name} "):
                counts[key] = int(raw.split()[-1])
    missing = set(_COUNTERS) - set(counts)
    assert not missing, f"usage counters missing from /metrics output:\n{body}"
    return counts


class TestUsageCounters:
    def test_a_fresh_engine_reports_zero_usage(self, app_client: Any) -> None:
        """Nothing has talked to a model yet: the counters read zero.

        The engine builds its client lazily and nothing in the lifespan makes
        an LLM call, so a fresh process has genuinely empty usage — the
        endpoint must say so instead of hiding the keys.
        """
        assert _usage_counts(app_client) == {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }

    def test_usage_grows_as_runs_complete(self, app_client: Any, request_factory: Any) -> None:
        """A completed run moves every counter the process publishes.

        The accumulated usage must be observable after the run that produced it
        — this is the money question the endpoint exists to answer.
        """
        before = _usage_counts(app_client)
        response = app_client.post(
            "/v1/runs",
            json=_body(request_factory, "metrics-usage-1", auto_resolve=True),
        )
        assert response.status_code == 200, response.text
        after = _usage_counts(app_client)
        assert after["calls"] > before["calls"]
        assert after["prompt_tokens"] > before["prompt_tokens"]
        assert after["completion_tokens"] > before["completion_tokens"]
