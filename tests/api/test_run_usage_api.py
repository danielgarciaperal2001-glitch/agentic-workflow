"""Per-run usage on the wire: the operation payloads and the usage endpoint.

``POST /v1/runs`` and the resolve routes return a ``RunDetail`` projected
from the outcome of the drive they just performed, so ``usage`` there is the
spend that operation caused. ``GET /v1/runs/{id}/usage`` answers "what has
this run cost so far" from the registry's accumulated attribution — useful
for a parked run, where no single operation payload sums the whole spend.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.api


def _body(request_factory: Any, run_id: str, **overrides: Any) -> dict[str, Any]:
    """Build a ``POST /v1/runs`` body from a domain request fixture."""
    body = request_factory(run_id=run_id).model_dump(mode="json")
    body.pop("content_hash", None)
    body.update(overrides)
    return body


class TestRunUsageApi:
    def test_completed_run_reports_usage_in_its_payload(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """An auto-resolved run answers "what did that cost" immediately.

        The start response's ``usage`` is the whole run, because a single
        auto-resolved call drove every gate.
        """
        response = app_client.post(
            "/v1/runs",
            json=_body(request_factory, "usage-api-1", auto_resolve=True),
        )
        assert response.status_code == 200, response.text
        usage = response.json()["usage"]
        assert usage["calls"] > 0
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0

    def test_parked_run_reports_partial_usage_in_its_payload(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """A run parked on its first gate reports that drive's spend.

        Not the whole run — the rest has not happened yet — but the money
        already spent on its behalf.
        """
        response = app_client.post("/v1/runs", json=_body(request_factory, "usage-api-2"))
        assert response.status_code == 202, response.text
        usage = response.json()["usage"]
        assert usage["calls"] > 0

    def test_usage_endpoint_reads_the_accumulated_total(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """A parked run answers "spent so far" without grepping logs.

        The registry accumulates every drive this process ran, so the endpoint
        reports the parked run's whole attribution to date.
        """
        run_id = "usage-api-3"
        started = app_client.post("/v1/runs", json=_body(request_factory, run_id))
        assert started.status_code == 202, started.text
        first_drive = started.json()["usage"]["calls"]

        usage = app_client.get(f"/v1/runs/{run_id}/usage")
        assert usage.status_code == 200, usage.text
        body = usage.json()
        assert body["run_id"] == run_id
        assert body["usage"]["calls"] == first_drive

    def test_usage_endpoint_404s_for_unknown_runs(self, app_client: Any) -> None:
        """An unknown run is the same 404 every other run route answers."""
        response = app_client.get("/v1/runs/does-not-exist/usage")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "run_not_found"
