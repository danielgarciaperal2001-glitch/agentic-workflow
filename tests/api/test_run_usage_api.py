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


class TestTheRunDetailAgreesWithTheUsageEndpoint:
    """``GET /v1/runs/{id}`` carries a ``usage`` field, so it has to be right.

    ``RunDetail.usage`` is on every payload the run routes return, which makes
    the detail the natural place a client looks. It is filled from the outcome,
    and the outcome a *read* produces is projected from the checkpoint — which is
    authoritative for status and report, and knows nothing at all about tokens.
    So the field was present, well-typed, and empty on exactly the reads that
    follow the operations that spent the money.
    """

    def test_reading_a_finished_run_reports_what_it_spent(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """A completed run's detail reports the same usage its POST reported.

        Not "some" usage — the same. A client that submits, reads the detail back
        and sees zeros has to decide whether the run was free or the field is
        broken, and neither guess is available from the payload alone. An
        operator watching spend through a dashboard built on this endpoint would
        conclude the deployment costs nothing.
        """
        run_id = "usage-read-1"
        started = app_client.post(
            "/v1/runs", json=_body(request_factory, run_id, auto_resolve=True)
        )
        assert started.status_code == 200, started.text
        submitted = started.json()["usage"]
        assert submitted["calls"] > 0, "this run has to have spent something first"

        detail = app_client.get(f"/v1/runs/{run_id}")
        assert detail.status_code == 200, detail.text

        assert detail.json()["usage"] == submitted

    def test_the_detail_and_the_usage_endpoint_do_not_disagree(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """Two endpoints answering the same question must answer it alike.

        The split is historical: ``/usage`` was added for a parked run, where no
        single operation payload sums the whole spend, and the detail's ``usage``
        came from the operation that last drove the run. Reading the same run
        twice and getting two different totals is worse than having neither, so
        the detail is held to the endpoint's number.
        """
        run_id = "usage-read-2"
        started = app_client.post("/v1/runs", json=_body(request_factory, run_id))
        assert started.status_code == 202, started.text

        detail = app_client.get(f"/v1/runs/{run_id}").json()
        usage = app_client.get(f"/v1/runs/{run_id}/usage").json()

        assert detail["usage"] == usage["usage"]

    def test_resolving_a_gate_updates_the_detail_as_well_as_the_endpoint(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """The accumulated figure moves when a drive settles, in both places.

        A run that parks, gets answered and is then read has been driven twice.
        The usage an operation payload reports is only that operation's share, so
        the accumulated attribution is the honest number for a parked run — and
        the detail is where a client actually looks for it.

        The resolve answers ``202`` rather than ``200`` because the graph parks on
        the next gate, and that is beside the point: what matters is that the
        figure grew and that both places grew it by the same amount.
        """
        run_id = "usage-read-3"
        started = app_client.post("/v1/runs", json=_body(request_factory, run_id))
        assert started.status_code == 202, started.text
        parked = app_client.get(f"/v1/runs/{run_id}").json()

        pending = parked["pending_approval"]
        resolved = app_client.post(
            f"/v1/approvals/{pending['approval_id']}/resolve",
            json={"decision": "approve", "reviewer": "tester"},
        )
        assert resolved.status_code in {200, 202}, resolved.text

        after = app_client.get(f"/v1/runs/{run_id}").json()
        assert after["usage"]["calls"] > parked["usage"]["calls"], (
            "a second drive spent tokens, so the accumulated figure must move"
        )
        assert after["usage"] == app_client.get(f"/v1/runs/{run_id}/usage").json()["usage"]
