"""The HTTP surface, exercised the way a client would exercise it.

The unit suite proves the error taxonomy maps to the right status. This suite
proves the *whole* surface: a run submitted over HTTP parks on a gate, a human
answers it over HTTP, the run finishes, and the answer is on the record. Those
are the three steps the README documents, and the only way to know they work is
to perform them.

The engine runs on the echo provider, so every response here is deterministic
and no test touches the network.

Status codes are load-bearing throughout, because the engine's whole vocabulary
is unusual: a run waiting for a person is a *success*, so reads of it answer
``202``. A client that treats that as an error retries a run that is working
exactly as designed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from agentic_workflow.api.routers.events import TERMINAL_EVENTS
from agentic_workflow.api.schemas import StartRunRequest
from tests.helpers import BUGGY_SOURCE

pytestmark = [pytest.mark.api, pytest.mark.integration]

#: The graph's own gate is what the pending stage is compared against, because
#: which gate a run stops at depends on the provider's answer.
PATCH_STAGE = "patch_apply"

#: Reused by the WebSocket tests so "the stream ended" and "the server closed
#: the socket" are the same statement, taken from the implementation rather than
#: restated here where the two could drift apart.
TERMINAL_EVENT_NAMES = set(TERMINAL_EVENTS)


@pytest.fixture
def fast_ws(checkpointer: Any) -> Iterator[TestClient]:
    """A client whose WebSocket heartbeat is fast enough to assert on.

    The production default is 20 seconds, which is right for a fleet of idle
    dashboards and far too slow to prove a socket is still open. Asserting the
    default instead would mean sleeping through it.

    Args:
        checkpointer: The in-memory checkpointer fixture.

    Yields:
        A bound test client with a 50 ms heartbeat.
    """
    from agentic_workflow.api.app import create_app
    from agentic_workflow.config import load_settings, reset_settings_cache
    from agentic_workflow.services.engine import WorkflowEngine

    reset_settings_cache()
    settings = load_settings(
        environment="development",
        llm_provider="echo",
        postgres_enabled=False,
        hitl_enabled=True,
        log_level="WARNING",
        ws_heartbeat_seconds=0.05,
    )
    engine = WorkflowEngine(settings, checkpointer=checkpointer)
    try:
        with TestClient(create_app(settings, engine=engine, configure_logs=False)) as client:
            yield client
    finally:
        reset_settings_cache()


def _read_until(socket: Any, until: set[str], *, limit: int = 200) -> list[dict[str, Any]]:
    """Read frames until one of *until* arrives, or the server closes the socket.

    Reading a fixed number of frames cannot work here: the server sends
    heartbeats forever while a run is parked, and closes the connection on a
    terminal event. Both outcomes are legitimate endings, and both are reported
    by returning.

    Args:
        socket: An open test WebSocket.
        until: Event names that end the read.
        limit: Safety bound, so a protocol change cannot hang the suite.

    Returns:
        The frames read, in arrival order.
    """
    frames: list[dict[str, Any]] = []
    for _ in range(limit):
        try:
            frame = socket.receive_json()
        except WebSocketDisconnect:
            break
        frames.append(frame)
        if str(frame.get("event")) in until:
            break
    return frames


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


def _parked(client: TestClient, request_factory: Any, run_id: str) -> str:
    """Submit a run and return the id of the gate it parked on.

    Args:
        client: The bound test client.
        request_factory: The ``ReviewRequest`` factory fixture.
        run_id: Identifier for the run.

    Returns:
        The pending approval id.
    """
    response = client.post("/v1/runs", json=_body(request_factory, run_id))
    assert response.status_code == 202, response.text
    return str(response.json()["pending_approval"]["approval_id"])


def _resolve(client: TestClient, approval_id: str, decision: str = "approve", **extra: Any) -> Any:
    """Answer a gate and return the response for assertions.

    Args:
        client: The bound test client.
        approval_id: The gate to answer.
        decision: ``approve``, ``edit`` or ``reject``.
        **extra: Additional fields for the request body.

    Returns:
        The ``httpx`` response.
    """
    return client.post(
        f"/v1/approvals/{approval_id}/resolve",
        json={"decision": decision, "reviewer": "alice", **extra},
    )


class TestHealth:
    """Probes an orchestrator uses to decide where to send traffic."""

    def test_liveness_needs_no_arguments(self, app_client: TestClient) -> None:
        response = app_client.get("/health/live")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]

    def test_readiness_checks_the_engine_not_just_the_process(self, app_client: TestClient) -> None:
        """``ready`` must mean "can serve", not "process exists".

        A liveness probe that passes while the checkpointer is unreachable is
        worse than no probe: the orchestrator keeps sending traffic to an
        instance that can only fail.
        """
        response = app_client.get("/health/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["checks"]["engine"]["ok"] is True
        assert body["checks"]["engine"]["durable"] is False

    def test_every_response_carries_a_request_id(self, app_client: TestClient) -> None:
        """Without it, a report from a user is not traceable to a log line."""
        response = app_client.get("/health/live")

        assert response.headers["X-Request-ID"].startswith("req_")

    def test_an_inbound_request_id_is_preserved(self, app_client: TestClient) -> None:
        """A gateway that already assigned an id must not have it overwritten.

        Two systems that each mint their own correlation id produce a log where
        neither id can be used to find the other half of a trace.
        """
        response = app_client.get("/health/live", headers={"X-Request-ID": "req_from_edge"})

        assert response.headers["X-Request-ID"] == "req_from_edge"


class TestRunLifecycle:
    """Submit, inspect, finish."""

    def test_a_submitted_run_parked_on_a_gate_answers_202(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """``202`` because waiting for a human is a normal outcome.

        Returning ``201`` would tell the client the work is done; returning an
        error would tell it the system broke. Neither is true, and a client that
        believes either will retry or give up.
        """
        response = app_client.post("/v1/runs", json=_body(request_factory, "api-park-1"))

        assert response.status_code == 202
        body = response.json()
        assert body["run_id"] == "api-park-1"
        assert body["status"] == "waiting_human"
        assert body["is_parked"] is True
        assert body["is_finished"] is False
        assert body["pending_approval"]["approval_id"]

    def test_an_auto_resolved_run_finishes_in_one_call(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """``auto_resolve`` is the batch path: one call, no human in the loop."""
        response = app_client.post(
            "/v1/runs",
            json=_body(request_factory, "api-auto-1", auto_resolve=True),
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["is_finished"] is True
        assert body["report"] is not None
        assert body["pending_approval"] is None

    def test_reading_a_parked_run_also_answers_202(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A read must report the same situation the write did.

        A client that polls for the report and sees ``200`` with an empty body
        has to guess whether the run finished. The status code answers it.
        """
        _parked(app_client, request_factory, "api-read-1")

        response = app_client.get("/v1/runs/api-read-1")

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "waiting_human"
        assert body["pending_approval"]["stage"]
        assert body["pending_approval"]["options"]

    def test_reading_a_finished_run_answers_200(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        app_client.post("/v1/runs", json=_body(request_factory, "api-read-2", auto_resolve=True))

        assert app_client.get("/v1/runs/api-read-2").status_code == 200

    def test_an_unknown_run_is_a_404_with_a_code(self, app_client: TestClient) -> None:
        """The body must be actionable, not a bare 404.

        A client that cannot distinguish "you typed the id wrong" from "that run
        was garbage collected" has no way to decide whether to retry.
        """
        response = app_client.get("/v1/runs/does-not-exist")

        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "run_not_found"
        assert error["retryable"] is False

    def test_a_duplicate_run_id_is_a_409(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """Reusing a run id would silently overwrite a live checkpoint thread."""
        body = _body(request_factory, "api-dup-1")
        assert app_client.post("/v1/runs", json=body).status_code == 202

        response = app_client.post("/v1/runs", json=body)

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "run_already_exists"

    def test_a_matching_content_hash_replays_instead_of_duplicating(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A client that retries a submission it never got an answer for.

        Without idempotency a network retry produces two runs, two sets of
        approvals and two reports — and the operator answers whichever one they
        happen to see first.
        """
        request = request_factory(run_id="api-idem-1")
        body = request.model_dump(mode="json")
        body.pop("content_hash", None)
        headers = {"X-Content-Hash": request.content_hash}

        first = app_client.post("/v1/runs", json=body, headers=headers)
        second = app_client.post("/v1/runs", json=body, headers=headers)

        assert first.status_code == 202
        assert second.status_code == 202
        assert second.json()["run_id"] == first.json()["run_id"]

    def test_a_lying_content_hash_is_refused(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The header is an integrity check; accepting a wrong one defeats it.

        A client that sends a stale hash alongside a changed body is either
        buggy or tampering, and both deserve an error rather than a run.
        """
        response = app_client.post(
            "/v1/runs",
            json=_body(request_factory, "api-hash-1"),
            headers={"X-Content-Hash": "sha256_of_something_else"},
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert "content" in error["message"].lower()

    def test_an_unknown_field_is_refused_rather_than_ignored(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A typo that is silently dropped becomes an anonymous decision later.

        ``extra="forbid"`` is the difference between a client that learns its
        payload was wrong and one that ships a run with no reviewer.
        """
        body = _body(request_factory, "api-extra-1")
        body["reviewer"] = "alice"

        response = app_client.post("/v1/runs", json=body)

        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert "reviewer" in response.text

    def test_an_unsafe_run_id_is_refused(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """Run ids become checkpoint thread keys and log fields.

        Accepting a newline or a shell metacharacter there is a log-injection
        and key-collision vector, so the alphabet is enforced at the edge.
        """
        body = _body(request_factory, "api-safe-1")
        body["run_id"] = "run with spaces\nand a newline"

        assert app_client.post("/v1/runs", json=body).status_code == 422

    def test_a_run_without_a_request_id_is_refused(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The business id is the only thing tying a run back to its ticket."""
        body = _body(request_factory, "api-noreq-1")
        del body["request_id"]

        assert app_client.post("/v1/runs", json=body).status_code == 422

    def test_the_report_of_a_completed_run_is_retrievable(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The report is the product; it must be addressable on its own.

        A UI lists runs and shows a verdict next to each. Fetching a whole run
        to read one field is what makes such a list slow.
        """
        app_client.post("/v1/runs", json=_body(request_factory, "api-rep-1", auto_resolve=True))

        response = app_client.get("/v1/runs/api-rep-1/report")

        assert response.status_code == 200
        body = response.json()
        assert body["markdown"]
        assert body["decision"] in ("approved", "rejected", "changes_requested")

    def test_a_parked_run_reports_no_report_yet(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """``202`` with no body, not a 404.

        A 404 would claim the run does not exist — it does, and it is healthy.
        Inventing a placeholder report would be a worse lie: the operator would
        read a verdict the agents never reached.
        """
        _parked(app_client, request_factory, "api-rep-2")

        response = app_client.get("/v1/runs/api-rep-2/report")

        assert response.status_code == 202
        assert response.json() is None

    def test_timings_are_exposed_per_node(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """``"the agents are slow"`` is unanswerable without per-node accounting."""
        app_client.post("/v1/runs", json=_body(request_factory, "api-time-1", auto_resolve=True))

        response = app_client.get("/v1/runs/api-time-1/timings")

        assert response.status_code == 200
        body = response.json()
        nodes = {entry["node"] for entry in body["items"]}
        assert {"triage", "programmer", "reviewer", "tester", "reporter"} <= nodes
        assert body["count"] == len(body["items"])
        assert body["total_ms"] >= 0
        assert body["slowest_node"] in nodes

    def test_runs_are_listable(self, app_client: TestClient, request_factory: Any) -> None:
        app_client.post("/v1/runs", json=_body(request_factory, "api-list-1", auto_resolve=True))

        response = app_client.get("/v1/runs")

        assert response.status_code == 200
        body = response.json()
        assert "api-list-1" in {item["run_id"] for item in body["items"]}
        assert body["count"] == len(body["items"])

    def test_the_listing_can_be_filtered_by_status(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A dashboard of "what is stuck" must not need to fetch every run."""
        _parked(app_client, request_factory, "api-filter-1")

        response = app_client.get("/v1/runs", params={"status": "waiting_human"})

        assert response.status_code == 200
        assert "api-filter-1" in {item["run_id"] for item in response.json()["items"]}


class TestHumanInTheLoop:
    """The three documented steps, performed over HTTP."""

    def test_the_inbox_lists_what_is_blocking_a_run(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """An operator arrives at the inbox, not at a run id they were given."""
        _parked(app_client, request_factory, "api-hitl-1")

        response = app_client.get("/v1/approvals")

        assert response.status_code == 200
        body = response.json()
        assert [item["run_id"] for item in body["items"]] == ["api-hitl-1"]
        assert body["items"][0]["diff_preview"]
        assert body["count"] == 1

    def test_a_looked_up_gate_offers_the_diff_a_human_needs(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A gate answered without reading the diff is a rubber stamp."""
        approval_id = _parked(app_client, request_factory, "api-hitl-2")

        response = app_client.get(f"/v1/approvals/{approval_id}/diff")

        assert response.status_code == 200
        body = response.json()
        assert "---" in body["diff"]
        assert body["stage"]
        assert isinstance(body["files"], list)

    def test_a_single_gate_is_addressable(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A deep link from a notification must resolve without a listing."""
        approval_id = _parked(app_client, request_factory, "api-hitl-2b")

        response = app_client.get(f"/v1/approvals/{approval_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == "api-hitl-2b"
        assert set(body["options"]) <= {"approve", "edit", "reject"}

    def test_approving_completes_the_run(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The end-to-end claim the README makes, asserted rather than assumed.

        The run parks more than once — the reporter's own sign-off is a gate —
        so the test answers gates the way an operator would: keep answering
        until the run stops asking.
        """
        approval_id = _parked(app_client, request_factory, "api-hitl-3")

        for _ in range(5):
            response = _resolve(app_client, approval_id, comment="ship it")
            assert response.status_code in (200, 202), response.text

            body = response.json()
            if body["status"] != "waiting_human":
                break
            approval_id = body["pending_approval"]["approval_id"]

        assert app_client.get("/v1/runs/api-hitl-3").json()["status"] == "completed"

    def test_rejecting_ends_the_run_as_rejected_not_failed(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A human saying no must not be reported as the system breaking.

        This is the assertion that would have caught the original defect, and it
        matters beyond the status code: a dashboard that counts `failed` would
        page an on-call engineer because a reviewer did their job.
        """
        approval_id = _parked(app_client, request_factory, "api-hitl-4")

        # Rejecting a patch gate routes the run to the reporter, which asks for
        # its own sign-off. A rejection is a verdict, so the reporter has to be
        # told "no" too before the run can reach a terminal status.
        for _ in range(5):
            response = _resolve(
                app_client, approval_id, decision="reject", comment="wrong approach"
            )
            assert response.status_code in (200, 202), response.text
            assert response.json()["decision"] == "reject"

            body = response.json()
            if body["status"] != "waiting_human":
                break
            approval_id = body["pending_approval"]["approval_id"]

        body = app_client.get("/v1/runs/api-hitl-4").json()
        assert body["status"] == "rejected"
        assert body["pending_approval"] is None
        assert body["error"] is None

    def test_a_rejection_still_produces_a_report(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The person who refused is the audience for the report.

        A status code with no explanation leaves them nothing to act on and
        nothing to forward to whoever proposed the change.
        """
        approval_id = _parked(app_client, request_factory, "api-hitl-5")
        for _ in range(5):
            response = _resolve(app_client, approval_id, decision="reject", comment="no")
            if response.json()["status"] != "waiting_human":
                break
            approval_id = response.json()["pending_approval"]["approval_id"]

        report = app_client.get("/v1/runs/api-hitl-5/report")

        assert report.status_code == 200
        assert report.json()["decision"] == "rejected"

    def test_the_decision_is_on_the_record(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """Who decided, at which gate, and what they typed must be recoverable."""
        approval_id = _parked(app_client, request_factory, "api-hitl-6")
        _resolve(app_client, approval_id, comment="reviewed the diff")

        response = app_client.get("/v1/runs/api-hitl-6/decisions")

        assert response.status_code == 200
        body = response.json()
        assert body["count"] >= 1
        first = body["items"][0]
        assert first["reviewer"] == "alice"
        assert first["stage"] in ("patch_review", PATCH_STAGE)
        assert first["decided_at"]
        assert first["latency_seconds"] >= 0

    def test_answering_a_gate_twice_is_a_409(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A double-clicking operator must not apply two decisions.

        The second answer would be recorded against a gate that no longer exists,
        and the audit trail would claim a person authorised something twice.
        """
        approval_id = _parked(app_client, request_factory, "api-hitl-7")
        assert _resolve(app_client, approval_id).status_code in (200, 202)

        second = _resolve(app_client, approval_id)

        assert second.status_code == 409
        assert second.json()["error"]["code"] == "approval_already_resolved"

    def test_replaying_a_decision_is_idempotent(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The explicit "I am sure" path for a client that lost the first answer.

        Refusing would force the operator to find a fresh approval to answer;
        re-applying would double-book. Returning the original resolution is the
        only answer that lets a client recover.
        """
        approval_id = _parked(app_client, request_factory, "api-hitl-8")
        _resolve(app_client, approval_id)

        response = app_client.post(
            f"/v1/approvals/{approval_id}/replay",
            json={"decision": "approve", "reviewer": "alice"},
        )

        assert response.status_code == 200
        assert response.json()["replayed"] is True

    def test_approving_someone_elses_gate_is_refused(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The graph's own guard must stay the authority on gate identity.

        A client that answers a superseded gate has a stale view of the run;
        applying its decision to whatever is pending now would let an approval
        authorise an action the approver never saw. ``/resume`` is the endpoint
        that accepts a client-declared id precisely so this guard has something
        to check.
        """
        _parked(app_client, request_factory, "api-hitl-9")

        response = app_client.post(
            "/v1/runs/api-hitl-9/resume",
            json={
                "approval_id": "apr_some_other_run_patch_apply_09_deadbeef",
                "decision": "approve",
                "reviewer": "alice",
            },
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "invalid_state"

    def test_an_edit_decision_records_what_the_human_wrote(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """An ``edit`` whose payload is dropped makes the human's work unrecoverable.

        This is the one decision where the human is the author, so the payload
        *is* the contribution — an audit trail without it records that somebody
        was involved without recording what they did.
        """
        approval_id = _parked(app_client, request_factory, "api-hitl-11")
        _resolve(
            app_client,
            approval_id,
            decision="edit",
            payload={"instruction": "use Decimal and add a regression test"},
        )

        decisions = app_client.get("/v1/runs/api-hitl-11/decisions").json()["items"]

        assert decisions[0]["decision"] == "edit"
        assert "Decimal" in str(decisions[0].get("payload"))

    def test_the_audit_endpoint_reports_on_the_signatures(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """Without a signing secret there is nothing to verify, and that is said.

        Reporting a failure here would train operators to ignore the field.
        """
        approval_id = _parked(app_client, request_factory, "api-hitl-10")
        _resolve(app_client, approval_id)

        response = app_client.get("/v1/approvals/by-run/api-hitl-10/audit")

        assert response.status_code == 200
        body = response.json()
        assert body["count"] >= 1
        assert body["unverified"] == 0
        assert body["signing_configured"] in (True, False)

    def test_inbox_stats_are_exposed(self, app_client: TestClient, request_factory: Any) -> None:
        """A dashboard needs a cheap poll; re-listing every approval is not one."""
        _parked(app_client, request_factory, "api-hitl-12")

        response = app_client.get("/v1/approvals/stats")

        assert response.status_code == 200
        assert response.json()["pending"] >= 1

    def test_the_public_stats_still_report_the_counters_the_cheap_one_omits(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The readiness probe got a cheap path; this route must not have.

        The probe counts from the registry, which cannot know whether an approval
        has expired or already been answered — so it reports zero for both. That
        trade is correct for a poll an orchestrator makes every few seconds and
        wrong here, where a human reads the number and acts on it. Both counters
        are present in this payload, which is what distinguishes it from the
        probe's.
        """
        _parked(app_client, request_factory, "api-hitl-12b")

        body = app_client.get("/v1/approvals/stats").json()

        assert set(body) == {"pending", "expired", "resolved_pending", "by_stage"}
        assert isinstance(body["expired"], int)
        assert isinstance(body["resolved_pending"], int)

    def test_an_unknown_approval_is_a_404(self, app_client: TestClient) -> None:
        response = app_client.get("/v1/approvals/apr_nope_00_deadbeef")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "approval_not_found"

    def test_resuming_a_run_that_is_not_parked_is_a_409(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """There is no gate to answer, and inventing one would be a lie."""
        app_client.post("/v1/runs", json=_body(request_factory, "api-hitl-13", auto_resolve=True))

        response = app_client.post(
            "/v1/runs/api-hitl-13/resume",
            json={
                "approval_id": "apr_api-hitl-13_patch_apply_01_deadbeef",
                "decision": "approve",
                "reviewer": "alice",
            },
        )

        assert response.status_code == 409


class TestCancellation:
    def test_a_parked_run_can_be_cancelled(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        _parked(app_client, request_factory, "api-cancel-1")

        response = app_client.post(
            "/v1/runs/api-cancel-1/cancel",
            json={"reason": "superseded by a newer patch"},
        )

        assert response.status_code in (200, 202)
        body = app_client.get("/v1/runs/api-cancel-1").json()
        assert body["status"] == "cancelled"
        assert body["pending_approval"] is None

    def test_a_cancelled_gate_can_no_longer_be_answered(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """Otherwise the operator would approve a run that is no longer running."""
        approval_id = _parked(app_client, request_factory, "api-cancel-2")
        app_client.post("/v1/runs/api-cancel-2/cancel", json={})

        assert _resolve(app_client, approval_id).status_code >= 400

    def test_cancelling_an_unknown_run_is_a_404(self, app_client: TestClient) -> None:
        assert app_client.post("/v1/runs/nope/cancel", json={}).status_code == 404

    def test_a_finished_run_cannot_be_cancelled(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """There is nothing left to stop, and reporting success would be a lie."""
        app_client.post("/v1/runs", json=_body(request_factory, "api-cancel-3", auto_resolve=True))

        response = app_client.post("/v1/runs/api-cancel-3/cancel", json={})

        assert response.status_code == 409


class TestTimeTravel:
    """Checkpoint history, point-in-time reads and replay."""

    def test_history_is_addressable(self, app_client: TestClient, request_factory: Any) -> None:
        """Every super-step is a checkpoint; that is what makes replay possible."""
        app_client.post("/v1/runs", json=_body(request_factory, "api-tt-1", auto_resolve=True))

        response = app_client.get("/v1/threads/api-tt-1/history")

        assert response.status_code == 200
        entries = response.json()["items"]
        assert len(entries) > 3
        assert all(entry["checkpoint_id"] for entry in entries)
        assert [entry["step"] for entry in entries] == sorted(
            (entry["step"] for entry in entries), reverse=True
        )

    def test_history_is_paginated(self, app_client: TestClient, request_factory: Any) -> None:
        """A run with many gates must not return an unbounded history."""
        app_client.post("/v1/runs", json=_body(request_factory, "api-tt-1b", auto_resolve=True))

        full = app_client.get("/v1/threads/api-tt-1b/history").json()["count"]
        limited = app_client.get("/v1/threads/api-tt-1b/history", params={"limit": 2}).json()

        assert limited["count"] == 2
        assert full > 2

    def test_state_can_be_read_as_of_a_checkpoint(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The question a reviewer actually asks is "what did you see then?"."""
        app_client.post("/v1/runs", json=_body(request_factory, "api-tt-2", auto_resolve=True))
        history = app_client.get("/v1/threads/api-tt-2/history").json()["items"]
        early = history[-1]["checkpoint_id"]

        response = app_client.get(f"/v1/threads/api-tt-2/checkpoints/{early}")

        assert response.status_code in (200, 202)
        body = response.json()
        assert body["run_id"] == "api-tt-2"
        assert body["checkpoint_id"] == early

    def test_an_unknown_checkpoint_is_a_404_not_the_head(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """Returning the head would be worse than an error.

        A UI that scrubs a run and asks for a checkpoint that does not exist must
        be told, or it shows the operator the final state for a request about
        the first iteration — and they believe it.
        """
        app_client.post("/v1/runs", json=_body(request_factory, "api-tt-3", auto_resolve=True))

        response = app_client.get("/v1/threads/api-tt-3/checkpoints/1-0-0-nope")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "checkpoint_not_found"

    def test_an_unknown_checkpoint_is_distinct_from_an_unknown_run(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The two demand opposite client behaviour: fix the id versus resubmit.

        Reporting a bad checkpoint id as a missing run would make a client's
        "the run is gone, start over" branch fire against a run that exists.
        """
        app_client.post("/v1/runs", json=_body(request_factory, "api-tt-3b", auto_resolve=True))

        bad_checkpoint = app_client.get("/v1/threads/api-tt-3b/checkpoints/9-9-9-nope")
        bad_run = app_client.get("/v1/threads/never-ran/history")

        assert bad_checkpoint.status_code == bad_run.status_code == 404
        assert bad_checkpoint.json()["error"]["code"] == "checkpoint_not_found"
        assert bad_run.json()["error"]["code"] == "run_not_found"

    def test_a_replay_appends_a_branch_instead_of_erasing_the_original(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """A replay must not destroy what it is replaying.

        LangGraph forks on the same thread, so the branch shares the run id and
        the history grows. That is only safe because the original checkpoints stay
        individually addressable — a design where a replay rewrote the thread
        would make the audit trail unfalsifiable, and this is the assertion that
        would notice.
        """
        app_client.post("/v1/runs", json=_body(request_factory, "api-tt-4", auto_resolve=True))
        before = app_client.get("/v1/threads/api-tt-4/history").json()["items"]
        original_head = before[0]["checkpoint_id"]

        response = app_client.post(
            "/v1/threads/api-tt-4/replay",
            json={
                "checkpoint_id": before[1]["checkpoint_id"],
                "reason": "the reviewer rejected iteration 2",
            },
        )

        assert response.status_code in (200, 201, 202), response.text
        assert response.json()["run_id"] == "api-tt-4"

        after = app_client.get("/v1/threads/api-tt-4/history").json()["items"]
        assert len(after) > len(before)
        assert {entry["checkpoint_id"] for entry in before} <= {
            entry["checkpoint_id"] for entry in after
        }
        # The pre-replay head is still readable as itself, not as the branch.
        assert app_client.get(f"/v1/threads/api-tt-4/checkpoints/{original_head}").status_code in (
            200,
            202,
        )

    def test_two_checkpoints_can_be_compared(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """``"what changed between these two steps?"`` answered in one call."""
        app_client.post("/v1/runs", json=_body(request_factory, "api-tt-5", auto_resolve=True))
        history = app_client.get("/v1/threads/api-tt-5/history").json()["items"]

        response = app_client.get(
            "/v1/threads/api-tt-5/checkpoints",
            params={"from": history[0]["checkpoint_id"], "to": history[2]["checkpoint_id"]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["changed_count"] == len(body["changed"])
        assert "review" in body["unchanged"] or "review" in {c["channel"] for c in body["changed"]}

    def test_the_history_of_an_unknown_run_is_a_404(self, app_client: TestClient) -> None:
        assert app_client.get("/v1/threads/never-ran/history").status_code == 404


class TestWebSocket:
    """Push updates, so a UI does not have to poll."""

    def test_every_terminal_status_ends_a_stream(self) -> None:
        """A terminal status the socket layer does not know about hangs a client.

        ``TERMINAL_EVENTS`` and the engine's status vocabulary are maintained in
        two files, and nothing forced them to agree. Adding a status on the
        engine side alone is the easy mistake, and its symptom is not a crash: the
        run ends, its event arrives, the subscription sees an event that is not
        in the set it is filtering on, and the client holds a socket open
        forever waiting for a close that will not come. No log line, no failing
        test, one permanently wedged dashboard per interrupted run.

        Asserted as an equality in both directions, so an event with no status
        behind it is caught too.
        """
        from agentic_workflow.persistence.repository import TERMINAL_STATUSES

        assert {f"run.{status}" for status in TERMINAL_STATUSES} == TERMINAL_EVENT_NAMES

    def test_a_run_stream_delivers_lifecycle_events(
        self, fast_ws: TestClient, request_factory: Any
    ) -> None:
        """The socket is the reason an operator can watch a run at all.

        Events must arrive in order and the stream must close on a terminal
        event: a socket that stays open after ``run.completed`` keeps a client
        waiting for an answer that is never coming.
        """
        with fast_ws.websocket_connect("/ws/runs/api-ws-1") as socket:
            assert socket.receive_json()["event"] == "stream.open"

            fast_ws.post("/v1/runs", json=_body(request_factory, "api-ws-1", auto_resolve=True))

            frames = _read_until(socket, TERMINAL_EVENT_NAMES)

        names = [frame["event"] for frame in frames]
        assert "run.started" in names
        assert "node.completed" in names
        assert names[-1] in TERMINAL_EVENT_NAMES
        assert names.index("run.started") < len(names) - 1

    def test_per_node_progress_names_the_node(
        self, fast_ws: TestClient, request_factory: Any
    ) -> None:
        """``node.completed`` without the node name is a progress bar, not a trace.

        An operator debugging a slow run needs to know *which* agent to look at,
        and a single "working..." frame tells them nothing.
        """
        with fast_ws.websocket_connect("/ws/runs/api-ws-1b") as socket:
            socket.receive_json()
            fast_ws.post("/v1/runs", json=_body(request_factory, "api-ws-1b", auto_resolve=True))

            frames = _read_until(socket, TERMINAL_EVENT_NAMES)

        nodes = {f["node"] for f in frames if f["event"] == "node.completed"}
        assert {"triage", "programmer", "reviewer", "tester", "reporter"} <= nodes

    def test_the_firehose_carries_every_run(
        self, fast_ws: TestClient, request_factory: Any
    ) -> None:
        """A dashboard opens one socket, not one per run."""
        with fast_ws.websocket_connect("/ws/events") as socket:
            assert socket.receive_json()["event"] == "stream.open"

            fast_ws.post("/v1/runs", json=_body(request_factory, "api-ws-2", auto_resolve=True))

            frames = _read_until(socket, {"run.started"})

        assert {frame["run_id"] for frame in frames if frame["event"] == "run.started"} == {
            "api-ws-2"
        }

    def test_a_parked_run_does_not_close_the_stream(
        self, fast_ws: TestClient, request_factory: Any
    ) -> None:
        """The next thing that happens to a parked run is the human's decision.

        Closing on ``run.parked`` would force the client to reconnect to receive
        the very answer it is waiting for — and reconnecting is exactly what a
        browser does not do reliably.
        """
        with fast_ws.websocket_connect("/ws/runs/api-ws-3") as socket:
            socket.receive_json()
            _parked(fast_ws, request_factory, "api-ws-3")

            parked = _read_until(socket, {"run.parked"})
            # Still open: the heartbeat that follows is only sent to a live
            # stream, so receiving one *is* the assertion.
            after = _read_until(socket, {"heartbeat"})

        assert parked[-1]["event"] == "run.parked"
        assert parked[-1]["pending_approval"]
        assert after[-1]["event"] == "heartbeat"

    def test_a_replay_of_a_known_run_opens_with_a_snapshot(
        self, fast_ws: TestClient, request_factory: Any
    ) -> None:
        """Reconnecting must not mean starting from nothing.

        An operator who reloads the page has lost nothing: the stream opens with
        the run's current state, so the UI does not have to fetch it separately
        and risk showing a projection that is already stale.
        """
        fast_ws.post("/v1/runs", json=_body(request_factory, "api-ws-4", auto_resolve=True))

        with fast_ws.websocket_connect("/ws/runs/api-ws-4") as socket:
            assert socket.receive_json()["event"] == "stream.open"
            snapshot = socket.receive_json()

        assert snapshot["event"] == "stream.snapshot"
        assert snapshot["run_id"] == "api-ws-4"
        assert snapshot["status"] == "completed"

    def test_subscribing_to_an_unknown_run_is_not_an_error(
        self, fast_ws: TestClient, request_factory: Any
    ) -> None:
        """The run may not have been submitted yet, and the socket is still useful.

        A client that opens a stream before POSTing the run must not be dropped:
        the first event it receives is the run starting.
        """
        with fast_ws.websocket_connect("/ws/runs/not-submitted-yet") as socket:
            assert socket.receive_json()["event"] == "stream.open"

            fast_ws.post("/v1/runs", json=_body(request_factory, "not-submitted-yet"))

            frames = _read_until(socket, {"run.started"})

        assert frames[-1]["run_id"] == "not-submitted-yet"

    def test_a_wildcard_subscription_survives_a_terminal_run(
        self, fast_ws: TestClient, request_factory: Any
    ) -> None:
        """A dashboard socket must not be closed by one run finishing.

        Closing on the first terminal event would make a single-socket dashboard
        show exactly one run and then go dark.
        """
        with fast_ws.websocket_connect("/ws/events") as socket:
            socket.receive_json()
            fast_ws.post("/v1/runs", json=_body(request_factory, "api-ws-5", auto_resolve=True))
            _read_until(socket, TERMINAL_EVENT_NAMES)

            fast_ws.post("/v1/runs", json=_body(request_factory, "api-ws-6", auto_resolve=True))
            frames = _read_until(socket, {"run.started"})

        assert "api-ws-6" in {frame.get("run_id") for frame in frames}


class TestRateLimiting:
    def test_the_budget_is_enforced_and_answers_429(self, checkpointer: Any) -> None:
        """A client that cannot be throttled can exhaust the engine's budget.

        The limit is applied per router rather than per handler, so a new
        endpoint cannot accidentally ship unthrottled.
        """
        from agentic_workflow.api.app import create_app
        from agentic_workflow.config import load_settings, reset_settings_cache
        from agentic_workflow.services.engine import WorkflowEngine

        reset_settings_cache()
        settings = load_settings(
            environment="development",
            llm_provider="echo",
            postgres_enabled=False,
            hitl_enabled=True,
            log_level="WARNING",
            api_rate_limit_per_minute=3,
        )
        engine = WorkflowEngine(settings, checkpointer=checkpointer)
        try:
            with TestClient(create_app(settings, engine=engine, configure_logs=False)) as client:
                codes = [client.get("/v1/runs").status_code for _ in range(6)]
        finally:
            reset_settings_cache()

        assert codes[:3] == [200, 200, 200]
        assert codes[3:] == [429, 429, 429]

    def test_a_throttled_response_asks_the_client_to_wait(self, checkpointer: Any) -> None:
        """``429`` without a hint is a dead end for the client that hit it."""
        from agentic_workflow.api.app import create_app
        from agentic_workflow.config import load_settings, reset_settings_cache
        from agentic_workflow.services.engine import WorkflowEngine

        reset_settings_cache()
        settings = load_settings(
            environment="development",
            llm_provider="echo",
            postgres_enabled=False,
            hitl_enabled=True,
            log_level="WARNING",
            api_rate_limit_per_minute=1,
        )
        engine = WorkflowEngine(settings, checkpointer=checkpointer)
        try:
            with TestClient(create_app(settings, engine=engine, configure_logs=False)) as client:
                client.get("/v1/runs")
                response = client.get("/v1/runs")
        finally:
            reset_settings_cache()

        assert response.status_code == 429
        assert response.json()["error"]["code"] == "rate_limited"
        assert response.json()["error"]["retryable"] is True

    def test_probes_are_not_throttled(self, app_client: TestClient) -> None:
        """A fleet of orchestrators polling probes must not exhaust the budget."""
        assert [app_client.get("/health/live").status_code for _ in range(30)] == [200] * 30


class TestDocumentation:
    def test_the_openapi_document_is_servable(self, app_client: TestClient) -> None:
        """A generated client is the cheapest integration test there is."""
        response = app_client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "agentic-workflow control plane"
        assert "/v1/runs" in schema["paths"]
        assert "/v1/approvals/{approval_id}/resolve" in schema["paths"]

    def test_every_route_is_documented(self, app_client: TestClient) -> None:
        """An undocumented endpoint is one nobody can use correctly.

        The summaries are written, not generated, precisely so a client author
        can decide whether a status code means what they think it means.
        """
        schema = app_client.get("/openapi.json").json()

        undocumented = [
            path
            for path, operations in schema["paths"].items()
            for method, operation in operations.items()
            if method in ("get", "post", "delete", "put", "patch") and not operation.get("summary")
        ]

        assert undocumented == []

    def test_the_202_outcome_is_advertised(self, app_client: TestClient) -> None:
        """A parked run is the *normal* answer; it must be in the contract.

        A client author reading only the status codes would otherwise treat
        ``202`` as an error case and invent a retry loop for a healthy run.
        """
        schema = app_client.get("/openapi.json").json()
        responses = schema["paths"]["/v1/runs"]["post"]["responses"]

        assert "202" in responses
        assert "human" in responses["202"]["description"].lower()

    def test_the_wire_schemas_forbid_unknown_fields(self, app_client: TestClient) -> None:
        """Pinned in the contract itself, not only in the handlers.

        A client SDK generated from this document then rejects typos too, which
        is the point of having the contract at all.
        """
        schema = app_client.get("/openapi.json").json()
        body = schema["components"]["schemas"]["StartRunRequest"]

        assert body.get("additionalProperties") is False


class TestSubmittedSourceIsPreserved:
    """The wire layer must not undo the domain's byte-exactness guarantee."""

    def test_indentation_survives_validation(self) -> None:
        """Leading whitespace is the payload.

        Stripping it turns every nested block into a syntax error and every
        review into a report about code the author never wrote. The domain
        declares a ``Verbatim`` alias for exactly this, and the wire model sets
        ``str_strip_whitespace`` globally — so this is the one place the two
        could disagree.
        """
        request = StartRunRequest(
            run_id="api-verbatim-1",
            request_id="req-verbatim",
            title="Fix the total",
            files=[{"path": "checkout/total.py", "content": BUGGY_SOURCE}],
        )

        assert request.files[0].content == BUGGY_SOURCE
        assert "    result = 0" in request.files[0].content

    def test_prose_fields_are_still_trimmed(self) -> None:
        """The contrast that makes the exception above deliberate.

        A title padded with whitespace is a display bug; a source file padded
        with whitespace is a syntax error. Treating them alike would fix one and
        cause the other.
        """
        request = StartRunRequest(
            run_id="api-verbatim-2",
            request_id="req-verbatim",
            title="  Fix the total  ",
        )

        assert request.title == "Fix the total"

    def test_a_submitted_file_reaches_the_run_untouched(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """End to end: the gate shows a path taken from the submitted file.

        The approval's payload names the file the agents worked on, so a wrong
        or stripped file would surface as a gate pointing at nothing.
        """
        _parked(app_client, request_factory, "api-verbatim-3")

        approvals = app_client.get("/v1/approvals").json()["items"]

        assert approvals[0]["run_id"] == "api-verbatim-3"
        assert approvals[0]["diff_preview"]
        assert "checkout/total.py" in approvals[0]["diff_preview"] or approvals[0]["payload"]
