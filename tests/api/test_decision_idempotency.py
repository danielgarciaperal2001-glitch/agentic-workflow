"""Two endpoints, one mistake, one answer.

The control plane offers two ways to answer a gate: the approval's own resource
at ``POST /v1/approvals/{approval_id}/resolve`` and the run's at
``POST /v1/runs/{run_id}/resume``. They exist for different clients, which is
legitimate. They must not disagree about what happened, because a client cannot
tell which one it is talking to when it needs to know.

A duplicate submission is the case that forces the question. It arrives when a
button is double-clicked, when a request times out and is retried, or when a
queue redelivers. Whatever the client does next follows from the answer it gets,
so the answer has to name the real cause.

These tests pin that agreement at the HTTP boundary, where a client actually
sees it.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
import pytest

pytestmark = pytest.mark.api


def _submit(client: TestClient, request_factory: Any, run_id: str) -> str:
    """Submit a run and return the id of the gate it parked on.

    Args:
        client: The bound test client.
        request_factory: The ``ReviewRequest`` factory fixture.
        run_id: Identifier for the run under test.

    Returns:
        The pending approval id.
    """
    body = request_factory(run_id=run_id).model_dump(mode="json")
    body.pop("content_hash", None)
    response = client.post("/v1/runs", json=body)
    assert response.status_code == 202, response.text
    return response.json()["pending_approval"]["approval_id"]


def _approve_body(approval_id: str) -> dict[str, Any]:
    """Build an approve decision for *approval_id*.

    Args:
        approval_id: The gate being answered.

    Returns:
        A request body.
    """
    return {
        "approval_id": approval_id,
        "decision": "approve",
        "reviewer": "alice",
        "comment": "looks right",
    }


def _resolve(client: TestClient, approval_id: str) -> Any:
    """Answer a gate through the approval resource.

    Args:
        client: The bound test client.
        approval_id: The gate being answered.

    Returns:
        The response.
    """
    return client.post(
        f"/v1/approvals/{approval_id}/resolve",
        json={"decision": "approve", "reviewer": "alice"},
    )


def _resume(client: TestClient, run_id: str, approval_id: str) -> Any:
    """Answer a gate through the run resource.

    Args:
        client: The bound test client.
        run_id: The parked run.
        approval_id: The gate being answered.

    Returns:
        The response.
    """
    return client.post(f"/v1/runs/{run_id}/resume", json=_approve_body(approval_id))


class TestDuplicateSubmission:
    """A re-sent decision is reported identically by both entry points."""

    def test_a_replayed_decision_reads_as_already_applied(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The run endpoint says the decision was applied, not that it went stale.

        The run has moved to a *different* gate by the time the duplicate
        arrives, so the graph's id check fires and reports a mismatch between the
        pending gate and the submitted id. That is arithmetically true and
        practically misleading: the decision was not too old, it was accepted
        when it was current. A client that believes it expired fetches the new
        gate and answers it too, and one approval has silently become two.

        The wording of the response is the fix. ``approval_already_resolved``
        plus the run id sends the client to the audit trail; ``invalid_state``
        plus a mismatch sends it back to the approval form.
        """
        approval_id = _submit(app_client, request_factory, "api-dup-1")

        first = _resume(app_client, "api-dup-1", approval_id)
        assert first.status_code == 202, first.text

        replay = _resume(app_client, "api-dup-1", approval_id)
        assert replay.status_code == 409, replay.text
        assert replay.json()["error"]["code"] == "approval_already_resolved"
        assert replay.json()["error"]["run_id"] == "api-dup-1"

    def test_both_entry_points_answer_identically(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The same mistake produces the same code through either endpoint.

        The two resources describe the same decision from different angles, and a
        client that cannot predict which error it will get cannot write the retry
        logic. The disagreement is also a structural warning: it means one path
        carries a guard the other lacks, which is how a check ends up written
        down and not applied.
        """
        through_run = _submit(app_client, request_factory, "api-dup-2")
        assert _resume(app_client, "api-dup-2", through_run).status_code == 202
        from_run = _resume(app_client, "api-dup-2", through_run)

        through_approval = _submit(app_client, request_factory, "api-dup-3")
        assert _resolve(app_client, through_approval).status_code == 202
        from_approval = _resolve(app_client, through_approval)

        assert from_run.status_code == from_approval.status_code == 409
        assert (
            from_run.json()["error"]["code"]
            == from_approval.json()["error"]["code"]
            == "approval_already_resolved"
        )

    def test_a_replayed_decision_does_not_reach_the_next_gate(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The audit log records the decision once, however often it is sent.

        The log is the answer to "who approved this". An entry per submission
        would make a single approval look like a room full of approvers, and
        would count toward a gate's decision total as though it were independent
        agreement.
        """
        approval_id = _submit(app_client, request_factory, "api-dup-4")
        _resume(app_client, "api-dup-4", approval_id)
        _resume(app_client, "api-dup-4", approval_id)

        audit = app_client.get("/v1/approvals/by-run/api-dup-4/audit")
        assert audit.status_code == 200, audit.text
        assert audit.json()["count"] == 1

    def test_a_decision_for_another_runs_gate_is_still_refused(
        self, app_client: TestClient, request_factory: Any
    ) -> None:
        """The replay guard does not weaken the stale-answer check.

        The two guards answer different questions — "you already said this" and
        "this is not the gate you think it is" — and the second must still fire
        for a decision that was never recorded. Otherwise the new check would
        have swallowed it: any id mismatch now has to be attributed to the
        right cause.
        """
        mine = _submit(app_client, request_factory, "api-dup-5")
        theirs = _submit(app_client, request_factory, "api-dup-6")

        response = _resume(app_client, "api-dup-5", theirs)

        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] != "approval_already_resolved"
        # The run is untouched: another run's decision never reached it.
        assert app_client.get("/v1/runs/api-dup-5").json()["status"] == "waiting_human"
        assert mine != theirs
