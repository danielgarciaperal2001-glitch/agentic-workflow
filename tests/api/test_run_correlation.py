"""The correlation data a client submits is already stored, and never read back.

``ReviewRequest.metadata`` is documented as "Arbitrary correlation data
propagated end-to-end", and every request fixture in the suite populates it
(``{"repo": "acme/checkout", "author": "team-payments"}``) as though it were
load-bearing. Measured against a real submitted run, it reached nothing:

    state['request'].metadata   {'ticket': 'OPS-77', 'tenant': 'acme', 'attempts': 3}
    registry metadata           {'request_id': 'PR-4242', 'title': 't', 'files': 1,
                                 'content_hash': 'sha256:3115...', 'auto_resolve': False}
    RunOutcome.metadata         attribute does not exist
    GET /v1/runs/{id}           request_id: None, metadata: absent

The interesting part is the first line. The metadata is **not** lost: the graph
seeds ``WorkflowState.request`` with the immutable request, the request carries
the metadata, and the whole thing is checkpointed. So it is already persisted
authoritatively, already survives a restart, and is already readable by any
replica. The defect is that nothing projects it onto the wire.

That is why this fix touches no persistence. The tempting alternative is to
read the *registry's* metadata, which also holds a ``request_id`` and is one
line away. It is the wrong source for two measured reasons: the registry is
per-process, so the field would answer on the replica that served the POST and
go blank everywhere else — and its dict is the engine's own annotation bag, so
projecting it raw hands the client keys it never sent (``auto_resolve``).

The tests below pin the source, because the two designs are indistinguishable
until the registry is removed underneath a live run.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_workflow.domain.schemas import MAX_METADATA_BYTES, MAX_METADATA_KEYS

pytestmark = pytest.mark.api


class TestTheRoundTrip:
    """What a client submits is what a client reads back, under the same names."""

    def test_client_metadata_comes_back_exactly_as_submitted(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """The submitted dict survives the round trip unaltered.

        Key by key, not merely present: a projection that flattened the values
        to strings, or that merged engine annotations in, would still look
        populated and would still be wrong.
        """
        submitted = {"ticket": "OPS-77", "tenant": "acme", "attempts": 3}
        body = request_factory(run_id="corr-1", metadata=submitted).model_dump(mode="json")
        body.pop("content_hash", None)

        started = app_client.post("/v1/runs", json=body)
        assert started.status_code == 202, started.text

        detail = app_client.get("/v1/runs/corr-1").json()
        assert detail["metadata"] == submitted

    def test_the_business_request_id_comes_back(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """A client can read back the ticket or PR it filed the run under.

        Without this a client's own correlation data is write-only: it submits
        ``PR-4242`` and can only ever hold it in its own memory, which is no use
        at all once the run outlives the process that started it.
        """
        body = request_factory(run_id="corr-2").model_dump(mode="json")
        body.pop("content_hash", None)
        body["request_id"] = "PR-4242"

        app_client.post("/v1/runs", json=body)
        assert app_client.get("/v1/runs/corr-2").json()["request_id"] == "PR-4242"

    def test_the_two_request_ids_do_not_collide(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """The HTTP correlation id and the business one stay distinct.

        Both are called ``request_id`` and that is not a naming slip to paper
        over: the middleware binds the HTTP one into the log envelope, and the
        body field is the client's own. The log plane is measured working (19
        of 28 records on a single submitted run carry both ``request_id`` and
        ``run_id``, joined on the gateway's value), so the two planes are live
        at once and a projection that conflated them would quietly break the
        join that already works.
        """
        body = request_factory(run_id="corr-3").model_dump(mode="json")
        body.pop("content_hash", None)

        app_client.post("/v1/runs", json=body, headers={"X-Request-ID": "gateway-abc-123"})

        detail = app_client.get("/v1/runs/corr-3").json()
        assert detail["request_id"] == "PR-1042", "the body field, not the header"
        assert detail["request_id"] != "gateway-abc-123"

    def test_a_run_submitted_without_metadata_reports_an_empty_mapping(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """No metadata is ``{}``, not a missing key and not an error.

        ``metadata`` is optional on the request, so the field has to be present
        for a client that reads it unconditionally. A ``None`` here would push
        the ``or {}`` onto every caller, and the key being absent entirely
        would break a generated client outright.
        """
        body = request_factory(run_id="corr-4").model_dump(mode="json")
        body.pop("content_hash", None)
        body["metadata"] = {}

        app_client.post("/v1/runs", json=body)
        detail = app_client.get("/v1/runs/corr-4").json()
        assert detail["metadata"] == {}


class TestTheSourceIsTheCheckpoint:
    """The projection reads the checkpoint, which is the only durable answer."""

    async def test_the_correlation_survives_the_registry_record_being_deleted(
        self, engine: Any, request_factory: Any
    ) -> None:
        """Drop the per-process record; the correlation data does not go with it.

        This is the test that separates the two designs. Both a
        checkpoint-backed and a registry-backed projection return the right
        answer on the happy path, because the registry does hold a
        ``request_id``. They differ the moment the process that served the POST
        is gone — and the registry is explicitly per-process and rebuilt at
        startup, so that moment is a normal deployment, not an edge case.
        """
        request = request_factory(run_id="corr-5", metadata={"ticket": "OPS-77"})
        outcome = await engine.start(request, metadata={"auto_resolve": False})

        assert outcome.metadata == {"ticket": "OPS-77"}, "read back before the removal"

        # The registry is a rebuildable projection, so removing it models a
        # restart or a read served by a different replica.
        engine._registry.delete("corr-5")

        after = await engine.status("corr-5")
        assert after.metadata == {"ticket": "OPS-77"}, (
            "the correlation data is checkpointed, so losing the per-process "
            f"record must not lose it; got {after.metadata!r}"
        )
        assert after.request_id == request.request_id

    def test_metadata_too_large_to_echo_is_refused(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """A correlation dict that would bloat every read of the run is rejected.

        The bound is part of this change rather than a separate concern: the dict
        was already stored, unbounded, but nothing returned it. Echoing it is
        what turns its size into a property of the read path, so the read path
        has to be what bounds it. Measured before the bound, a 19.5 MiB metadata
        was accepted with ``202`` and came back on the following ``GET``.
        """
        # One key, well over the byte bound: key count alone would let this
        # through, which is the case a naive `max_length` fix misses.
        oversized = {"blob": "x" * (MAX_METADATA_BYTES * 4)}
        body = request_factory(run_id="corr-7").model_dump(mode="json")
        body.pop("content_hash", None)
        body["metadata"] = oversized

        response = app_client.post("/v1/runs", json=body)
        assert response.status_code == 422, response.text
        assert "metadata" in response.text
        assert "4096" in response.text, "the limit is named, so a client can fix it"

    def test_too_many_keys_is_refused_even_when_each_one_is_tiny(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """The key count is its own bound, not a side effect of the byte one.

        Sixty-five one-character keys are 4 KiB of payload nowhere near, so this
        is the case a bytes-only bound silently accepts.
        """
        body = request_factory(run_id="corr-9").model_dump(mode="json")
        body.pop("content_hash", None)
        body["metadata"] = {f"k{i}": 1 for i in range(MAX_METADATA_KEYS + 1)}

        response = app_client.post("/v1/runs", json=body)
        assert response.status_code == 422, response.text
        assert "metadata" in response.text

    def test_a_wide_but_small_dict_is_still_accepted(
        self, app_client: Any, request_factory: Any
    ) -> None:
        """The byte bound does not quietly become a key-count bound.

        Fifty keys of a few bytes each is plausible correlation data and must go
        through, so a fix that only counted keys would be tightening the contract
        for no measured reason.
        """
        body = request_factory(run_id="corr-8").model_dump(mode="json")
        body.pop("content_hash", None)
        body["metadata"] = {f"k{i}": i for i in range(50)}

        response = app_client.post("/v1/runs", json=body)
        assert response.status_code == 202, response.text
        assert app_client.get("/v1/runs/corr-8").json()["metadata"] == body["metadata"]

    async def test_engine_annotations_are_not_handed_to_the_client(
        self, engine: Any, request_factory: Any
    ) -> None:
        """The registry's annotation bag is not the client's metadata.

        ``engine.start`` records ``{"auto_resolve": False}`` in the registry for
        a client that never sent it. Projecting the registry dict raw would
        return that key as though the client had submitted it, which is worse
        than returning nothing: it looks like the round trip works.
        """
        request = request_factory(run_id="corr-6", metadata={"ticket": "OPS-77"})
        outcome = await engine.start(request, metadata={"auto_resolve": False})

        assert "auto_resolve" in engine._registry.find("corr-6").metadata, (
            "the annotation really is in the registry, so this test can fail"
        )
        assert "auto_resolve" not in outcome.metadata
        assert outcome.metadata == {"ticket": "OPS-77"}
