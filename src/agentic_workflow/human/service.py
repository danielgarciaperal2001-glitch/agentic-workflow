"""Approval inbox: the service behind the human-in-the-loop endpoints.

A parked run is only useful if a human can *find* it. This module is the query
and command layer over that:

* :meth:`ApprovalService.inbox` — every actionable approval, oldest first.
* :meth:`ApprovalService.resolve` — answer one, with idempotency, expiry
  enforcement and signature verification.
* :meth:`ApprovalService.expire_stale` — sweep approvals nobody answered.

Idempotency is the subtle part. Operators double-click; retrying a request
after a network blip is normal; a browser tab can be open for hours. Answering
the same approval twice must therefore be *safe* and *visible* rather than a
double-applied write, which is what :func:`assert_not_already_resolved` guards
in the graph and what the decision log here records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentic_workflow.domain.schemas import ApprovalDecision, ApprovalRequest, Decision
from agentic_workflow.errors import (
    ApprovalAlreadyResolvedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalRejectedError,
    InvalidStateError,
    RunNotFoundError,
)
from agentic_workflow.human.gates import (
    assert_not_already_resolved,
    run_id_from_approval_id,
    sign_decision,
    verify_decision,
)
from agentic_workflow.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    # The engine depends on this package, so the dependency is declared for typing
    # only. At runtime the service is handed an engine by its caller.
    from agentic_workflow.services.engine import RunOutcome, WorkflowEngine

log = get_logger(__name__)

#: Fields an :class:`ApprovalDecision` declares, derived from the model so the two
#: cannot drift. A decision-log entry is a strict superset — it adds the gate's
#: stage and the response latency — and rebuilding a decision from one therefore
#: means projecting, not validating.
_DECISION_FIELDS: frozenset[str] = frozenset(ApprovalDecision.model_fields)


@dataclass(frozen=True, slots=True)
class ApprovalView:
    """An approval request enriched with the run context a human needs.

    Attributes:
        request: The approval itself.
        run_id: Owning run.
        run_status: Current status of that run.
        iteration: Feedback-loop iteration the gate belongs to.
        available_actions: Decisions this approval accepts, refined by policy.
        expires_in_seconds: Remaining window; negative once expired.
        already_resolved: Whether a decision was already recorded for this id.
    """

    request: ApprovalRequest
    run_status: str
    iteration: int
    available_actions: tuple[str, ...]
    expires_in_seconds: float
    already_resolved: bool

    @property
    def approval_id(self) -> str:
        """Convenience accessor for the approval identifier."""
        return self.request.approval_id

    @property
    def is_expired(self) -> bool:
        """Whether the approval window has closed."""
        return self.expires_in_seconds <= 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the approval inbox UI."""
        return {
            **self.request.model_dump(mode="json"),
            "run_status": self.run_status,
            "iteration": self.iteration,
            "available_actions": list(self.available_actions),
            "expires_in_seconds": round(self.expires_in_seconds, 3),
            "already_resolved": self.already_resolved,
        }


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Outcome of answering an approval.

    Attributes:
        run_id: The run that was resumed.
        outcome: The run's state after the resume, when the resume succeeded.
        decision: The validated, signed decision that was applied.
        replayed: ``True`` when the decision was a no-op because it had already
            been applied — the client gets a success and a clear signal.
    """

    run_id: str
    decision: ApprovalDecision
    outcome: RunOutcome | None = None
    replayed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "run_id": self.run_id,
            "decision": self.decision.model_dump(mode="json"),
            "replayed": self.replayed,
            "status": self.outcome.status if self.outcome else "unknown",
            "pending_approval": (
                self.outcome.pending.model_dump(mode="json")
                if self.outcome and self.outcome.pending
                else None
            ),
        }


class ApprovalService:
    """Query and resolve the approvals blocking parked runs.

    Example:
        --------
        >>> service = ApprovalService(engine)  # doctest: +SKIP
        >>> inbox = await service.inbox()  # doctest: +SKIP
        >>> result = await service.resolve(  # doctest: +SKIP
        ...     inbox[0].approval_id, decision="approve", reviewer="alice"
        ... )
    """

    def __init__(self, engine: WorkflowEngine) -> None:
        """Bind the service to an engine.

        Args:
            engine: The engine that owns the runs and the graph.
        """
        self._engine = engine

    @property
    def engine(self) -> WorkflowEngine:
        """The bound engine."""
        return self._engine

    # -------------------------------------------------------------- read #
    async def inbox(self, *, run_id: str | None = None) -> list[ApprovalView]:
        """List every actionable approval, oldest first.

        Args:
            run_id: Restrict the inbox to a single run.

        Returns:
            Enriched approval views. Expired approvals are included but flagged,
            because hiding them would leave a human wondering where their run went.
        """
        return [
            self._to_view(outcome.pending, outcome, resolved_ids=_resolved_ids(outcome))
            for outcome in await self._engine.parked_outcomes(run_id)
            if outcome.pending is not None
        ]

    async def get(self, approval_id: str) -> ApprovalView:
        """Fetch one approval by id.

        Reads the one run that could own the approval instead of sweeping the inbox
        to find it. An approval id names its own run —
        ``apr_<run>_<stage>_<iteration>_<digest>`` — so the sweep was answering a
        question the id had already answered. Measured against a real PostgreSQL
        checkpointer with 40 parked runs, ``GET /v1/approvals/{id}`` cost 363 ms
        because it built all 40 views to return one; this is one read.

        It is also *more* correct than the sweep it replaces, not merely faster.
        The sweep asked the engine for its known runs, which is a per-process
        registry, so a run a sibling replica had parked was invisible to it. This
        reads the checkpoint directly, so it finds the approval wherever it lives.

        A parseable id is answered from that one run and only that run, with no
        sweep on failure. That is not a shortcut taken for speed: the id names the
        run it was minted for, the approval lives in that run's state, and the
        digest is derived from run, stage and iteration — so an id naming a run
        that is missing, or parked on a later gate, cannot be current for any other
        run. Both are 404s, and scanning would only re-derive that.

        The sweep is kept for the one case it can actually answer: an id that does
        not parse, which is what :func:`run_id_from_approval_id`'s ``None`` means —
        "ask the store", not "no such run".

        Args:
            approval_id: The approval to fetch.

        Returns:
            The enriched view.

        Raises:
            ApprovalNotFoundError: If no run is currently parked on that id.
        """
        owner = run_id_from_approval_id(approval_id)
        if owner is None:
            return await self._get_by_scan(approval_id)
        try:
            outcome = await self._engine.status(owner)
        except RunNotFoundError as exc:
            raise ApprovalNotFoundError(
                "no pending approval with that id", approval_id=approval_id
            ) from exc
        if outcome.pending is not None and outcome.pending.approval_id == approval_id:
            return self._to_view(outcome.pending, outcome, resolved_ids=_resolved_ids(outcome))
        # Either the run holds a different gate now — so this id is stale rather
        # than unknown — or it is not parked at all. Both are a 404, and returning
        # the run's *current* gate here would hand the caller a different approval
        # than the one they asked about.
        raise ApprovalNotFoundError("no pending approval with that id", approval_id=approval_id)

    async def _get_by_scan(self, approval_id: str) -> ApprovalView:
        """Locate an approval by building the inbox.

        The fallback for :meth:`get` when the id does not name its own run, which
        is the only case a scan can answer.

        Args:
            approval_id: The approval to find.

        Returns:
            The enriched view.

        Raises:
            ApprovalNotFoundError: If no run is currently parked on that id.
        """
        for view in await self.inbox():
            if view.approval_id == approval_id:
                return view
        raise ApprovalNotFoundError("no pending approval with that id", approval_id=approval_id)

    # ----------------------------------------------------------- resolve #
    async def resolve(
        self,
        approval_id: str,
        *,
        decision: str | Decision,
        reviewer: str,
        comment: str = "",
        payload: dict[str, Any] | None = None,
        context: Any = None,
    ) -> ResolutionResult:
        """Answer an approval and resume the run.

        The decision is signed here rather than in the graph so the audit trail
        records a verifiable artefact even when the resume later fails.

        Args:
            approval_id: The approval being answered.
            decision: ``approve``, ``edit`` or ``reject``.
            reviewer: Who answered. Recorded in the immutable log.
            comment: Free text. Required in spirit for ``reject``.
            payload: Edited content, used with ``decision="edit"``.
            context: Optional per-run runtime context.

        Returns:
            A :class:`ResolutionResult`.

        Raises:
            ApprovalNotFoundError: If the approval is not pending.
            ApprovalExpiredError: If the window has closed.
            ApprovalAlreadyResolvedError: If the approval was already answered.
            ApprovalRejectedError: If the human rejected the action.
            InvalidStateError: If the payload cannot be interpreted.
        """
        # The "already answered" check comes *before* the pending lookup, and the
        # order is load-bearing. Once a gate is answered it is no longer pending,
        # so asking the inbox first made every double-click a 404 — "no such
        # approval" — when the truth is "that approval exists and you already
        # answered it". The two send a client to opposite places: one retries,
        # the other gives up on a run that is still healthy.
        owner = run_id_from_approval_id(approval_id)
        if owner is None:
            # Not one of our derived ids. Fall back to searching, so a foreign or
            # hand-built id still gets the same answer rather than a false 404.
            try:
                owner = await self._locate_run(approval_id)
            except ApprovalNotFoundError:
                owner = ""
        if owner:
            logged = [
                entry
                for entry in await self._decisions_for(owner)
                if entry.get("approval_id") == approval_id
            ]
            if logged:
                raise ApprovalAlreadyResolvedError(
                    "this approval was already resolved",
                    approval_id=approval_id,
                    run_id=owner,
                    decided_at=logged[0].get("decided_at"),
                )

        view = await self.get(approval_id)
        request = view.request

        if view.is_expired:
            raise ApprovalExpiredError(
                "approval window elapsed before a decision was recorded",
                approval_id=approval_id,
                run_id=request.run_id,
            )

        resolved = self._build_decision(request, decision, reviewer, comment, payload)
        assert_not_already_resolved(resolved, await self._decisions_for(request.run_id))

        try:
            outcome = await self._engine.resume(
                request.run_id, resolved.model_dump(mode="json"), context=context
            )
        except ApprovalRejectedError:
            # A rejection is a legitimate outcome, not a transport failure: record
            # it and let the caller see the run ended, rather than 500-ing.
            log.info(
                "approval.rejected",
                approval_id=approval_id,
                run_id=request.run_id,
                reviewer=reviewer,
            )
            raise
        return ResolutionResult(
            run_id=request.run_id, decision=resolved, outcome=outcome, replayed=False
        )

    async def replay(
        self, approval_id: str, *, reviewer: str, decision: str | Decision = "approve"
    ) -> ResolutionResult:
        """Re-submit a decision that was already applied, idempotently.

        Clients that retry after a timeout need this: without it, every retry is
        an error, and operators learn to fear the button.

        The recorded decision wins. A client that retries with a *different*
        verdict is telling us its state is stale, and silently returning the old
        decision would hide that — so a conflict is raised instead.

        Args:
            approval_id: The approval being replayed.
            reviewer: Who is replaying, for the conflict check.
            decision: The verdict the client believes it sent, for the conflict
                check. Defaults to ``approve``.

        Returns:
            A result with ``replayed=True`` and no graph invocation.

        Raises:
            ApprovalNotFoundError: If the approval is not known at all.
            ApprovalAlreadyResolvedError: If the retry disagrees with the log.
        """
        wanted = str(getattr(decision, "value", decision)).lower()
        run_id = await self._locate_run(approval_id)
        recorded = await self._decisions_for(run_id)
        for entry in recorded:
            if entry.get("approval_id") != approval_id:
                continue
            # Compare against the *log entry*, not a re-validated model. The entry
            # is the record this endpoint exists to honour, and it is deliberately
            # a superset of the decision — it carries the stage and the response
            # latency. Validating it as an `ApprovalDecision` failed on those extra
            # keys and turned every retry into a 500, on the one path whose whole
            # purpose is to recover from a lost answer.
            mismatch = {
                key: {"recorded": recorded_value, "received": received}
                for key, recorded_value, received in (
                    ("decision", str(entry.get("decision", "")).lower(), wanted),
                    ("reviewer", str(entry.get("reviewer", "")), reviewer),
                )
                if recorded_value != received
            }
            if mismatch:
                raise ApprovalAlreadyResolvedError(
                    "replayed decision does not match the recorded one",
                    approval_id=approval_id,
                    run_id=run_id,
                    conflict=mismatch,
                )
            return ResolutionResult(
                run_id=run_id,
                decision=ApprovalDecision.model_validate(
                    {key: value for key, value in entry.items() if key in _DECISION_FIELDS}
                ),
                outcome=await self._engine.status(run_id),
                replayed=True,
            )
        raise ApprovalNotFoundError("this approval was never resolved", approval_id=approval_id)

    # ---------------------------------------------------------- sweeper #
    async def expire_stale(self) -> list[str]:
        """Return the ids of approvals whose window has closed.

        The runs stay parked — closing an approval is a *human* decision, and the
        service refuses to make it on their behalf. The sweeper exists so the UI
        can show an operator which runs are stuck and for how long.

        Returns:
            The expired approval ids.
        """
        expired = [view.approval_id for view in await self.inbox() if view.is_expired]
        if expired:
            log.warning("approval.expired_pending", count=len(expired), approval_ids=expired[:20])
        return expired

    async def stats(self) -> dict[str, Any]:
        """Return inbox counters, authoritative.

        This is the ``GET /v1/approvals/stats`` payload, so it is computed the way
        the rest of the inbox is: every known run read, every view built. That
        costs O(runs) and it is the right trade here, because a human or a
        dashboard asks occasionally and acts on the answer — including on
        ``expired`` and ``resolved_pending``, which need the approval object and
        the decision log and cannot be derived from the registry.

        The readiness probe wants the same shape continuously and cannot afford it;
        it uses :meth:`probe_stats` instead. Keeping the two apart is deliberate:
        a probe that quietly returned the cheap numbers would be reporting zeros
        for two counters a dashboard trusts.
        """
        views = await self.inbox()
        by_stage: dict[str, int] = {}
        for view in views:
            by_stage[view.request.stage] = by_stage.get(view.request.stage, 0) + 1
        return {
            "pending": len(views),
            "expired": sum(1 for v in views if v.is_expired),
            "resolved_pending": sum(1 for v in views if v.already_resolved),
            "by_stage": by_stage,
        }

    async def probe_stats(self) -> dict[str, Any]:
        """Return inbox counters in the shape a readiness probe can afford.

        Counted from the run registry, with no store access at all, because the
        caller is polled every few seconds whether or not anything is wrong.

        Building the views in order to count them is not a small thing. The inbox
        sweeps every known run and then reads each parked one a second time to
        enrich it. Measured against a real PostgreSQL checkpointer, per probe:

            parked runs      old (``stats``)    new (``probe_stats``)
                       25            181 ms                2.3 ms
                      100            728 ms                2.3 ms
                      800          5,788 ms                4.7 ms

        ``PROBE_TIMEOUT_SECONDS`` is 5, so the old cost turned a healthy instance
        into a 503 — and so out of the load balancer — at around 700 parked runs,
        which is a queue an ordinary backlog reaches. At that moment the store
        check was also a no-op, so the expensive path was incidentally the only
        thing still noticing an outage.

        Two counters are reported as zero because they cannot be had for free:
        ``expired`` lives on the approval object and ``resolved_pending`` needs the
        decision log, and the registry keeps neither. Zero is a lie if a reader
        takes it literally, which is why this method is not ``stats`` and why
        :meth:`stats` remains the one a human reads.

        The count is per-process. A run parked by another replica is not in this
        registry, exactly as :meth:`~agentic_workflow.services.engine.WorkflowEngine.list_runs`
        already documents; fanning out to every replica to produce a global count
        is the cost being removed.
        """
        counts = self._engine.parked_summary()
        return {
            "pending": counts["pending"],
            "expired": counts["expired"],
            "resolved_pending": counts["resolved_pending"],
            "by_stage": counts["by_stage"],
        }

    async def verify_log(self, run_id: str) -> list[dict[str, Any]]:
        """Re-verify every recorded decision signature for a run.

        The decision log lives inside the graph state, so it inherits the trust
        level of whatever can write to the checkpoint store. Re-deriving the HMAC
        turns "we recorded that alice approved this" into something checkable,
        which is the whole point of signing.

        Args:
            run_id: The run to audit.

        Returns:
            One entry per recorded decision with a ``verified`` flag. ``None``
            means the deployment has no signing secret configured, so the decision
            could not be verified either way.
        """
        secret = self._signing_secret()
        findings: list[dict[str, Any]] = []
        for entry in await self._decisions_for(run_id):
            approval_id = str(entry.get("approval_id") or "")
            verified: bool | None
            if not secret:
                verified = None
            else:
                try:
                    verified = verify_decision(ApprovalDecision.model_validate(entry), secret)
                except Exception as exc:  # a malformed row is a finding
                    log.warning("audit.unreadable", run_id=run_id, approval_id=approval_id)
                    findings.append(
                        {
                            "approval_id": approval_id,
                            "decision": entry.get("decision"),
                            "reviewer": entry.get("reviewer"),
                            "verified": False,
                            "reason": f"unreadable record: {type(exc).__name__}",
                        }
                    )
                    continue
            findings.append(
                {
                    "approval_id": approval_id,
                    "decision": entry.get("decision"),
                    "reviewer": entry.get("reviewer"),
                    "verified": verified,
                    "reason": None if verified is not False else "signature mismatch",
                }
            )
        return findings

    # -------------------------------------------------------- internals #
    def _signing_secret(self) -> str:
        """Return the secret decisions are signed with, or ``""`` when unset.

        Delegates to :meth:`Settings.resolved_signing_secret`. The duplicate
        copy that used to live here is what allowed the service and the agent
        context to disagree about the key, and therefore about whether a
        decision was signed at all.
        """
        return self._engine.settings.resolved_signing_secret()

    def _to_view(
        self,
        request: ApprovalRequest,
        outcome: RunOutcome,
        *,
        resolved_ids: set[str],
    ) -> ApprovalView:
        """Enrich a raw approval request with run context."""
        from agentic_workflow.domain.schemas import utcnow

        remaining = (request.expires_at - utcnow()).total_seconds() if request.expires_at else 0.0
        return ApprovalView(
            request=request,
            run_status=outcome.status,
            iteration=int(outcome.state.get("iteration", 0)),
            available_actions=tuple(option.value for option in request.options),
            expires_in_seconds=remaining,
            already_resolved=request.approval_id in resolved_ids,
        )

    def _build_decision(
        self,
        request: ApprovalRequest,
        decision: str | Decision,
        reviewer: str,
        comment: str,
        payload: dict[str, Any] | None,
    ) -> ApprovalDecision:
        """Validate and sign the decision.

        Raises:
            InvalidStateError: If the decision value or reviewer is unusable, or
                an ``edit`` arrives without the content to apply.
        """
        if not reviewer or not reviewer.strip():
            raise InvalidStateError("a reviewer is required", approval_id=request.approval_id)
        try:
            resolved = Decision(str(getattr(decision, "value", decision)).lower())
        except ValueError as exc:
            raise InvalidStateError(
                f"unknown decision {decision!r}; expected one of {[d.value for d in Decision]}",
                approval_id=request.approval_id,
            ) from exc

        if resolved is Decision.EDIT and not (payload or comment):
            raise InvalidStateError(
                "an `edit` decision must carry the replacement content in `payload`",
                approval_id=request.approval_id,
            )

        built = ApprovalDecision(
            approval_id=request.approval_id,
            decision=resolved,
            reviewer=reviewer.strip(),
            comment=comment,
            payload=payload or {},
        )
        secret = self._signing_secret()
        if self._engine.settings.hitl_require_signature and secret:
            built.signature = sign_decision(built, secret)
        return built

    async def _decisions_for(self, run_id: str) -> list[dict[str, Any]]:
        """Read the recorded decision log for a run."""
        try:
            outcome = await self._engine.status(run_id)
        except RunNotFoundError:
            return []
        return list(outcome.decisions)

    async def _locate_run(self, approval_id: str) -> str:
        """Find the run an approval id belongs to.

        Raises:
            ApprovalNotFoundError: If no run references the id.
        """
        for record in self._engine.registry.list_runs(limit=10_000):
            try:
                outcome = await self._engine.status(record.run_id)
            except RunNotFoundError:
                continue
            for entry in outcome.decisions:
                if entry.get("approval_id") == approval_id:
                    return record.run_id
            if outcome.pending is not None and outcome.pending.approval_id == approval_id:
                return record.run_id
        raise ApprovalNotFoundError("no run references that approval", approval_id=approval_id)


def _resolved_ids(outcome: RunOutcome) -> set[str]:
    """Collect the approval ids already answered in a run's decision log."""
    return {
        str(entry.get("approval_id")) for entry in outcome.decisions if entry.get("approval_id")
    }


__all__ = ["ApprovalService", "ApprovalView", "ResolutionResult"]
