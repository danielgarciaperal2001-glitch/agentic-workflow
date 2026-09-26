"""Builders for the domain objects the test suite needs.

Kept out of ``conftest.py`` on purpose: fixtures are for *setup*, and these are
*values*. A test that imports a builder is explicit about what data it depends
on, which is what makes a failing test readable.

The payloads here are deliberately non-trivial — real file contents, several
acceptance criteria, a metadata bag — because the agents behave differently on
``title="x"`` than on a request that resembles production traffic, and a test
suite that only exercises the degenerate case tests the wrong thing.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import count
from typing import Any

from agentic_workflow.domain.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    Decision,
    ReviewRequest,
    SourceFile,
)

#: Monotonic suffix so every generated ``run_id`` is a distinct thread key.
_COUNTER = count(1)

#: A source file with a real defect, so the reviewer has something to find.
BUGGY_SOURCE = '''\
"""Order total calculation."""


def total(items: list[dict]) -> float:
    """Sum the price of every item.

    >>> total([{"price": 10.0}, {"price": 5.5}])
    15.5
    """
    result = 0
    for item in items:
        result = result + float(item["price"])
    return result
'''

#: The same file with the defect fixed.
FIXED_SOURCE = '''\
"""Order total calculation."""

from decimal import Decimal


def total(items: list[dict]) -> Decimal:
    """Sum the price of every item, without binary float drift.

    >>> total([{"price": 10.0}, {"price": 5.5}])
    Decimal('15.5')
    """
    result = Decimal("0")
    for item in items:
        result += Decimal(str(item["price"]))
    return result
'''


def unique_run_id(prefix: str = "run") -> str:
    """Return a run id that has not been used in this process.

    Args:
        prefix: Readable prefix for the identifier.

    Returns:
        An identifier matching the domain's safe alphabet.
    """
    return f"{prefix}-{next(_COUNTER):04d}"


def make_request(
    run_id: str | None = None,
    *,
    title: str = "Fix floating point drift in order totals",
    description: str = (
        "The checkout service accumulates prices with binary floats, so large "
        "orders drift by cents. Return exact monetary values instead."
    ),
    files: list[SourceFile] | None = None,
    acceptance_criteria: list[str] | None = None,
    constraints: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReviewRequest:
    """Build a realistic :class:`ReviewRequest`.

    Args:
        run_id: Explicit run id, or a unique one when omitted.
        title: Subject line.
        description: Problem statement handed to the agents.
        files: Files under review; defaults to a single buggy module.
        acceptance_criteria: Conditions the report must satisfy.
        constraints: Hard limits the agents must respect.
        metadata: Correlation data.

    Returns:
        A validated :class:`~agentic_workflow.domain.schemas.ReviewRequest`.
    """
    return ReviewRequest(
        run_id=run_id or unique_run_id(),
        request_id="PR-1042",
        title=title,
        description=description,
        language="python",
        files=files
        if files is not None
        else [SourceFile(path="checkout/total.py", content=BUGGY_SOURCE)],
        acceptance_criteria=acceptance_criteria
        or [
            "The result must be exact for two-decimal currency values.",
            "Existing callers of `total` must keep working.",
        ],
        constraints=constraints or ["Do not add third-party dependencies."],
        metadata=metadata or {"repo": "acme/checkout", "author": "team-payments"},
    )


def make_approval(
    *,
    run_id: str | None = None,
    stage: str = "patch_apply",
    node: str = "apply_patch",
    decision_options: list[Decision] | None = None,
    expires_in_seconds: float | None = None,
) -> ApprovalRequest:
    """Build an :class:`ApprovalRequest` for service-level tests.

    Args:
        run_id: Owning run; generated when omitted.
        stage: Coarse stage label.
        node: Graph node that requested approval.
        decision_options: Offered actions.
        expires_in_seconds: Window length; ``None`` means no expiry.

    Returns:
        A validated approval request with a deterministic id.
    """
    from datetime import timedelta

    from agentic_workflow.domain.schemas import utcnow

    resolved = run_id or unique_run_id()
    return ApprovalRequest.new(
        run_id=resolved,
        node=node,
        stage=stage,
        title="Apply the proposed patch to checkout/total.py?",
        approval_id=f"apr_{resolved}_{stage}_00_testfixture",
        rationale="A human must approve writes to production source files.",
        payload={"files_changed": ["checkout/total.py"]},
        options=decision_options or [Decision.APPROVE, Decision.EDIT, Decision.REJECT],
        confidence=0.62,
        expires_at=(
            utcnow() + timedelta(seconds=expires_in_seconds)
            if expires_in_seconds is not None
            else None
        ),
        diff_preview="--- a/checkout/total.py\n+++ b/checkout/total.py\n-    result = 0.0\n+    result = Decimal('0')",
    )


def make_decision(
    approval_id: str,
    *,
    decision: Decision = Decision.APPROVE,
    reviewer: str = "alice",
    comment: str = "looks right",
    payload: dict[str, Any] | None = None,
) -> ApprovalDecision:
    """Build an :class:`ApprovalDecision`.

    Args:
        approval_id: The gate being answered.
        decision: The verdict.
        reviewer: Who decided.
        comment: Justification.
        payload: Replacement content, required for ``edit``.

    Returns:
        A validated decision.
    """
    return ApprovalDecision(
        approval_id=approval_id,
        decision=decision,
        reviewer=reviewer,
        comment=comment,
        payload=payload or {},
    )


#: Ready-made factory aliases, for tests that want to import them directly.
request_factory: Callable[..., ReviewRequest] = make_request
approval_factory: Callable[..., ApprovalRequest] = make_approval
decision_factory: Callable[..., ApprovalDecision] = make_decision

__all__ = [
    "BUGGY_SOURCE",
    "FIXED_SOURCE",
    "approval_factory",
    "decision_factory",
    "make_approval",
    "make_decision",
    "make_request",
    "request_factory",
    "unique_run_id",
]
