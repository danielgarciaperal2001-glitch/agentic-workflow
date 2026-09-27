"""Deterministic, offline LLM provider.

``EchoLLM`` is not a toy: it is the reference implementation the whole test
suite runs against. It answers by *deriving* a schema-valid object from the
prompt using deterministic heuristics and lightweight static analysis, which
gives us three properties the project depends on:

* **CI needs no credentials, no network and no spend.**
* **Every run is byte-reproducible**, so evaluation results are comparable and
  regressions are unambiguous.
* **The graph is exercised end-to-end** (routing, HITL, checkpointing, evals)
  even in environments where no real model is available.

It is also genuinely useful as a *baseline* in the evaluation suite: comparing
a real model against a deterministic null model is how you prove the model adds
value rather than just adding words.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from agentic_workflow.llm.base import Completion, LLMClient, Message, Usage
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Rule catalogue keyed by a token that appears in the system prompt. The graph
#: names its agent in the system message, which is how the echo provider knows
#: which schema-shaped answer to synthesise.
_RULES: tuple[tuple[str, str], ...] = (
    ("task_brief", "task_brief"),
    ("patch", "patch"),
    ("reviewresult", "review"),
    ("testreport", "test_report"),
    ("finalreport", "report"),
)

#: Lightweight security heuristics. These are intentionally conservative: the
#: provider is a null model, so it should be *sensitive* but not noisy.
_SECURITY_RULES: tuple[tuple[str, str, str], ...] = (
    (r"\beval\s*\(", "Use of eval() enables arbitrary code execution", "security"),
    (r"\bexec\s*\(", "Use of exec() enables arbitrary code execution", "security"),
    # SQL built by interpolating a value into the statement. The pattern used to
    # require a literal `{}`, which is the *only* form that no longer occurs in
    # practice: an f-string carries a name (`{term}`) and a `.format()` carries
    # a `{}` outside the quotes entirely. Matching only the literal brace form
    # meant the rule scored zero on the single most common injection shape.
    (
        r"(?i)\b(sql|query|statement)\w*\s*(?:=|:)\s*f?[\"'][^\n]*\{",
        "SQL built by string interpolation is vulnerable to injection",
        "security",
    ),
    (
        r"(?i)\b(sql|query|statement)\w*[^\n]*\.(?:format|replace)\s*\(",
        "SQL built by string formatting is vulnerable to injection",
        "security",
    ),
    # A hard-coded credential. The pattern used to start with `\b`, which cannot
    # match after an underscore because `_` is a word character — so the rule
    # missed `PAYMENTS_API_KEY = "sk_live_..."`, the single most conventional
    # way to write a secret in Python, while matching `api_key = "..."`.
    (
        (
            r"(?i)(?<![a-z0-9])(?:[a-z0-9]*_)?"
            r"(?:password|passwd|secret|token|api_key|apikey)"
            r"\s*[:=]\s*[\"'][^\"']{6,}[\"']"
        ),
        "Hard-coded credential detected",
        "security",
    ),
    (r"\bos\.system\s*\(", "shell=True invokes a shell; pass an argument list instead", "security"),
    (r"(?i)verify\s*=\s*False", "TLS verification disabled enables MITM attacks", "security"),
    (r"except\s*:\s*(?:\n\s*)?pass", "Bare except swallows errors silently", "maintainability"),
    (r"#\s*type:\s*ignore", "Unchecked type suppression hides real defects", "maintainability"),
    (r"(?i)todo|fixme|xxx", "Unresolved TODO left in the change", "maintainability"),
    (
        r"\bwhile\s+True\s*:\s*(?:\n\s*)?(?:[^\n]*\n){0,4}?(?:\n\s*)?\s*(?:break|return)?\s*$",
        "Possible infinite loop",
        "correctness",
    ),
)

#: A payload label is treated as a file path only if it looks like one. A
#: description block is scanned by the rules but never cited, because
#: attributing a finding to a file that does not exist is the exact hallucination
#: the evaluation suite exists to catch.
_PATH_LIKE = re.compile(r"[\w./-]+\.\w+")


class EchoLLM(LLMClient):
    """Deterministic provider that synthesises schema-shaped answers.

    The implementation is a *rule router*: it inspects the system prompt for the
    target schema name, then runs the corresponding synthesis function. Every
    answer is a valid instance of the requested model, so the downstream
    validation path is always exercised.

    Example:
        --------
        >>> import asyncio
        >>> from agentic_workflow.llm.echo import EchoLLM
        >>> client = EchoLLM(model="echo-1")
        >>> reply = asyncio.run(client.complete([Message(role="user", content="hello")]))
        >>> reply.model
        'echo-1'
    """

    def __init__(self, *, model: str = "echo-1", latency_ms: float = 1.0, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.latency_ms = latency_ms

    async def _complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002 - accepted for signature compatibility
    ) -> Completion:
        """Return a deterministic response derived from the prompt."""
        prompt = "\n".join(m.content for m in messages)
        target = self._detect_target(prompt, response_format)
        payload = self._synthesise(target, prompt)

        # Report a plausible token count so budget dashboards are not empty.
        usage = Usage(
            prompt_tokens=_approx_tokens(prompt),
            completion_tokens=_approx_tokens(json.dumps(payload)),
            calls=1,
        )
        log.debug("llm.echo", target=target, model=self.model, chars=len(prompt))
        return Completion(
            content=json.dumps(payload, sort_keys=True),
            model=self.model,
            usage=usage,
            latency_ms=self.latency_ms,
            raw={"provider": "echo", "target": target},
        )

    # -------------------------------------------------------- synthesis #
    @staticmethod
    def _detect_target(prompt: str, response_format: dict[str, Any] | None) -> str:
        """Infer which schema the caller expects from prompt or response format.

        Matching is done on a *normalised* form (lowercased, underscores removed)
        so ``TaskBrief``, ``task_brief`` and ``TASK-BRIEF`` all resolve to the
        same target.
        """
        if response_format:
            name = response_format.get("json_schema", {}).get("name")
            if name:
                needle = str(name).lower().replace("_", "")
                for token, target in _RULES:
                    if token.replace("_", "") in needle:
                        return target
                return str(name)
        normalised = prompt.lower().replace("_", "")
        best: tuple[int, str] = (0, "generic")
        for token, target in _RULES:
            needle = token.replace("_", "")
            if needle in normalised and len(needle) > best[0]:
                best = (len(needle), target)
        return best[1]

    def _synthesise(self, target: str, prompt: str) -> dict[str, Any]:
        """Build the payload for the detected target."""
        iteration = _extract_int(prompt, "iteration", default=1)
        # A stable per-prompt jitter keeps confidence values varied (so the
        # escalation threshold is actually exercised) yet reproducible.
        jitter = _stable_unit_interval(prompt)

        builders = {
            "task_brief": self._build_task_brief,
            "patch": self._build_patch,
            "review": self._build_review,
            "test_report": self._build_test_report,
            "report": self._build_report,
        }
        builder = builders.get(target)
        if builder is None:
            return {"summary": f"echo: unsupported target {target!r}"}
        return builder(prompt, iteration, jitter)

    @staticmethod
    def _build_task_brief(prompt: str, iteration: int, jitter: float) -> dict[str, Any]:
        title = _extract_quoted(prompt, "title") or _first_line(prompt) or "Unnamed change"
        return {
            "objective": f"Review and harden: {title}",
            "scope": _extract_list(prompt, "acceptance_criteria")
            or ["correctness", "security", "maintainability"],
            "out_of_scope": ["unrelated refactors", "dependency upgrades"],
            "risks": ["behavioural regression", "unvalidated input at trust boundary"],
            "success_criteria": ["no critical/high findings", "all tests pass"],
            "estimated_effort": "small" if iteration <= 2 else "medium",
            "confidence": round(0.55 + 0.35 * jitter, 2),
        }

    @staticmethod
    def _build_patch(prompt: str, iteration: int, jitter: float) -> dict[str, Any]:
        files = re.findall(r"(?:^|\n)\s*file:\s*(\S+)", prompt) or _paths_in(prompt)
        findings = _finding_titles(prompt)
        body = "\n".join(
            [
                f"# iteration {iteration}: address {len(findings)} finding(s)",
                *(f"- {title}" for title in findings[:10]),
            ]
        )
        diff_lines = [
            "--- a/" + (files[0] if files else "README.md"),
            "+++ b/" + (files[0] if files else "README.md"),
        ]
        diff_lines += [f"+# {line}" for line in body.splitlines()]
        return {
            "diff": "\n".join(diff_lines),
            "files_changed": files[:20] or ["README.md"],
            "summary": body,
            "strategy": "direct" if iteration == 1 else "incremental",
            "confidence": round(0.5 + 0.4 * jitter, 2),
        }

    @staticmethod
    def _build_review(prompt: str, iteration: int, jitter: float) -> dict[str, Any]:
        # Later iterations "see" fewer issues: the loop must converge, otherwise
        # the iteration cap is the only thing stopping the graph.
        detected = _synthesise_findings(prompt)
        remaining = max(0, min(3 - (iteration - 1), len(detected)))
        findings = detected[:remaining]
        blocking = [f["id"] for f in findings if f["severity"] in {"high", "critical"}]
        verdict = "approved" if not findings else "changes_requested"
        return {
            "verdict": verdict,
            "findings": findings,
            "summary": f"iteration {iteration}: {len(findings)} finding(s), "
            f"{len(blocking)} blocking",
            "confidence": round(0.6 + 0.3 * jitter, 2),
            "blocking_findings": blocking,
        }

    @staticmethod
    def _build_test_report(
        prompt: str,  # noqa: ARG004 - builders share one uniform signature
        iteration: int,
        jitter: float,
    ) -> dict[str, Any]:
        total = 12 + iteration
        failed = 0 if iteration >= 2 else max(0, 2 - iteration)
        return {
            "passed": failed == 0,
            "total": total,
            "failed": failed,
            "skipped": 0,
            "coverage": round(0.6 + 0.3 * jitter, 2),
            "failing_tests": [f"test_case_{i}" for i in range(failed)],
            "generated_tests": [f"test_generated_{i}" for i in range(3)],
            "regressions": [],
            "summary": f"{total - failed}/{total} passed",
        }

    @staticmethod
    def _build_report(prompt: str, iteration: int, jitter: float) -> dict[str, Any]:
        findings = _finding_titles(prompt)
        body_lines = [
            "# Automated Review Report",
            "",
            "## Decision",
            "Approved" if iteration >= 2 else "Approved with comments",
            "",
            "## Findings",
            *(f"- {title}" for title in (findings[:10] or ["none"])),
        ]
        citations = _paths_in(prompt)[:10]
        return {
            "report_id": _stable_id(prompt),
            "markdown": "\n".join(body_lines),
            "decision": "approved" if iteration >= 2 else "changes_requested",
            "findings": [],
            "metrics": {"iterations": float(iteration), "confidence": round(0.6 + 0.3 * jitter, 2)},
            "citations": citations,
        }


# --------------------------------------------------------------------------- #
# Prompt-scraping helpers
# --------------------------------------------------------------------------- #
def _extract_int(text: str, key: str, *, default: int) -> int:
    """Read an integer from ``key: N`` / ``key=N`` / ``"key": N`` patterns."""
    match = re.search(rf"{key}\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else default


def _extract_quoted(text: str, key: str) -> str:
    """Read a quoted value following *key* in the prompt."""
    match = re.search(rf'{key}\s*[:=]\s*"([^"]*)"', text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _first_line(text: str) -> str:
    """Return the first non-empty, non-boilerplate line of the prompt."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "You are", "Respond with")):
            return stripped[:120]
    return ""


def _extract_list(text: str, key: str) -> list[str]:
    """Read a bracketed list of strings following *key*."""
    match = re.search(rf"{key}\s*[:=]\s*\[(.*?)\]", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return [item.strip().strip("\"'") for item in match.group(1).split(",") if item.strip()]


def _paths_in(prompt: str) -> list[str]:
    """Extract repo-relative file paths mentioned in the prompt."""
    return list(dict.fromkeys(re.findall(r"(?m)^\s*(?:file|path):\s*(\S+\.\w+)\s*$", prompt)))


def _finding_titles(prompt: str) -> list[str]:
    """Extract finding titles from a serialised ``findings:`` block."""
    block = re.search(r"findings:\s*(\[.*?\]|\n\s+-.*?)(?=\n\w+:|\Z)", prompt, flags=re.DOTALL)
    if not block:
        return []
    return [m.strip()[:120] for m in re.findall(r'"title"\s*:\s*"([^"]+)"', block.group(1))]


def _attributed_lines(prompt: str) -> list[tuple[str, int, str]]:
    """Split the prompt into ``(file, line_in_file, text)`` triples.

    Only lines inside a delimited payload whose label is a real file path are
    returned. That restriction is the point, and it is what makes a finding
    trustworthy:

    * A rule cannot fire on the *prompt's own* text. The word ``token`` in an
      acceptance criterion must not read as a hard-coded credential, and a
      description that quotes the vulnerable pattern must not be reported as
      containing it.
    * Every finding can name the file it is about, which is the difference
      between a review a human can act on and a title they have to go hunting
      for — and the evaluation suite's ``path_grounding`` and
      ``citation_coverage`` metrics can then check the claim instead of
      assuming it.

    Args:
        prompt: The rendered prompt.

    Returns:
        One triple per scannable line, in document order. Line numbers are
        relative to the file, which is what a reader needs to jump to it.
    """
    triples: list[tuple[str, int, str]] = []
    path: str | None = None
    offset = 0
    for line in prompt.splitlines():
        if line.startswith("<<<UNTRUSTED_PAYLOAD:"):
            label = line.split(":", 1)[1].strip()
            path = label if _PATH_LIKE.fullmatch(label) else None
            offset = 0
            continue
        if line.startswith("UNTRUSTED_PAYLOAD>>>"):
            path = None
            continue
        if path is not None:
            offset += 1
            triples.append((path, offset, line))
    return triples


def _synthesise_findings(prompt: str) -> list[dict[str, Any]]:
    """Run the static rule set over the prompt and build a finding list.

    Rules that fire on several lines are collapsed into a single finding (the
    first occurrence wins) so the count reflects distinct problems rather than
    raw match volume. Findings are returned in first-seen order, which makes the
    result stable across runs.

    Args:
        prompt: The rendered prompt containing the code under review.

    Returns:
        Schema-shaped finding dicts, each with a unique identity.
    """
    seen: dict[str, tuple[str, str, int]] = {}
    for path, lineno, line in _attributed_lines(prompt):
        for pattern, title, category in _SECURITY_RULES:
            if re.search(pattern, line):
                seen.setdefault(title, (category, path, lineno))

    if not seen:
        return [
            {
                "id": "",
                "title": "No obvious issue detected; consider adding a regression test",
                "detail": (
                    "The rule set found nothing in this change. That is the absence "
                    "of a match, not proof of correctness: a regex cannot read "
                    "intent, so treat this as 'not covered', never as 'clean'."
                ),
                "severity": "medium",
                "category": "testing",
                "file": None,
                "line": None,
                "recommendation": "Apply the documented fix and add a regression test.",
                "confidence": 0.2,
            }
        ]

    out: list[dict[str, Any]] = []
    for title, (category, path, lineno) in seen.items():
        out.append(
            {
                "id": "",
                "title": title,
                "detail": f"Detected by the deterministic rule set at {path}:{lineno}.",
                "severity": "high" if category == "security" else "medium",
                "category": category,
                "file": path,
                "line": lineno,
                "recommendation": "Apply the documented fix and add a regression test.",
                "confidence": 0.7,
            }
        )
    return out


def _approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for usage reporting."""
    return max(1, len(text) // 4)


def _stable_unit_interval(*parts: str) -> float:
    """Deterministic float in ``[0, 1)`` derived from *parts*.

    Uses a cryptographic digest rather than :func:`hash` because the builtin is
    salted per process (PYTHONHASHSEED) and would break reproducibility.
    """
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _stable_id(*parts: str) -> str:
    """Deterministic short identifier derived from *parts*."""
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


__all__ = ["EchoLLM"]
