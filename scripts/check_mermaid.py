#!/usr/bin/env python3
"""Validate a Mermaid block from the README against the real parser.

The README's diagram is the first thing a reader sees and the only thing that is
not checked by any test, so a syntax error in it ships silently. This posts the
block to mermaid.ink and reports the parser's own error message, which is far
more useful than a guess about which construct is wrong.

Usage: check_mermaid.py [path-to-.md]
Exits 0 when the diagram parses, 1 otherwise.
"""

from __future__ import annotations

# S310 (urllib without a scheme allowlist) is disabled for this file as a whole.
# The only URL it ever opens is ENDPOINT below — a literal https origin — with
# the diagram appended as base64. Neither the scheme nor the host can be
# influenced by the file being checked, and a per-call allowlist would have to
# be repeated on both `Request` and `urlopen` to silence the same false
# positive twice.
# ruff: noqa: S310
import base64
import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://mermaid.ink/svg/"


def extract(markdown: str) -> str:
    """Return the first fenced ```mermaid block.

    Args:
        markdown: The file's contents.

    Returns:
        The diagram source.

    Raises:
        SystemExit: If there is no mermaid block.
    """
    match = re.search(r"```mermaid\n(.*?)\n```", markdown, re.DOTALL)
    if match is None:
        raise SystemExit("no ```mermaid block found")
    return match.group(1)


def validate(code: str) -> tuple[bool, str]:
    """Ask the Mermaid parser to render *code*.

    Args:
        code: Mermaid source.

    Returns:
        ``(ok, detail)`` where detail is the parser's message on failure or the
        rendered SVG's length on success.
    """
    payload = json.dumps({"code": code})
    token = urllib.parse.quote(base64.b64encode(payload.encode()).decode(), safe="")
    # An explicit User-Agent: the default urllib one is rejected by the CDN in
    # front of the renderer with a 1010, which reads like a network failure and
    # is not one.
    request = urllib.request.Request(
        ENDPOINT + token, headers={"User-Agent": "agentic-workflow-ci/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return True, f"{len(response.read())} bytes of SVG"
    except urllib.error.HTTPError as exc:
        return False, exc.read().decode(errors="replace").strip()
    except urllib.error.URLError as exc:
        return False, f"network unavailable: {exc.reason}"


def main(argv: list[str]) -> int:
    """Validate the diagram and report.

    Args:
        argv: Optional path to a Markdown file.

    Returns:
        ``0`` if the diagram parses, ``1`` otherwise.
    """
    path = pathlib.Path(argv[1] if len(argv) > 1 else "README.md")
    code = extract(path.read_text(encoding="utf-8"))
    ok, detail = validate(code)
    if ok:
        print(f"{path}: mermaid diagram is valid ({detail})")
        return 0
    print(f"{path}: mermaid diagram is INVALID\n{detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
