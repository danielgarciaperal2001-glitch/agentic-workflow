"""Quality evaluation for the review workflow.

What lives here
---------------

* :mod:`evals.datasets` — the golden set: reviewed changes with the defects a
  human reviewer was known to catch, and the validation that keeps a case
  honest about its own code.
* :mod:`evals.metrics` — deterministic, offline metrics computed from the
  artefacts a run already produced. No judge model, no network, no cost, and
  the same score twice for the same commit.
* :mod:`evals.external` — optional Ragas and DeepEval scorers, imported only
  when asked for, for the question the native metrics deliberately cannot
  answer: is this finding actually *right*?
* :mod:`evals.runners` — drives the real graph over the dataset and aggregates
  the scores.
* :mod:`evals.report` — renders a suite report for a terminal and for a PR
  comment.

Deliberately not part of the ``src`` package
--------------------------------------------

The evaluation harness is a development and CI tool, not a runtime dependency.
Shipping it inside ``agentic_workflow`` would make ``import evals`` a promise
the installed library cannot keep without also shipping a golden dataset and a
report renderer to every production host. The ``Dockerfile`` copies this
directory alongside ``src`` so the image can run ``awf eval``, and ``awf
eval`` adds the repository root to ``sys.path`` before importing — so the
harness works from a checkout without being a dependency of the wheel.
"""

from __future__ import annotations

__all__ = ["datasets", "external", "metrics", "report", "runners"]
