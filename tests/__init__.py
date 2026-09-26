"""Test suite for :mod:`agentic_workflow`.

Layered by marker so a contributor can run the cheap subset while iterating and
the full suite before pushing:

* :mod:`tests.unit` — pure logic, no I/O.
* :mod:`tests.integration` — engine, graph and checkpointer wired together.
* :mod:`tests.api` — HTTP and WebSocket control plane.
* :mod:`tests.postgres` — durable checkpointer (auto-skipped without a database).
* :mod:`tests.eval` — automated quality suite (opt-in via ``-m eval``).
"""
