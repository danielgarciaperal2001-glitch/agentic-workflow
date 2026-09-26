"""HTTP routers for the control plane.

One module per resource group. Each exposes a single ``router`` that
:func:`agentic_workflow.api.app.create_app` mounts, so adding an endpoint never
touches the application factory.
"""

from __future__ import annotations

from agentic_workflow.api.routers import approvals, events, health, runs, threads

#: Every router, in the order the application mounts them. Exported so a test can
#: assert the control plane's surface without instantiating the app.
ALL_ROUTERS = (health.router, runs.router, approvals.router, threads.router, events.router)

__all__ = ["ALL_ROUTERS", "approvals", "events", "health", "runs", "threads"]
