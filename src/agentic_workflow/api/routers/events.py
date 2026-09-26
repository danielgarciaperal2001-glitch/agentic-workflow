"""WebSocket endpoints streaming run events.

The polling alternative (``GET /v1/runs/{id}`` every second) is worse on every
axis that matters: it multiplies checkpoint reads by the number of connected
operators, it makes an approval visible up to a second late, and it offers no
ordering guarantee. A socket gives ordered, push-based delivery with one
connection per operator.

The connection is intentionally *lossy but never wrong*:

* Events are advisory. Every one of them can be recovered over REST, so a dropped
  event costs the UI a refresh, not correctness.
* A heartbeat frame is emitted on every idle tick. Without it, proxies and load
  balancers with an idle timeout silently close healthy sockets and the client
  reconnects in a loop it cannot explain.
* The first frame after ``accept`` is a *snapshot*, not a change. A stream that
  only reports deltas leaves a late subscriber with no idea where the run
  currently is.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

from agentic_workflow.api.deps import websocket_auth
from agentic_workflow.api.events import ANY_RUN, ConnectionLimitError, EventHub, Subscription
from agentic_workflow.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["events"])

#: Events after which a run will not produce another one on its own.
#:
#: A rejection counts: the human said no, the run is over, and a client that kept
#: the socket open would wait for an answer that is never coming.
TERMINAL_EVENTS: frozenset[str] = frozenset(
    {"run.completed", "run.failed", "run.cancelled", "run.rejected"}
)


@router.websocket("/ws/runs/{run_id}")
async def stream_run(
    websocket: WebSocket,
    run_id: str,
    token: str | None = Query(default=None, description="Bearer token, for browsers."),
) -> None:
    """Stream one run's events to a WebSocket client.

    The stream ends when the run reaches a terminal status. A *parked* run keeps
    the socket open, because the next event — the human's decision — arrives on
    it.

    Args:
        websocket: The inbound socket.
        run_id: The run to observe. Use ``*`` for every run.
        token: Bearer token, required when ``api_auth_enabled`` is set. A browser
            ``WebSocket`` cannot set headers, so the token travels in the query
            string; it is checked during the handshake, before any data flows.
    """
    await _serve(websocket, None if run_id == ANY_RUN else run_id, token)


@router.websocket("/ws/events")
async def stream_all(
    websocket: WebSocket,
    token: str | None = Query(default=None, description="Bearer token, for browsers."),
) -> None:
    """Stream every run's events on one socket.

    Intended for dashboards. This is a firehose — a busy deployment emits events
    from every run — so a UI should filter client-side.

    Args:
        websocket: The inbound socket.
        token: Bearer token, required when ``api_auth_enabled`` is set.
    """
    await _serve(websocket, None, token)


# --------------------------------------------------------------------------- #
# Connection lifecycle
# --------------------------------------------------------------------------- #
async def _serve(websocket: WebSocket, run_id: str | None, token: str | None) -> None:
    """Authenticate, subscribe, pump events and clean up on disconnect.

    Args:
        websocket: The inbound socket.
        run_id: Run to observe, or ``None`` for every run.
        token: Bearer token for the handshake.
    """
    try:
        websocket_auth(websocket, token)
    except WebSocketDisconnect:
        # Rejected before the handshake completes, which is the only way to
        # refuse a WebSocket: the client observes a 1008 close.
        return

    hub: EventHub | None = getattr(websocket.app.state, "hub", None)
    engine = getattr(websocket.app.state, "engine", None)
    if hub is None or engine is None:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="service is not ready")
        return

    bucket = run_id or ANY_RUN
    try:
        sub = await hub.subscribe(bucket)
    except ConnectionLimitError as exc:
        log.warning("ws.rejected", run_id=bucket, error=str(exc))
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc)[:120])
        return

    settings = websocket.app.state.settings
    heartbeat = settings.ws_heartbeat_seconds
    send_timeout = settings.ws_send_timeout_seconds

    await websocket.accept()
    log.info("ws.connected", run_id=bucket)
    try:
        await _send(
            websocket,
            {
                "event": "stream.open",
                "run_id": bucket,
                "heartbeat_seconds": heartbeat,
            },
            send_timeout,
        )
        if run_id is not None:
            await _send_snapshot(websocket, engine, run_id, send_timeout)
        await _pump(websocket, sub, heartbeat, send_timeout)
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unsubscribe(sub)
        log.info("ws.closed", run_id=bucket, dropped=sub.dropped)
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()


async def _pump(
    websocket: WebSocket,
    sub: Subscription,
    heartbeat: float,
    send_timeout: float,
) -> None:
    """Forward queued events until the run terminates or the client leaves.

    Args:
        websocket: The live socket.
        sub: This client's subscription.
        heartbeat: Idle interval between keep-alive frames.
        send_timeout: Bound on a single send, so one stuck socket cannot pin the
            event loop.

    Raises:
        WebSocketDisconnect: If the client vanished or a send timed out.
    """
    while True:
        event = await sub.get(timeout=heartbeat)
        if event is None:
            await _send(websocket, {"event": "heartbeat"}, send_timeout)
            continue
        await _send(websocket, event, send_timeout)
        name = str(event.get("event") or "")
        if not sub.is_wildcard and name in TERMINAL_EVENTS:
            # `event=` is the structlog event slot, so the terminal name goes in
            # as the message and the details ride along as fields.
            log.info("ws.terminal", run_id=sub.run_id, terminal_event=name)
            return


async def _send(websocket: WebSocket, payload: dict[str, Any], timeout: float) -> None:
    """Send one JSON frame, bounded so a stalled client cannot pin the loop.

    Args:
        websocket: The live socket.
        payload: The frame to send.
        timeout: Bound on the send, in seconds.

    Raises:
        WebSocketDisconnect: If the client vanished or the send timed out.
    """
    frame = json.dumps(payload, default=str)
    try:
        await asyncio.wait_for(websocket.send_text(frame), timeout=timeout)
    except TimeoutError as exc:
        # A client that accepts the TCP connection but never reads is worse than
        # one that disconnects: the send would block forever and leak a coroutine
        # per attempt. Closing is the only safe answer.
        log.warning("ws.send_timeout", path=websocket.url.path)
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION) from exc


async def _send_snapshot(websocket: WebSocket, engine: Any, run_id: str, timeout: float) -> None:
    """Send the run's current state as the first content frame.

    Args:
        websocket: The live socket.
        engine: The workflow engine.
        run_id: The run being observed.
        timeout: Bound on the send.
    """
    try:
        outcome = await engine.status(run_id)
    except Exception as exc:
        # A brand-new run may not have a checkpoint yet. That is not an error:
        # the client is told the stream is open and will hear the first event.
        log.debug("ws.snapshot_unavailable", run_id=run_id, error=str(exc))
        return
    if websocket.client_state is not WebSocketState.CONNECTED:  # pragma: no cover
        return
    await _send(websocket, {"event": "stream.snapshot", **outcome.to_dict()}, timeout)


__all__ = ["TERMINAL_EVENTS", "router", "stream_all", "stream_run"]
