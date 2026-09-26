"""FastAPI dependencies: the wiring seam between the transport and the engine.

Everything long-lived — the :class:`~agentic_workflow.services.engine.WorkflowEngine`,
the :class:`~agentic_workflow.human.service.ApprovalService` and the
:class:`~agentic_workflow.api.events.EventHub` — is created once in the app's
lifespan and reachable through :func:`get_engine` and friends.

Resolving them from ``request.app.state`` rather than from module globals is
deliberate: it means two ``TestClient`` instances in the same process get two
independent engines, and a dependency override in a test replaces exactly one
seam instead of the whole process.
"""

from __future__ import annotations

from collections import OrderedDict, deque
import hmac
import time
from typing import Annotated, Any, Final

from fastapi import Depends, Header, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from agentic_workflow.api.events import EventHub
from agentic_workflow.config import Settings
from agentic_workflow.errors import RateLimitedError
from agentic_workflow.human.service import ApprovalService
from agentic_workflow.logging import get_logger
from agentic_workflow.services.engine import WorkflowEngine

log = get_logger(__name__)

#: How many distinct clients the rate limiter remembers before evicting.
_RATE_LIMIT_TRACKED_CLIENTS: Final[int] = 10_000


def get_settings_dep(request: Request) -> Settings:
    """Return the settings bound to this application instance.

    Args:
        request: The inbound request.

    Returns:
        The application :class:`~agentic_workflow.config.Settings`.
    """
    return request.app.state.settings  # type: ignore[no-any-return]


def get_engine(request: Request) -> WorkflowEngine:
    """Return the workflow engine owned by this application instance.

    Args:
        request: The inbound request.

    Returns:
        The engine.

    Raises:
        HTTPException: 503 if the engine was never started, which means the
            lifespan failed and the process is not fit to serve traffic.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="workflow engine is not initialised",
        )
    return engine  # type: ignore[no-any-return]


def get_approvals(request: Request) -> ApprovalService:
    """Return the approval inbox service.

    Args:
        request: The inbound request.

    Returns:
        The approval service.
    """
    service = getattr(request.app.state, "approvals", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="approval service is not initialised",
        )
    return service  # type: ignore[no-any-return]


def get_hub(request: Request) -> EventHub:
    """Return the WebSocket event hub.

    Args:
        request: The inbound request.

    Returns:
        The event hub.
    """
    hub = getattr(request.app.state, "hub", None)
    if hub is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="event hub is not initialised",
        )
    return hub  # type: ignore[no-any-return]


def get_app_state(request: Request) -> Any:
    """Return the raw application state, for the readiness probe."""
    return request.app.state


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
async def require_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Enforce the bearer token when ``api_auth_enabled`` is set.

    The comparison is constant-time. A control plane whose only secret is a
    static token has no business leaking its length or prefix through timing,
    and ``hmac.compare_digest`` costs one import.

    Authentication is off by default so ``docker compose up`` gives a working
    local stack. :meth:`Settings._validate_cross_field` refuses to start with
    ``api_auth_enabled=true`` and no token, so the "secure but broken" state is
    not reachable.

    Args:
        request: The inbound request, used to read settings.
        authorization: Raw ``Authorization`` header.

    Raises:
        HTTPException: 401 when a token is required and absent or wrong.
    """
    settings: Settings = request.app.state.settings
    if not settings.api_auth_enabled:
        return
    expected = settings.api_auth_token
    if expected is None:  # pragma: no cover - blocked by the settings validator
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication is not configured"
        )
    header = authorization or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(token, expected.get_secret_value()):
        log.warning(
            "api.auth_failed",
            path=request.url.path,
            request_id=getattr(request.state, "request_id", None),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Fixed-window rate limiter keyed by client identity.

    Deliberately in-process and deliberately simple. A real deployment puts a
    shared limiter at the edge; duplicating a distributed algorithm here would
    add a dependency and a failure mode without improving correctness, because
    the ``max_parallel_runs`` ceiling already bounds the real resource.

    Attributes:
        limit: Requests allowed per window. ``0`` disables the limiter.
        window: Window length in seconds.
        hits: Per-key request timestamps, least-recent first.
    """

    def __init__(self, limit: int, window: float = 60.0) -> None:
        """Build the limiter.

        Args:
            limit: Requests allowed per window. ``0`` disables it.
            window: Window length in seconds.
        """
        self.limit = limit
        self.window = window
        self.hits: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, key: str, *, now: float | None = None) -> None:
        """Record a request and reject it when the budget is spent.

        Args:
            key: Client identity.
            now: Current epoch seconds; injectable for tests.

        Raises:
            RateLimitedError: When the client exceeded its budget.
        """
        if self.limit <= 0:
            return
        moment = time.monotonic() if now is None else now
        bucket = self.hits.get(key)
        if bucket is None:
            if len(self.hits) >= _RATE_LIMIT_TRACKED_CLIENTS:
                self.hits.popitem(last=False)
            self.hits[key] = deque([moment])
            return
        while bucket and moment - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.limit:
            raise RateLimitedError(
                "request budget exceeded",
                retry_after_seconds=round(self.window - (moment - bucket[0]), 3),
                limit=self.limit,
            )
        bucket.append(moment)


def client_key(
    request: Request | WebSocket,
    *,
    authorization: str | None = None,
) -> str:
    """Derive the rate-limiter identity of a caller.

    Uses the authenticated identity when there is one, falling back to the
    forwarded client address. Keying on the *token* rather than the IP matters
    behind a proxy, where every request otherwise appears to come from the load
    balancer and a single busy user throttles everybody.

    Args:
        request: The inbound request or socket.
        authorization: Raw ``Authorization`` header, when available.

    Returns:
        A stable client key.
    """
    if authorization:
        return f"token:{authorization[-16:]}"
    forwarded = None
    if isinstance(request, Request):
        forwarded = request.headers.get("x-forwarded-for")
    host = request.client.host if request.client else "unknown"
    return f"ip:{(forwarded or host).split(',')[0].strip()}"


def rate_limited(request: Request) -> None:
    """Apply the per-client request budget.

    The limiter itself lives on ``app.state`` rather than on ``Settings``: the
    settings object is frozen and shared, and a limiter is per-process mutable
    state. Lazy creation also means a test that swaps the budget on a fresh app
    gets a fresh limiter instead of inheriting a stale window.

    The typed :class:`~agentic_workflow.errors.RateLimitedError` is raised
    rather than a bare ``HTTPException``: the domain handler turns it into the
    standard envelope, which reports it as ``retryable`` — the one thing a
    throttled client needs to know — and keeps the limit and the remaining window
    that the limiter computed. A hand-rolled 429 answered "do not retry" while
    simultaneously attaching a ``Retry-After`` header.

    Args:
        request: The inbound request.

    Raises:
        RateLimitedError: When the caller exceeded its budget.
    """
    settings: Settings = request.app.state.settings
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        limiter = RateLimiter(settings.api_rate_limit_per_minute)
        request.app.state.rate_limiter = limiter
    limiter.check(client_key(request, authorization=request.headers.get("authorization")))


# --------------------------------------------------------------------------- #
# Typed dependency aliases
# --------------------------------------------------------------------------- #
#: Dependency aliases for route signatures.
#:
#: These are ``Annotated[None, Depends(...)]`` rather than a bare ``Depends(...)``
#: because FastAPI only reads a ``Depends`` out of ``Annotated`` metadata or out
#: of a *default value*; a bare ``Depends`` used as the annotation makes FastAPI
#: treat the parameter as a request field and hand the ``Depends`` object to
#: pydantic, which fails at import time. ``Annotated[None, ...]`` is the form
#: that composes cleanly with the typed aliases above.
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
EngineDep = Annotated[WorkflowEngine, Depends(get_engine)]
ApprovalsDep = Annotated[ApprovalService, Depends(get_approvals)]
HubDep = Annotated[EventHub, Depends(get_hub)]
AuthDep = Annotated[None, Depends(require_auth)]
RateLimitDep = Annotated[None, Depends(rate_limited)]
ClientKeyDep = Annotated[str, Depends(lambda request: client_key(request))]


def websocket_auth(websocket: WebSocket, token: str | None = None) -> None:
    """Authenticate a WebSocket connection.

    A browser ``WebSocket`` cannot set an ``Authorization`` header, so the token
    arrives as a query parameter. That is acceptable *only* because the socket is
    rejected during the handshake, before any workflow data flows.

    Args:
        websocket: The inbound socket.
        token: Token supplied as ``?token=``.

    Raises:
        WebSocketDisconnect: If authentication is required and fails. Closing
            during the handshake is the only way to reject a WebSocket.
    """
    settings: Settings = websocket.app.state.settings
    if not settings.api_auth_enabled:
        return
    expected = settings.api_auth_token
    if expected is None or not token or token != expected.get_secret_value():
        log.warning("ws.auth_failed", path=websocket.url.path)
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION, reason="unauthorised")


__all__ = [
    "ApprovalsDep",
    "AuthDep",
    "ClientKeyDep",
    "EngineDep",
    "HubDep",
    "RateLimitDep",
    "RateLimiter",
    "SettingsDep",
    "client_key",
    "get_app_state",
    "get_approvals",
    "get_engine",
    "get_hub",
    "get_settings_dep",
    "rate_limited",
    "require_auth",
    "websocket_auth",
]
