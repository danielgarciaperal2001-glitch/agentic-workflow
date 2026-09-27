"""Runtime context injected into every graph node.

LangGraph 1.x separates *graph state* (must be JSON-serialisable, it is
checkpointed) from *runtime context* (live objects, never persisted). That
split is exactly what this project needs: the LLM client, the escalation policy
and the event sink must be available inside nodes, but must never leak into a
checkpoint payload.

Example:
    --------
    >>> from agentic_workflow.config import load_settings
    >>> from agentic_workflow.llm import EchoLLM
    >>> ctx = AgentContext(settings=load_settings(), llm=EchoLLM(model="echo-1"))
    >>> ctx.policy.enabled
    True
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentic_workflow.config import Settings, load_settings
from agentic_workflow.human.policy import EscalationPolicy
from agentic_workflow.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    pass

log = get_logger(__name__)

#: Callback invoked with a structured event dict. The API wires this to a
#: WebSocket broadcast; tests wire it to a list.
EventSink = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class AgentContext:
    """Live dependencies available to every node.

    Attributes:
        settings: Immutable application configuration.
        llm: Provider-agnostic LLM client.
        policy: Escalation policy driving the human gates.
        emit: Optional async callback receiving node-level events.
        extra: Free-form bag for application-specific dependencies.
    """

    settings: Settings = field(default_factory=load_settings)
    llm: Any = None
    policy: EscalationPolicy | None = None
    emit: EventSink | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build the derived policy when the caller did not supply one."""
        if self.policy is None:
            self.policy = EscalationPolicy.from_settings(self.settings)

    # ------------------------------------------------------------- events #
    async def publish(self, event: str, **fields: Any) -> None:
        """Emit a structured event to the configured sink.

        Emission must never break the run: a failing WebSocket is an
        observability problem, not a business one, so errors are logged and
        swallowed.

        Args:
            event: Event name, e.g. ``"node.started"``.
            **fields: Structured payload.
        """
        if self.emit is None:
            return
        from agentic_workflow.domain.schemas import utcnow

        payload = {"event": event, "ts": utcnow().isoformat(), **fields}
        try:
            await self.emit(payload)
        except Exception as exc:
            log.warning("event.emit_failed", failed_event=event, error=str(exc))

    # ----------------------------------------------------------- helpers #
    @property
    def client(self) -> Any:
        """Return the LLM client, raising a clear error when unset.

        Raises:
            ConfigurationError: If the context was built without a client.
        """
        if self.llm is None:
            from agentic_workflow.llm.base import build_llm_client

            self.llm = build_llm_client(self.settings)
            log.info("llm.lazy_initialised", provider=self.settings.llm_provider.value)
        return self.llm

    @property
    def escalation(self) -> EscalationPolicy:
        """The escalation policy (never ``None`` after ``__post_init__``)."""
        assert self.policy is not None  # noqa: S101 - guaranteed by __post_init__
        return self.policy

    @property
    def sign_human_decisions(self) -> bool:
        """Whether resolved decisions must carry an HMAC signature."""
        return self.settings.hitl_require_signature

    @property
    def signing_secret(self) -> str:
        """Secret used to sign human decisions, or ``""`` when there is none.

        Delegates to :meth:`Settings.resolved_signing_secret` so the resolution
        and its fallback order live in one place. This used to be a second copy
        of that logic, which is how the two drifted into disagreeing about which
        secret to use.
        """
        return self.settings.resolved_signing_secret()


def context_from_runtime(runtime: Any) -> AgentContext:
    """Extract the :class:`AgentContext` from a LangGraph ``Runtime``.

    Args:
        runtime: The second argument LangGraph passes to each node.

    Returns:
        The injected context, or a default one built from settings when the
        graph is invoked without a context (which happens in notebooks and in
        bare ``graph.ainvoke`` calls).

    Raises:
        ConfigurationError: If ``runtime.context`` is a non-``AgentContext`` type,
            which indicates a wiring bug worth failing loudly on.
    """
    from agentic_workflow.errors import ConfigurationError

    ctx = getattr(runtime, "context", None)
    if ctx is None:
        return AgentContext()
    if isinstance(ctx, AgentContext):
        return ctx
    raise ConfigurationError(
        f"unexpected runtime context type: {type(ctx).__name__}; "
        "expected agentic_workflow.graph.context.AgentContext",
        received=type(ctx).__name__,
    )


__all__ = ["AgentContext", "EventSink", "context_from_runtime"]
