"""Checkpointer selection and lifecycle.

The graph is compiled with a checkpointer, and which one you pass is a *runtime
and durability* decision:

* :class:`~langgraph.checkpoint.memory.InMemorySaver` — fast, zero setup, used by
  tests, the offline demo and the ``echo`` provider. State dies with the process.
* ``AsyncPostgresSaver`` — durable, shared across processes, required for
  production and for any deployment where a human approval may arrive after a
  restart.

Both are built with :func:`~agentic_workflow.persistence.serializer.build_serializer`
so domain models round-trip as real classes rather than degrading to dicts.

Connection pooling
------------------
``AsyncPostgresSaver`` owns an ``AsyncConnectionPool``. Two details matter in
production and are easy to get wrong:

1. ``max_size`` bounds *concurrent* connections, not total opens. Set it from
   ``postgres_pool_max_size``.
2. The pool must be ``open()``-ed before the first query. We do that in
   :meth:`PostgresCheckpointer.__aenter__` and rely on LangGraph's own lazy
   connection for the synchronous path.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

from agentic_workflow.config import Settings, load_settings
from agentic_workflow.errors import ConfigurationError, PersistenceError
from agentic_workflow.logging import get_logger
from agentic_workflow.persistence.serializer import build_serializer

log = get_logger(__name__)


class PostgresCheckpointer:
    """Async context manager owning an ``AsyncPostgresSaver`` and its pool.

    Instances are reusable: the saver is created lazily on first use so that
    importing this module never opens a socket.

    Example:
        --------
        >>> async with PostgresCheckpointer(settings) as cp:  # doctest: +SKIP
        ...     graph = build_graph(settings, checkpointer=cp.saver)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Store configuration without connecting.

        Args:
            settings: Application configuration; defaults to the process settings.
        """
        self._settings = settings or load_settings()
        self._saver: Any | None = None
        self._entered = False

    # ------------------------------------------------------------------ #
    @property
    def settings(self) -> Settings:
        """The configuration this checkpointer was built from."""
        return self._settings

    @property
    def saver(self) -> Any:
        """The underlying ``AsyncPostgresSaver``, created on first access.

        Raises:
            PersistenceError: If ``langgraph-checkpoint-postgres`` is not
                installed, or the schema could not be prepared.
        """
        if self._saver is None:
            self._saver = self._build_saver()
        return self._saver

    def _build_saver(self) -> Any:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise ConfigurationError(
                "PostgreSQL persistence requested but "
                "`langgraph-checkpoint-postgres` is not installed. "
                "Install the `postgres` extra: pip install 'agentic-workflow[postgres]'",
                setting="postgres_enabled",
            ) from exc

        settings = self._settings
        log.info(
            "checkpointer.postgres.building",
            pool_min=settings.postgres_pool_min_size,
            pool_max=settings.postgres_pool_max_size,
            schema=settings.postgres_schema,
            auto_setup=settings.postgres_auto_setup,
        )
        try:
            pool = AsyncConnectionFactory(
                dsn=settings.dsn_with_timeout,
                max_size=settings.postgres_pool_max_size,
                min_size=settings.postgres_pool_min_size,
                timeout=settings.postgres_pool_timeout_seconds,
                open_on_start=False,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": None,
                    "sslmode": settings.postgres_sslmode,
                },
            ).build()
            return AsyncPostgresSaver(conn=pool, serde=build_serializer())
        except Exception as exc:
            raise PersistenceError(
                f"could not build the PostgreSQL checkpointer: {exc}", stage="checkpointer"
            ) from exc

    # ------------------------------------------------------------ lifecycle #
    async def setup(self) -> None:
        """Open the pool and create the checkpoint tables (idempotent).

        Raises:
            PersistenceError: If the database is unreachable or the DDL fails.
        """
        saver = self.saver
        if self._entered:
            return
        conn = getattr(saver, "conn", None)
        pool = getattr(conn, "_pool", conn)
        opener = getattr(pool, "open", None) or getattr(conn, "open", None)
        if opener is not None:
            try:
                await opener()
            except Exception as exc:
                raise PersistenceError(
                    f"could not open the PostgreSQL pool: {exc}", stage="checkpointer"
                ) from exc
        if self._settings.postgres_auto_setup:
            try:
                await saver.setup()
            except Exception as exc:
                raise PersistenceError(
                    f"could not create the checkpoint schema: {exc}", stage="checkpointer"
                ) from exc
        self._entered = True
        log.info("checkpointer.postgres.ready", schema=self._settings.postgres_schema)

    async def close(self) -> None:
        """Close the pool. Safe to call more than once."""
        if not self._entered:
            return
        saver = self._saver
        self._entered = False
        if saver is None:
            return
        conn = getattr(saver, "conn", None)
        pool = getattr(conn, "_pool", conn)
        closer = getattr(pool, "close", None) or getattr(conn, "close", None)
        if closer is not None:
            try:
                await closer()
            except Exception as exc:
                log.warning("checkpointer.close_failed", error=str(exc))

    async def __aenter__(self) -> Self:
        """Open the pool and return ``self``."""
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the pool on exit."""
        await self.close()

    def __repr__(self) -> str:
        return (
            f"PostgresCheckpointer(schema={self._settings.postgres_schema!r}, "
            f"ready={self._entered})"
        )


def build_checkpointer(settings: Settings | None = None) -> Any:
    """Return the checkpointer the configuration asks for.

    Args:
        settings: Application configuration; defaults to the process settings.

    Returns:
        A LangGraph checkpointer. For the in-memory variant this is the saver
        itself; for PostgreSQL it is a :class:`PostgresCheckpointer`, which must
        be entered (or :meth:`PostgresCheckpointer.setup` awaited) before use.

    Raises:
        ConfigurationError: If ``postgres_enabled`` is set but the extra is
            missing.
    """
    settings = settings or load_settings()
    if settings.use_durable_checkpointer:
        return PostgresCheckpointer(settings)
    return build_memory_checkpointer()


def build_memory_checkpointer() -> Any:
    """Build an ``InMemorySaver`` with the project serializer.

    Returns:
        A checkpointer suitable for tests, the CLI demo and the evaluation suite.

    Raises:
        RuntimeError: If the installed LangGraph version is unsupported.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver(serde=build_serializer())


class AsyncConnectionFactory:
    """Build a lazily-opened ``AsyncConnectionPool`` from a DSN.

    Extracted as a class so the import of ``psycopg_pool`` happens in one place
    and can be reported as a :class:`ConfigurationError` with an actionable
    message instead of a bare ``ImportError`` from deep inside LangGraph.
    """

    def __init__(
        self,
        *,
        dsn: str,
        max_size: int,
        min_size: int,
        timeout: float,
        open_on_start: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Store pool parameters; the pool is built on construction.

        Args:
            dsn: libpq connection string.
            max_size: Maximum concurrent connections.
            min_size: Minimum eagerly-opened connections.
            timeout: Seconds to wait for a free connection.
            open_on_start: Passed through to the pool; kept ``False`` so boot never
                blocks on the database.
            kwargs: Extra driver keyword arguments.
        """
        self.dsn = dsn
        self.max_size = max_size
        self.min_size = min_size
        self.timeout = timeout
        self.open_on_start = open_on_start
        self.kwargs = kwargs or {}

    def build(self) -> Any:
        """Create the pool.

        Returns:
            An ``AsyncConnectionPool`` instance.

        Raises:
            ConfigurationError: If ``psycopg``/``psycopg_pool`` is missing.
        """
        try:
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:
            raise ConfigurationError(
                "`psycopg[pool]` is required for PostgreSQL persistence. "
                "Install the `postgres` extra: pip install 'agentic-workflow[postgres]'",
                setting="postgres_enabled",
            ) from exc
        log.debug("checkpointer.pool.building", max_size=self.max_size, min_size=self.min_size)
        return AsyncConnectionPool(
            conninfo=self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            timeout=self.timeout,
            open=self.open_on_start,
            kwargs=self.kwargs,
            check=AsyncConnectionPool.check_connection,
        )


__all__ = [
    "AsyncConnectionFactory",
    "PostgresCheckpointer",
    "build_checkpointer",
    "build_memory_checkpointer",
]
