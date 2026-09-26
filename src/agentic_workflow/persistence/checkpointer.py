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

from agentic_workflow.config import DEFAULT_SCHEMA, Settings, load_settings
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
        self._pool: Any | None = None
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
                dsn=settings.checkpointer_dsn,
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
            # Held rather than recovered from the saver later. The previous
            # `getattr(saver.conn, "_pool", saver.conn)` idiom was a guess, and
            # a wrong one: psycopg's `AsyncConnectionPool` has a `_pool`
            # attribute which is its internal `deque` of idle connections, not
            # the pool. It happened to work only because the code then fell
            # through to `saver.conn.open`, so the deque was discarded — by
            # luck, in the one place that used it.
            self._pool = pool
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
        if self._entered:
            return
        try:
            saver = self.saver
            # The held pool rather than digging the saver apart. See _build_saver
            # for why the old guess was wrong.
            pool = self.pool
            opener = getattr(pool, "open", None)
            if opener is not None:
                try:
                    await opener()
                except Exception as exc:
                    raise PersistenceError(
                        f"could not open the PostgreSQL pool: {exc}", stage="checkpointer"
                    ) from exc
            if self._settings.postgres_auto_setup:
                try:
                    await self._ensure_schema(pool)
                    await saver.setup()
                except Exception as exc:
                    raise PersistenceError(
                        f"could not create the checkpoint schema: {exc}", stage="checkpointer"
                    ) from exc
        except BaseException:
            # Release the pool on *this* failure rather than waiting for a
            # later close(). The caller is `WorkflowEngine.startup`, which
            # propagates the error and so never reaches its own `close`; a boot
            # refused by the database — a read-only role, a schema another role
            # owns — would otherwise leave real sockets open for the lifetime of
            # the process, on the one path where the operator is already looking
            # at something broken. `BaseException` because a cancelled setup
            # leaves exactly the same mess, and a leaked pool is not the thing
            # to fix by not cleaning up on the way out.
            await self.close()
            raise
        self._entered = True
        log.info("checkpointer.postgres.ready", schema=self._settings.postgres_schema)

    @property
    def pool(self) -> Any:
        """The connection pool, built on first access.

        Returns:
            The ``AsyncConnectionPool`` backing the saver.
        """
        if self._pool is None:
            self.saver  # noqa: B018 - building the saver is what builds the pool
        return self._pool

    async def _ensure_schema(self, pool: Any) -> None:
        """Create the configured schema if it does not exist.

        Necessary because the schema is applied as a ``search_path``, and a
        ``search_path`` naming a schema that does not exist does not create one.
        Every unqualified ``CREATE TABLE`` the saver's migrations issue would
        then land in ``public`` anyway — precisely the behaviour this method
        exists to remove.

        The name is interpolated rather than bound because PostgreSQL will not
        accept a parameter in ``CREATE SCHEMA IF NOT EXISTS``. That is safe only
        because :attr:`Settings.postgres_schema` is pattern-constrained to a
        bare unquoted SQL identifier; the constraint and this method have to
        change together, and a test that feeds the field an injectable name is
        what catches them drifting apart.

        Args:
            pool: The open connection pool.
        """
        schema = self._settings.postgres_schema
        if schema == DEFAULT_SCHEMA:
            return
        # A checked-out connection rather than ``pool.cursor()``: the latter
        # acquires a connection from under the pool's own lock, which a pool
        # built with ``open_on_start=False`` does not release until its warm-up
        # has finished. The call blocks there rather than failing, which is a
        # far harder failure to diagnose than a refused one.
        async with pool.connection() as connection:
            await connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

    async def close(self) -> None:
        """Close the pool. Safe to call more than once.

        Keyed on whether a pool was *built*, not on whether :meth:`setup`
        finished. Those differ exactly when it matters: a pool is opened before
        the DDL runs, so a boot that fails on ``CREATE SCHEMA`` — a permission
        error, a read-only volume, a name another role already owns — has an
        open pool holding real sockets and never sets ``_entered``. Returning
        early on ``_entered`` therefore leaked the connections on precisely the
        failure path where the operator is already looking at a problem, and
        the "safe to call more than once" property was being bought with it.

        The saver is dropped along with the pool, so that a later :meth:`setup`
        builds a fresh one. Keeping it would leave a *closed* pool wired into the
        saver and the second ``setup`` would then try to open a pool that no
        longer exists — the difference between "this instance can be reused" and
        "this instance is now quietly broken".
        """
        self._entered = False
        pool = self._pool
        self._pool = None
        self._saver = None
        if pool is None:
            return
        closer = getattr(pool, "close", None)
        if closer is None:
            return
        try:
            await closer()
        except Exception as exc:
            log.warning("checkpointer.close_failed", error=str(exc))

    async def list_thread_ids(self, *, limit: int = 10_000) -> list[str]:
        """Return the checkpoint thread ids this database holds.

        The engine needs this to rehydrate its run registry on boot. Without it
        a restarted process knows about nothing that was in flight, and every
        operator action that consults the registry — cancelling a run above all
        — 404s for a run that demonstrably exists.

        Enumerated through the saver's own ``alist`` rather than a hand-written
        ``SELECT DISTINCT thread_id``: the table name, the ``checkpoint_ns``
        column and the ordering are all the saver's business, and duplicating
        them here is how a library upgrade silently breaks the boot path.

        Args:
            limit: Upper bound on the number of threads to return.

        Returns:
            Distinct thread ids, most recently checkpointed first. Empty when
            enumeration is unsupported rather than when the database is empty —
            the two are distinguished by the log line, not by the return value,
            because a caller cannot act on either.
        """
        if not self._entered:
            await self.setup()
        threads: list[str] = []
        seen: set[str] = set()
        try:
            async for item in self.saver.alist(None, limit=limit):
                thread_id = (
                    (getattr(item, "config", None) or {}).get("configurable", {}).get("thread_id")
                )
                if isinstance(thread_id, str) and thread_id not in seen:
                    seen.add(thread_id)
                    threads.append(thread_id)
        except Exception as exc:
            log.warning("checkpointer.thread_listing_failed", error=str(exc))
            return []
        log.info("checkpointer.threads_listed", found=len(threads))
        return threads

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
