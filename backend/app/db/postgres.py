"""PostgreSQL database connection and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""

    pass


# Global engine and session factory
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    """Get async database URL from settings."""
    url = str(settings.database_url)
    # Convert postgresql:// to postgresql+asyncpg://
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif not url.startswith("postgresql+asyncpg://"):
        url = f"postgresql+asyncpg://{url.split('://', 1)[-1]}"
    return url


async def init_postgres() -> AsyncEngine:
    """Initialize PostgreSQL connection pool.

    Returns:
        Configured async engine instance.
    """
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    database_url = get_database_url()

    _engine = create_async_engine(
        database_url,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    return _engine


async def close_postgres() -> None:
    """Close PostgreSQL connection pool."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection.

    Yields:
        AsyncSession instance that auto-closes after use.
    """
    if _session_factory is None:
        await init_postgres()

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Get database session as context manager.

    Usage:
        async with get_session_context() as session:
            result = await session.execute(query)
    """
    if _session_factory is None:
        await init_postgres()

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_postgres_connection() -> dict[str, str | bool]:
    """Check PostgreSQL connectivity for health checks.

    Returns:
        Dict with connection status and details.
    """
    try:
        if _engine is None:
            return {
                "connected": False,
                "status": "not_initialized",
                "message": "Database engine not initialized",
            }

        async with _engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()

        return {
            "connected": True,
            "status": "healthy",
            "message": "PostgreSQL connection successful",
        }
    except Exception as e:
        return {
            "connected": False,
            "status": "error",
            "message": str(e),
        }


async def create_tables() -> None:
    """Create all database tables.

    Should be called during application startup in development.
    In production, use Alembic migrations instead.
    """
    if _engine is None:
        await init_postgres()

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables() -> None:
    """Drop all database tables.

    WARNING: This will delete all data. Use only in development/testing.
    """
    if _engine is None:
        await init_postgres()

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
