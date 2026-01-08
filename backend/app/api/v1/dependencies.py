"""API dependencies for dependency injection."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session, get_neo4j_session
from neo4j import AsyncSession as Neo4jSession


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL database session.

    Usage:
        @router.get("/items")
        async def get_items(db: Annotated[AsyncSession, Depends(get_db)]):
            ...
    """
    async for session in get_session():
        yield session


async def get_graph_db() -> AsyncGenerator[Neo4jSession, None]:
    """Get Neo4j database session.

    Usage:
        @router.get("/nodes")
        async def get_nodes(graph: Annotated[Neo4jSession, Depends(get_graph_db)]):
            ...
    """
    async for session in get_neo4j_session():
        yield session


# Type aliases for cleaner dependency injection
DBSession = Annotated[AsyncSession, Depends(get_db)]
GraphSession = Annotated[Neo4jSession, Depends(get_graph_db)]
