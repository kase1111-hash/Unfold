"""Neo4j graph database connection and operations."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from app.config import get_settings

settings = get_settings()

# Global driver instance
_driver: AsyncDriver | None = None


async def init_neo4j() -> AsyncDriver:
    """Initialize Neo4j connection.

    Returns:
        Configured async driver instance.
    """
    global _driver

    if _driver is not None:
        return _driver

    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_acquisition_timeout=60,
    )

    return _driver


async def close_neo4j() -> None:
    """Close Neo4j connection."""
    global _driver

    if _driver is not None:
        await _driver.close()
        _driver = None


async def get_neo4j_session() -> AsyncGenerator[AsyncSession, None]:
    """Get Neo4j session for dependency injection.

    Yields:
        AsyncSession instance that auto-closes after use.
    """
    if _driver is None:
        await init_neo4j()

    async with _driver.session() as session:
        yield session


@asynccontextmanager
async def get_neo4j_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Get Neo4j session as context manager.

    Usage:
        async with get_neo4j_session_context() as session:
            result = await session.run(query)
    """
    if _driver is None:
        await init_neo4j()

    async with _driver.session() as session:
        yield session


async def check_neo4j_connection() -> dict[str, str | bool]:
    """Check Neo4j connectivity for health checks.

    Returns:
        Dict with connection status and details.
    """
    try:
        if _driver is None:
            return {
                "connected": False,
                "status": "not_initialized",
                "message": "Neo4j driver not initialized",
            }

        async with _driver.session() as session:
            result = await session.run("RETURN 1 as n")
            await result.single()

        return {
            "connected": True,
            "status": "healthy",
            "message": "Neo4j connection successful",
        }
    except AuthError as e:
        return {
            "connected": False,
            "status": "auth_error",
            "message": f"Authentication failed: {e}",
        }
    except ServiceUnavailable as e:
        return {
            "connected": False,
            "status": "unavailable",
            "message": f"Service unavailable: {e}",
        }
    except Exception as e:
        return {
            "connected": False,
            "status": "error",
            "message": str(e),
        }


# ============================================================================
# Knowledge Graph Operations
# ============================================================================


async def create_node(
    session: AsyncSession,
    node_type: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    """Create a node in the knowledge graph.

    Args:
        session: Neo4j session
        node_type: Node label (Concept, Author, Paper, etc.)
        properties: Node properties

    Returns:
        Created node data
    """
    query = f"""
    CREATE (n:{node_type} $props)
    RETURN n, elementId(n) as id
    """
    result = await session.run(query, props=properties)
    record = await result.single()
    return {"id": record["id"], "properties": dict(record["n"])}


async def create_relationship(
    session: AsyncSession,
    source_id: str,
    target_id: str,
    rel_type: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a relationship between two nodes.

    Args:
        session: Neo4j session
        source_id: Source node element ID
        target_id: Target node element ID
        rel_type: Relationship type (EXPLAINS, CITES, etc.)
        properties: Optional relationship properties

    Returns:
        Created relationship data
    """
    props = properties or {}
    query = f"""
    MATCH (a), (b)
    WHERE elementId(a) = $source_id AND elementId(b) = $target_id
    CREATE (a)-[r:{rel_type} $props]->(b)
    RETURN r, elementId(r) as id
    """
    result = await session.run(query, source_id=source_id, target_id=target_id, props=props)
    record = await result.single()
    if record is None:
        raise ValueError(f"Could not create relationship: nodes not found")
    return {"id": record["id"], "properties": dict(record["r"])}


async def get_node_by_id(
    session: AsyncSession,
    node_id: str,
) -> dict[str, Any] | None:
    """Get a node by its element ID.

    Args:
        session: Neo4j session
        node_id: Node element ID

    Returns:
        Node data or None if not found
    """
    query = """
    MATCH (n)
    WHERE elementId(n) = $node_id
    RETURN n, labels(n) as labels, elementId(n) as id
    """
    result = await session.run(query, node_id=node_id)
    record = await result.single()
    if record is None:
        return None
    return {
        "id": record["id"],
        "labels": record["labels"],
        "properties": dict(record["n"]),
    }


async def search_nodes(
    session: AsyncSession,
    label: str | None = None,
    properties: dict[str, Any] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search for nodes by label and/or properties.

    Args:
        session: Neo4j session
        label: Optional node label filter
        properties: Optional property filters
        limit: Maximum results

    Returns:
        List of matching nodes
    """
    label_clause = f":{label}" if label else ""
    where_clauses = []
    params = {"limit": limit}

    if properties:
        for i, (key, value) in enumerate(properties.items()):
            param_name = f"prop_{i}"
            where_clauses.append(f"n.{key} = ${param_name}")
            params[param_name] = value

    where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
    MATCH (n{label_clause})
    {where_clause}
    RETURN n, labels(n) as labels, elementId(n) as id
    LIMIT $limit
    """
    result = await session.run(query, **params)
    records = await result.data()
    return [
        {
            "id": r["id"],
            "labels": r["labels"],
            "properties": dict(r["n"]),
        }
        for r in records
    ]


async def traverse_graph(
    session: AsyncSession,
    start_node_id: str,
    relationship_types: list[str] | None = None,
    direction: str = "OUTGOING",
    max_depth: int = 3,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Traverse the graph from a starting node.

    Args:
        session: Neo4j session
        start_node_id: Starting node element ID
        relationship_types: Optional filter for relationship types
        direction: OUTGOING, INCOMING, or BOTH
        max_depth: Maximum traversal depth
        limit: Maximum results

    Returns:
        List of paths/nodes found
    """
    rel_filter = "|".join(relationship_types) if relationship_types else ""
    rel_pattern = f"[r:{rel_filter}*1..{max_depth}]" if rel_filter else f"[r*1..{max_depth}]"

    if direction == "OUTGOING":
        pattern = f"-{rel_pattern}->"
    elif direction == "INCOMING":
        pattern = f"<-{rel_pattern}-"
    else:
        pattern = f"-{rel_pattern}-"

    query = f"""
    MATCH (start){pattern}(end)
    WHERE elementId(start) = $start_id
    RETURN DISTINCT end, labels(end) as labels, elementId(end) as id
    LIMIT $limit
    """
    result = await session.run(query, start_id=start_node_id, limit=limit)
    records = await result.data()
    return [
        {
            "id": r["id"],
            "labels": r["labels"],
            "properties": dict(r["end"]),
        }
        for r in records
    ]


async def delete_node(
    session: AsyncSession,
    node_id: str,
    detach: bool = True,
) -> bool:
    """Delete a node from the graph.

    Args:
        session: Neo4j session
        node_id: Node element ID
        detach: If True, also delete relationships

    Returns:
        True if deleted, False if not found
    """
    detach_clause = "DETACH " if detach else ""
    query = f"""
    MATCH (n)
    WHERE elementId(n) = $node_id
    {detach_clause}DELETE n
    RETURN count(n) as deleted
    """
    result = await session.run(query, node_id=node_id)
    record = await result.single()
    return record["deleted"] > 0


async def create_indexes() -> None:
    """Create necessary indexes for knowledge graph queries."""
    if _driver is None:
        await init_neo4j()

    async with _driver.session() as session:
        # Create indexes for common node types
        indexes = [
            "CREATE INDEX concept_label IF NOT EXISTS FOR (n:Concept) ON (n.label)",
            "CREATE INDEX concept_node_id IF NOT EXISTS FOR (n:Concept) ON (n.node_id)",
            "CREATE INDEX author_name IF NOT EXISTS FOR (n:Author) ON (n.name)",
            "CREATE INDEX paper_doi IF NOT EXISTS FOR (n:Paper) ON (n.doi)",
            "CREATE INDEX paper_title IF NOT EXISTS FOR (n:Paper) ON (n.title)",
            "CREATE INDEX method_label IF NOT EXISTS FOR (n:Method) ON (n.label)",
            "CREATE INDEX dataset_label IF NOT EXISTS FOR (n:Dataset) ON (n.label)",
        ]

        for index_query in indexes:
            try:
                await session.run(index_query)
            except Exception:
                # Index might already exist
                pass
