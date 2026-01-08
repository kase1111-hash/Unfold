"""Database connections and session management.

This module provides a unified interface for all database connections:
- PostgreSQL (via SQLAlchemy async)
- Neo4j (graph database)
- Vector Store (FAISS local or Pinecone cloud)
"""

from app.db.neo4j import (
    check_neo4j_connection,
    close_neo4j,
    create_indexes as create_neo4j_indexes,
    create_node,
    create_relationship,
    delete_node,
    get_neo4j_session,
    get_neo4j_session_context,
    get_node_by_id,
    init_neo4j,
    search_nodes,
    traverse_graph,
)
from app.db.postgres import (
    Base,
    check_postgres_connection,
    close_postgres,
    create_tables,
    drop_tables,
    get_database_url,
    get_session,
    get_session_context,
    init_postgres,
)
from app.db.vector import (
    VectorStore,
    check_faiss_connection,
    check_pinecone_connection,
    close_faiss,
    close_pinecone,
    faiss_add_vectors,
    faiss_delete,
    faiss_search,
    init_faiss,
    init_pinecone,
    pinecone_delete,
    pinecone_query,
    pinecone_upsert,
    save_faiss_index,
)

__all__ = [
    # PostgreSQL
    "Base",
    "init_postgres",
    "close_postgres",
    "get_session",
    "get_session_context",
    "get_database_url",
    "check_postgres_connection",
    "create_tables",
    "drop_tables",
    # Neo4j
    "init_neo4j",
    "close_neo4j",
    "get_neo4j_session",
    "get_neo4j_session_context",
    "check_neo4j_connection",
    "create_neo4j_indexes",
    "create_node",
    "create_relationship",
    "get_node_by_id",
    "search_nodes",
    "traverse_graph",
    "delete_node",
    # Vector Store - FAISS
    "init_faiss",
    "close_faiss",
    "save_faiss_index",
    "faiss_add_vectors",
    "faiss_search",
    "faiss_delete",
    "check_faiss_connection",
    # Vector Store - Pinecone
    "init_pinecone",
    "close_pinecone",
    "pinecone_upsert",
    "pinecone_query",
    "pinecone_delete",
    "check_pinecone_connection",
    # Unified Vector Store
    "VectorStore",
]


async def init_all_databases() -> None:
    """Initialize all database connections.

    Call this during application startup.
    """
    # Initialize PostgreSQL
    await init_postgres()

    # Initialize Neo4j (optional, may fail if not configured)
    try:
        await init_neo4j()
        await create_neo4j_indexes()
    except Exception:
        pass  # Neo4j is optional

    # Initialize FAISS (local vector store)
    try:
        await init_faiss()
    except ImportError:
        pass  # FAISS is optional


async def close_all_databases() -> None:
    """Close all database connections.

    Call this during application shutdown.
    """
    await close_postgres()
    await close_neo4j()
    await close_faiss()
    await close_pinecone()


async def check_all_connections() -> dict[str, dict]:
    """Check all database connections for health.

    Returns:
        Dict with status for each database
    """
    return {
        "postgresql": await check_postgres_connection(),
        "neo4j": await check_neo4j_connection(),
        "vector_store": await check_faiss_connection(),
    }
