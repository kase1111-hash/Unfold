"""Vector store connection for embeddings (FAISS and Pinecone)."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings

settings = get_settings()

# ============================================================================
# FAISS Local Vector Store
# ============================================================================

_faiss_index: Any = None
_faiss_id_map: dict[str, int] = {}
_faiss_metadata: dict[str, dict] = {}
_faiss_dimension: int = 3072  # text-embedding-3-large dimension


async def init_faiss(dimension: int = 3072, index_path: str | None = None) -> None:
    """Initialize FAISS index for local vector storage.

    Args:
        dimension: Embedding dimension (3072 for text-embedding-3-large)
        index_path: Optional path to load existing index
    """
    global _faiss_index, _faiss_id_map, _faiss_metadata, _faiss_dimension

    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

    _faiss_dimension = dimension

    if index_path and Path(index_path).exists():
        # Load existing index
        _faiss_index = faiss.read_index(index_path)
        metadata_path = Path(index_path).with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
                _faiss_id_map = data.get("id_map", {})
                _faiss_metadata = data.get("metadata", {})
    else:
        # Create new index with L2 distance
        _faiss_index = faiss.IndexFlatL2(dimension)


async def close_faiss() -> None:
    """Close FAISS index."""
    global _faiss_index, _faiss_id_map, _faiss_metadata
    _faiss_index = None
    _faiss_id_map = {}
    _faiss_metadata = {}


async def save_faiss_index(index_path: str) -> None:
    """Save FAISS index to disk.

    Args:
        index_path: Path to save the index
    """
    global _faiss_index, _faiss_id_map, _faiss_metadata

    if _faiss_index is None:
        raise ValueError("FAISS index not initialized")

    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS not installed")

    # Save index
    faiss.write_index(_faiss_index, index_path)

    # Save metadata
    metadata_path = Path(index_path).with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump({"id_map": _faiss_id_map, "metadata": _faiss_metadata}, f)


async def faiss_add_vectors(
    vectors: list[list[float]],
    ids: list[str],
    metadata: list[dict] | None = None,
) -> None:
    """Add vectors to FAISS index.

    Args:
        vectors: List of embedding vectors
        ids: List of unique IDs for each vector
        metadata: Optional metadata for each vector
    """
    global _faiss_index, _faiss_id_map, _faiss_metadata

    if _faiss_index is None:
        await init_faiss()

    vectors_np = np.array(vectors, dtype=np.float32)

    # Get current index size for mapping
    start_idx = _faiss_index.ntotal

    # Add vectors
    _faiss_index.add(vectors_np)

    # Update mappings
    for i, vec_id in enumerate(ids):
        _faiss_id_map[vec_id] = start_idx + i
        if metadata:
            _faiss_metadata[vec_id] = metadata[i]


async def faiss_search(
    query_vector: list[float],
    k: int = 10,
    filter_metadata: dict | None = None,
) -> list[dict[str, Any]]:
    """Search for similar vectors in FAISS index.

    Args:
        query_vector: Query embedding vector
        k: Number of results to return
        filter_metadata: Optional metadata filter (post-filter)

    Returns:
        List of matches with id, score, and metadata
    """
    global _faiss_index, _faiss_id_map, _faiss_metadata

    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []

    query_np = np.array([query_vector], dtype=np.float32)

    # Search with extra results for post-filtering
    search_k = k * 3 if filter_metadata else k
    distances, indices = _faiss_index.search(query_np, min(search_k, _faiss_index.ntotal))

    # Reverse ID map for lookup
    idx_to_id = {v: k for k, v in _faiss_id_map.items()}

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        vec_id = idx_to_id.get(idx)
        if vec_id is None:
            continue

        metadata = _faiss_metadata.get(vec_id, {})

        # Apply metadata filter
        if filter_metadata:
            if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                continue

        results.append({
            "id": vec_id,
            "score": float(1 / (1 + dist)),  # Convert L2 distance to similarity
            "metadata": metadata,
        })

        if len(results) >= k:
            break

    return results


async def faiss_delete(ids: list[str]) -> int:
    """Delete vectors from FAISS index.

    Note: FAISS doesn't support true deletion. We mark as deleted in metadata.

    Args:
        ids: List of vector IDs to delete

    Returns:
        Number of vectors marked as deleted
    """
    global _faiss_metadata

    deleted = 0
    for vec_id in ids:
        if vec_id in _faiss_metadata:
            _faiss_metadata[vec_id]["_deleted"] = True
            deleted += 1

    return deleted


async def check_faiss_connection() -> dict[str, str | bool]:
    """Check FAISS availability for health checks.

    Returns:
        Dict with status information
    """
    global _faiss_index

    try:
        import faiss

        if _faiss_index is None:
            return {
                "connected": False,
                "status": "not_initialized",
                "message": "FAISS index not initialized",
            }

        return {
            "connected": True,
            "status": "healthy",
            "message": f"FAISS index active with {_faiss_index.ntotal} vectors",
            "vector_count": _faiss_index.ntotal,
        }
    except ImportError:
        return {
            "connected": False,
            "status": "not_installed",
            "message": "FAISS library not installed",
        }
    except Exception as e:
        return {
            "connected": False,
            "status": "error",
            "message": str(e),
        }


# ============================================================================
# Pinecone Cloud Vector Store
# ============================================================================

_pinecone_index: Any = None


async def init_pinecone() -> Any:
    """Initialize Pinecone connection.

    Returns:
        Pinecone index instance
    """
    global _pinecone_index

    if _pinecone_index is not None:
        return _pinecone_index

    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not configured")

    try:
        from pinecone import Pinecone
    except ImportError:
        raise ImportError("Pinecone not installed. Run: pip install pinecone-client")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    _pinecone_index = pc.Index(settings.pinecone_index_name)

    return _pinecone_index


async def close_pinecone() -> None:
    """Close Pinecone connection."""
    global _pinecone_index
    _pinecone_index = None


async def pinecone_upsert(
    vectors: list[dict[str, Any]],
    namespace: str = "",
) -> dict[str, int]:
    """Upsert vectors to Pinecone.

    Args:
        vectors: List of dicts with id, values, and metadata
        namespace: Optional namespace for organization

    Returns:
        Upsert statistics
    """
    global _pinecone_index

    if _pinecone_index is None:
        await init_pinecone()

    response = _pinecone_index.upsert(vectors=vectors, namespace=namespace)
    return {"upserted_count": response.upserted_count}


async def pinecone_query(
    vector: list[float],
    top_k: int = 10,
    namespace: str = "",
    filter: dict | None = None,
    include_metadata: bool = True,
) -> list[dict[str, Any]]:
    """Query Pinecone for similar vectors.

    Args:
        vector: Query embedding vector
        top_k: Number of results
        namespace: Optional namespace
        filter: Optional metadata filter
        include_metadata: Whether to include metadata

    Returns:
        List of matches
    """
    global _pinecone_index

    if _pinecone_index is None:
        await init_pinecone()

    response = _pinecone_index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        filter=filter,
        include_metadata=include_metadata,
    )

    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata if include_metadata else {},
        }
        for match in response.matches
    ]


async def pinecone_delete(
    ids: list[str] | None = None,
    delete_all: bool = False,
    namespace: str = "",
    filter: dict | None = None,
) -> None:
    """Delete vectors from Pinecone.

    Args:
        ids: List of vector IDs to delete
        delete_all: Delete all vectors in namespace
        namespace: Optional namespace
        filter: Optional metadata filter
    """
    global _pinecone_index

    if _pinecone_index is None:
        await init_pinecone()

    _pinecone_index.delete(
        ids=ids,
        delete_all=delete_all,
        namespace=namespace,
        filter=filter,
    )


async def check_pinecone_connection() -> dict[str, str | bool]:
    """Check Pinecone connectivity for health checks.

    Returns:
        Dict with connection status
    """
    global _pinecone_index

    try:
        if not settings.pinecone_api_key:
            return {
                "connected": False,
                "status": "not_configured",
                "message": "Pinecone API key not set",
            }

        if _pinecone_index is None:
            return {
                "connected": False,
                "status": "not_initialized",
                "message": "Pinecone index not initialized",
            }

        # Get index stats to verify connection
        stats = _pinecone_index.describe_index_stats()

        return {
            "connected": True,
            "status": "healthy",
            "message": "Pinecone connection successful",
            "vector_count": stats.total_vector_count,
        }
    except Exception as e:
        return {
            "connected": False,
            "status": "error",
            "message": str(e),
        }


# ============================================================================
# Unified Vector Store Interface
# ============================================================================


class VectorStore:
    """Unified interface for vector operations.

    Supports both FAISS (local) and Pinecone (cloud) backends.
    """

    def __init__(self, backend: str = "faiss"):
        """Initialize vector store.

        Args:
            backend: "faiss" or "pinecone"
        """
        self.backend = backend

    async def initialize(self) -> None:
        """Initialize the vector store backend."""
        if self.backend == "faiss":
            await init_faiss()
        else:
            await init_pinecone()

    async def add(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Add vectors to the store."""
        if self.backend == "faiss":
            await faiss_add_vectors(vectors, ids, metadata)
        else:
            vector_data = [
                {"id": id_, "values": vec, "metadata": meta or {}}
                for id_, vec, meta in zip(ids, vectors, metadata or [{}] * len(ids))
            ]
            await pinecone_upsert(vector_data)

    async def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        if self.backend == "faiss":
            return await faiss_search(query_vector, k, filter)
        else:
            return await pinecone_query(query_vector, k, filter=filter)

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        if self.backend == "faiss":
            await faiss_delete(ids)
        else:
            await pinecone_delete(ids=ids)

    async def health_check(self) -> dict[str, str | bool]:
        """Check vector store health."""
        if self.backend == "faiss":
            return await check_faiss_connection()
        else:
            return await check_pinecone_connection()
