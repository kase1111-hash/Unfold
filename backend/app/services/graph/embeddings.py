"""Embedding service for generating vector representations."""

import asyncio
from typing import Any

from app.config import get_settings
from app.db.vector import VectorStore, faiss_add_vectors, faiss_search

settings = get_settings()


class EmbeddingService:
    """Service for generating and managing text embeddings."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 100,
    ):
        """Initialize embedding service.

        Args:
            model: OpenAI embedding model name
            batch_size: Maximum texts per API call
        """
        self.model = model or settings.openai_embedding_model
        self.batch_size = batch_size
        self._client = None
        self._dimension = 3072  # text-embedding-3-large default

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")

            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)

        return self._client

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = self._get_client()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Clean texts (remove newlines, limit length)
            cleaned_batch = [
                text.replace("\n", " ")[:8191]  # Max 8191 tokens
                for text in batch
            ]

            response = client.embeddings.create(
                model=self.model,
                input=cleaned_batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_and_store(
        self,
        texts: list[str],
        ids: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Generate embeddings and store in vector database.

        Args:
            texts: Texts to embed
            ids: Unique IDs for each text
            metadata: Optional metadata for each text
        """
        embeddings = await self.embed_texts(texts)

        await faiss_add_vectors(
            vectors=embeddings,
            ids=ids,
            metadata=metadata,
        )

    async def search_similar(
        self,
        query: str,
        k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar texts using vector similarity.

        Args:
            query: Query text
            k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of matches with id, score, and metadata
        """
        query_embedding = await self.embed_text(query)

        results = await faiss_search(
            query_vector=query_embedding,
            k=k,
            filter_metadata=filter_metadata,
        )

        return results

    def get_dimension(self) -> int:
        """Get embedding dimension for current model.

        Returns:
            Embedding dimension
        """
        # Model dimensions
        dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 3072)


class LocalEmbeddingService:
    """Local embedding service using sentence-transformers (no API required)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize local embedding service.

        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    async def embed_and_store(
        self,
        texts: list[str],
        ids: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Generate embeddings and store in vector database.

        Args:
            texts: Texts to embed
            ids: Unique IDs for each text
            metadata: Optional metadata for each text
        """
        embeddings = await self.embed_texts(texts)

        await faiss_add_vectors(
            vectors=embeddings,
            ids=ids,
            metadata=metadata,
        )

    async def search_similar(
        self,
        query: str,
        k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar texts using vector similarity.

        Args:
            query: Query text
            k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of matches with id, score, and metadata
        """
        query_embedding = await self.embed_text(query)

        results = await faiss_search(
            query_vector=query_embedding,
            k=k,
            filter_metadata=filter_metadata,
        )

        return results

    def get_dimension(self) -> int:
        """Get embedding dimension for current model.

        Returns:
            Embedding dimension
        """
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


# Global embedding service instances
_openai_service: EmbeddingService | None = None
_local_service: LocalEmbeddingService | None = None


def get_embedding_service(use_openai: bool = True) -> EmbeddingService | LocalEmbeddingService:
    """Get embedding service instance.

    Args:
        use_openai: Whether to use OpenAI API

    Returns:
        Embedding service instance
    """
    global _openai_service, _local_service

    if use_openai:
        if _openai_service is None:
            _openai_service = EmbeddingService()
        return _openai_service
    else:
        if _local_service is None:
            _local_service = LocalEmbeddingService()
        return _local_service
