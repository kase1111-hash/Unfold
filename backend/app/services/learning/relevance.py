"""
Relevance Scoring Engine using TF-IDF and BERT embeddings.
Scores text passages by relevance to user's learning goals.
"""

import re
import math
from typing import Optional, Any
from collections import Counter

import numpy as np

# Optional import for sentence_transformers
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    HAS_SENTENCE_TRANSFORMERS = False

from app.config import settings


class RelevanceScorer:
    """
    Combines TF-IDF and semantic (BERT) relevance scoring.
    """

    def __init__(self):
        self._model: Optional[Any] = None
        self._idf_cache: dict[str, float] = {}
        self._corpus_size = 0

    @property
    def model(self) -> Any:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence_transformers is required for semantic scoring. "
                    "Install with: pip install sentence-transformers"
                )
            model_name = (
                getattr(settings, "EMBEDDING_MODEL", None) or "all-MiniLM-L6-v2"
            )
            self._model = SentenceTransformer(model_name)
        return self._model

    def tokenize(self, text: str) -> list[str]:
        """Simple tokenization with lowercasing and cleaning."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove very short tokens and stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "it",
        }
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def compute_tf(self, tokens: list[str]) -> dict[str, float]:
        """Compute term frequency for tokens."""
        counter = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        return {term: count / total for term, count in counter.items()}

    def update_idf(self, documents: list[str]) -> None:
        """
        Update IDF values based on a corpus of documents.

        Args:
            documents: List of document texts to build IDF from
        """
        doc_freq: dict[str, int] = {}

        for doc in documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self._corpus_size = len(documents)

        # Compute IDF with smoothing
        for term, freq in doc_freq.items():
            self._idf_cache[term] = math.log((self._corpus_size + 1) / (freq + 1)) + 1

    def compute_tfidf_score(
        self,
        text: str,
        query: str,
    ) -> float:
        """
        Compute TF-IDF based relevance score.

        Args:
            text: The text passage to score
            query: The query/learning goal

        Returns:
            TF-IDF relevance score (0-1)
        """
        text_tokens = self.tokenize(text)
        query_tokens = self.tokenize(query)

        if not text_tokens or not query_tokens:
            return 0.0

        text_tf = self.compute_tf(text_tokens)
        query_tf = self.compute_tf(query_tokens)

        # Compute TF-IDF vectors
        all_terms = set(text_tf.keys()) | set(query_tf.keys())

        text_vector = []
        query_vector = []

        for term in all_terms:
            idf = self._idf_cache.get(term, 1.0)
            text_vector.append(text_tf.get(term, 0) * idf)
            query_vector.append(query_tf.get(term, 0) * idf)

        # Cosine similarity
        text_arr = np.array(text_vector)
        query_arr = np.array(query_vector)

        dot_product = np.dot(text_arr, query_arr)
        text_norm = np.linalg.norm(text_arr)
        query_norm = np.linalg.norm(query_arr)

        if text_norm == 0 or query_norm == 0:
            return 0.0

        return float(dot_product / (text_norm * query_norm))

    def compute_semantic_score(
        self,
        text: str,
        query: str,
    ) -> float:
        """
        Compute BERT-based semantic similarity score.

        Args:
            text: The text passage to score
            query: The query/learning goal

        Returns:
            Semantic similarity score (0-1)
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            # Fallback to TF-IDF based score when semantic model unavailable
            return self.compute_tfidf_score(text, query)

        embeddings = self.model.encode([text, query], normalize_embeddings=True)
        similarity = np.dot(embeddings[0], embeddings[1])
        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        return float((similarity + 1) / 2)

    def score_relevance(
        self,
        text: str,
        query: str,
        tfidf_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> dict:
        """
        Compute combined relevance score using TF-IDF and semantic similarity.

        Args:
            text: The text passage to score
            query: The query/learning goal
            tfidf_weight: Weight for TF-IDF score (default 0.3)
            semantic_weight: Weight for semantic score (default 0.7)

        Returns:
            Dictionary with individual and combined scores
        """
        tfidf_score = self.compute_tfidf_score(text, query)

        # Use TF-IDF only if sentence_transformers not available
        if HAS_SENTENCE_TRANSFORMERS:
            semantic_score = self.compute_semantic_score(text, query)
            combined_score = (
                tfidf_weight * tfidf_score + semantic_weight * semantic_score
            )
        else:
            semantic_score = tfidf_score  # Fallback
            combined_score = tfidf_score

        return {
            "tfidf_score": round(tfidf_score, 4),
            "semantic_score": round(semantic_score, 4),
            "combined_score": round(combined_score, 4),
            "relevance_level": self._get_relevance_level(combined_score),
            "using_semantic": HAS_SENTENCE_TRANSFORMERS,
        }

    def _get_relevance_level(self, score: float) -> str:
        """Convert numeric score to human-readable relevance level."""
        if score >= 0.8:
            return "highly_relevant"
        elif score >= 0.6:
            return "relevant"
        elif score >= 0.4:
            return "somewhat_relevant"
        elif score >= 0.2:
            return "marginally_relevant"
        else:
            return "not_relevant"

    def rank_passages(
        self,
        passages: list[str],
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Rank multiple passages by relevance to a query.

        Args:
            passages: List of text passages to rank
            query: The query/learning goal
            top_k: Return only top K results (None for all)

        Returns:
            List of passages with scores, sorted by relevance
        """
        results = []

        for i, passage in enumerate(passages):
            score_data = self.score_relevance(passage, query)
            results.append(
                {
                    "index": i,
                    "passage": passage[:200] + "..." if len(passage) > 200 else passage,
                    "full_passage": passage,
                    **score_data,
                }
            )

        # Sort by combined score descending
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def compute_focus_order(
        self,
        sections: list[dict],
        learning_goal: str,
    ) -> list[dict]:
        """
        Compute optimal reading order for focus mode based on learning goals.

        Args:
            sections: List of document sections with 'title' and 'content'
            learning_goal: User's stated learning goal

        Returns:
            Sections reordered by relevance with focus scores
        """
        scored_sections = []

        for section in sections:
            text = f"{section.get('title', '')} {section.get('content', '')}"
            score_data = self.score_relevance(text, learning_goal)

            scored_sections.append(
                {
                    **section,
                    "focus_score": score_data["combined_score"],
                    "relevance_level": score_data["relevance_level"],
                    "should_expand": score_data["combined_score"] >= 0.5,
                }
            )

        # Sort by focus score
        scored_sections.sort(key=lambda x: x["focus_score"], reverse=True)

        # Add focus order index
        for i, section in enumerate(scored_sections):
            section["focus_order"] = i + 1

        return scored_sections


# Singleton instance
_relevance_scorer: Optional[RelevanceScorer] = None


def get_relevance_scorer() -> RelevanceScorer:
    """Get or create singleton RelevanceScorer instance."""
    global _relevance_scorer
    if _relevance_scorer is None:
        _relevance_scorer = RelevanceScorer()
    return _relevance_scorer
