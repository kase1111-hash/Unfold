"""Entity extraction service using spaCy and LLM."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.models.graph import NodeType, RelationType


class EntityType(str, Enum):
    """Entity types for extraction."""

    CONCEPT = "CONCEPT"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    METHOD = "METHOD"
    DATASET = "DATASET"
    TECHNOLOGY = "TECHNOLOGY"
    THEORY = "THEORY"
    TERM = "TERM"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from text."""

    text: str
    entity_type: EntityType
    start_char: int
    end_char: int
    confidence: float
    metadata: dict[str, Any] | None = None

    def to_node_type(self) -> NodeType:
        """Convert entity type to graph node type."""
        mapping = {
            EntityType.CONCEPT: NodeType.CONCEPT,
            EntityType.PERSON: NodeType.AUTHOR,
            EntityType.ORGANIZATION: NodeType.INSTITUTION,
            EntityType.METHOD: NodeType.METHOD,
            EntityType.DATASET: NodeType.DATASET,
            EntityType.TECHNOLOGY: NodeType.CONCEPT,
            EntityType.THEORY: NodeType.CONCEPT,
            EntityType.TERM: NodeType.TERM,
        }
        return mapping.get(self.entity_type, NodeType.CONCEPT)


@dataclass
class ExtractedRelation:
    """Represents an extracted relation between entities."""

    source_text: str
    target_text: str
    relation_type: RelationType
    confidence: float
    context: str | None = None


class EntityExtractor:
    """Extract entities from text using spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize entity extractor.

        Args:
            model_name: spaCy model to use
        """
        self._nlp = None
        self._model_name = model_name

    def _load_model(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load(self._model_name)
            except OSError:
                # Model not installed, try downloading
                import spacy.cli

                spacy.cli.download(self._model_name)
                import spacy

                self._nlp = spacy.load(self._model_name)
        return self._nlp

    def extract_entities(
        self,
        text: str,
        include_noun_chunks: bool = True,
        min_confidence: float = 0.5,
    ) -> list[ExtractedEntity]:
        """Extract entities from text using spaCy NER.

        Args:
            text: Input text to extract entities from
            include_noun_chunks: Whether to include noun chunks as concepts
            min_confidence: Minimum confidence threshold

        Returns:
            List of extracted entities
        """
        nlp = self._load_model()
        doc = nlp(text)
        entities = []

        # Map spaCy entity labels to our types
        label_mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.ORGANIZATION,  # Geopolitical entities
            "PRODUCT": EntityType.TECHNOLOGY,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LAW": EntityType.CONCEPT,
            "EVENT": EntityType.CONCEPT,
            "NORP": EntityType.CONCEPT,  # Nationalities, religious/political groups
            "FAC": EntityType.CONCEPT,  # Facilities
        }

        # Extract named entities
        for ent in doc.ents:
            entity_type = label_mapping.get(ent.label_, EntityType.TERM)

            entities.append(
                ExtractedEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.85,  # spaCy doesn't provide confidence
                    metadata={"spacy_label": ent.label_},
                )
            )

        # Extract noun chunks as potential concepts
        if include_noun_chunks:
            seen_spans = {(e.start_char, e.end_char) for e in entities}

            for chunk in doc.noun_chunks:
                # Skip if overlaps with existing entity
                if (chunk.start_char, chunk.end_char) in seen_spans:
                    continue

                # Skip very short or generic chunks
                if len(chunk.text) < 3 or chunk.text.lower() in {
                    "it",
                    "this",
                    "that",
                    "they",
                }:
                    continue

                # Check if it's a meaningful concept (has noun as root)
                if chunk.root.pos_ in {"NOUN", "PROPN"}:
                    entities.append(
                        ExtractedEntity(
                            text=chunk.text,
                            entity_type=EntityType.CONCEPT,
                            start_char=chunk.start_char,
                            end_char=chunk.end_char,
                            confidence=0.6,
                            metadata={"source": "noun_chunk"},
                        )
                    )

        # Filter by confidence
        entities = [e for e in entities if e.confidence >= min_confidence]

        # Deduplicate by text (keep highest confidence)
        seen_texts: dict[str, ExtractedEntity] = {}
        for entity in entities:
            key = entity.text.lower()
            if key not in seen_texts or entity.confidence > seen_texts[key].confidence:
                seen_texts[key] = entity

        return list(seen_texts.values())

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Extract keywords from text using TF-IDF-like scoring.

        Args:
            text: Input text
            top_k: Number of keywords to return

        Returns:
            List of (keyword, score) tuples
        """
        nlp = self._load_model()
        doc = nlp(text)

        # Count term frequencies for nouns and proper nouns
        term_freq: dict[str, int] = {}

        for token in doc:
            if (
                token.pos_ in {"NOUN", "PROPN"}
                and not token.is_stop
                and len(token.text) > 2
            ):
                lemma = token.lemma_.lower()
                term_freq[lemma] = term_freq.get(lemma, 0) + 1

        # Sort by frequency and return top_k
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)

        # Normalize scores
        max_freq = sorted_terms[0][1] if sorted_terms else 1
        return [(term, freq / max_freq) for term, freq in sorted_terms[:top_k]]

    def extract_sentences_with_entities(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """Extract sentences along with their entities.

        Args:
            text: Input text

        Returns:
            List of dicts with sentence and entities
        """
        nlp = self._load_model()
        doc = nlp(text)

        results = []
        for sent in doc.sents:
            sent_entities = []

            for ent in sent.ents:
                sent_entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char - sent.start_char,
                        "end": ent.end_char - sent.start_char,
                    }
                )

            results.append(
                {
                    "sentence": sent.text,
                    "entities": sent_entities,
                    "start_char": sent.start_char,
                    "end_char": sent.end_char,
                }
            )

        return results


# Global extractor instance
_extractor: EntityExtractor | None = None


def get_entity_extractor() -> EntityExtractor:
    """Get or create global entity extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor
