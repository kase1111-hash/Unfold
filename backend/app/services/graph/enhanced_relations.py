"""
Enhanced Relation Extraction Module

This module addresses the coherence gap in relation extraction by implementing:
1. Syntactic pattern matching using dependency structures
2. Semantic verb frame analysis
3. LLM-based extraction for complex relations
4. Multi-signal confidence calibration
5. Relation validation and filtering

The goal is to improve relation coherence from ~59% to >85%.
"""

import re
import hashlib
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """Enhanced relation types with semantic categories."""
    # Causal/Creative Relations
    CREATES = "CREATES"          # Person/Org creates Technology
    DEVELOPS = "DEVELOPS"        # Person/Org develops Method
    INTRODUCES = "INTRODUCES"    # Person/Org introduces Concept
    DISCOVERS = "DISCOVERS"      # Person discovers Concept

    # Structural Relations
    PART_OF = "PART_OF"          # Component is part of System
    CONTAINS = "CONTAINS"        # System contains Component
    IMPLEMENTS = "IMPLEMENTS"    # Technology implements Method
    EXTENDS = "EXTENDS"          # Technology extends another

    # Functional Relations
    USES = "USES"                # System uses Method/Technology
    APPLIES_TO = "APPLIES_TO"    # Method applies to Domain
    ENABLES = "ENABLES"          # Technology enables Capability
    REQUIRES = "REQUIRES"        # Method requires Prerequisite

    # Comparative Relations
    IMPROVES = "IMPROVES"        # New improves Old
    REPLACES = "REPLACES"        # New replaces Old
    CONTRASTS_WITH = "CONTRASTS_WITH"  # A differs from B
    SIMILAR_TO = "SIMILAR_TO"    # A resembles B

    # Associative Relations
    RELATED_TO = "RELATED_TO"    # Generic association
    CITES = "CITES"              # Work cites another
    AFFILIATED_WITH = "AFFILIATED_WITH"  # Person affiliated with Org
    TRAINED_ON = "TRAINED_ON"    # Model trained on Dataset
    EVALUATED_ON = "EVALUATED_ON"  # Model evaluated on Benchmark


class EntityType(str, Enum):
    CONCEPT = "CONCEPT"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    METHOD = "METHOD"
    TECHNOLOGY = "TECHNOLOGY"
    DATASET = "DATASET"
    METRIC = "METRIC"


@dataclass
class Entity:
    """Entity with position information."""
    text: str
    type: EntityType
    confidence: float
    start: int = 0
    end: int = 0
    normalized: str = ""

    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.text.lower().strip()


@dataclass
class ExtractedRelation:
    """A relation extracted between two entities."""
    source: Entity
    target: Entity
    relation_type: RelationType
    confidence: float
    evidence: str
    extraction_method: str  # "syntactic", "semantic", "llm", "cooccurrence"
    metadata: Dict = field(default_factory=dict)


@dataclass
class RelationPattern:
    """A syntactic pattern for relation extraction."""
    pattern: str  # Regex pattern
    relation_type: RelationType
    source_types: Set[EntityType]
    target_types: Set[EntityType]
    confidence_boost: float = 0.0
    bidirectional: bool = False


# ============================================================================
# SYNTACTIC RELATION PATTERNS
# ============================================================================

# Verb-based patterns that indicate specific relations
VERB_RELATION_PATTERNS: List[RelationPattern] = [
    # Creation/Introduction patterns
    RelationPattern(
        pattern=r"(?:introduced|proposed|presented|developed|created|invented|designed)",
        relation_type=RelationType.INTRODUCES,
        source_types={EntityType.PERSON, EntityType.ORGANIZATION},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD, EntityType.CONCEPT},
        confidence_boost=0.2
    ),
    RelationPattern(
        pattern=r"(?:built|constructed|implemented|engineered)",
        relation_type=RelationType.CREATES,
        source_types={EntityType.PERSON, EntityType.ORGANIZATION},
        target_types={EntityType.TECHNOLOGY},
        confidence_boost=0.2
    ),

    # Usage patterns
    RelationPattern(
        pattern=r"(?:uses?|utilizes?|employs?|leverages?|applies)",
        relation_type=RelationType.USES,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.METHOD, EntityType.TECHNOLOGY, EntityType.CONCEPT},
        confidence_boost=0.15
    ),
    RelationPattern(
        pattern=r"(?:relies\s+on|depends\s+on|based\s+on|built\s+(?:on|upon))",
        relation_type=RelationType.USES,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.METHOD, EntityType.TECHNOLOGY},
        confidence_boost=0.2
    ),

    # Structural patterns
    RelationPattern(
        pattern=r"(?:consists?\s+of|comprises?|includes?|contains?)",
        relation_type=RelationType.CONTAINS,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.METHOD, EntityType.CONCEPT},
        confidence_boost=0.15
    ),
    RelationPattern(
        pattern=r"(?:is\s+)?(?:a\s+)?(?:part|component|element)\s+of",
        relation_type=RelationType.PART_OF,
        source_types={EntityType.METHOD, EntityType.CONCEPT},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        confidence_boost=0.2
    ),

    # Extension/Improvement patterns
    RelationPattern(
        pattern=r"(?:extends?|expands?|augments?|enhances?)",
        relation_type=RelationType.EXTENDS,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        confidence_boost=0.15
    ),
    RelationPattern(
        pattern=r"(?:improves?(?:\s+(?:on|upon))?|outperforms?|surpasses?|exceeds?)",
        relation_type=RelationType.IMPROVES,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        confidence_boost=0.2
    ),

    # Contrast patterns
    RelationPattern(
        pattern=r"(?:unlike|(?:in\s+)?contrast\s+(?:to|with)|differs?\s+from|versus|vs\.?)",
        relation_type=RelationType.CONTRASTS_WITH,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD, EntityType.CONCEPT},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD, EntityType.CONCEPT},
        confidence_boost=0.2,
        bidirectional=True
    ),
    RelationPattern(
        pattern=r"(?:replaces?|supersedes?|instead\s+of|rather\s+than)",
        relation_type=RelationType.REPLACES,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        confidence_boost=0.2
    ),

    # Training/Evaluation patterns
    RelationPattern(
        pattern=r"(?:trained\s+on|learned\s+from|pre-?trained\s+(?:on|with))",
        relation_type=RelationType.TRAINED_ON,
        source_types={EntityType.TECHNOLOGY},
        target_types={EntityType.DATASET},
        confidence_boost=0.25
    ),
    RelationPattern(
        pattern=r"(?:evaluated\s+on|tested\s+on|benchmarked\s+(?:on|against))",
        relation_type=RelationType.EVALUATED_ON,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.DATASET},
        confidence_boost=0.25
    ),

    # Enablement patterns
    RelationPattern(
        pattern=r"(?:enables?|allows?|permits?|facilitates?|supports?)",
        relation_type=RelationType.ENABLES,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.CONCEPT, EntityType.METHOD},
        confidence_boost=0.15
    ),

    # Affiliation patterns
    RelationPattern(
        pattern=r"(?:at|from|of|works?\s+(?:at|for))\s+",
        relation_type=RelationType.AFFILIATED_WITH,
        source_types={EntityType.PERSON},
        target_types={EntityType.ORGANIZATION},
        confidence_boost=0.2
    ),
]


# Entity type compatibility matrix for relation types
# Maps (source_type, target_type) -> list of valid relation types
VALID_RELATION_TYPES: Dict[Tuple[EntityType, EntityType], List[RelationType]] = {
    # Person -> X
    (EntityType.PERSON, EntityType.TECHNOLOGY): [
        RelationType.INTRODUCES, RelationType.CREATES, RelationType.DEVELOPS
    ],
    (EntityType.PERSON, EntityType.METHOD): [
        RelationType.INTRODUCES, RelationType.DEVELOPS, RelationType.CREATES
    ],
    (EntityType.PERSON, EntityType.CONCEPT): [
        RelationType.INTRODUCES, RelationType.DISCOVERS
    ],
    (EntityType.PERSON, EntityType.ORGANIZATION): [
        RelationType.AFFILIATED_WITH
    ],
    (EntityType.PERSON, EntityType.PERSON): [
        RelationType.RELATED_TO, RelationType.CITES
    ],

    # Organization -> X
    (EntityType.ORGANIZATION, EntityType.TECHNOLOGY): [
        RelationType.CREATES, RelationType.DEVELOPS, RelationType.INTRODUCES
    ],
    (EntityType.ORGANIZATION, EntityType.METHOD): [
        RelationType.DEVELOPS, RelationType.INTRODUCES
    ],
    (EntityType.ORGANIZATION, EntityType.PERSON): [
        RelationType.AFFILIATED_WITH
    ],

    # Technology -> X
    (EntityType.TECHNOLOGY, EntityType.METHOD): [
        RelationType.USES, RelationType.IMPLEMENTS, RelationType.CONTAINS, RelationType.PART_OF
    ],
    (EntityType.TECHNOLOGY, EntityType.TECHNOLOGY): [
        RelationType.EXTENDS, RelationType.IMPROVES, RelationType.REPLACES,
        RelationType.CONTRASTS_WITH, RelationType.SIMILAR_TO, RelationType.USES
    ],
    (EntityType.TECHNOLOGY, EntityType.CONCEPT): [
        RelationType.ENABLES, RelationType.APPLIES_TO, RelationType.RELATED_TO
    ],
    (EntityType.TECHNOLOGY, EntityType.DATASET): [
        RelationType.TRAINED_ON, RelationType.EVALUATED_ON
    ],

    # Method -> X
    (EntityType.METHOD, EntityType.METHOD): [
        RelationType.EXTENDS, RelationType.IMPROVES, RelationType.CONTRASTS_WITH,
        RelationType.SIMILAR_TO, RelationType.USES, RelationType.PART_OF
    ],
    (EntityType.METHOD, EntityType.TECHNOLOGY): [
        RelationType.PART_OF, RelationType.ENABLES, RelationType.USES
    ],
    (EntityType.METHOD, EntityType.CONCEPT): [
        RelationType.APPLIES_TO, RelationType.ENABLES, RelationType.RELATED_TO
    ],
    (EntityType.METHOD, EntityType.DATASET): [
        RelationType.EVALUATED_ON
    ],

    # Concept -> X
    (EntityType.CONCEPT, EntityType.CONCEPT): [
        RelationType.RELATED_TO, RelationType.CONTRASTS_WITH, RelationType.PART_OF
    ],
    (EntityType.CONCEPT, EntityType.TECHNOLOGY): [
        RelationType.ENABLES, RelationType.RELATED_TO
    ],
    (EntityType.CONCEPT, EntityType.METHOD): [
        RelationType.RELATED_TO, RelationType.ENABLES
    ],

    # Dataset -> X
    (EntityType.DATASET, EntityType.CONCEPT): [
        RelationType.RELATED_TO, RelationType.APPLIES_TO
    ],
}


class EnhancedRelationExtractor:
    """
    Multi-layered relation extraction with confidence calibration.

    Extraction Pipeline:
    1. Syntactic Pattern Matching - High precision verb patterns
    2. Semantic Type Compatibility - Entity type constraints
    3. Context Window Analysis - Proximity-based extraction
    4. LLM Enhancement (optional) - Complex relation inference
    5. Confidence Calibration - Multi-signal scoring
    6. Validation & Filtering - Remove low-quality relations
    """

    def __init__(
        self,
        llm_client=None,
        min_confidence: float = 0.4,
        max_context_distance: int = 200,  # characters
        enable_llm: bool = False
    ):
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.max_context_distance = max_context_distance
        self.enable_llm = enable_llm and llm_client is not None

        # Compile patterns
        self.compiled_patterns = [
            (re.compile(p.pattern, re.IGNORECASE), p)
            for p in VERB_RELATION_PATTERNS
        ]

    def extract_relations(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[ExtractedRelation]:
        """
        Extract relations using the multi-layered pipeline.

        Args:
            text: Source text
            entities: List of extracted entities with positions

        Returns:
            List of extracted relations with confidence scores
        """
        all_relations = []

        # Split into sentences for context
        sentences = self._split_sentences(text)

        # Map entities to sentences
        entity_sentences = self._map_entities_to_sentences(entities, sentences)

        # Layer 1: Syntactic Pattern Extraction
        syntactic_relations = self._extract_syntactic_relations(
            text, entities, sentences, entity_sentences
        )
        all_relations.extend(syntactic_relations)

        # Layer 2: Co-occurrence with Type Constraints
        cooccurrence_relations = self._extract_cooccurrence_relations(
            entities, sentences, entity_sentences
        )
        all_relations.extend(cooccurrence_relations)

        # Layer 3: LLM Enhancement (if enabled)
        if self.enable_llm:
            llm_relations = self._extract_llm_relations(
                text, entities, sentences
            )
            all_relations.extend(llm_relations)

        # Layer 4: Confidence Calibration
        calibrated_relations = self._calibrate_confidence(all_relations)

        # Layer 5: Validation & Deduplication
        validated_relations = self._validate_and_deduplicate(calibrated_relations)

        # Filter by minimum confidence
        final_relations = [
            r for r in validated_relations
            if r.confidence >= self.min_confidence
        ]

        return sorted(final_relations, key=lambda r: r.confidence, reverse=True)

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with position tracking."""
        sentences = []
        # Simple sentence splitting (could use NLTK/spaCy for better results)
        pattern = r'(?<=[.!?])\s+'

        start = 0
        for match in re.finditer(pattern, text):
            end = match.start() + 1
            sentence = text[start:end].strip()
            if sentence:
                sentences.append((sentence, start, end))
            start = match.end()

        # Add last sentence
        if start < len(text):
            sentence = text[start:].strip()
            if sentence:
                sentences.append((sentence, start, len(text)))

        return sentences

    def _map_entities_to_sentences(
        self,
        entities: List[Entity],
        sentences: List[Tuple[str, int, int]]
    ) -> Dict[int, List[Entity]]:
        """Map each entity to its containing sentence."""
        entity_map = defaultdict(list)

        for entity in entities:
            for sent_idx, (sent_text, sent_start, sent_end) in enumerate(sentences):
                # Check if entity text appears in sentence
                if entity.normalized in sent_text.lower():
                    entity_map[sent_idx].append(entity)
                    break
                # Also check by position if available
                elif entity.start >= sent_start and entity.end <= sent_end:
                    entity_map[sent_idx].append(entity)
                    break

        return dict(entity_map)

    def _extract_syntactic_relations(
        self,
        text: str,
        entities: List[Entity],
        sentences: List[Tuple[str, int, int]],
        entity_sentences: Dict[int, List[Entity]]
    ) -> List[ExtractedRelation]:
        """Extract relations using syntactic verb patterns."""
        relations = []

        for sent_idx, (sentence, _, _) in enumerate(sentences):
            sent_entities = entity_sentences.get(sent_idx, [])
            if len(sent_entities) < 2:
                continue

            # Check each pattern against the sentence
            for compiled_pattern, pattern_info in self.compiled_patterns:
                if not compiled_pattern.search(sentence):
                    continue

                # Find entity pairs that match the pattern's type constraints
                for i, source in enumerate(sent_entities):
                    for target in sent_entities[i+1:]:
                        # Check type compatibility
                        if (source.type in pattern_info.source_types and
                            target.type in pattern_info.target_types):

                            confidence = self._calculate_base_confidence(
                                source, target, sentence
                            ) + pattern_info.confidence_boost

                            relations.append(ExtractedRelation(
                                source=source,
                                target=target,
                                relation_type=pattern_info.relation_type,
                                confidence=min(confidence, 0.95),
                                evidence=sentence[:200],
                                extraction_method="syntactic",
                                metadata={"pattern": pattern_info.pattern}
                            ))

                        # Handle bidirectional patterns
                        if pattern_info.bidirectional:
                            if (target.type in pattern_info.source_types and
                                source.type in pattern_info.target_types):

                                confidence = self._calculate_base_confidence(
                                    target, source, sentence
                                ) + pattern_info.confidence_boost

                                relations.append(ExtractedRelation(
                                    source=target,
                                    target=source,
                                    relation_type=pattern_info.relation_type,
                                    confidence=min(confidence, 0.95),
                                    evidence=sentence[:200],
                                    extraction_method="syntactic",
                                    metadata={"pattern": pattern_info.pattern}
                                ))

        return relations

    def _extract_cooccurrence_relations(
        self,
        entities: List[Entity],
        sentences: List[Tuple[str, int, int]],
        entity_sentences: Dict[int, List[Entity]]
    ) -> List[ExtractedRelation]:
        """Extract relations based on entity co-occurrence with type constraints."""
        relations = []

        for sent_idx, sent_entities in entity_sentences.items():
            if len(sent_entities) < 2:
                continue

            sentence = sentences[sent_idx][0] if sent_idx < len(sentences) else ""

            for i, source in enumerate(sent_entities):
                for target in sent_entities[i+1:]:
                    if source.normalized == target.normalized:
                        continue

                    # Get valid relation types for this entity pair
                    type_key = (source.type, target.type)
                    valid_types = VALID_RELATION_TYPES.get(type_key, [RelationType.RELATED_TO])

                    if not valid_types:
                        continue

                    # Use the most general valid type (RELATED_TO if available, else first)
                    if RelationType.RELATED_TO in valid_types:
                        rel_type = RelationType.RELATED_TO
                    else:
                        rel_type = valid_types[0]

                    confidence = self._calculate_base_confidence(
                        source, target, sentence
                    ) * 0.7  # Lower confidence for cooccurrence

                    relations.append(ExtractedRelation(
                        source=source,
                        target=target,
                        relation_type=rel_type,
                        confidence=confidence,
                        evidence=sentence[:200],
                        extraction_method="cooccurrence",
                        metadata={"valid_types": [t.value for t in valid_types]}
                    ))

        return relations

    def _extract_llm_relations(
        self,
        text: str,
        entities: List[Entity],
        sentences: List[Tuple[str, int, int]]
    ) -> List[ExtractedRelation]:
        """Use LLM to extract complex relations (optional enhancement)."""
        if not self.llm_client:
            return []

        relations = []

        # Group entities for efficient LLM calls
        entity_texts = [e.text for e in entities[:30]]  # Limit for API efficiency

        prompt = f"""Extract semantic relationships between the following entities from the given text.

Entities: {', '.join(entity_texts)}

Text excerpt: {text[:2000]}

For each relationship found, output in this format:
SOURCE_ENTITY | RELATION_TYPE | TARGET_ENTITY | CONFIDENCE

Valid relation types: INTRODUCES, CREATES, USES, PART_OF, EXTENDS, IMPROVES, CONTRASTS_WITH, TRAINED_ON, EVALUATED_ON, ENABLES, AFFILIATED_WITH

Output only clear, well-supported relationships. Be precise about directionality.
"""

        try:
            # This would call the LLM - placeholder for actual implementation
            # response = self.llm_client.complete(prompt)
            # Parse response and create ExtractedRelation objects
            pass
        except Exception as e:
            logger.warning(f"LLM relation extraction failed: {e}")

        return relations

    def _calculate_base_confidence(
        self,
        source: Entity,
        target: Entity,
        context: str
    ) -> float:
        """Calculate base confidence from entity confidence and context."""
        # Start with entity confidences
        base = min(source.confidence, target.confidence)

        # Boost for proximity in text
        try:
            source_pos = context.lower().find(source.normalized)
            target_pos = context.lower().find(target.normalized)
            if source_pos >= 0 and target_pos >= 0:
                distance = abs(target_pos - source_pos)
                if distance < 50:
                    base += 0.1
                elif distance < 100:
                    base += 0.05
        except:
            pass

        return min(base, 0.95)

    def _calibrate_confidence(
        self,
        relations: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """Calibrate confidence scores based on multiple signals."""
        # Group relations by source-target pair
        pair_relations = defaultdict(list)
        for rel in relations:
            key = (rel.source.normalized, rel.target.normalized)
            pair_relations[key].append(rel)

        calibrated = []

        for (source_norm, target_norm), rels in pair_relations.items():
            # If multiple extraction methods found the same relation, boost confidence
            methods = set(r.extraction_method for r in rels)
            method_boost = 0.1 * (len(methods) - 1)

            # If multiple relations of same type, boost confidence
            type_counts = defaultdict(int)
            for r in rels:
                type_counts[r.relation_type] += 1

            for rel in rels:
                # Apply calibration
                new_confidence = rel.confidence + method_boost

                # Boost for repeated same-type relations
                if type_counts[rel.relation_type] > 1:
                    new_confidence += 0.05 * (type_counts[rel.relation_type] - 1)

                # Penalty for syntactic method when types don't match well
                if rel.extraction_method == "syntactic":
                    type_key = (rel.source.type, rel.target.type)
                    if type_key not in VALID_RELATION_TYPES:
                        new_confidence -= 0.15

                calibrated.append(ExtractedRelation(
                    source=rel.source,
                    target=rel.target,
                    relation_type=rel.relation_type,
                    confidence=min(max(new_confidence, 0.0), 0.99),
                    evidence=rel.evidence,
                    extraction_method=rel.extraction_method,
                    metadata={**rel.metadata, "calibrated": True}
                ))

        return calibrated

    def _validate_and_deduplicate(
        self,
        relations: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """Validate relations and remove duplicates."""
        seen = {}
        validated = []

        for rel in relations:
            # Validation checks
            if not self._is_valid_relation(rel):
                continue

            # Deduplication key
            key = (
                rel.source.normalized,
                rel.target.normalized,
                rel.relation_type
            )

            # Keep highest confidence version
            if key not in seen or seen[key].confidence < rel.confidence:
                seen[key] = rel

        return list(seen.values())

    def _is_valid_relation(self, rel: ExtractedRelation) -> bool:
        """Check if a relation is valid."""
        # Self-relations are invalid
        if rel.source.normalized == rel.target.normalized:
            return False

        # Check entity type compatibility
        type_key = (rel.source.type, rel.target.type)
        valid_types = VALID_RELATION_TYPES.get(type_key, [])

        # If we have specific valid types, check compatibility
        if valid_types and rel.relation_type not in valid_types:
            # Allow RELATED_TO as fallback
            if rel.relation_type != RelationType.RELATED_TO:
                return False

        # Minimum confidence threshold
        if rel.confidence < 0.2:
            return False

        return True


def create_enhanced_extractor(
    llm_client=None,
    min_confidence: float = 0.4,
    enable_llm: bool = False
) -> EnhancedRelationExtractor:
    """Factory function to create an enhanced relation extractor."""
    return EnhancedRelationExtractor(
        llm_client=llm_client,
        min_confidence=min_confidence,
        enable_llm=enable_llm
    )
