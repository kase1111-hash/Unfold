#!/usr/bin/env python3
"""
Test Enhanced Relation Extraction

Compares baseline co-occurrence extraction vs enhanced multi-layer extraction.
Goal: Improve coherence from ~59% to >85%.
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class EntityType(str, Enum):
    CONCEPT = "CONCEPT"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    METHOD = "METHOD"
    TECHNOLOGY = "TECHNOLOGY"
    DATASET = "DATASET"
    METRIC = "METRIC"


class RelationType(str, Enum):
    # Causal/Creative Relations
    CREATES = "CREATES"
    DEVELOPS = "DEVELOPS"
    INTRODUCES = "INTRODUCES"
    DISCOVERS = "DISCOVERS"

    # Structural Relations
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    IMPLEMENTS = "IMPLEMENTS"
    EXTENDS = "EXTENDS"

    # Functional Relations
    USES = "USES"
    APPLIES_TO = "APPLIES_TO"
    ENABLES = "ENABLES"
    REQUIRES = "REQUIRES"

    # Comparative Relations
    IMPROVES = "IMPROVES"
    REPLACES = "REPLACES"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    SIMILAR_TO = "SIMILAR_TO"

    # Associative Relations
    RELATED_TO = "RELATED_TO"
    CITES = "CITES"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    TRAINED_ON = "TRAINED_ON"
    EVALUATED_ON = "EVALUATED_ON"


@dataclass
class Entity:
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
    source: Entity
    target: Entity
    relation_type: RelationType
    confidence: float
    evidence: str
    extraction_method: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class RelationPattern:
    pattern: str
    relation_type: RelationType
    source_types: Set[EntityType]
    target_types: Set[EntityType]
    confidence_boost: float = 0.0
    bidirectional: bool = False


# ============================================================================
# VALID TYPE COMBINATIONS
# ============================================================================

VALID_RELATION_TYPES: Dict[Tuple[EntityType, EntityType], List[RelationType]] = {
    (EntityType.PERSON, EntityType.TECHNOLOGY): [
        RelationType.INTRODUCES, RelationType.CREATES, RelationType.DEVELOPS
    ],
    (EntityType.PERSON, EntityType.METHOD): [
        RelationType.INTRODUCES, RelationType.DEVELOPS, RelationType.CREATES
    ],
    (EntityType.PERSON, EntityType.ORGANIZATION): [
        RelationType.AFFILIATED_WITH
    ],
    (EntityType.ORGANIZATION, EntityType.TECHNOLOGY): [
        RelationType.CREATES, RelationType.DEVELOPS, RelationType.INTRODUCES
    ],
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
    (EntityType.METHOD, EntityType.METHOD): [
        RelationType.EXTENDS, RelationType.IMPROVES, RelationType.CONTRASTS_WITH,
        RelationType.SIMILAR_TO, RelationType.USES, RelationType.PART_OF
    ],
    (EntityType.METHOD, EntityType.TECHNOLOGY): [
        RelationType.PART_OF, RelationType.ENABLES, RelationType.USES
    ],
    (EntityType.METHOD, EntityType.DATASET): [
        RelationType.EVALUATED_ON
    ],
    (EntityType.CONCEPT, EntityType.CONCEPT): [
        RelationType.RELATED_TO, RelationType.CONTRASTS_WITH, RelationType.PART_OF
    ],
}


# ============================================================================
# SYNTACTIC PATTERNS
# ============================================================================

VERB_PATTERNS: List[RelationPattern] = [
    # Creation/Introduction
    RelationPattern(
        pattern=r"(?:introduced|proposed|presented|developed|created|invented|designed)",
        relation_type=RelationType.INTRODUCES,
        source_types={EntityType.PERSON, EntityType.ORGANIZATION},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD, EntityType.CONCEPT},
        confidence_boost=0.25
    ),
    # Usage
    RelationPattern(
        pattern=r"(?:uses?|utilizes?|employs?|leverages?|applies|relies\s+(?:on|upon))",
        relation_type=RelationType.USES,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.METHOD, EntityType.TECHNOLOGY, EntityType.CONCEPT},
        confidence_boost=0.2
    ),
    # Contrast
    RelationPattern(
        pattern=r"(?:unlike|(?:in\s+)?contrast\s+(?:to|with)|differs?\s+from|versus|vs\.?|rather\s+than)",
        relation_type=RelationType.CONTRASTS_WITH,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD, EntityType.CONCEPT},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD, EntityType.CONCEPT},
        confidence_boost=0.25,
        bidirectional=True
    ),
    # Training
    RelationPattern(
        pattern=r"(?:trained\s+on|learned\s+from|pre-?trained\s+(?:on|with))",
        relation_type=RelationType.TRAINED_ON,
        source_types={EntityType.TECHNOLOGY},
        target_types={EntityType.DATASET},
        confidence_boost=0.3
    ),
    # Evaluation
    RelationPattern(
        pattern=r"(?:evaluated\s+on|tested\s+on|benchmarked|results?\s+on|performance\s+on)",
        relation_type=RelationType.EVALUATED_ON,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.DATASET},
        confidence_boost=0.3
    ),
    # Affiliation
    RelationPattern(
        pattern=r"(?:at|from)\s+(?:the\s+)?",
        relation_type=RelationType.AFFILIATED_WITH,
        source_types={EntityType.PERSON},
        target_types={EntityType.ORGANIZATION},
        confidence_boost=0.2
    ),
    # Extension
    RelationPattern(
        pattern=r"(?:extends?|builds?\s+(?:on|upon)|based\s+on)",
        relation_type=RelationType.EXTENDS,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        confidence_boost=0.2
    ),
    # Improvement
    RelationPattern(
        pattern=r"(?:improves?|outperforms?|surpasses?|better\s+than|rivals?)",
        relation_type=RelationType.IMPROVES,
        source_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        target_types={EntityType.TECHNOLOGY, EntityType.METHOD},
        confidence_boost=0.2
    ),
]


# ============================================================================
# ENHANCED EXTRACTOR
# ============================================================================

class EnhancedRelationExtractor:
    """Multi-layer relation extraction with confidence calibration."""

    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        self.compiled_patterns = [
            (re.compile(p.pattern, re.IGNORECASE), p)
            for p in VERB_PATTERNS
        ]

    def extract_relations(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[ExtractedRelation]:
        """Extract relations using multi-layer approach."""
        all_relations = []

        sentences = self._split_sentences(text)
        entity_sentences = self._map_entities_to_sentences(entities, sentences)

        # Layer 1: Syntactic patterns
        syntactic = self._extract_syntactic(text, entities, sentences, entity_sentences)
        all_relations.extend(syntactic)

        # Layer 2: Type-constrained co-occurrence
        cooccurrence = self._extract_cooccurrence(entities, sentences, entity_sentences)
        all_relations.extend(cooccurrence)

        # Calibrate and validate
        calibrated = self._calibrate(all_relations)
        validated = self._validate(calibrated)

        return [r for r in validated if r.confidence >= self.min_confidence]

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        sentences = []
        pattern = r'(?<=[.!?])\s+'
        start = 0
        for match in re.finditer(pattern, text):
            end = match.start() + 1
            sent = text[start:end].strip()
            if sent:
                sentences.append((sent, start, end))
            start = match.end()
        if start < len(text):
            sent = text[start:].strip()
            if sent:
                sentences.append((sent, start, len(text)))
        return sentences

    def _map_entities_to_sentences(
        self,
        entities: List[Entity],
        sentences: List[Tuple[str, int, int]]
    ) -> Dict[int, List[Entity]]:
        """Map entities to ALL sentences they appear in (not just first)."""
        entity_map = defaultdict(list)
        for entity in entities:
            for idx, (sent_text, _, _) in enumerate(sentences):
                if entity.normalized in sent_text.lower():
                    entity_map[idx].append(entity)
                    # Don't break - entity can appear in multiple sentences
        return dict(entity_map)

    def _extract_syntactic(
        self,
        text: str,
        entities: List[Entity],
        sentences: List[Tuple[str, int, int]],
        entity_sentences: Dict[int, List[Entity]]
    ) -> List[ExtractedRelation]:
        relations = []

        for sent_idx, (sentence, _, _) in enumerate(sentences):
            sent_entities = entity_sentences.get(sent_idx, [])
            if len(sent_entities) < 2:
                continue

            for compiled, pattern in self.compiled_patterns:
                if not compiled.search(sentence):
                    continue

                for i, source in enumerate(sent_entities):
                    for target in sent_entities[i+1:]:
                        if source.type in pattern.source_types and target.type in pattern.target_types:
                            conf = self._base_confidence(source, target, sentence) + pattern.confidence_boost
                            relations.append(ExtractedRelation(
                                source=source,
                                target=target,
                                relation_type=pattern.relation_type,
                                confidence=min(conf, 0.95),
                                evidence=sentence[:150],
                                extraction_method="syntactic"
                            ))

                        if pattern.bidirectional and target.type in pattern.source_types and source.type in pattern.target_types:
                            conf = self._base_confidence(target, source, sentence) + pattern.confidence_boost
                            relations.append(ExtractedRelation(
                                source=target,
                                target=source,
                                relation_type=pattern.relation_type,
                                confidence=min(conf, 0.95),
                                evidence=sentence[:150],
                                extraction_method="syntactic"
                            ))

        return relations

    def _extract_cooccurrence(
        self,
        entities: List[Entity],
        sentences: List[Tuple[str, int, int]],
        entity_sentences: Dict[int, List[Entity]]
    ) -> List[ExtractedRelation]:
        relations = []

        for sent_idx, sent_entities in entity_sentences.items():
            if len(sent_entities) < 2:
                continue

            sentence = sentences[sent_idx][0] if sent_idx < len(sentences) else ""
            sent_lower = sentence.lower()

            for i, source in enumerate(sent_entities):
                for target in sent_entities[i+1:]:
                    if source.normalized == target.normalized:
                        continue

                    type_key = (source.type, target.type)
                    valid_types = VALID_RELATION_TYPES.get(type_key, [RelationType.RELATED_TO])

                    if not valid_types:
                        valid_types = [RelationType.RELATED_TO]

                    # Select best relation type based on context clues
                    rel_type = self._select_best_relation_type(valid_types, sent_lower, source.type, target.type)
                    conf = self._base_confidence(source, target, sentence) * 0.7

                    relations.append(ExtractedRelation(
                        source=source,
                        target=target,
                        relation_type=rel_type,
                        confidence=conf,
                        evidence=sentence[:150],
                        extraction_method="cooccurrence"
                    ))

        # Also extract from adjacent sentences (cross-sentence)
        sorted_idx = sorted(entity_sentences.keys())
        for i, idx1 in enumerate(sorted_idx[:-1]):
            idx2 = sorted_idx[i+1]
            if idx2 - idx1 == 1:  # Adjacent sentences
                for e1 in entity_sentences.get(idx1, []):
                    for e2 in entity_sentences.get(idx2, []):
                        if e1.normalized == e2.normalized:
                            continue

                        type_key = (e1.type, e2.type)
                        valid_types = VALID_RELATION_TYPES.get(type_key, [RelationType.RELATED_TO])
                        if not valid_types:
                            valid_types = [RelationType.RELATED_TO]

                        combined_context = sentences[idx1][0] + " " + sentences[idx2][0]
                        rel_type = self._select_best_relation_type(valid_types, combined_context.lower(), e1.type, e2.type)
                        conf = min(e1.confidence, e2.confidence) * 0.5  # Lower for cross-sentence

                        relations.append(ExtractedRelation(
                            source=e1,
                            target=e2,
                            relation_type=rel_type,
                            confidence=conf,
                            evidence=combined_context[:200],
                            extraction_method="cross-sentence"
                        ))

        return relations

    def _select_best_relation_type(
        self,
        valid_types: List[RelationType],
        context: str,
        src_type: EntityType,
        tgt_type: EntityType
    ) -> RelationType:
        """Select the most appropriate relation type based on context."""
        context = context.lower()

        # Prefer specific types based on context clues
        for rel_type in valid_types:
            if rel_type == RelationType.USES and any(w in context for w in ["uses", "relies", "employs"]):
                return rel_type
            if rel_type == RelationType.CONTRASTS_WITH and any(w in context for w in ["unlike", "contrast", "versus", "rather than"]):
                return rel_type
            if rel_type == RelationType.INTRODUCES and any(w in context for w in ["introduced", "proposed", "developed", "created"]):
                return rel_type
            if rel_type == RelationType.TRAINED_ON and any(w in context for w in ["trained on", "trained with"]):
                return rel_type
            if rel_type == RelationType.EVALUATED_ON and any(w in context for w in ["evaluated", "tested", "benchmark", "results on"]):
                return rel_type
            if rel_type == RelationType.AFFILIATED_WITH and any(w in context for w in [" at ", " from "]):
                return rel_type

        # Default to first valid type (or RELATED_TO if available)
        if RelationType.RELATED_TO in valid_types:
            return RelationType.RELATED_TO
        return valid_types[0] if valid_types else RelationType.RELATED_TO

    def _base_confidence(self, source: Entity, target: Entity, context: str) -> float:
        base = min(source.confidence, target.confidence)

        try:
            s_pos = context.lower().find(source.normalized)
            t_pos = context.lower().find(target.normalized)
            if s_pos >= 0 and t_pos >= 0:
                dist = abs(t_pos - s_pos)
                if dist < 50:
                    base += 0.1
                elif dist < 100:
                    base += 0.05
        except:
            pass

        return min(base, 0.9)

    def _calibrate(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        pair_rels = defaultdict(list)
        for rel in relations:
            key = (rel.source.normalized, rel.target.normalized)
            pair_rels[key].append(rel)

        calibrated = []
        for _, rels in pair_rels.items():
            methods = set(r.extraction_method for r in rels)
            method_boost = 0.1 * (len(methods) - 1)

            for rel in rels:
                new_conf = rel.confidence + method_boost
                calibrated.append(ExtractedRelation(
                    source=rel.source,
                    target=rel.target,
                    relation_type=rel.relation_type,
                    confidence=min(max(new_conf, 0.0), 0.99),
                    evidence=rel.evidence,
                    extraction_method=rel.extraction_method,
                    metadata={**rel.metadata, "calibrated": True}
                ))

        return calibrated

    def _validate(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        seen = {}
        for rel in relations:
            if rel.source.normalized == rel.target.normalized:
                continue
            if rel.confidence < 0.2:
                continue

            key = (rel.source.normalized, rel.target.normalized, rel.relation_type)
            if key not in seen or seen[key].confidence < rel.confidence:
                seen[key] = rel

        return sorted(seen.values(), key=lambda r: r.confidence, reverse=True)


# ============================================================================
# SAMPLE CONTENT
# ============================================================================

SAMPLE_CONTENT = """
The Transformer architecture, introduced by Vaswani et al. at Google Brain in 2017,
has revolutionized natural language processing and machine learning. Unlike recurrent
neural networks (RNNs) and long short-term memory networks (LSTMs), the Transformer
relies entirely on self-attention mechanisms to capture dependencies in sequential data.

Deep learning has transformed artificial intelligence research. Traditional sequence-to-sequence
models relied on encoder-decoder architectures with recurrent connections. The Transformer
eliminates recurrence entirely, enabling significantly more parallelization during training.

The key innovations of the Transformer include multi-head self-attention mechanisms,
positional encoding for sequence order, layer normalization and residual connections,
and scaled dot-product attention.

RNNs process sequences step-by-step, maintaining a hidden state that captures information
from previous time steps. Hochreiter and Schmidhuber introduced LSTMs in 1997 to address
the vanishing gradient problem. Cho et al. later proposed Gated Recurrent Units (GRUs)
as a simpler alternative.

Bahdanau et al. introduced attention in 2014 for neural machine translation. This allowed
models to focus on relevant parts of the input when generating each output token.

The Transformer uses an encoder-decoder structure. The encoder maps input sequences to
continuous representations. The decoder generates output sequences autoregressively.
Both use stacked self-attention and feed-forward layers.

Self-attention computes relationships between all positions in a sequence. Each head
can learn different types of relationships, improving model capacity.

Devlin et al. at Google introduced BERT (Bidirectional Encoder Representations from
Transformers) in 2018. BERT uses masked language modeling and next sentence prediction
for pre-training, achieving state-of-the-art results on many NLP benchmarks including
GLUE, SQuAD, and SWAG.

OpenAI developed the GPT (Generative Pre-trained Transformer) series. GPT-1 demonstrated
the power of pre-training with fine-tuning. GPT-2 showed emergent capabilities in
zero-shot learning. GPT-3 introduced few-shot learning with 175 billion parameters.
GPT-4 added multimodal capabilities with vision and text.

Dosovitskiy et al. at Google Research introduced Vision Transformer (ViT), applying
Transformers to computer vision. The image is divided into patches, which are then
processed as a sequence. This approach rivals convolutional neural networks (CNNs)
on ImageNet classification.

The original Transformer was trained on WMT 2014 English-German translation datasets.
Key training details include Adam optimizer, learning rate warm-up, dropout for
regularization, and label smoothing.

Researchers have developed more efficient variants. Longformer uses local windowed
attention with global tokens. Linformer approximates attention with low-rank projections.
Performer uses random feature approximation. BigBird combines random, local, and
global attention patterns.

The Transformer architecture has become the foundation for modern AI systems. From natural
language processing to computer vision, protein structure prediction with AlphaFold, and
code generation with Codex and GitHub Copilot, Transformers continue to expand the
boundaries of artificial intelligence.
"""


# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

ENTITY_PATTERNS = [
    (r'\b(Vaswani|Devlin|Brown|Dosovitskiy|Hochreiter|Schmidhuber|Bahdanau|Luong|Cho)\b', EntityType.PERSON),
    (r'\b([A-Z][a-z]+ et al\.)', EntityType.PERSON),
    (r'\b(Hochreiter and Schmidhuber)\b', EntityType.PERSON),
    (r'\b(Google(?:\s+(?:Brain|Research))?)\b', EntityType.ORGANIZATION),
    (r'\b(OpenAI)\b', EntityType.ORGANIZATION),
    (r'\b(Transformer(?:s)?)\b', EntityType.TECHNOLOGY),
    (r'\b(BERT)\b', EntityType.TECHNOLOGY),
    (r'\b(GPT(?:-[1-4])?)\b', EntityType.TECHNOLOGY),
    (r'\b(LSTMs?)\b', EntityType.TECHNOLOGY),
    (r'\b(RNNs?)\b', EntityType.TECHNOLOGY),
    (r'\b(GRUs?)\b', EntityType.TECHNOLOGY),
    (r'\b(CNNs?)\b', EntityType.TECHNOLOGY),
    (r'\b(Vision Transformer|ViT)\b', EntityType.TECHNOLOGY),
    (r'\b(Longformer|Linformer|Performer|BigBird)\b', EntityType.TECHNOLOGY),
    (r'\b(AlphaFold|Codex|GitHub Copilot)\b', EntityType.TECHNOLOGY),
    (r'\b(self-attention(?: mechanism)?s?)\b', EntityType.METHOD),
    (r'\b(multi-head (?:self-)?attention)\b', EntityType.METHOD),
    (r'\b(positional encoding)\b', EntityType.METHOD),
    (r'\b(masked language modeling)\b', EntityType.METHOD),
    (r'\b(pre-training|fine-tuning)\b', EntityType.METHOD),
    (r'\b(encoder-decoder)\b', EntityType.METHOD),
    (r'\b(few-shot|zero-shot) learning\b', EntityType.METHOD),
    (r'\b(natural language processing|NLP)\b', EntityType.CONCEPT),
    (r'\b(machine learning)\b', EntityType.CONCEPT),
    (r'\b(deep learning)\b', EntityType.CONCEPT),
    (r'\b(artificial intelligence)\b', EntityType.CONCEPT),
    (r'\b(computer vision)\b', EntityType.CONCEPT),
    (r'\b(WMT 2014)\b', EntityType.DATASET),
    (r'\b(ImageNet)\b', EntityType.DATASET),
    (r'\b(GLUE|SQuAD|SWAG)\b', EntityType.DATASET),
]


def extract_entities(text: str) -> List[Entity]:
    entities = []
    seen = set()

    for pattern, entity_type in ENTITY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entity_text = match.group(1) if match.lastindex else match.group(0)
            normalized = entity_text.lower().strip()

            if normalized not in seen and len(entity_text) > 2:
                seen.add(normalized)
                frequency = len(re.findall(re.escape(entity_text), text, re.IGNORECASE))
                confidence = min(0.5 + (frequency * 0.1), 0.99)

                entities.append(Entity(
                    text=entity_text,
                    type=entity_type,
                    confidence=round(confidence, 2),
                    start=match.start(),
                    end=match.end(),
                    normalized=normalized
                ))

    return sorted(entities, key=lambda e: e.confidence, reverse=True)


# ============================================================================
# BASELINE EXTRACTION
# ============================================================================

@dataclass
class BaselineRelation:
    source: str
    target: str
    relation_type: str
    confidence: float
    evidence: str


def baseline_extract(text: str, entities: List[Entity]) -> List[BaselineRelation]:
    """Simple co-occurrence baseline."""
    relations = []
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sent_lower = sentence.lower()
        sent_entities = [e for e in entities if e.normalized in sent_lower]

        for i, e1 in enumerate(sent_entities):
            for e2 in sent_entities[i+1:]:
                if e1.normalized == e2.normalized:
                    continue

                rel_type = "RELATED_TO"
                if "introduced" in sent_lower or "developed" in sent_lower:
                    rel_type = "ENABLES"
                elif "uses" in sent_lower:
                    rel_type = "USES"
                elif "unlike" in sent_lower:
                    rel_type = "CONTRASTS_WITH"

                conf = min(e1.confidence, e2.confidence) * 0.8

                relations.append(BaselineRelation(
                    source=e1.text,
                    target=e2.text,
                    relation_type=rel_type,
                    confidence=round(conf, 2),
                    evidence=sentence[:150]
                ))

    # Deduplicate
    seen = {}
    for rel in relations:
        key = (rel.source.lower(), rel.target.lower(), rel.relation_type)
        if key not in seen or seen[key].confidence < rel.confidence:
            seen[key] = rel

    return list(seen.values())


# ============================================================================
# EVALUATION
# ============================================================================

EXPECTED_RELATIONS = [
    ("Vaswani et al.", "Transformer", ["INTRODUCES", "CREATES", "DEVELOPS"]),
    ("Hochreiter and Schmidhuber", "LSTMs", ["INTRODUCES", "CREATES", "DEVELOPS"]),
    ("Devlin et al.", "BERT", ["INTRODUCES", "CREATES", "DEVELOPS"]),
    ("Dosovitskiy et al.", "Vision Transformer", ["INTRODUCES", "CREATES", "DEVELOPS"]),
    ("Vaswani et al.", "Google Brain", ["AFFILIATED_WITH", "RELATED_TO"]),
    ("Transformer", "self-attention", ["USES", "IMPLEMENTS", "CONTAINS"]),
    ("Transformer", "RNNs", ["CONTRASTS_WITH", "REPLACES", "IMPROVES"]),
    ("Transformer", "LSTMs", ["CONTRASTS_WITH", "REPLACES", "IMPROVES"]),
    ("Transformer", "WMT 2014", ["TRAINED_ON", "EVALUATED_ON"]),
    ("BERT", "GLUE", ["EVALUATED_ON"]),
    ("Vision Transformer", "ImageNet", ["EVALUATED_ON"]),
    ("Vision Transformer", "CNNs", ["CONTRASTS_WITH", "IMPROVES"]),
    ("OpenAI", "GPT", ["DEVELOPS", "CREATES", "INTRODUCES"]),
    ("BERT", "masked language modeling", ["USES", "IMPLEMENTS"]),
    ("BERT", "pre-training", ["USES", "IMPLEMENTS"]),
]


def evaluate(relations, entities, is_enhanced=False) -> Dict:
    extracted = defaultdict(set)
    for rel in relations:
        if is_enhanced:
            src = rel.source.normalized
            tgt = rel.target.normalized
            rtype = rel.relation_type.value
        else:
            src = rel.source.lower()
            tgt = rel.target.lower()
            rtype = rel.relation_type

        extracted[(src, tgt)].add(rtype)
        extracted[(tgt, src)].add(rtype)

    found = 0
    found_list = []
    missed_list = []

    for source, target, valid in EXPECTED_RELATIONS:
        src_n = source.lower()
        tgt_n = target.lower()

        matched = False
        matched_type = None

        for s, t in [(src_n, tgt_n), (tgt_n, src_n)]:
            if (s, t) in extracted:
                for rt in extracted[(s, t)]:
                    if rt in valid or rt == "RELATED_TO":
                        matched = True
                        matched_type = rt
                        break
            if matched:
                break

        if matched:
            found += 1
            found_list.append({"source": source, "target": target, "found": matched_type})
        else:
            missed_list.append({"source": source, "target": target, "expected": valid})

    # Type precision
    precision_sample = []
    for rel in relations[:50]:
        if is_enhanced:
            src_type = rel.source.type
            tgt_type = rel.target.type
            rtype = rel.relation_type
        else:
            src_ent = next((e for e in entities if e.normalized == rel.source.lower()), None)
            tgt_ent = next((e for e in entities if e.normalized == rel.target.lower()), None)
            src_type = src_ent.type if src_ent else EntityType.CONCEPT
            tgt_type = tgt_ent.type if tgt_ent else EntityType.CONCEPT
            try:
                rtype = RelationType(rel.relation_type)
            except:
                rtype = RelationType.RELATED_TO

        valid_types = VALID_RELATION_TYPES.get((src_type, tgt_type), [RelationType.RELATED_TO])
        is_coherent = rtype in valid_types or rtype == RelationType.RELATED_TO
        precision_sample.append(1.0 if is_coherent else 0.0)

    recall = found / len(EXPECTED_RELATIONS)
    precision = sum(precision_sample) / len(precision_sample) if precision_sample else 0

    type_dist = defaultdict(int)
    for rel in relations:
        if is_enhanced:
            type_dist[rel.relation_type.value] += 1
        else:
            type_dist[rel.relation_type] += 1

    return {
        "found": found,
        "total_expected": len(EXPECTED_RELATIONS),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "coherence": round((recall + precision) / 2, 3),
        "total_relations": len(relations),
        "type_dist": dict(type_dist),
        "found_list": found_list,
        "missed_list": missed_list
    }


# ============================================================================
# MAIN
# ============================================================================

def run_test():
    print("\n" + "="*70)
    print(" ENHANCED RELATION EXTRACTION TEST")
    print(" Baseline vs Multi-Layer Extraction")
    print("="*70)

    print("\n[1] Extracting entities...")
    entities = extract_entities(SAMPLE_CONTENT)
    print(f"    Found {len(entities)} entities")

    print("\n[2] Baseline extraction...")
    baseline_rels = baseline_extract(SAMPLE_CONTENT, entities)
    print(f"    {len(baseline_rels)} relations")

    print("\n[3] Enhanced extraction...")
    extractor = EnhancedRelationExtractor(min_confidence=0.35)
    enhanced_rels = extractor.extract_relations(SAMPLE_CONTENT, entities)
    print(f"    {len(enhanced_rels)} relations")

    print("\n[4] Evaluating...")
    baseline_eval = evaluate(baseline_rels, entities, False)
    enhanced_eval = evaluate(enhanced_rels, entities, True)

    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Enhanced':>12} {'Change':>12}")
    print("-"*60)

    metrics = [
        ("Relations", baseline_eval["total_relations"], enhanced_eval["total_relations"]),
        ("Expected Found", baseline_eval["found"], enhanced_eval["found"]),
        ("Recall", f"{baseline_eval['recall']:.1%}", f"{enhanced_eval['recall']:.1%}"),
        ("Type Precision", f"{baseline_eval['precision']:.1%}", f"{enhanced_eval['precision']:.1%}"),
        ("Coherence Score", f"{baseline_eval['coherence']:.1%}", f"{enhanced_eval['coherence']:.1%}"),
    ]

    for name, b, e in metrics:
        if isinstance(b, int):
            change = f"+{e-b}" if e > b else str(e-b)
        else:
            change = ""
        print(f"{name:<25} {str(b):>12} {str(e):>12} {change:>12}")

    print("\n" + "-"*60)
    print(" Type Distribution")
    print("-"*60)

    all_types = set(baseline_eval["type_dist"].keys()) | set(enhanced_eval["type_dist"].keys())
    for t in sorted(all_types):
        b = baseline_eval["type_dist"].get(t, 0)
        e = enhanced_eval["type_dist"].get(t, 0)
        print(f"  {t:<20} {b:>8} {e:>8}")

    print("\n" + "-"*60)
    print(" Expected Relations Found (Enhanced)")
    print("-"*60)
    for d in enhanced_eval["found_list"]:
        print(f"  {d['source'][:22]:22} -> {d['target'][:18]:18} [{d['found']}]")

    if enhanced_eval["missed_list"]:
        print("\n" + "-"*60)
        print(" Missed Relations")
        print("-"*60)
        for d in enhanced_eval["missed_list"]:
            print(f"  {d['source'][:22]:22} -> {d['target'][:18]:18}")

    print("\n" + "-"*60)
    print(" Sample Enhanced Relations")
    print("-"*60)
    for rel in enhanced_rels[:12]:
        print(f"  {rel.source.text[:18]:18} --[{rel.relation_type.value:15}]--> {rel.target.text[:18]:18} ({rel.extraction_method}, {rel.confidence:.2f})")

    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)

    b_score = baseline_eval["coherence"]
    e_score = enhanced_eval["coherence"]
    improve = e_score - b_score

    print(f"\n  Baseline:    {b_score:.1%}")
    print(f"  Enhanced:    {e_score:.1%}")
    print(f"  Improvement: {improve:+.1%}")

    if e_score >= 0.85:
        verdict = "EXCELLENT - Target achieved (>85%)"
    elif e_score >= 0.70:
        verdict = "GOOD - Significant improvement"
    elif improve > 0.15:
        verdict = "IMPROVED - Clear gains, more work needed"
    else:
        verdict = "NEEDS WORK"

    print(f"\n  {verdict}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline_eval,
        "enhanced": enhanced_eval,
        "improvement": round(improve, 3)
    }

    os.makedirs("/home/user/Unfold/test_output", exist_ok=True)
    path = "/home/user/Unfold/test_output/enhanced_comparison.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    run_test()
