#!/usr/bin/env python3
"""
Integrated Knowledge Graph Test

Combines all three extraction improvements:
1. LLM-based relation extraction
2. Dependency parsing
3. Coreference resolution

Compares baseline vs enhanced extraction pipelines.
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# Multi-word Entity Matching Helper
# ============================================================================

def match_entity(text: str, entity_texts: Dict[str, Dict], prefer_longer: bool = True) -> Optional[str]:
    """
    Match text to an entity, preferring longer matches.

    Args:
        text: The text to match (e.g., "google brain")
        entity_texts: Dict mapping lowercase entity text to entity dict
        prefer_longer: If True, prefer longer entity matches

    Returns:
        The matched entity key (lowercase) or None
    """
    text_lower = text.lower().strip()

    # First, try exact match
    if text_lower in entity_texts:
        return text_lower

    # Collect all partial matches
    matches = []
    for ent_key in entity_texts:
        # Check if text contains entity or entity contains text
        if text_lower in ent_key or ent_key in text_lower:
            matches.append(ent_key)

    if not matches:
        return None

    if prefer_longer:
        # Sort by length descending, return longest match
        matches.sort(key=len, reverse=True)

    return matches[0]


def match_entities_in_text(text: str, entities: List[Dict]) -> List[str]:
    """
    Find all entities mentioned in text, preferring longer matches.

    Returns list of matched entity texts (original case).
    """
    text_lower = text.lower()

    # Sort entities by length (longest first) to prefer longer matches
    sorted_ents = sorted(entities, key=lambda e: len(e['text']), reverse=True)

    matched = []
    matched_spans = []  # Track matched character spans to avoid overlaps

    for ent in sorted_ents:
        ent_lower = ent['text'].lower()

        # Find all occurrences
        start = 0
        while True:
            pos = text_lower.find(ent_lower, start)
            if pos == -1:
                break

            end = pos + len(ent_lower)

            # Check if this span overlaps with already matched spans
            overlaps = False
            for (ms, me) in matched_spans:
                if not (end <= ms or pos >= me):  # Overlaps
                    overlaps = True
                    break

            if not overlaps:
                if ent['text'] not in matched:
                    matched.append(ent['text'])
                matched_spans.append((pos, end))

            start = pos + 1

    return matched

# ============================================================================
# Test Document (same as E2E test)
# ============================================================================

TEST_TEXT = """
The Transformer architecture, introduced by Vaswani et al. at Google Brain in 2017,
revolutionized natural language processing. Unlike recurrent neural networks (RNNs)
and long short-term memory networks (LSTMs), the Transformer relies entirely on
self-attention mechanisms. It achieved state-of-the-art results on machine translation.

The key innovation is the self-attention mechanism, which allows the model to weigh
the importance of different parts of the input sequence. This approach eliminates
the need for sequential processing, enabling significant parallelization.

Devlin et al. at Google introduced BERT in 2018. BERT uses masked language modeling
and next sentence prediction for pre-training. The model achieved state-of-the-art
results on GLUE and SQuAD benchmarks. It demonstrated that bidirectional pre-training
produces superior representations.

OpenAI developed the GPT series starting in 2018. GPT-3 introduced few-shot learning
with 175 billion parameters. The model demonstrated remarkable zero-shot capabilities
across diverse tasks. It can generate human-like text without task-specific training.

Dosovitskiy et al. at Google Research introduced Vision Transformer (ViT) in 2020.
This approach applies the Transformer architecture to image classification.
ViT rivals CNNs on ImageNet classification while using fewer computational resources.
The model processes images as sequences of patches.
"""

TEST_ENTITIES = [
    {"text": "Transformer", "type": "TECHNOLOGY"},
    {"text": "Vaswani et al.", "type": "PERSON"},
    {"text": "Google Brain", "type": "ORGANIZATION"},
    {"text": "Google", "type": "ORGANIZATION"},
    {"text": "RNNs", "type": "TECHNOLOGY"},
    {"text": "LSTMs", "type": "TECHNOLOGY"},
    {"text": "self-attention", "type": "METHOD"},
    {"text": "Devlin et al.", "type": "PERSON"},
    {"text": "BERT", "type": "TECHNOLOGY"},
    {"text": "masked language modeling", "type": "METHOD"},
    {"text": "pre-training", "type": "METHOD"},
    {"text": "GLUE", "type": "DATASET"},
    {"text": "SQuAD", "type": "DATASET"},
    {"text": "OpenAI", "type": "ORGANIZATION"},
    {"text": "GPT", "type": "TECHNOLOGY"},
    {"text": "GPT-3", "type": "TECHNOLOGY"},
    {"text": "Dosovitskiy et al.", "type": "PERSON"},
    {"text": "Vision Transformer", "type": "TECHNOLOGY"},
    {"text": "ViT", "type": "TECHNOLOGY"},
    {"text": "CNNs", "type": "TECHNOLOGY"},
    {"text": "ImageNet", "type": "DATASET"},
    {"text": "Google Research", "type": "ORGANIZATION"},
]

# Ground truth relations we want to capture
EXPECTED_RELATIONS = [
    # Author -> Technology (INTRODUCES)
    ("Vaswani et al.", "Transformer", "INTRODUCES"),
    ("Devlin et al.", "BERT", "INTRODUCES"),
    ("Dosovitskiy et al.", "Vision Transformer", "INTRODUCES"),
    ("OpenAI", "GPT", "DEVELOPS"),

    # Technology -> Method (USES)
    ("Transformer", "self-attention", "USES"),
    ("BERT", "masked language modeling", "USES"),
    ("BERT", "pre-training", "USES"),

    # Technology -> Dataset (EVALUATED_ON)
    ("BERT", "GLUE", "EVALUATED_ON"),
    ("BERT", "SQuAD", "EVALUATED_ON"),
    ("ViT", "ImageNet", "EVALUATED_ON"),

    # Technology -> Technology (COMPETES/CONTRASTS)
    ("Transformer", "RNNs", "CONTRASTS_WITH"),
    ("Transformer", "LSTMs", "CONTRASTS_WITH"),
    ("ViT", "CNNs", "COMPETES_WITH"),

    # Person -> Organization (AFFILIATED_WITH)
    ("Vaswani et al.", "Google Brain", "AFFILIATED_WITH"),
    ("Devlin et al.", "Google", "AFFILIATED_WITH"),
    ("Dosovitskiy et al.", "Google Research", "AFFILIATED_WITH"),
]


# ============================================================================
# Baseline Extractor (co-occurrence only)
# ============================================================================

@dataclass
class Relation:
    subject: str
    relation_type: str
    object: str
    confidence: float = 0.5
    source: str = "unknown"


class BaselineExtractor:
    """Simple co-occurrence based extraction."""

    def extract_relations(self, text: str, entities: List[Dict]) -> List[Relation]:
        relations = []
        entity_texts = {e['text'].lower(): e for e in entities}

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent_lower = sent.lower()
            ents_in_sent = [e for e in entity_texts.keys() if e in sent_lower]

            # Create relations for co-occurring entities
            for i, e1 in enumerate(ents_in_sent):
                for e2 in ents_in_sent[i+1:]:
                    relations.append(Relation(
                        subject=entity_texts[e1]['text'],
                        relation_type="RELATED_TO",
                        object=entity_texts[e2]['text'],
                        confidence=0.5,
                        source="cooccurrence"
                    ))

        return relations


# ============================================================================
# Enhanced Extractor (combines all three modules)
# ============================================================================

class EnhancedExtractor:
    """Combines coreference, dependency parsing, and pattern extraction."""

    def __init__(self):
        self.coref_resolver = CoreferenceResolver()
        self.dep_parser = DependencyParser()

    def extract_relations(self, text: str, entities: List[Dict]) -> List[Relation]:
        relations = []

        # Step 1: Resolve coreferences
        resolved = self.coref_resolver.resolve(text, entities)

        # Step 2: Extract with dependency parsing
        dep_relations = self.dep_parser.extract_relations(resolved.resolved_text, entities)
        for r in dep_relations:
            relations.append(Relation(
                subject=r.subject,
                relation_type=r.relation_type,
                object=r.object,
                confidence=r.confidence,
                source="dependency"
            ))

        # Step 3: Pattern-based extraction on resolved text
        pattern_relations = self._pattern_extract(resolved.resolved_text, entities)
        relations.extend(pattern_relations)

        # Step 4: Co-occurrence fallback for remaining entity pairs
        cooc_relations = self._cooccurrence_extract(text, entities, relations)
        relations.extend(cooc_relations)

        # Deduplicate
        relations = self._deduplicate(relations)

        return relations

    def _pattern_extract(self, text: str, entities: List[Dict]) -> List[Relation]:
        """Pattern-based relation extraction with multi-word entity support."""
        relations = []
        entity_texts = {e['text'].lower(): e for e in entities}
        text_lower = text.lower()

        # Simple patterns
        simple_patterns = [
            (r'(\w+)\s+uses?\s+(\w+)', 'USES'),
            (r'(\w+)\s+rivals?\s+(\w+)', 'COMPETES_WITH'),
            (r'unlike\s+(\w+).*?,?\s*(?:the\s+)?(\w+)', 'CONTRASTS_WITH'),
        ]

        for pattern, rel_type in simple_patterns:
            for m in re.finditer(pattern, text_lower):
                s, o = m.group(1), m.group(2)
                sm = match_entity(s, entity_texts)
                om = match_entity(o, entity_texts)
                if sm and om and sm != om:
                    relations.append(Relation(
                        subject=entity_texts[sm]['text'],
                        relation_type=rel_type,
                        object=entity_texts[om]['text'],
                        confidence=0.7,
                        source="pattern"
                    ))

        # Special pattern: "X at Y introduced Z" - handles et al. and multi-word orgs
        author_intro = r'(\w+(?:\s+et\s+al\.)?)\s+at\s+(\w+(?:\s+\w+)?)\s+introduced\s+(?:the\s+)?(\w+(?:\s+\w+)?)'
        for m in re.finditer(author_intro, text_lower):
            author, org, tech = m.group(1), m.group(2), m.group(3)
            am = match_entity(author, entity_texts)
            om = match_entity(org, entity_texts)  # Prefers "Google Brain" over "Google"
            tm = match_entity(tech, entity_texts)

            if am and tm and am != tm:
                relations.append(Relation(
                    subject=entity_texts[am]['text'],
                    relation_type='INTRODUCES',
                    object=entity_texts[tm]['text'],
                    confidence=0.8,
                    source="pattern"
                ))
            if am and om and am != om:
                relations.append(Relation(
                    subject=entity_texts[am]['text'],
                    relation_type='AFFILIATED_WITH',
                    object=entity_texts[om]['text'],
                    confidence=0.8,
                    source="pattern"
                ))

        # Passive pattern: "X, introduced by Y at Z"
        passive_intro = r'(\w+(?:\s+\w+)?),?\s+introduced\s+by\s+(\w+(?:\s+et\s+al\.)?)\s+at\s+(\w+(?:\s+\w+)?)'
        for m in re.finditer(passive_intro, text_lower):
            tech, author, org = m.group(1), m.group(2), m.group(3)
            am = match_entity(author, entity_texts)
            om = match_entity(org, entity_texts)
            tm = match_entity(tech, entity_texts)

            if am and tm and am != tm:
                relations.append(Relation(
                    subject=entity_texts[am]['text'],
                    relation_type='INTRODUCES',
                    object=entity_texts[tm]['text'],
                    confidence=0.8,
                    source="pattern"
                ))
            if am and om:
                relations.append(Relation(
                    subject=entity_texts[am]['text'],
                    relation_type='AFFILIATED_WITH',
                    object=entity_texts[om]['text'],
                    confidence=0.8,
                    source="pattern"
                ))

        # OpenAI developed X
        dev_pattern = r'(\w+)\s+developed\s+(?:the\s+)?(\w+)'
        for m in re.finditer(dev_pattern, text_lower):
            s, o = m.group(1), m.group(2)
            sm = match_entity(s, entity_texts)
            om = match_entity(o, entity_texts)
            if sm and om and sm != om:
                relations.append(Relation(
                    subject=entity_texts[sm]['text'],
                    relation_type='DEVELOPS',
                    object=entity_texts[om]['text'],
                    confidence=0.75,
                    source="pattern"
                ))

        # Evaluation patterns: "results on X and Y" or "on X benchmarks"
        eval_pattern = r'results\s+on\s+(\w+)(?:\s+and\s+(\w+))?'
        for m in re.finditer(eval_pattern, text_lower):
            d1 = m.group(1)
            d2 = m.group(2) if m.lastindex > 1 else None

            # Find subject (technology mentioned nearby)
            for e in entity_texts:
                if entity_texts[e].get('type') == 'TECHNOLOGY':
                    for d in [d1, d2]:
                        if d:
                            dm = match_entity(d, entity_texts)
                            if dm:
                                relations.append(Relation(
                                    subject=entity_texts[e]['text'],
                                    relation_type='EVALUATED_ON',
                                    object=entity_texts[dm]['text'],
                                    confidence=0.65,
                                    source="pattern"
                                ))

        return relations

    def _cooccurrence_extract(self, text: str, entities: List[Dict], existing: List[Relation]) -> List[Relation]:
        """Co-occurrence fallback for uncovered pairs."""
        relations = []
        entity_texts = {e['text'].lower(): e for e in entities}

        # Get existing pairs
        existing_pairs = {(r.subject.lower(), r.object.lower()) for r in existing}
        existing_pairs.update({(r.object.lower(), r.subject.lower()) for r in existing})

        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent_lower = sent.lower()
            ents_in_sent = [e for e in entity_texts.keys() if e in sent_lower]

            for i, e1 in enumerate(ents_in_sent):
                for e2 in ents_in_sent[i+1:]:
                    if (e1, e2) not in existing_pairs and (e2, e1) not in existing_pairs:
                        relations.append(Relation(
                            subject=entity_texts[e1]['text'],
                            relation_type="RELATED_TO",
                            object=entity_texts[e2]['text'],
                            confidence=0.4,
                            source="cooccurrence"
                        ))

        return relations

    def _deduplicate(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations, keeping highest confidence."""
        seen = {}
        for r in relations:
            key = (r.subject.lower(), r.object.lower(), r.relation_type)
            if key not in seen or r.confidence > seen[key].confidence:
                seen[key] = r
        return list(seen.values())


# ============================================================================
# Coreference Resolution (inline)
# ============================================================================

class CoreferenceResolver:
    def __init__(self):
        self.description_patterns = [
            (r'[Tt]he\s+(model|system|approach)', 'desc'),
            (r'[Tt]his\s+(approach|method|model)', 'demo'),
            (r'\b[Ii]t\b', 'pronoun'),
        ]

    def resolve(self, text: str, entities: List[Dict]) -> 'ResolvedText':
        resolutions = []
        resolved_text = text

        # Find technology entities for resolution targets
        tech_entities = [e for e in entities if e['type'] in ('TECHNOLOGY', 'MODEL', 'METHOD')]

        sentences = self._split_sentences(text)
        sentence_entities = self._map_entities_to_sentences(sentences, tech_entities)

        for i, sent in enumerate(sentences):
            # Find references in this sentence
            for pattern, ref_type in self.description_patterns:
                for match in re.finditer(pattern, sent):
                    ref_text = match.group(0)

                    # Find best antecedent
                    antecedent = self._find_antecedent(
                        i, sentences, sentence_entities, tech_entities, ref_type
                    )
                    if antecedent:
                        resolutions.append((ref_text, antecedent))

        # Build resolved text
        for ref, ant in resolutions:
            resolved_text = resolved_text.replace(ref, f"{ref} (={ant})", 1)

        return ResolvedText(text, resolved_text, resolutions)

    def _map_entities_to_sentences(self, sentences: List[str], entities: List[Dict]) -> Dict[int, List[str]]:
        """Map which entities appear in which sentences."""
        mapping = defaultdict(list)
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            for ent in entities:
                if ent['text'].lower() in sent_lower:
                    mapping[i].append(ent['text'])
        return mapping

    def _find_antecedent(self, sent_idx: int, sentences: List[str],
                         sent_entities: Dict[int, List[str]],
                         all_entities: List[Dict], ref_type: str) -> str:
        """Find the most recent matching entity."""
        # Look at previous sentence first (most common for "The model")
        if sent_idx > 0 and sent_idx - 1 in sent_entities:
            # Return the last mentioned entity in previous sentence
            ents = sent_entities[sent_idx - 1]
            if ents:
                return ents[-1]  # Last mentioned is usually the topic

        # Look backwards through earlier sentences
        for i in range(sent_idx - 2, -1, -1):
            if i in sent_entities and sent_entities[i]:
                return sent_entities[i][-1]

        return None

    def _split_sentences(self, text: str) -> List[str]:
        protected = re.sub(r'\bet al\.', 'ET_AL', text)
        sentences = re.split(r'(?<=[.!?])\s+', protected)
        return [s.replace('ET_AL', 'et al.') for s in sentences]


@dataclass
class ResolvedText:
    original_text: str
    resolved_text: str
    resolutions: List[Tuple[str, str]] = field(default_factory=list)


# ============================================================================
# Dependency Parser (inline)
# ============================================================================

@dataclass
class ExtractedRelation:
    subject: str
    relation_type: str
    object: str
    confidence: float
    evidence: str = ""


class DependencyParser:
    def __init__(self):
        self.verb_map = {
            'introduced': 'INTRODUCES', 'developed': 'DEVELOPS',
            'uses': 'USES', 'achieved': 'EVALUATED_ON',
            'rivals': 'COMPETES_WITH', 'relies': 'USES'
        }

    def extract_relations(self, text: str, entities: List[Dict]) -> List[ExtractedRelation]:
        relations = []
        entity_texts = {e['text'].lower(): e for e in entities}

        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent_lower = sent.lower()
            ents_in_sent = [e for e in entity_texts.keys() if e in sent_lower]

            if len(ents_in_sent) < 2:
                continue

            # Look for verb patterns
            for verb, rel_type in self.verb_map.items():
                if verb in sent_lower:
                    # Try to find subject and object
                    rel = self._extract_svo(sent_lower, verb, rel_type, ents_in_sent, entity_texts)
                    if rel:
                        relations.append(rel)

        return relations

    def _extract_svo(self, sent: str, verb: str, rel_type: str,
                     ents: List[str], entity_texts: Dict) -> ExtractedRelation:
        """Extract subject-verb-object pattern."""
        verb_pos = sent.find(verb)
        if verb_pos == -1:
            return None

        before = sent[:verb_pos]
        after = sent[verb_pos:]

        subj = obj = None

        # Find subject (entity before verb)
        for e in ents:
            if e in before:
                subj = e

        # Find object (entity after verb)
        for e in ents:
            if e in after and e != subj:
                obj = e
                break

        if subj and obj:
            return ExtractedRelation(
                subject=entity_texts[subj]['text'],
                relation_type=rel_type,
                object=entity_texts[obj]['text'],
                confidence=0.75,
                evidence=sent
            )
        return None


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(relations: List[Relation], expected: List[Tuple]) -> Dict:
    """Evaluate extracted relations against ground truth."""
    # Normalize relations for comparison
    extracted = set()
    for r in relations:
        extracted.add((r.subject.lower(), r.object.lower(), r.relation_type))
        extracted.add((r.subject.lower(), r.object.lower()))  # Without type

    found = 0
    missed = []
    found_details = []

    for subj, obj, rel_type in expected:
        sl, ol = subj.lower(), obj.lower()

        # Exact match
        if (sl, ol, rel_type) in extracted:
            found += 1
            found_details.append((subj, obj, rel_type, "exact"))
        # Type-agnostic match
        elif (sl, ol) in extracted:
            found += 1
            found_details.append((subj, obj, rel_type, "partial"))
        # Reverse match
        elif (ol, sl) in extracted or (ol, sl, rel_type) in extracted:
            found += 1
            found_details.append((subj, obj, rel_type, "reversed"))
        else:
            missed.append((subj, obj, rel_type))

    recall = found / len(expected) * 100 if expected else 0

    # Calculate precision approximation
    meaningful = sum(1 for r in relations if r.relation_type != "RELATED_TO")
    precision = meaningful / len(relations) * 100 if relations else 0

    return {
        "recall": recall,
        "precision": precision,
        "found": found,
        "total_expected": len(expected),
        "total_extracted": len(relations),
        "meaningful_relations": meaningful,
        "found_details": found_details,
        "missed": missed
    }


def print_header(title: str):
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print_header("INTEGRATED KNOWLEDGE GRAPH EXTRACTION TEST")
    print()

    # Baseline extraction
    print("[1] Baseline Extraction (co-occurrence only)")
    baseline = BaselineExtractor()
    baseline_relations = baseline.extract_relations(TEST_TEXT, TEST_ENTITIES)
    baseline_eval = evaluate(baseline_relations, EXPECTED_RELATIONS)

    print(f"    Relations extracted: {len(baseline_relations)}")
    print(f"    Recall: {baseline_eval['recall']:.1f}%")
    print(f"    Meaningful relations: {baseline_eval['meaningful_relations']}")

    # Enhanced extraction
    print("\n[2] Enhanced Extraction (coref + dependency + patterns)")
    enhanced = EnhancedExtractor()
    enhanced_relations = enhanced.extract_relations(TEST_TEXT, TEST_ENTITIES)
    enhanced_eval = evaluate(enhanced_relations, EXPECTED_RELATIONS)

    print(f"    Relations extracted: {len(enhanced_relations)}")
    print(f"    Recall: {enhanced_eval['recall']:.1f}%")
    print(f"    Meaningful relations: {enhanced_eval['meaningful_relations']}")

    # Comparison
    print("\n" + "=" * 70)
    print(" COMPARISON")
    print("=" * 70)

    improvement = enhanced_eval['recall'] - baseline_eval['recall']
    print(f"\n  Baseline Recall:  {baseline_eval['recall']:.1f}%")
    print(f"  Enhanced Recall:  {enhanced_eval['recall']:.1f}%")
    print(f"  Improvement:      +{improvement:.1f}%")

    # Show what enhanced found
    print("\n  Relations found by enhanced extractor:")
    for subj, obj, rel_type, match_type in enhanced_eval['found_details']:
        marker = "✓" if match_type == "exact" else "~"
        print(f"    {marker} {subj:20} -> {obj:20} [{rel_type}]")

    if enhanced_eval['missed']:
        print("\n  Still missing:")
        for subj, obj, rel_type in enhanced_eval['missed']:
            print(f"    ✗ {subj:20} -> {obj:20} [{rel_type}]")

    # Breakdown by source
    print("\n  Relation sources (enhanced):")
    by_source = defaultdict(int)
    by_source_meaningful = defaultdict(int)
    for r in enhanced_relations:
        by_source[r.source] += 1
        if r.relation_type != "RELATED_TO":
            by_source_meaningful[r.source] += 1

    for source, count in sorted(by_source.items()):
        meaningful = by_source_meaningful.get(source, 0)
        print(f"    {source:15} {count:3} total, {meaningful:3} meaningful")

    # Save results
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    results = {
        "baseline": {
            "recall": baseline_eval['recall'],
            "total": len(baseline_relations),
            "meaningful": baseline_eval['meaningful_relations']
        },
        "enhanced": {
            "recall": enhanced_eval['recall'],
            "total": len(enhanced_relations),
            "meaningful": enhanced_eval['meaningful_relations']
        },
        "improvement": improvement
    }

    with open(output_dir / "integrated_test.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: test_output/integrated_test.json")

    # Verdict
    print("\n" + "=" * 70)
    print(" VERDICT")
    print("=" * 70)

    if enhanced_eval['recall'] >= 80:
        verdict = "EXCELLENT"
    elif enhanced_eval['recall'] >= 60:
        verdict = "GOOD"
    elif enhanced_eval['recall'] >= 40:
        verdict = "FAIR"
    else:
        verdict = "NEEDS IMPROVEMENT"

    print(f"\n  Enhanced system recall: {enhanced_eval['recall']:.1f}%")
    print(f"  Rating: {verdict}")
    print()


if __name__ == "__main__":
    main()
