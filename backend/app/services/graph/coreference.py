"""
Coreference Resolution Module for Knowledge Graph Construction

Resolves pronouns and anaphoric references to their antecedents:
- Personal pronouns: he, she, it, they
- Demonstratives: this, that, these, those
- Definite descriptions: "the model", "the system", "this approach"

Supports multiple resolution strategies:
1. Rule-based: Recency + type matching heuristics
2. LLM-based: Uses language model for complex cases
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import re

logger = logging.getLogger(__name__)


class CoreferenceError(Exception):
    """Exception raised for coreference resolution errors."""
    pass


class ReferenceType(Enum):
    """Types of referring expressions."""
    PRONOUN_IT = "it"
    PRONOUN_THEY = "they"
    PRONOUN_THIS = "this"
    PRONOUN_THAT = "that"
    PRONOUN_THESE = "these"
    PRONOUN_THOSE = "those"
    DEFINITE_DESC = "definite_description"  # "the model", "the system"
    DEMONSTRATIVE_DESC = "demonstrative_description"  # "this approach", "that method"


@dataclass
class Reference:
    """A referring expression that needs resolution."""
    text: str
    ref_type: ReferenceType
    sentence_idx: int
    char_start: int
    char_end: int
    resolved_to: Optional[str] = None
    confidence: float = 0.0


@dataclass
class Entity:
    """An entity that can be an antecedent."""
    text: str
    entity_type: str
    sentence_idx: int
    char_start: int
    char_end: int
    salience: float = 1.0  # How prominent/important this entity is


@dataclass
class ResolvedText:
    """Text with coreferences resolved."""
    original_text: str
    resolved_text: str
    resolutions: List[Tuple[str, str]]  # (reference, antecedent) pairs


# Patterns for detecting referring expressions
PRONOUN_PATTERNS = {
    ReferenceType.PRONOUN_IT: r'\b[Ii]t\b',
    ReferenceType.PRONOUN_THEY: r'\b[Tt]hey\b',
    ReferenceType.PRONOUN_THIS: r'\b[Tt]his\b(?!\s+\w)',  # "this" alone, not "this model"
    ReferenceType.PRONOUN_THAT: r'\b[Tt]hat\b(?!\s+\w)',
    ReferenceType.PRONOUN_THESE: r'\b[Tt]hese\b(?!\s+\w)',
    ReferenceType.PRONOUN_THOSE: r'\b[Tt]hose\b(?!\s+\w)',
}

# Definite and demonstrative descriptions
DESCRIPTION_PATTERNS = {
    ReferenceType.DEFINITE_DESC: [
        r'[Tt]he\s+(model|system|architecture|network|method|approach|algorithm|framework|technique)',
        r'[Tt]he\s+(transformer|attention|mechanism|layer|module)',
    ],
    ReferenceType.DEMONSTRATIVE_DESC: [
        r'[Tt]his\s+(model|system|architecture|network|method|approach|algorithm|framework|technique)',
        r'[Tt]hat\s+(model|system|architecture|network|method|approach|algorithm|framework|technique)',
        r'[Tt]hese\s+(models|systems|architectures|networks|methods|approaches|algorithms|frameworks|techniques)',
    ],
}

# Entity type compatibility for resolution
# Maps reference type patterns to compatible entity types
TYPE_COMPATIBILITY = {
    'model': {'TECHNOLOGY', 'MODEL', 'SYSTEM'},
    'system': {'TECHNOLOGY', 'MODEL', 'SYSTEM'},
    'architecture': {'TECHNOLOGY', 'MODEL', 'ARCHITECTURE'},
    'network': {'TECHNOLOGY', 'MODEL', 'NETWORK'},
    'method': {'METHOD', 'TECHNIQUE', 'ALGORITHM'},
    'approach': {'METHOD', 'TECHNIQUE', 'TECHNOLOGY'},
    'algorithm': {'METHOD', 'ALGORITHM'},
    'framework': {'TECHNOLOGY', 'FRAMEWORK'},
    'technique': {'METHOD', 'TECHNIQUE'},
    'transformer': {'TECHNOLOGY', 'MODEL'},
    'attention': {'METHOD', 'MECHANISM'},
    'mechanism': {'METHOD', 'MECHANISM'},
    'layer': {'COMPONENT', 'MODULE'},
    'module': {'COMPONENT', 'MODULE'},
}


class CoreferenceResolver:
    """
    Resolves coreferences in text using rule-based heuristics.
    Falls back to LLM for complex cases when available.
    """

    def __init__(self, use_llm: bool = False, llm_provider=None, verbose: bool = True):
        """
        Initialize the resolver.

        Args:
            use_llm: Whether to use LLM for complex resolution
            llm_provider: Optional LLM provider for complex cases
            verbose: Whether to print progress messages
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.verbose = verbose
        self._log(f"Initialized (LLM: {use_llm})")

    def _log(self, message: str, level: str = "info"):
        """Log a message with appropriate formatting."""
        prefix = "[CoreferenceResolver]"
        if level == "error":
            logger.error(f"{prefix} {message}")
            if self.verbose:
                print(f"  {prefix} ✗ ERROR: {message}", file=sys.stderr)
        elif level == "warning":
            logger.warning(f"{prefix} {message}")
            if self.verbose:
                print(f"  {prefix} ⚠ {message}")
        elif level == "success":
            logger.info(f"{prefix} {message}")
            if self.verbose:
                print(f"  {prefix} ✓ {message}")
        else:
            logger.info(f"{prefix} {message}")
            if self.verbose:
                print(f"  {prefix} {message}")

    def find_references(self, text: str) -> List[Reference]:
        """Find all referring expressions in text."""
        references = []

        # Split into sentences for tracking
        sentences = self._split_sentences(text)
        sent_offsets = self._get_sentence_offsets(text, sentences)

        # Find pronouns
        for ref_type, pattern in PRONOUN_PATTERNS.items():
            for match in re.finditer(pattern, text):
                sent_idx = self._get_sentence_index(match.start(), sent_offsets)
                references.append(Reference(
                    text=match.group(),
                    ref_type=ref_type,
                    sentence_idx=sent_idx,
                    char_start=match.start(),
                    char_end=match.end()
                ))

        # Find definite descriptions
        for ref_type, patterns in DESCRIPTION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    sent_idx = self._get_sentence_index(match.start(), sent_offsets)
                    references.append(Reference(
                        text=match.group(),
                        ref_type=ref_type,
                        sentence_idx=sent_idx,
                        char_start=match.start(),
                        char_end=match.end()
                    ))

        # Sort by position
        references.sort(key=lambda r: r.char_start)
        return references

    def find_entities(self, text: str, entities: List[Dict]) -> List[Entity]:
        """Map entity dictionaries to Entity objects with positions."""
        result = []
        sentences = self._split_sentences(text)
        sent_offsets = self._get_sentence_offsets(text, sentences)
        text_lower = text.lower()

        for ent in entities:
            ent_text = ent['text']
            ent_lower = ent_text.lower()

            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(ent_lower, start)
                if pos == -1:
                    break

                sent_idx = self._get_sentence_index(pos, sent_offsets)

                # Calculate salience based on position and frequency
                salience = 1.0 - (pos / len(text)) * 0.3  # Earlier = more salient

                result.append(Entity(
                    text=ent_text,
                    entity_type=ent.get('type', 'UNKNOWN'),
                    sentence_idx=sent_idx,
                    char_start=pos,
                    char_end=pos + len(ent_text),
                    salience=salience
                ))

                start = pos + 1

        return result

    def resolve(self, text: str, entities: List[Dict]) -> ResolvedText:
        """
        Resolve all coreferences in text.

        Args:
            text: Input text
            entities: List of entity dicts with 'text' and 'type' keys

        Returns:
            ResolvedText with original and resolved versions

        Raises:
            CoreferenceError: If resolution fails critically
        """
        self._log("Starting coreference resolution...")

        try:
            # Validate inputs
            if not text or not text.strip():
                self._log("Empty text provided, skipping resolution", "warning")
                return ResolvedText("", "", [])

            if not entities:
                self._log("No entities provided, skipping resolution", "warning")
                return ResolvedText(text, text, [])

            # Find references
            try:
                references = self.find_references(text)
                self._log(f"Found {len(references)} references to resolve")
            except Exception as e:
                self._log(f"Failed to find references: {e}", "error")
                raise CoreferenceError(f"Reference finding failed: {e}") from e

            # Find entity positions
            try:
                entity_objs = self.find_entities(text, entities)
                self._log(f"Mapped {len(entity_objs)} entity occurrences")
            except Exception as e:
                self._log(f"Failed to map entities: {e}", "error")
                raise CoreferenceError(f"Entity mapping failed: {e}") from e

            if not references:
                self._log("No references found, nothing to resolve", "success")
                return ResolvedText(text, text, [])

            # Resolve references
            resolutions = []
            resolved_count = 0
            failed_count = 0

            for ref in references:
                try:
                    antecedent = self._find_antecedent(ref, entity_objs, resolutions)
                    if antecedent:
                        ref.resolved_to = antecedent.text
                        ref.confidence = self._calculate_confidence(ref, antecedent)
                        resolutions.append((ref.text, antecedent.text))
                        resolved_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    self._log(f"Failed to resolve '{ref.text}': {e}", "warning")
                    failed_count += 1

            # Build resolved text
            try:
                resolved_text = self._build_resolved_text(text, references)
            except Exception as e:
                self._log(f"Failed to build resolved text: {e}", "error")
                resolved_text = text

            # Report completion
            self._log(f"Completed: {resolved_count} resolved, {failed_count} unresolved", "success")
            return ResolvedText(text, resolved_text, resolutions)

        except CoreferenceError:
            raise
        except Exception as e:
            self._log(f"Unexpected error during resolution: {e}", "error")
            raise CoreferenceError(f"Resolution failed: {e}") from e

    def _find_antecedent(
        self,
        ref: Reference,
        entities: List[Entity],
        prior_resolutions: List[Tuple[str, str]]
    ) -> Optional[Entity]:
        """Find the best antecedent for a reference."""
        candidates = []

        for ent in entities:
            # Must appear before the reference
            if ent.char_end > ref.char_start:
                continue

            # Check compatibility
            score = self._score_candidate(ref, ent)
            if score > 0:
                candidates.append((ent, score))

        if not candidates:
            return None

        # Sort by score (descending) then by recency (closer = better)
        candidates.sort(key=lambda x: (x[1], -abs(ref.char_start - x[0].char_end)), reverse=True)

        return candidates[0][0]

    def _score_candidate(self, ref: Reference, ent: Entity) -> float:
        """Score how well an entity matches a reference."""
        score = 0.0

        # Base score from salience
        score += ent.salience * 0.3

        # Sentence distance penalty
        sent_dist = ref.sentence_idx - ent.sentence_idx
        if sent_dist == 0:
            score += 0.3  # Same sentence
        elif sent_dist == 1:
            score += 0.2  # Previous sentence
        elif sent_dist <= 3:
            score += 0.1  # Within 3 sentences
        else:
            score -= 0.1  # Too far

        # Type compatibility for descriptions
        if ref.ref_type in (ReferenceType.DEFINITE_DESC, ReferenceType.DEMONSTRATIVE_DESC):
            # Extract the noun from "the model", "this approach", etc.
            match = re.search(r'(model|system|architecture|network|method|approach|algorithm|framework|technique|transformer|attention|mechanism|layer|module)s?', ref.text.lower())
            if match:
                noun = match.group(1)
                compatible_types = TYPE_COMPATIBILITY.get(noun, set())
                if ent.entity_type.upper() in compatible_types:
                    score += 0.4  # Strong type match
                elif ent.entity_type.upper() in {'TECHNOLOGY', 'MODEL', 'METHOD'}:
                    score += 0.2  # Weak type match

        # For pronouns, prefer recent entities
        elif ref.ref_type in (ReferenceType.PRONOUN_IT, ReferenceType.PRONOUN_THIS, ReferenceType.PRONOUN_THAT):
            # "it/this/that" usually refers to singular technology/model
            if ent.entity_type.upper() in {'TECHNOLOGY', 'MODEL', 'METHOD'}:
                score += 0.3

        elif ref.ref_type in (ReferenceType.PRONOUN_THEY, ReferenceType.PRONOUN_THESE, ReferenceType.PRONOUN_THOSE):
            # "they/these/those" might refer to plural or group
            if ent.entity_type.upper() in {'ORGANIZATION', 'PERSON'}:
                score += 0.2

        # Recency bonus
        char_dist = ref.char_start - ent.char_end
        if char_dist < 100:
            score += 0.2
        elif char_dist < 200:
            score += 0.1

        return score

    def _calculate_confidence(self, ref: Reference, antecedent: Entity) -> float:
        """Calculate confidence in the resolution."""
        base_confidence = 0.5

        # Type-based confidence
        if ref.ref_type in (ReferenceType.DEFINITE_DESC, ReferenceType.DEMONSTRATIVE_DESC):
            base_confidence += 0.2  # Descriptions are more reliable

        # Distance-based confidence
        sent_dist = ref.sentence_idx - antecedent.sentence_idx
        if sent_dist == 0:
            base_confidence += 0.2
        elif sent_dist == 1:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _build_resolved_text(self, text: str, references: List[Reference]) -> str:
        """Build text with resolved references."""
        # Sort references by position (reverse for replacement)
        refs_to_replace = [r for r in references if r.resolved_to]
        refs_to_replace.sort(key=lambda r: r.char_start, reverse=True)

        resolved = text
        for ref in refs_to_replace:
            # Replace reference with "reference (=antecedent)"
            replacement = f"{ref.text} (={ref.resolved_to})"
            resolved = resolved[:ref.char_start] + replacement + resolved[ref.char_end:]

        return resolved

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, protecting abbreviations."""
        protected = text
        abbrevs = [
            (r'\bet al\.', 'ET_AL_PROT'),
            (r'\bi\.e\.', 'IE_PROT'),
            (r'\be\.g\.', 'EG_PROT'),
            (r'\bvs\.', 'VS_PROT'),
        ]
        for pattern, replacement in abbrevs:
            protected = re.sub(pattern, replacement, protected)

        sentences = re.split(r'(?<=[.!?])\s+', protected)

        # Restore abbreviations
        restored = []
        for sent in sentences:
            for pattern, replacement in abbrevs:
                original = pattern.replace(r'\b', '').replace('\\', '')
                sent = sent.replace(replacement, original)
            restored.append(sent)

        return restored

    def _get_sentence_offsets(self, text: str, sentences: List[str]) -> List[Tuple[int, int]]:
        """Get character offsets for each sentence."""
        offsets = []
        pos = 0
        for sent in sentences:
            # Find this sentence in the original text
            start = text.find(sent[:20], pos)  # Use first 20 chars
            if start == -1:
                start = pos
            end = start + len(sent)
            offsets.append((start, end))
            pos = end
        return offsets

    def _get_sentence_index(self, char_pos: int, offsets: List[Tuple[int, int]]) -> int:
        """Get the sentence index for a character position."""
        for i, (start, end) in enumerate(offsets):
            if start <= char_pos < end:
                return i
        return len(offsets) - 1


class CoreferenceEnhancedExtractor:
    """
    Wraps relation extraction with coreference resolution.
    Resolves references before extracting relations for better coverage.
    """

    def __init__(self, base_extractor=None, resolver: CoreferenceResolver = None):
        """
        Initialize with a base relation extractor.

        Args:
            base_extractor: Any relation extractor with extract_relations(text, entities) method
            resolver: CoreferenceResolver instance
        """
        self.base_extractor = base_extractor
        self.resolver = resolver or CoreferenceResolver()

    def extract_relations(self, text: str, entities: List[Dict]) -> Tuple[List, ResolvedText]:
        """
        Extract relations with coreference resolution.

        Returns:
            Tuple of (relations, resolved_text)
        """
        # First, resolve coreferences
        resolved = self.resolver.resolve(text, entities)

        # Create augmented entities (add resolved references as entity mentions)
        augmented_entities = self._augment_entities(entities, resolved)

        # Extract relations from resolved text
        if self.base_extractor:
            relations = self.base_extractor.extract_relations(resolved.resolved_text, augmented_entities)
        else:
            relations = []

        return relations, resolved

    def _augment_entities(self, entities: List[Dict], resolved: ResolvedText) -> List[Dict]:
        """Add resolved references as additional entity mentions."""
        augmented = list(entities)

        for ref_text, antecedent in resolved.resolutions:
            # Find the entity type of the antecedent
            ent_type = 'UNKNOWN'
            for ent in entities:
                if ent['text'].lower() == antecedent.lower():
                    ent_type = ent.get('type', 'UNKNOWN')
                    break

            # Add the reference as an alias
            augmented.append({
                'text': f"{ref_text} (={antecedent})",
                'type': ent_type,
                'is_resolved_reference': True,
                'original_reference': ref_text,
                'resolved_to': antecedent
            })

        return augmented


def create_resolver(use_llm: bool = False) -> CoreferenceResolver:
    """Factory function to create a coreference resolver."""
    return CoreferenceResolver(use_llm=use_llm)
