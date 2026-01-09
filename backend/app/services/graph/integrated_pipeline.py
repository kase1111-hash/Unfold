"""
Integrated Relation Extraction Pipeline

Combines multiple extraction methods for comprehensive relation discovery:
1. Coreference resolution - Links pronouns to entities
2. Dependency parsing - Syntactic structure analysis
3. LLM extraction - Semantic understanding (Ollama/llama.cpp default)
4. Pattern matching - Rule-based fallback

This pipeline maximizes recall while maintaining precision through
multi-signal confidence scoring.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum

from app.models.graph import RelationType
from app.services.graph.extractor import ExtractedEntity

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRelation:
    """Unified relation format for the integrated pipeline."""
    source_text: str
    target_text: str
    relation_type: RelationType
    confidence: float
    context: Optional[str] = None
    extraction_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExtractionMethod(Enum):
    """Extraction methods used in the pipeline."""
    COREFERENCE = "coreference"
    DEPENDENCY = "dependency"
    LLM = "llm"
    PATTERN = "pattern"
    COOCCURRENCE = "cooccurrence"


class IntegratedRelationExtractor:
    """
    Integrated pipeline combining multiple relation extraction methods.

    Default configuration uses Ollama for LLM extraction, with fallback
    to pattern-based methods when LLM is unavailable.
    """

    def __init__(
        self,
        use_coreference: bool = True,
        use_dependency: bool = True,
        use_llm: bool = True,
        use_patterns: bool = True,
        llm_provider: str = "ollama",
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        llama_model_path: Optional[str] = None,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the integrated extractor.

        Args:
            use_coreference: Enable coreference resolution
            use_dependency: Enable dependency parsing
            use_llm: Enable LLM-based extraction
            use_patterns: Enable pattern-based extraction
            llm_provider: Preferred LLM provider ("ollama", "llama_cpp", "openai", "anthropic")
            ollama_model: Ollama model name
            ollama_url: Ollama server URL
            llama_model_path: Path to GGUF model for llama.cpp
            openai_key: OpenAI API key
            anthropic_key: Anthropic API key
            min_confidence: Minimum confidence threshold
        """
        self.use_coreference = use_coreference
        self.use_dependency = use_dependency
        self.use_llm = use_llm
        self.use_patterns = use_patterns
        self.min_confidence = min_confidence

        # Lazy-loaded components
        self._coref_resolver = None
        self._dep_parser = None
        self._llm_extractor = None

        # LLM configuration
        self._llm_config = {
            "preferred_provider": llm_provider,
            "ollama_model": ollama_model,
            "ollama_url": ollama_url,
            "llama_model_path": llama_model_path,
            "openai_key": openai_key,
            "anthropic_key": anthropic_key,
        }

        logger.info(f"[IntegratedPipeline] Initialized with LLM provider: {llm_provider}")

    @property
    def coref_resolver(self):
        """Lazy load coreference resolver."""
        if self._coref_resolver is None and self.use_coreference:
            try:
                from app.services.graph.coreference import CoreferenceResolver
                self._coref_resolver = CoreferenceResolver(use_llm=False)
            except ImportError as e:
                logger.warning(f"Coreference module unavailable: {e}")
        return self._coref_resolver

    @property
    def dep_parser(self):
        """Lazy load dependency parser."""
        if self._dep_parser is None and self.use_dependency:
            try:
                from app.services.graph.dependency_parsing import DependencyParser
                self._dep_parser = DependencyParser(use_spacy=True)
            except ImportError as e:
                logger.warning(f"Dependency parsing module unavailable: {e}")
        return self._dep_parser

    @property
    def llm_extractor(self):
        """Lazy load LLM extractor."""
        if self._llm_extractor is None and self.use_llm:
            try:
                from app.services.graph.llm_relations import LLMRelationExtractor
                self._llm_extractor = LLMRelationExtractor(
                    preferred_provider=self._llm_config["preferred_provider"],
                    ollama_model=self._llm_config["ollama_model"],
                    ollama_url=self._llm_config["ollama_url"],
                    llama_model_path=self._llm_config["llama_model_path"],
                    openai_api_key=self._llm_config["openai_key"],
                    anthropic_api_key=self._llm_config["anthropic_key"],
                )
            except ImportError as e:
                logger.warning(f"LLM relations module unavailable: {e}")
        return self._llm_extractor

    def extract_relations(
        self,
        text: str,
        entities: List[ExtractedEntity],
        max_relations: int = 100,
    ) -> List[ExtractedRelation]:
        """
        Extract relations using all available methods.

        Args:
            text: Source text
            entities: List of extracted entities
            max_relations: Maximum relations to return

        Returns:
            List of extracted relations, deduplicated and ranked
        """
        if not entities or len(entities) < 2:
            return []

        # Convert entities to dict format for modules that expect it
        entity_dicts = [
            {"text": e.text, "type": e.entity_type.value}
            for e in entities
        ]

        all_relations: List[ExtractedRelation] = []

        # Step 1: Coreference resolution
        resolved_text = text
        if self.coref_resolver:
            try:
                resolved = self.coref_resolver.resolve(text, entity_dicts)
                resolved_text = resolved.resolved_text
                logger.debug(f"Resolved {len(resolved.resolutions)} coreferences")
            except Exception as e:
                logger.warning(f"Coreference resolution failed: {e}")

        # Step 2: Dependency parsing extraction
        if self.dep_parser:
            try:
                dep_relations = self.dep_parser.extract_relations(resolved_text, entity_dicts)
                for rel in dep_relations:
                    all_relations.append(ExtractedRelation(
                        source_text=rel.subject,
                        target_text=rel.object,
                        relation_type=self._map_relation_type(rel.relation_type),
                        confidence=rel.confidence,
                        context=rel.evidence,
                        extraction_method=ExtractionMethod.DEPENDENCY.value,
                    ))
                logger.debug(f"Dependency parsing: {len(dep_relations)} relations")
            except Exception as e:
                logger.warning(f"Dependency extraction failed: {e}")

        # Step 3: LLM extraction
        if self.llm_extractor and self.llm_extractor.is_available():
            try:
                from app.services.graph.llm_relations import EntityPair

                # Build entity pairs for LLM
                entity_pairs = self._build_entity_pairs(resolved_text, entity_dicts)

                if entity_pairs:
                    llm_relations = self.llm_extractor.extract_relations(entity_pairs)
                    for rel in llm_relations:
                        all_relations.append(ExtractedRelation(
                            source_text=rel.source,
                            target_text=rel.target,
                            relation_type=self._map_llm_relation_type(rel.relation_type.value),
                            confidence=rel.confidence,
                            context=rel.reasoning,
                            extraction_method=ExtractionMethod.LLM.value,
                        ))
                    logger.debug(f"LLM extraction: {len(llm_relations)} relations")
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

        # Step 4: Pattern-based extraction
        if self.use_patterns:
            try:
                pattern_relations = self._extract_patterns(resolved_text, entity_dicts)
                all_relations.extend(pattern_relations)
                logger.debug(f"Pattern extraction: {len(pattern_relations)} relations")
            except Exception as e:
                logger.warning(f"Pattern extraction failed: {e}")

        # Step 5: Co-occurrence fallback for uncovered pairs
        if len(all_relations) < max_relations // 2:
            cooc_relations = self._extract_cooccurrence(resolved_text, entity_dicts, all_relations)
            all_relations.extend(cooc_relations)
            logger.debug(f"Co-occurrence: {len(cooc_relations)} relations")

        # Deduplicate and filter
        relations = self._deduplicate_relations(all_relations)
        relations = [r for r in relations if r.confidence >= self.min_confidence]
        relations.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(f"[IntegratedPipeline] Extracted {len(relations)} relations")
        return relations[:max_relations]

    def _build_entity_pairs(
        self,
        text: str,
        entities: List[Dict],
        max_pairs: int = 50
    ) -> List[Any]:
        """Build entity pairs for LLM extraction."""
        from app.services.graph.llm_relations import EntityPair

        pairs = []
        sentences = self._split_sentences(text)

        for sent in sentences:
            sent_lower = sent.lower()

            # Find entities in this sentence
            sent_entities = [
                e for e in entities
                if e['text'].lower() in sent_lower
            ]

            # Create pairs
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i+1:]:
                    if e1['text'].lower() != e2['text'].lower():
                        pairs.append(EntityPair(
                            source_text=e1['text'],
                            source_type=e1.get('type', 'UNKNOWN'),
                            target_text=e2['text'],
                            target_type=e2.get('type', 'UNKNOWN'),
                            context=sent
                        ))

                        if len(pairs) >= max_pairs:
                            return pairs

        return pairs

    def _extract_patterns(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[ExtractedRelation]:
        """Pattern-based relation extraction."""
        import re

        relations = []
        entity_texts = {e['text'].lower(): e for e in entities}
        text_lower = text.lower()

        patterns = [
            (r'(\w+(?:\s+et\s+al\.)?)\s+introduced\s+(?:the\s+)?(\w+)', 'INTRODUCES'),
            (r'(\w+(?:\s+et\s+al\.)?)\s+developed\s+(?:the\s+)?(\w+)', 'DEVELOPS'),
            (r'(\w+)\s+uses?\s+(\w+)', 'USES'),
            (r'(\w+)\s+trained\s+on\s+(\w+)', 'TRAINED_ON'),
            (r'(\w+)\s+evaluated\s+on\s+(\w+)', 'EVALUATED_ON'),
            (r'(\w+)\s+at\s+(\w+(?:\s+\w+)?)', 'AFFILIATED_WITH'),
            (r'(\w+)\s+outperforms?\s+(\w+)', 'OUTPERFORMS'),
            (r'unlike\s+(\w+).*?,?\s*(?:the\s+)?(\w+)', 'CONTRASTS_WITH'),
        ]

        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text_lower):
                source = match.group(1)
                target = match.group(2)

                source_match = self._match_entity(source, entity_texts)
                target_match = self._match_entity(target, entity_texts)

                if source_match and target_match and source_match != target_match:
                    relations.append(ExtractedRelation(
                        source_text=entity_texts[source_match]['text'],
                        target_text=entity_texts[target_match]['text'],
                        relation_type=self._map_relation_type(rel_type),
                        confidence=0.7,
                        context=match.group(0),
                        extraction_method=ExtractionMethod.PATTERN.value,
                    ))

        return relations

    def _extract_cooccurrence(
        self,
        text: str,
        entities: List[Dict],
        existing: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """Co-occurrence based relation extraction."""
        relations = []
        entity_texts = {e['text'].lower(): e for e in entities}

        # Get existing pairs
        existing_pairs = {
            (r.source_text.lower(), r.target_text.lower())
            for r in existing
        }

        sentences = self._split_sentences(text)

        for sent in sentences:
            sent_lower = sent.lower()

            # Find entities in sentence
            sent_entities = [
                e for e in entities
                if e['text'].lower() in sent_lower
            ]

            # Create pairs for uncovered entities
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i+1:]:
                    e1_lower = e1['text'].lower()
                    e2_lower = e2['text'].lower()

                    if e1_lower != e2_lower:
                        pair = (e1_lower, e2_lower)
                        pair_rev = (e2_lower, e1_lower)

                        if pair not in existing_pairs and pair_rev not in existing_pairs:
                            relations.append(ExtractedRelation(
                                source_text=e1['text'],
                                target_text=e2['text'],
                                relation_type=RelationType.RELATED_TO,
                                confidence=0.5,
                                context=sent[:100],
                                extraction_method=ExtractionMethod.COOCCURRENCE.value,
                            ))
                            existing_pairs.add(pair)

        return relations

    def _match_entity(
        self,
        text: str,
        entity_texts: Dict[str, Dict],
        prefer_longer: bool = True
    ) -> Optional[str]:
        """Match text to an entity, preferring longer matches."""
        text_lower = text.lower().strip()

        # Exact match
        if text_lower in entity_texts:
            return text_lower

        # Collect partial matches
        matches = []
        for ent_key in entity_texts:
            if text_lower in ent_key or ent_key in text_lower:
                matches.append(ent_key)

        if not matches:
            return None

        if prefer_longer:
            matches.sort(key=len, reverse=True)

        return matches[0]

    def _deduplicate_relations(
        self,
        relations: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """Remove duplicate relations, keeping highest confidence."""
        seen = {}

        for rel in relations:
            # Handle relation_type as either enum or string
            rel_type_value = (
                rel.relation_type.value
                if hasattr(rel.relation_type, 'value')
                else str(rel.relation_type)
            )

            key = (
                rel.source_text.lower(),
                rel.target_text.lower(),
                rel_type_value,
            )

            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel

        return list(seen.values())

    def _map_relation_type(self, type_str: str) -> RelationType:
        """Map string relation type to RelationType enum."""
        mapping = {
            "INTRODUCES": RelationType.RELATED_TO,
            "DEVELOPS": RelationType.DERIVES_FROM,
            "CREATES": RelationType.DERIVES_FROM,
            "USES": RelationType.USES_METHOD,
            "TRAINED_ON": RelationType.USES_DATASET,
            "EVALUATED_ON": RelationType.USES_DATASET,
            "AFFILIATED_WITH": RelationType.AFFILIATED_WITH,
            "OUTPERFORMS": RelationType.CONTRASTS_WITH,
            "CONTRASTS_WITH": RelationType.CONTRASTS_WITH,
            "EXPLAINS": RelationType.EXPLAINS,
            "CITES": RelationType.CITES,
            "DERIVES_FROM": RelationType.DERIVES_FROM,
            "PART_OF": RelationType.PART_OF,
            "RELATED_TO": RelationType.RELATED_TO,
        }
        return mapping.get(type_str.upper(), RelationType.RELATED_TO)

    def _map_llm_relation_type(self, type_str: str) -> RelationType:
        """Map LLM relation type to RelationType enum."""
        mapping = {
            "CREATES": RelationType.DERIVES_FROM,
            "DEVELOPS": RelationType.DERIVES_FROM,
            "INTRODUCES": RelationType.RELATED_TO,
            "PART_OF": RelationType.PART_OF,
            "CONTAINS": RelationType.PART_OF,
            "IMPLEMENTS": RelationType.USES_METHOD,
            "EXTENDS": RelationType.DERIVES_FROM,
            "USES": RelationType.USES_METHOD,
            "ENABLES": RelationType.EXPLAINS,
            "IMPROVES": RelationType.CONTRASTS_WITH,
            "REPLACES": RelationType.CONTRASTS_WITH,
            "CONTRASTS_WITH": RelationType.CONTRASTS_WITH,
            "SIMILAR_TO": RelationType.RELATED_TO,
            "RELATED_TO": RelationType.RELATED_TO,
            "AFFILIATED_WITH": RelationType.AFFILIATED_WITH,
            "TRAINED_ON": RelationType.USES_DATASET,
            "EVALUATED_ON": RelationType.USES_DATASET,
        }
        return mapping.get(type_str.upper(), RelationType.RELATED_TO)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, protecting abbreviations."""
        import re

        protected = text
        abbrevs = [
            (r'\bet al\.', 'ET_AL_P'),
            (r'\bi\.e\.', 'IE_P'),
            (r'\be\.g\.', 'EG_P'),
            (r'\bvs\.', 'VS_P'),
        ]
        for pattern, repl in abbrevs:
            protected = re.sub(pattern, repl, protected)

        sentences = re.split(r'(?<=[.!?])\s+', protected)

        restored = []
        for sent in sentences:
            for pattern, repl in abbrevs:
                original = pattern.replace(r'\b', '').replace('\\', '')
                sent = sent.replace(repl, original)
            restored.append(sent)

        return [s.strip() for s in restored if s.strip()]

    def get_status(self) -> Dict[str, bool]:
        """Get availability status of each extraction method."""
        return {
            "coreference": self.coref_resolver is not None,
            "dependency": self.dep_parser is not None,
            "llm": self.llm_extractor is not None and self.llm_extractor.is_available(),
            "patterns": self.use_patterns,
        }


# Global instance
_integrated_extractor: Optional[IntegratedRelationExtractor] = None


def get_integrated_extractor(
    use_llm: bool = True,
    llm_provider: str = "ollama",
    **kwargs
) -> IntegratedRelationExtractor:
    """
    Get or create integrated extractor instance.

    Args:
        use_llm: Whether to enable LLM extraction
        llm_provider: Preferred LLM provider
        **kwargs: Additional configuration

    Returns:
        IntegratedRelationExtractor instance
    """
    global _integrated_extractor

    if _integrated_extractor is None:
        _integrated_extractor = IntegratedRelationExtractor(
            use_llm=use_llm,
            llm_provider=llm_provider,
            **kwargs
        )

    return _integrated_extractor


def reset_extractor():
    """Reset the global extractor instance."""
    global _integrated_extractor
    _integrated_extractor = None
