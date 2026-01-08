"""Relation extraction service using LLM."""

import json
from typing import Any

from app.config import get_settings
from app.models.graph import RelationType
from app.services.graph.extractor import ExtractedEntity, ExtractedRelation

settings = get_settings()

# Relation extraction prompt template
RELATION_EXTRACTION_PROMPT = """You are an expert at extracting relationships between concepts from academic and technical text.

Given the following text and list of entities, identify relationships between the entities.

Text:
{text}

Entities:
{entities}

For each relationship found, provide:
1. source_entity: The entity where the relationship originates
2. target_entity: The entity the relationship points to
3. relation_type: One of: EXPLAINS, CITES, CONTRASTS_WITH, DERIVES_FROM, RELATED_TO, USES_METHOD, USES_DATASET, PART_OF
4. confidence: A score from 0.0 to 1.0 indicating your confidence
5. context: A brief phrase from the text supporting this relationship

Respond ONLY with a JSON array of relationships. Example format:
[
  {{"source_entity": "quantum computing", "target_entity": "qubits", "relation_type": "USES_METHOD", "confidence": 0.9, "context": "quantum computing relies on qubits"}}
]

If no relationships are found, respond with an empty array: []

Relationships:"""


class RelationExtractor:
    """Extract relations between entities using LLM."""

    def __init__(self, use_openai: bool = True):
        """Initialize relation extractor.

        Args:
            use_openai: Whether to use OpenAI API (vs local model)
        """
        self.use_openai = use_openai
        self._client = None

    def _get_openai_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")

            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)

        return self._client

    async def extract_relations(
        self,
        text: str,
        entities: list[ExtractedEntity],
        min_confidence: float = 0.5,
    ) -> list[ExtractedRelation]:
        """Extract relations between entities using LLM.

        Args:
            text: Source text
            entities: List of extracted entities
            min_confidence: Minimum confidence threshold

        Returns:
            List of extracted relations
        """
        if not entities or len(entities) < 2:
            return []

        # Format entities for prompt
        entity_list = "\n".join(
            f"- {e.text} ({e.entity_type.value})"
            for e in entities
        )

        prompt = RELATION_EXTRACTION_PROMPT.format(
            text=text[:4000],  # Limit text length
            entities=entity_list,
        )

        try:
            response = await self._call_llm(prompt)
            relations = self._parse_relations_response(response, min_confidence)
            return relations
        except Exception as e:
            # Log error and return empty list
            print(f"Relation extraction error: {e}")
            return []

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API for relation extraction.

        Args:
            prompt: Formatted prompt

        Returns:
            LLM response text
        """
        if self.use_openai:
            return await self._call_openai(prompt)
        else:
            return await self._call_local_model(prompt)

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API.

        Args:
            prompt: Formatted prompt

        Returns:
            Response text
        """
        client = self._get_openai_client()

        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting structured relationships from text. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        return response.choices[0].message.content or "[]"

    async def _call_local_model(self, prompt: str) -> str:
        """Call local model (placeholder for future implementation).

        Args:
            prompt: Formatted prompt

        Returns:
            Response text
        """
        # Placeholder for local model integration (e.g., Ollama, llama.cpp)
        raise NotImplementedError("Local model not yet implemented")

    def _parse_relations_response(
        self,
        response: str,
        min_confidence: float,
    ) -> list[ExtractedRelation]:
        """Parse LLM response into ExtractedRelation objects.

        Args:
            response: LLM response text
            min_confidence: Minimum confidence threshold

        Returns:
            List of extracted relations
        """
        relations = []

        try:
            # Clean response (remove markdown code blocks if present)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            data = json.loads(response)

            if not isinstance(data, list):
                return []

            for item in data:
                try:
                    relation_type = self._map_relation_type(item.get("relation_type", ""))
                    confidence = float(item.get("confidence", 0.5))

                    if confidence < min_confidence:
                        continue

                    relations.append(
                        ExtractedRelation(
                            source_text=item.get("source_entity", ""),
                            target_text=item.get("target_entity", ""),
                            relation_type=relation_type,
                            confidence=confidence,
                            context=item.get("context"),
                        )
                    )
                except (KeyError, ValueError):
                    continue

        except json.JSONDecodeError:
            pass

        return relations

    def _map_relation_type(self, type_str: str) -> RelationType:
        """Map string relation type to enum.

        Args:
            type_str: Relation type string from LLM

        Returns:
            RelationType enum value
        """
        mapping = {
            "EXPLAINS": RelationType.EXPLAINS,
            "CITES": RelationType.CITES,
            "CONTRASTS_WITH": RelationType.CONTRASTS_WITH,
            "DERIVES_FROM": RelationType.DERIVES_FROM,
            "RELATED_TO": RelationType.RELATED_TO,
            "USES_METHOD": RelationType.USES_METHOD,
            "USES_DATASET": RelationType.USES_DATASET,
            "PART_OF": RelationType.PART_OF,
            "AUTHORED_BY": RelationType.AUTHORED_BY,
            "AFFILIATED_WITH": RelationType.AFFILIATED_WITH,
        }

        return mapping.get(type_str.upper(), RelationType.RELATED_TO)


class RuleBasedRelationExtractor:
    """Extract relations using rule-based patterns (no LLM required)."""

    def __init__(self):
        """Initialize rule-based extractor."""
        self._nlp = None

    def _load_spacy(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract_relations(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        """Extract relations using dependency parsing and patterns.

        Args:
            text: Source text
            entities: List of extracted entities

        Returns:
            List of extracted relations
        """
        nlp = self._load_spacy()
        doc = nlp(text)
        relations = []

        # Create entity lookup by text
        entity_texts = {e.text.lower() for e in entities}

        # Pattern-based extraction
        for sent in doc.sents:
            relations.extend(
                self._extract_from_sentence(sent, entity_texts)
            )

        return relations

    def _extract_from_sentence(
        self,
        sent: Any,
        entity_texts: set[str],
    ) -> list[ExtractedRelation]:
        """Extract relations from a single sentence.

        Args:
            sent: spaCy sentence span
            entity_texts: Set of entity texts (lowercase)

        Returns:
            List of extracted relations
        """
        relations = []

        # Find verb-mediated relations
        for token in sent:
            if token.pos_ == "VERB":
                # Look for subject and object
                subjects = [
                    child for child in token.children
                    if child.dep_ in {"nsubj", "nsubjpass"}
                ]
                objects = [
                    child for child in token.children
                    if child.dep_ in {"dobj", "pobj", "attr"}
                ]

                for subj in subjects:
                    subj_text = self._get_full_phrase(subj)
                    if subj_text.lower() not in entity_texts:
                        continue

                    for obj in objects:
                        obj_text = self._get_full_phrase(obj)
                        if obj_text.lower() not in entity_texts:
                            continue

                        relation_type = self._infer_relation_type(token.lemma_)

                        relations.append(
                            ExtractedRelation(
                                source_text=subj_text,
                                target_text=obj_text,
                                relation_type=relation_type,
                                confidence=0.6,
                                context=sent.text,
                            )
                        )

        return relations

    def _get_full_phrase(self, token: Any) -> str:
        """Get full noun phrase for a token.

        Args:
            token: spaCy token

        Returns:
            Full phrase text
        """
        # Get the noun chunk containing this token
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text
        return token.text

    def _infer_relation_type(self, verb_lemma: str) -> RelationType:
        """Infer relation type from verb.

        Args:
            verb_lemma: Lemmatized verb

        Returns:
            Inferred relation type
        """
        verb_mappings = {
            # EXPLAINS
            "explain": RelationType.EXPLAINS,
            "describe": RelationType.EXPLAINS,
            "define": RelationType.EXPLAINS,
            "illustrate": RelationType.EXPLAINS,
            "clarify": RelationType.EXPLAINS,
            # CITES
            "cite": RelationType.CITES,
            "reference": RelationType.CITES,
            "mention": RelationType.CITES,
            "quote": RelationType.CITES,
            # DERIVES_FROM
            "derive": RelationType.DERIVES_FROM,
            "base": RelationType.DERIVES_FROM,
            "build": RelationType.DERIVES_FROM,
            "extend": RelationType.DERIVES_FROM,
            "develop": RelationType.DERIVES_FROM,
            # CONTRASTS_WITH
            "contrast": RelationType.CONTRASTS_WITH,
            "differ": RelationType.CONTRASTS_WITH,
            "oppose": RelationType.CONTRASTS_WITH,
            "compare": RelationType.CONTRASTS_WITH,
            # USES_METHOD
            "use": RelationType.USES_METHOD,
            "apply": RelationType.USES_METHOD,
            "employ": RelationType.USES_METHOD,
            "utilize": RelationType.USES_METHOD,
            # PART_OF
            "include": RelationType.PART_OF,
            "contain": RelationType.PART_OF,
            "comprise": RelationType.PART_OF,
            "consist": RelationType.PART_OF,
        }

        return verb_mappings.get(verb_lemma, RelationType.RELATED_TO)


# Global extractors
_llm_extractor: RelationExtractor | None = None
_rule_extractor: RuleBasedRelationExtractor | None = None


def get_relation_extractor(use_llm: bool = True) -> RelationExtractor | RuleBasedRelationExtractor:
    """Get relation extractor instance.

    Args:
        use_llm: Whether to use LLM-based extraction

    Returns:
        Extractor instance
    """
    global _llm_extractor, _rule_extractor

    if use_llm:
        if _llm_extractor is None:
            _llm_extractor = RelationExtractor()
        return _llm_extractor
    else:
        if _rule_extractor is None:
            _rule_extractor = RuleBasedRelationExtractor()
        return _rule_extractor
