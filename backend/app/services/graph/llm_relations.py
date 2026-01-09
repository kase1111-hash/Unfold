"""
LLM-Based Relation Extraction Module

Uses large language models to extract complex semantic relations that
pattern-based approaches miss.

Supported providers (in default priority order):
1. Ollama (default) - Local LLM server for offline use
2. llama.cpp - Direct model loading for fully offline inference
3. OpenAI - Cloud API (requires OPENAI_API_KEY)
4. Anthropic - Cloud API (requires ANTHROPIC_API_KEY)

Key features:
- Multiple LLM provider support with automatic fallback
- Ollama as default for local/offline operation
- Batch processing for efficiency
- Caching to minimize API calls
- Graceful fallback when LLM unavailable
- Structured output parsing

Setup for Ollama (recommended for offline use):
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull llama3.2
    ollama serve

Setup for llama.cpp (fully offline):
    pip install llama-cpp-python
    export LLAMA_MODEL_PATH=/path/to/model.gguf
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """Relation types for LLM extraction."""
    CREATES = "CREATES"
    DEVELOPS = "DEVELOPS"
    INTRODUCES = "INTRODUCES"
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    IMPLEMENTS = "IMPLEMENTS"
    EXTENDS = "EXTENDS"
    USES = "USES"
    ENABLES = "ENABLES"
    IMPROVES = "IMPROVES"
    REPLACES = "REPLACES"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    SIMILAR_TO = "SIMILAR_TO"
    RELATED_TO = "RELATED_TO"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    TRAINED_ON = "TRAINED_ON"
    EVALUATED_ON = "EVALUATED_ON"
    NONE = "NONE"


@dataclass
class EntityPair:
    """A pair of entities to analyze."""
    source_text: str
    source_type: str
    target_text: str
    target_type: str
    context: str


@dataclass
class LLMRelation:
    """A relation extracted by the LLM."""
    source: str
    target: str
    relation_type: RelationType
    confidence: float
    reasoning: str = ""


# ============================================================================
# LLM PROVIDER INTERFACE
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def extract_relations(
        self,
        entity_pairs: List[EntityPair]
    ) -> List[LLMRelation]:
        """Extract relations from entity pairs."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


# ============================================================================
# OPENAI PROVIDER
# ============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI-based relation extraction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client = None

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def extract_relations(
        self,
        entity_pairs: List[EntityPair]
    ) -> List[LLMRelation]:
        if not self.is_available():
            return []

        relations = []
        client = self._get_client()

        # Process in batches of 10
        batch_size = 10
        for i in range(0, len(entity_pairs), batch_size):
            batch = entity_pairs[i:i + batch_size]
            batch_relations = self._process_batch(client, batch)
            relations.extend(batch_relations)

        return relations

    def _process_batch(
        self,
        client,
        batch: List[EntityPair]
    ) -> List[LLMRelation]:
        """Process a batch of entity pairs."""

        # Build the prompt
        pairs_text = "\n".join([
            f"{idx+1}. Source: \"{p.source_text}\" ({p.source_type}) | "
            f"Target: \"{p.target_text}\" ({p.target_type})\n"
            f"   Context: \"{p.context[:200]}...\""
            for idx, p in enumerate(batch)
        ])

        prompt = f"""Analyze the semantic relationships between entity pairs based on their context.

For each pair, determine the most accurate relationship type from this list:
- CREATES: Person/Org created the technology/method
- DEVELOPS: Person/Org developed or built something
- INTRODUCES: Person/Org introduced or proposed something new
- USES: Technology/Method uses another method or component
- IMPLEMENTS: Technology implements a method
- CONTAINS: System contains a component
- PART_OF: Component is part of a system
- EXTENDS: Technology extends or builds upon another
- IMPROVES: Technology/Method improves upon another
- REPLACES: New technology replaces old one
- CONTRASTS_WITH: Two things are compared as different
- SIMILAR_TO: Two things are similar
- AFFILIATED_WITH: Person is affiliated with organization
- TRAINED_ON: Model was trained on dataset
- EVALUATED_ON: Model was evaluated on benchmark/dataset
- ENABLES: Technology enables a capability
- RELATED_TO: Generic relationship (use only if no specific type fits)
- NONE: No meaningful relationship exists

Entity Pairs to Analyze:
{pairs_text}

Respond with a JSON array. For each pair, provide:
{{"pair": 1, "relation": "RELATION_TYPE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Focus on the specific semantic relationship implied by the context. Be precise.
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise semantic relation extraction system. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            return self._parse_response(content, batch)

        except Exception as e:
            logger.warning(f"OpenAI extraction failed: {e}")
            return []

    def _parse_response(
        self,
        content: str,
        batch: List[EntityPair]
    ) -> List[LLMRelation]:
        """Parse LLM response into relations."""
        relations = []

        try:
            data = json.loads(content)

            # Handle both array and object with "relations" key
            if isinstance(data, dict):
                items = data.get("relations", data.get("results", []))
            else:
                items = data

            for item in items:
                pair_idx = item.get("pair", 1) - 1
                if 0 <= pair_idx < len(batch):
                    pair = batch[pair_idx]
                    rel_type_str = item.get("relation", "RELATED_TO").upper()

                    try:
                        rel_type = RelationType(rel_type_str)
                    except ValueError:
                        rel_type = RelationType.RELATED_TO

                    if rel_type != RelationType.NONE:
                        relations.append(LLMRelation(
                            source=pair.source_text,
                            target=pair.target_text,
                            relation_type=rel_type,
                            confidence=float(item.get("confidence", 0.7)),
                            reasoning=item.get("reasoning", "")
                        ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return relations


# ============================================================================
# ANTHROPIC PROVIDER
# ============================================================================

class AnthropicProvider(LLMProvider):
    """Anthropic Claude-based relation extraction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.1
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client = None

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def extract_relations(
        self,
        entity_pairs: List[EntityPair]
    ) -> List[LLMRelation]:
        if not self.is_available():
            return []

        relations = []
        client = self._get_client()

        # Process in batches
        batch_size = 10
        for i in range(0, len(entity_pairs), batch_size):
            batch = entity_pairs[i:i + batch_size]
            batch_relations = self._process_batch(client, batch)
            relations.extend(batch_relations)

        return relations

    def _process_batch(
        self,
        client,
        batch: List[EntityPair]
    ) -> List[LLMRelation]:
        """Process a batch of entity pairs."""

        pairs_text = "\n".join([
            f"{idx+1}. \"{p.source_text}\" ({p.source_type}) → \"{p.target_text}\" ({p.target_type})\n"
            f"   Context: {p.context[:150]}"
            for idx, p in enumerate(batch)
        ])

        prompt = f"""Extract semantic relationships between these entity pairs.

Valid relation types: CREATES, DEVELOPS, INTRODUCES, USES, IMPLEMENTS, CONTAINS, PART_OF, EXTENDS, IMPROVES, REPLACES, CONTRASTS_WITH, SIMILAR_TO, AFFILIATED_WITH, TRAINED_ON, EVALUATED_ON, ENABLES, RELATED_TO, NONE

Entity Pairs:
{pairs_text}

Output JSON array with format:
[{{"pair": 1, "relation": "TYPE", "confidence": 0.8, "reasoning": "..."}}]

Be precise. Use NONE if no clear relationship exists."""

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text

            # Extract JSON from response
            json_match = content
            if "```json" in content:
                json_match = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_match = content.split("```")[1].split("```")[0]

            return self._parse_response(json_match.strip(), batch)

        except Exception as e:
            logger.warning(f"Anthropic extraction failed: {e}")
            return []

    def _parse_response(
        self,
        content: str,
        batch: List[EntityPair]
    ) -> List[LLMRelation]:
        """Parse response into relations."""
        relations = []

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                data = data.get("relations", [])

            for item in data:
                pair_idx = item.get("pair", 1) - 1
                if 0 <= pair_idx < len(batch):
                    pair = batch[pair_idx]
                    rel_type_str = item.get("relation", "RELATED_TO").upper()

                    try:
                        rel_type = RelationType(rel_type_str)
                    except ValueError:
                        rel_type = RelationType.RELATED_TO

                    if rel_type != RelationType.NONE:
                        relations.append(LLMRelation(
                            source=pair.source_text,
                            target=pair.target_text,
                            relation_type=rel_type,
                            confidence=float(item.get("confidence", 0.7)),
                            reasoning=item.get("reasoning", "")
                        ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response: {e}")

        return relations


# ============================================================================
# OLLAMA PROVIDER (Local LLM - Default)
# ============================================================================

class OllamaProvider(LLMProvider):
    """
    Ollama-based relation extraction for local/offline LLM inference.

    Ollama runs open-source models locally. Install from: https://ollama.ai
    Recommended models: llama3.2, mistral, qwen2.5

    Usage:
        1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
        2. Pull a model: ollama pull llama3.2
        3. Start server: ollama serve
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 60
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        if self._available is not None:
            return self._available

        try:
            import requests
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                self._available = self.model.split(":")[0] in model_names
                if not self._available:
                    logger.info(f"Ollama model '{self.model}' not found. Available: {model_names}")
            else:
                self._available = False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False

        return self._available

    def extract_relations(
        self,
        entity_pairs: List[EntityPair]
    ) -> List[LLMRelation]:
        if not self.is_available():
            return []

        relations = []

        # Process in smaller batches for local models
        batch_size = 5
        for i in range(0, len(entity_pairs), batch_size):
            batch = entity_pairs[i:i + batch_size]
            batch_relations = self._process_batch(batch)
            relations.extend(batch_relations)

        return relations

    def _process_batch(self, batch: List[EntityPair]) -> List[LLMRelation]:
        """Process a batch of entity pairs with Ollama."""
        import requests

        pairs_text = "\n".join([
            f"{idx+1}. \"{p.source_text}\" ({p.source_type}) → \"{p.target_text}\" ({p.target_type})\n"
            f"   Context: {p.context[:150]}"
            for idx, p in enumerate(batch)
        ])

        prompt = f"""You are a relation extraction system. Analyze entity pairs and determine their semantic relationship.

Valid relation types:
CREATES, DEVELOPS, INTRODUCES, USES, IMPLEMENTS, CONTAINS, PART_OF, EXTENDS, IMPROVES, REPLACES, CONTRASTS_WITH, SIMILAR_TO, AFFILIATED_WITH, TRAINED_ON, EVALUATED_ON, ENABLES, RELATED_TO, NONE

Entity Pairs:
{pairs_text}

Respond with ONLY a JSON array (no other text):
[{{"pair": 1, "relation": "TYPE", "confidence": 0.8, "reasoning": "brief reason"}}]"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                    }
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                content = response.json().get("response", "")
                return self._parse_response(content, batch)
            else:
                logger.warning(f"Ollama request failed: {response.status_code}")
                return []

        except Exception as e:
            logger.warning(f"Ollama extraction failed: {e}")
            return []

    def _parse_response(
        self,
        content: str,
        batch: List[EntityPair]
    ) -> List[LLMRelation]:
        """Parse Ollama response into relations."""
        relations = []

        try:
            # Try to extract JSON from response
            json_str = content.strip()

            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            # Find JSON array in response
            start = json_str.find("[")
            end = json_str.rfind("]") + 1
            if start != -1 and end > start:
                json_str = json_str[start:end]

            data = json.loads(json_str)

            for item in data:
                pair_idx = item.get("pair", 1) - 1
                if 0 <= pair_idx < len(batch):
                    pair = batch[pair_idx]
                    rel_type_str = item.get("relation", "RELATED_TO").upper()

                    try:
                        rel_type = RelationType(rel_type_str)
                    except ValueError:
                        rel_type = RelationType.RELATED_TO

                    if rel_type != RelationType.NONE:
                        relations.append(LLMRelation(
                            source=pair.source_text,
                            target=pair.target_text,
                            relation_type=rel_type,
                            confidence=float(item.get("confidence", 0.7)),
                            reasoning=item.get("reasoning", "")
                        ))

        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse Ollama response as JSON: {e}")
            # Try to extract relations using simple pattern matching as fallback
            relations = self._fallback_parse(content, batch)

        return relations

    def _fallback_parse(
        self,
        content: str,
        batch: List[EntityPair]
    ) -> List[LLMRelation]:
        """Fallback parsing when JSON extraction fails."""
        import re
        relations = []

        # Look for patterns like "1. INTRODUCES" or "pair 1: DEVELOPS"
        pattern = r'(?:pair\s*)?(\d+)[:\.\s]+([A-Z_]+)'
        matches = re.findall(pattern, content.upper())

        for pair_num, rel_type_str in matches:
            pair_idx = int(pair_num) - 1
            if 0 <= pair_idx < len(batch):
                try:
                    rel_type = RelationType(rel_type_str)
                    if rel_type != RelationType.NONE:
                        pair = batch[pair_idx]
                        relations.append(LLMRelation(
                            source=pair.source_text,
                            target=pair.target_text,
                            relation_type=rel_type,
                            confidence=0.6,  # Lower confidence for fallback
                            reasoning="Extracted via fallback pattern"
                        ))
                except ValueError:
                    pass

        return relations


# ============================================================================
# LLAMA.CPP PROVIDER (Offline via llama-cpp-python)
# ============================================================================

class LlamaCppProvider(LLMProvider):
    """
    llama.cpp-based relation extraction for fully offline inference.

    Uses llama-cpp-python for direct model loading without a server.

    Install: pip install llama-cpp-python

    Usage:
        provider = LlamaCppProvider(model_path="/path/to/model.gguf")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: int = 4,
        temperature: float = 0.1,
        max_tokens: int = 512
    ):
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH")
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = None
        self._available = None

    def is_available(self) -> bool:
        """Check if llama-cpp-python is installed and model exists."""
        if self._available is not None:
            return self._available

        try:
            from llama_cpp import Llama
            if self.model_path and os.path.exists(self.model_path):
                self._available = True
            else:
                logger.debug(f"llama.cpp model not found: {self.model_path}")
                self._available = False
        except ImportError:
            logger.debug("llama-cpp-python not installed")
            self._available = False

        return self._available

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None and self.is_available():
            from llama_cpp import Llama
            logger.info(f"Loading llama.cpp model: {self.model_path}")
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
        return self._model

    def extract_relations(
        self,
        entity_pairs: List[EntityPair]
    ) -> List[LLMRelation]:
        if not self.is_available():
            return []

        model = self._get_model()
        if model is None:
            return []

        relations = []

        # Process one at a time for smaller local models
        for pair in entity_pairs:
            rel = self._extract_single(model, pair)
            if rel:
                relations.append(rel)

        return relations

    def _extract_single(self, model, pair: EntityPair) -> Optional[LLMRelation]:
        """Extract relation for a single pair."""

        prompt = f"""Determine the relationship between these entities:
Source: "{pair.source_text}" (type: {pair.source_type})
Target: "{pair.target_text}" (type: {pair.target_type})
Context: "{pair.context[:200]}"

Choose ONE relationship type:
CREATES, DEVELOPS, INTRODUCES, USES, AFFILIATED_WITH, TRAINED_ON, EVALUATED_ON, CONTRASTS_WITH, RELATED_TO, NONE

Answer with just the relationship type:"""

        try:
            output = model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["\n", ".", ","]
            )

            response = output["choices"][0]["text"].strip().upper()

            # Extract relation type from response
            for rel_type in RelationType:
                if rel_type.value in response:
                    if rel_type != RelationType.NONE:
                        return LLMRelation(
                            source=pair.source_text,
                            target=pair.target_text,
                            relation_type=rel_type,
                            confidence=0.65,
                            reasoning="Extracted via llama.cpp"
                        )
                    break

        except Exception as e:
            logger.debug(f"llama.cpp extraction failed for pair: {e}")

        return None

class RelationCache:
    """Simple in-memory cache for extracted relations."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, LLMRelation] = {}

    def _make_key(self, pair: EntityPair) -> str:
        """Create cache key from entity pair."""
        content = f"{pair.source_text}|{pair.target_text}|{pair.context[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, pair: EntityPair) -> Optional[LLMRelation]:
        """Get cached relation if available."""
        key = self._make_key(pair)
        return self._cache.get(key)

    def set(self, pair: EntityPair, relation: LLMRelation):
        """Cache a relation."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entries (simple LRU approximation)
            keys_to_remove = list(self._cache.keys())[:self.max_size // 4]
            for key in keys_to_remove:
                del self._cache[key]

        key = self._make_key(pair)
        self._cache[key] = relation

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# ============================================================================
# LLM RELATION EXTRACTOR
# ============================================================================

class LLMRelationExtractor:
    """
    Main class for LLM-based relation extraction.

    Supports multiple providers with automatic fallback.

    Provider priority (default):
    1. Ollama (local, default) - requires Ollama server running
    2. llama.cpp (offline) - requires model file
    3. OpenAI - requires API key
    4. Anthropic - requires API key
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        llama_model_path: Optional[str] = None,
        preferred_provider: str = "ollama",  # Default to Ollama
        enable_cache: bool = True,
        min_confidence: float = 0.5
    ):
        self.min_confidence = min_confidence
        self.cache = RelationCache() if enable_cache else None

        # Initialize providers in priority order
        self.providers: Dict[str, LLMProvider] = {}

        # 1. Ollama (default - local LLM)
        self.providers["ollama"] = OllamaProvider(
            model=ollama_model,
            base_url=ollama_url
        )

        # 2. llama.cpp (offline)
        llama_path = llama_model_path or os.getenv("LLAMA_MODEL_PATH")
        if llama_path:
            self.providers["llama_cpp"] = LlamaCppProvider(model_path=llama_path)

        # 3. OpenAI (cloud)
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            self.providers["openai"] = OpenAIProvider(api_key=openai_api_key)

        # 4. Anthropic (cloud)
        if anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"):
            self.providers["anthropic"] = AnthropicProvider(api_key=anthropic_api_key)

        self.preferred_provider = preferred_provider

    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        return any(p.is_available() for p in self.providers.values())

    def get_available_provider(self) -> Optional[LLMProvider]:
        """Get the best available provider."""
        # Try preferred first
        if self.preferred_provider in self.providers:
            provider = self.providers[self.preferred_provider]
            if provider.is_available():
                return provider

        # Fall back to any available
        for provider in self.providers.values():
            if provider.is_available():
                return provider

        return None

    def extract_relations(
        self,
        entity_pairs: List[EntityPair],
        use_cache: bool = True
    ) -> List[LLMRelation]:
        """
        Extract relations for entity pairs using LLM.

        Args:
            entity_pairs: List of entity pairs with context
            use_cache: Whether to use cached results

        Returns:
            List of extracted relations
        """
        if not entity_pairs:
            return []

        provider = self.get_available_provider()
        if not provider:
            logger.warning("No LLM provider available")
            return []

        relations = []
        pairs_to_process = []

        # Check cache first
        for pair in entity_pairs:
            if use_cache and self.cache:
                cached = self.cache.get(pair)
                if cached:
                    relations.append(cached)
                    continue
            pairs_to_process.append(pair)

        # Process uncached pairs
        if pairs_to_process:
            logger.info(f"Processing {len(pairs_to_process)} entity pairs with LLM")
            new_relations = provider.extract_relations(pairs_to_process)

            # Cache and collect results
            for rel in new_relations:
                if rel.confidence >= self.min_confidence:
                    relations.append(rel)

                    # Find matching pair and cache
                    if self.cache:
                        for pair in pairs_to_process:
                            if pair.source_text == rel.source and pair.target_text == rel.target:
                                self.cache.set(pair, rel)
                                break

        return relations

    def extract_from_text(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        max_pairs: int = 50
    ) -> List[LLMRelation]:
        """
        Extract relations from text and entity list.

        Args:
            text: Source text
            entities: List of entities with 'text', 'type' keys
            max_pairs: Maximum number of pairs to process

        Returns:
            List of extracted relations
        """
        # Build entity pairs from sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        entity_pairs = []

        for sentence in sentences:
            sent_lower = sentence.lower()

            # Find entities in this sentence
            sent_entities = [
                e for e in entities
                if e.get('text', '').lower() in sent_lower
            ]

            # Create pairs
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i+1:]:
                    if e1['text'].lower() != e2['text'].lower():
                        entity_pairs.append(EntityPair(
                            source_text=e1['text'],
                            source_type=e1.get('type', 'UNKNOWN'),
                            target_text=e2['text'],
                            target_type=e2.get('type', 'UNKNOWN'),
                            context=sentence
                        ))

                        if len(entity_pairs) >= max_pairs:
                            break
                if len(entity_pairs) >= max_pairs:
                    break
            if len(entity_pairs) >= max_pairs:
                break

        return self.extract_relations(entity_pairs)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_llm_extractor(
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    ollama_model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    llama_model_path: Optional[str] = None,
    preferred_provider: str = "ollama",
    **kwargs
) -> LLMRelationExtractor:
    """
    Factory function to create an LLM relation extractor.

    Args:
        openai_key: OpenAI API key (or set OPENAI_API_KEY env var)
        anthropic_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        ollama_model: Ollama model name (default: llama3.2)
        ollama_url: Ollama server URL (default: http://localhost:11434)
        llama_model_path: Path to GGUF model for llama.cpp (or set LLAMA_MODEL_PATH)
        preferred_provider: Which provider to try first: "ollama" (default),
                          "llama_cpp", "openai", or "anthropic"
        **kwargs: Additional arguments passed to LLMRelationExtractor

    Returns:
        Configured LLMRelationExtractor instance

    Example:
        # Use Ollama (default, local)
        extractor = create_llm_extractor()

        # Use llama.cpp (fully offline)
        extractor = create_llm_extractor(
            llama_model_path="/models/llama-3.2-3b.gguf",
            preferred_provider="llama_cpp"
        )

        # Use OpenAI (cloud)
        extractor = create_llm_extractor(
            openai_key="sk-...",
            preferred_provider="openai"
        )
    """
    return LLMRelationExtractor(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        llama_model_path=llama_model_path,
        preferred_provider=preferred_provider,
        **kwargs
    )
