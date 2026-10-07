"""
Flashcard Generator using LLM-based question synthesis.
Generates Q&A pairs for spaced repetition learning.
"""

import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from enum import Enum

import httpx

from app.config import settings
from app.models.learning import FlashcardDifficulty

logger = logging.getLogger(__name__)

# Whole documents are passed in; keep the prompt well inside the model's context
MAX_PROMPT_TEXT_CHARS = 48_000

# Longer "sentences" are usually extraction noise (tables, missing punctuation)
MAX_FALLBACK_SENTENCE_CHARS = 500

# Capitalized words that open sentences but make useless answers ("She", "The")
NON_ANSWER_WORDS = frozenset(
    {
        "A", "An", "The", "This", "That", "These", "Those",
        "I", "He", "She", "It", "We", "They", "You",
        "My", "His", "Her", "Its", "Our", "Their", "Your",
        "In", "On", "At", "By", "For", "From", "With", "As", "Of", "To",
        "After", "Before", "During", "When", "While", "Since", "Although",
        "However", "Then", "There", "Here", "But", "And", "Or", "If",
    }
)

# Generator difficulty labels -> stored flashcard difficulty
DIFFICULTY_LEVELS = {
    "beginner": FlashcardDifficulty.EASY,
    "intermediate": FlashcardDifficulty.MEDIUM,
    "advanced": FlashcardDifficulty.HARD,
}


def to_card_difficulty(level: object) -> FlashcardDifficulty:
    """Map a generator difficulty (beginner/intermediate/advanced) to easy/medium/hard."""
    if isinstance(level, str):
        level = level.strip().lower()
        if level in DIFFICULTY_LEVELS:
            return DIFFICULTY_LEVELS[level]
        try:
            return FlashcardDifficulty(level)
        except ValueError:
            pass
    return FlashcardDifficulty.MEDIUM


class QuestionType(str, Enum):
    """Types of flashcard questions."""

    DEFINITION = "definition"
    CONCEPT = "concept"
    COMPARISON = "comparison"
    APPLICATION = "application"
    RECALL = "recall"


class FlashcardGenerator:
    """
    Generates flashcards from text using LLM-based question synthesis.

    This class manages an HTTP client for API calls. Use as a context manager
    or call close() explicitly to ensure proper cleanup.
    """

    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model = "gpt-4o-mini"  # Cost-effective for flashcard generation
        self._client: Optional[httpx.AsyncClient] = None
        self._owns_client = True  # Track if we created the client

    async def __aenter__(self) -> "FlashcardGenerator":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup resources."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
            self._owns_client = True
        return self._client

    async def generate_flashcards(
        self,
        text: str,
        num_cards: int = 5,
        question_types: Optional[list[QuestionType]] = None,
        difficulty: str = "intermediate",
        context: Optional[str] = None,
    ) -> list[dict]:
        """
        Generate flashcards from text using LLM.

        Args:
            text: Source text to generate flashcards from
            num_cards: Number of flashcards to generate
            question_types: Types of questions to generate
            difficulty: Difficulty level (beginner, intermediate, advanced)
            context: Additional context about the subject

        Returns:
            List of flashcard dictionaries with question, answer, type, etc.
        """
        if not self.api_key:
            return self._generate_fallback_flashcards(text, num_cards)

        if question_types is None:
            question_types = list(QuestionType)

        types_str = ", ".join([qt.value for qt in question_types])

        prompt = f"""Generate {num_cards} high-quality flashcards from the following text for spaced repetition learning.

Text:
{text[:MAX_PROMPT_TEXT_CHARS]}

{f'Context: {context}' if context else ''}

Requirements:
1. Generate exactly {num_cards} flashcards
2. Question types to include: {types_str}
3. Difficulty level: {difficulty}
4. Each flashcard should test understanding, not just memorization
5. Answers should be concise but complete
6. Include key terms and concepts

Return the flashcards as a JSON array with this structure:
[
  {{
    "question": "The question text",
    "answer": "The answer text",
    "type": "definition|concept|comparison|application|recall",
    "difficulty": "beginner|intermediate|advanced",
    "key_concepts": ["concept1", "concept2"],
    "hint": "Optional hint for the question"
  }}
]

Return ONLY the JSON array, no additional text."""

        try:
            client = await self._ensure_client()
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert educator who creates effective flashcards for learning. Generate flashcards that promote deep understanding and long-term retention.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                },
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            flashcards = self._parse_flashcards_json(content)
            if not flashcards:
                logger.warning("LLM returned no parseable flashcards")
                return self._generate_fallback_flashcards(text, num_cards)

            # Add metadata
            for card in flashcards:
                card["card_id"] = str(uuid.uuid4())
                card["source_hash"] = hash(text) % 1000000

            return flashcards

        except httpx.HTTPStatusError as e:
            logger.warning(f"LLM API returned error status: {e.response.status_code}")
            return self._generate_fallback_flashcards(text, num_cards)
        except httpx.RequestError as e:
            logger.warning(f"LLM API request failed: {e}")
            return self._generate_fallback_flashcards(text, num_cards)
        except Exception as e:
            logger.warning(f"LLM flashcard generation failed: {e}")
            return self._generate_fallback_flashcards(text, num_cards)

    def _parse_flashcards_json(self, content: str) -> list[dict]:
        """Parse flashcards JSON from LLM response."""
        # Try to extract JSON array from response
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = re.sub(r"```(?:json)?\n?", "", content)
            content = content.strip()

        try:
            flashcards = json.loads(content)
            if isinstance(flashcards, list):
                return flashcards
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in content
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return []

    def _generate_fallback_flashcards(
        self,
        text: str,
        num_cards: int,
    ) -> list[dict]:
        """
        Generate basic flashcards without LLM using rule-based extraction.
        """
        flashcards = []
        # PDF text keeps hard line wraps; collapse them so terms aren't split
        text = re.sub(r"\s+", " ", text)
        sentences = re.split(r"[.!?]+", text)
        sentences = [
            s.strip()
            for s in sentences
            if 30 < len(s.strip()) <= MAX_FALLBACK_SENTENCE_CHARS
        ]

        for sentence in sentences[:num_cards]:
            # Extract key terms (capitalized words, quoted terms)
            matches = re.findall(
                r'"([^"]+)"|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', sentence
            )
            key_terms = []
            for quoted, capitalized in matches:
                if quoted:
                    key_terms.append(quoted)
                    continue
                # Drop leading pronouns/articles: "She" -> skip, "The Moon" -> "Moon"
                words = capitalized.split()
                while words and words[0] in NON_ANSWER_WORDS:
                    words.pop(0)
                if words:
                    key_terms.append(" ".join(words))

            # Create a fill-in-the-blank style question
            if key_terms:
                term = key_terms[0]
                question = sentence.replace(term, "_____")
                answer = term
                q_type = QuestionType.RECALL
            else:
                # Create a True/False or understanding question
                question = f"What does this statement describe? '{sentence[:100]}...'"
                answer = sentence
                q_type = QuestionType.CONCEPT

            flashcards.append(
                {
                    "card_id": str(uuid.uuid4()),
                    "question": question,
                    "answer": answer,
                    "type": q_type.value,
                    "difficulty": "intermediate",
                    "key_concepts": key_terms[:3],
                    "hint": None,
                    "source_hash": hash(text) % 1000000,
                }
            )

        return flashcards

    async def generate_cloze_deletions(
        self,
        text: str,
        num_deletions: int = 3,
    ) -> list[dict]:
        """
        Generate cloze deletion (fill-in-the-blank) flashcards.

        Args:
            text: Source text
            num_deletions: Number of cloze deletions to create

        Returns:
            List of cloze deletion flashcards
        """
        # Find important terms to blank out
        # Look for: defined terms, technical vocabulary, key concepts
        patterns = [
            r'"([^"]+)"',  # Quoted terms
            r"called\s+(\w+)",  # "called X"
            r"known\s+as\s+(\w+)",  # "known as X"
            r"(\w+)\s+is\s+defined",  # "X is defined"
            r"the\s+(\w+)\s+(?:process|method|technique|approach)",  # Technical terms
        ]

        terms = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(
                matches
                if isinstance(matches[0] if matches else "", str)
                else [m[0] for m in matches]
            )

        # Deduplicate and limit
        terms = list(dict.fromkeys(terms))[:num_deletions]

        cloze_cards = []
        for i, term in enumerate(terms):
            # Find the sentence containing this term
            for sentence in re.split(r"[.!?]+", text):
                if term.lower() in sentence.lower():
                    cloze_text = re.sub(
                        re.escape(term),
                        "{{c1::" + term + "}}",
                        sentence,
                        flags=re.IGNORECASE,
                        count=1,
                    )
                    cloze_cards.append(
                        {
                            "card_id": f"cloze_{i}",
                            "cloze_text": cloze_text.strip(),
                            "answer": term,
                            "type": "cloze",
                            "difficulty": "intermediate",
                        }
                    )
                    break

        return cloze_cards

    async def enhance_flashcard(
        self,
        flashcard: dict,
        source_text: str,
    ) -> dict:
        """
        Enhance a flashcard with additional context and hints.

        Args:
            flashcard: Existing flashcard to enhance
            source_text: Original source text

        Returns:
            Enhanced flashcard with additional fields
        """
        if not self.api_key:
            return flashcard

        prompt = f"""Enhance this flashcard with additional learning aids.

Question: {flashcard['question']}
Answer: {flashcard['answer']}
Source text excerpt: {source_text[:500]}

Provide:
1. A helpful hint (without giving away the answer)
2. A mnemonic or memory aid if applicable
3. Related concepts to explore
4. Common mistakes to avoid

Return as JSON:
{{
  "hint": "...",
  "mnemonic": "...",
  "related_concepts": ["..."],
  "common_mistakes": ["..."]
}}"""

        try:
            client = await self._ensure_client()
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500,
                },
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse enhancement JSON
            content = re.sub(r"```(?:json)?\n?", "", content.strip())
            enhancements = json.loads(content)

            return {**flashcard, **enhancements}

        except Exception as e:
            logger.warning(f"Flashcard enhancement failed: {e}")
            return flashcard

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
            logger.debug("FlashcardGenerator HTTP client closed")


# Module-level client for use with application lifespan
_flashcard_generator: Optional[FlashcardGenerator] = None


async def init_flashcard_generator() -> FlashcardGenerator:
    """Initialize the global flashcard generator. Call during app startup."""
    global _flashcard_generator
    if _flashcard_generator is None:
        _flashcard_generator = FlashcardGenerator()
        await _flashcard_generator._ensure_client()
    return _flashcard_generator


async def close_flashcard_generator() -> None:
    """Close the global flashcard generator. Call during app shutdown."""
    global _flashcard_generator
    if _flashcard_generator is not None:
        await _flashcard_generator.close()
        _flashcard_generator = None


def get_flashcard_generator() -> FlashcardGenerator:
    """Get the global FlashcardGenerator instance.

    For proper resource management, prefer using the generator as a context manager
    for one-off operations, or ensure init_flashcard_generator() is called at startup
    and close_flashcard_generator() at shutdown.
    """
    global _flashcard_generator
    if _flashcard_generator is None:
        _flashcard_generator = FlashcardGenerator()
    return _flashcard_generator


@asynccontextmanager
async def flashcard_generator_context() -> AsyncIterator[FlashcardGenerator]:
    """Context manager for using FlashcardGenerator with automatic cleanup.

    Use this for one-off flashcard generation:

        async with flashcard_generator_context() as generator:
            cards = await generator.generate_flashcards(text)
    """
    generator = FlashcardGenerator()
    try:
        await generator._ensure_client()
        yield generator
    finally:
        await generator.close()
