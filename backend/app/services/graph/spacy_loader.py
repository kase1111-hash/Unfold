"""
spaCy Model Loader with Caching and Fallback

Provides a unified interface for loading spaCy models with:
- Singleton pattern to avoid reloading
- Graceful fallback when model unavailable
- Model availability detection
- Performance optimizations
"""

from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Singleton storage
_nlp_instance = None
_spacy_available = None
_model_name = "en_core_web_sm"


def is_spacy_available() -> bool:
    """Check if spaCy and the required model are available."""
    global _spacy_available

    if _spacy_available is not None:
        return _spacy_available

    try:
        import spacy

        spacy.load(_model_name)  # Verify model is available
        _spacy_available = True
        logger.info(f"spaCy model '{_model_name}' is available")
    except ImportError:
        _spacy_available = False
        logger.warning("spaCy is not installed")
    except OSError:
        _spacy_available = False
        logger.warning(f"spaCy model '{_model_name}' is not installed")

    return _spacy_available


def get_nlp():
    """
    Get the spaCy NLP pipeline (singleton).

    Returns:
        spaCy Language object or None if unavailable
    """
    global _nlp_instance

    if _nlp_instance is not None:
        return _nlp_instance

    if not is_spacy_available():
        return None

    try:
        import spacy

        _nlp_instance = spacy.load(_model_name)

        # Optimize for our use case (disable unused components)
        # Keep: tagger, parser, ner
        # Could disable: lemmatizer, textcat if not needed

        logger.info(f"Loaded spaCy model: {_model_name}")
        return _nlp_instance

    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        return None


def get_nlp_with_status() -> Tuple[Optional[object], bool]:
    """
    Get the spaCy NLP pipeline with availability status.

    Returns:
        Tuple of (nlp_object_or_None, is_available_bool)
    """
    nlp = get_nlp()
    return nlp, nlp is not None


def parse_text(text: str) -> Optional[object]:
    """
    Parse text with spaCy if available.

    Args:
        text: Text to parse

    Returns:
        spaCy Doc object or None if spaCy unavailable
    """
    nlp = get_nlp()
    if nlp is None:
        return None
    return nlp(text)


def get_model_info() -> dict:
    """
    Get information about the loaded spaCy model.

    Returns:
        Dict with model info or empty dict if unavailable
    """
    nlp = get_nlp()
    if nlp is None:
        return {"available": False, "model": None}

    return {
        "available": True,
        "model": _model_name,
        "pipeline": nlp.pipe_names,
        "vocab_size": len(nlp.vocab),
    }


def reset():
    """Reset the singleton (useful for testing)."""
    global _nlp_instance, _spacy_available
    _nlp_instance = None
    _spacy_available = None


# Convenience functions for common NLP tasks


def extract_entities(text: str) -> list:
    """
    Extract named entities from text using spaCy.

    Returns:
        List of (text, label, start, end) tuples or empty list if unavailable
    """
    doc = parse_text(text)
    if doc is None:
        return []

    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]


def extract_noun_chunks(text: str) -> list:
    """
    Extract noun chunks (noun phrases) from text.

    Returns:
        List of (text, root, start, end) tuples or empty list
    """
    doc = parse_text(text)
    if doc is None:
        return []

    return [
        (chunk.text, chunk.root.text, chunk.start, chunk.end)
        for chunk in doc.noun_chunks
    ]


def get_dependencies(text: str) -> list:
    """
    Get dependency parse for text.

    Returns:
        List of (token, dep, head) tuples or empty list
    """
    doc = parse_text(text)
    if doc is None:
        return []

    return [(token.text, token.dep_, token.head.text) for token in doc]
