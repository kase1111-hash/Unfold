"""External API integrations."""

from app.services.external.semantic_scholar import (
    Author,
    Paper,
    SemanticScholarLinker,
    get_semantic_scholar_linker,
)
from app.services.external.wikipedia import (
    WikipediaLinker,
    WikipediaResult,
    get_wikipedia_linker,
)

__all__ = [
    "WikipediaLinker",
    "WikipediaResult",
    "get_wikipedia_linker",
    "SemanticScholarLinker",
    "Author",
    "Paper",
    "get_semantic_scholar_linker",
]
