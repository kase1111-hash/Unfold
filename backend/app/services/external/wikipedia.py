"""Wikipedia API integration for concept linking."""

from dataclasses import dataclass

import httpx

from app.config import get_settings

settings = get_settings()

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


@dataclass
class WikipediaResult:
    """Result from Wikipedia search."""

    title: str
    page_id: int
    url: str
    snippet: str | None = None
    extract: str | None = None
    categories: list[str] | None = None


class WikipediaLinker:
    """Service for linking concepts to Wikipedia articles."""

    def __init__(self, timeout: float = 10.0):
        """Initialize Wikipedia linker.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[WikipediaResult]:
        """Search Wikipedia for matching articles.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of Wikipedia results
        """
        client = await self._get_client()

        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet",
        }

        try:
            response = await client.get(WIKIPEDIA_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                results.append(
                    WikipediaResult(
                        title=item["title"],
                        page_id=item["pageid"],
                        url=f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                        snippet=item.get("snippet"),
                    )
                )

            return results

        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []

    async def get_page_info(
        self,
        title: str,
        include_extract: bool = True,
        include_categories: bool = False,
    ) -> WikipediaResult | None:
        """Get detailed information about a Wikipedia page.

        Args:
            title: Page title
            include_extract: Whether to include page extract/summary
            include_categories: Whether to include categories

        Returns:
            WikipediaResult with page info, or None if not found
        """
        client = await self._get_client()

        props = ["info"]
        if include_extract:
            props.append("extracts")
        if include_categories:
            props.append("categories")

        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "|".join(props),
            "exintro": "1",  # Only intro section
            "explaintext": "1",  # Plain text, no HTML
            "exsentences": "3",  # First 3 sentences
        }

        try:
            response = await client.get(WIKIPEDIA_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if page_id == "-1":
                    return None

                categories = None
                if include_categories:
                    categories = [
                        cat["title"].replace("Category:", "")
                        for cat in page_data.get("categories", [])
                    ]

                return WikipediaResult(
                    title=page_data["title"],
                    page_id=int(page_id),
                    url=f"https://en.wikipedia.org/wiki/{page_data['title'].replace(' ', '_')}",
                    extract=page_data.get("extract"),
                    categories=categories,
                )

            return None

        except Exception as e:
            print(f"Wikipedia page info error: {e}")
            return None

    async def find_best_match(
        self,
        concept: str,
    ) -> WikipediaResult | None:
        """Find the best matching Wikipedia article for a concept.

        Args:
            concept: Concept to match

        Returns:
            Best matching WikipediaResult, or None if no good match
        """
        results = await self.search(concept, limit=3)

        if not results:
            return None

        # Return first result (usually most relevant)
        best = results[0]

        # Get full page info
        page_info = await self.get_page_info(best.title)

        return page_info or best

    async def link_entities(
        self,
        entities: list[str],
    ) -> dict[str, WikipediaResult | None]:
        """Link multiple entities to Wikipedia articles.

        Args:
            entities: List of entity names

        Returns:
            Dict mapping entity name to Wikipedia result
        """
        results = {}

        for entity in entities:
            results[entity] = await self.find_best_match(entity)

        return results


# Global linker instance
_wikipedia_linker: WikipediaLinker | None = None


def get_wikipedia_linker() -> WikipediaLinker:
    """Get Wikipedia linker instance."""
    global _wikipedia_linker
    if _wikipedia_linker is None:
        _wikipedia_linker = WikipediaLinker()
    return _wikipedia_linker
