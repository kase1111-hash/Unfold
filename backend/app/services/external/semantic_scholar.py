"""Semantic Scholar API integration for academic paper linking."""

from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import get_settings

settings = get_settings()

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"


@dataclass
class Author:
    """Semantic Scholar author."""

    author_id: str
    name: str
    url: str | None = None


@dataclass
class Paper:
    """Semantic Scholar paper result."""

    paper_id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    authors: list[Author] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)


class SemanticScholarLinker:
    """Service for linking to Semantic Scholar papers and authors."""

    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        """Initialize Semantic Scholar linker.

        Args:
            api_key: Optional API key for higher rate limits
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key

            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        year_range: tuple[int, int] | None = None,
        fields_of_study: list[str] | None = None,
    ) -> list[Paper]:
        """Search for papers by query.

        Args:
            query: Search query
            limit: Maximum results
            year_range: Optional (min_year, max_year) filter
            fields_of_study: Optional field filter

        Returns:
            List of Paper results
        """
        client = await self._get_client()

        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,abstract,year,citationCount,referenceCount,url,externalIds,venue,authors,fieldsOfStudy",
        }

        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        try:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/search",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("data", []):
                papers.append(self._parse_paper(item))

            return papers

        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []

    async def get_paper(self, paper_id: str) -> Paper | None:
        """Get paper by Semantic Scholar ID, DOI, or arXiv ID.

        Args:
            paper_id: Paper identifier (S2 ID, DOI:xxx, ARXIV:xxx, etc.)

        Returns:
            Paper if found, None otherwise
        """
        client = await self._get_client()

        fields = "paperId,title,abstract,year,citationCount,referenceCount,url,externalIds,venue,authors,fieldsOfStudy"

        try:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}",
                params={"fields": fields},
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_paper(data)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            print(f"Semantic Scholar get paper error: {e}")
            return None

    async def get_paper_by_doi(self, doi: str) -> Paper | None:
        """Get paper by DOI.

        Args:
            doi: Digital Object Identifier

        Returns:
            Paper if found, None otherwise
        """
        return await self.get_paper(f"DOI:{doi}")

    async def get_paper_by_arxiv(self, arxiv_id: str) -> Paper | None:
        """Get paper by arXiv ID.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            Paper if found, None otherwise
        """
        return await self.get_paper(f"ARXIV:{arxiv_id}")

    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> list[Paper]:
        """Get papers that cite a given paper.

        Args:
            paper_id: Paper identifier
            limit: Maximum results

        Returns:
            List of citing papers
        """
        client = await self._get_client()

        fields = "paperId,title,abstract,year,citationCount,authors"

        try:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}/citations",
                params={"fields": fields, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("data", []):
                citing_paper = item.get("citingPaper", {})
                if citing_paper:
                    papers.append(self._parse_paper(citing_paper))

            return papers

        except Exception as e:
            print(f"Semantic Scholar citations error: {e}")
            return []

    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> list[Paper]:
        """Get papers referenced by a given paper.

        Args:
            paper_id: Paper identifier
            limit: Maximum results

        Returns:
            List of referenced papers
        """
        client = await self._get_client()

        fields = "paperId,title,abstract,year,citationCount,authors"

        try:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}/references",
                params={"fields": fields, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("data", []):
                cited_paper = item.get("citedPaper", {})
                if cited_paper:
                    papers.append(self._parse_paper(cited_paper))

            return papers

        except Exception as e:
            print(f"Semantic Scholar references error: {e}")
            return []

    async def search_authors(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Author]:
        """Search for authors by name.

        Args:
            query: Author name query
            limit: Maximum results

        Returns:
            List of Author results
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/author/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "authorId,name,url",
                },
            )
            response.raise_for_status()
            data = response.json()

            authors = []
            for item in data.get("data", []):
                authors.append(
                    Author(
                        author_id=item.get("authorId", ""),
                        name=item.get("name", ""),
                        url=item.get("url"),
                    )
                )

            return authors

        except Exception as e:
            print(f"Semantic Scholar author search error: {e}")
            return []

    async def get_author(self, author_id: str) -> Author | None:
        """Get author by Semantic Scholar ID.

        Args:
            author_id: Author identifier

        Returns:
            Author if found, None otherwise
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/author/{author_id}",
                params={"fields": "authorId,name,url,paperCount,citationCount,hIndex"},
            )
            response.raise_for_status()
            data = response.json()

            return Author(
                author_id=data.get("authorId", ""),
                name=data.get("name", ""),
                url=data.get("url"),
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            print(f"Semantic Scholar get author error: {e}")
            return None

    def _parse_paper(self, data: dict[str, Any]) -> Paper:
        """Parse paper data from API response.

        Args:
            data: API response data

        Returns:
            Paper object
        """
        external_ids = data.get("externalIds", {}) or {}

        authors = []
        for author_data in data.get("authors", []) or []:
            authors.append(
                Author(
                    author_id=author_data.get("authorId", ""),
                    name=author_data.get("name", ""),
                )
            )

        return Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            year=data.get("year"),
            citation_count=data.get("citationCount"),
            reference_count=data.get("referenceCount"),
            url=data.get("url"),
            doi=external_ids.get("DOI"),
            arxiv_id=external_ids.get("ArXiv"),
            venue=data.get("venue"),
            authors=authors,
            fields_of_study=data.get("fieldsOfStudy") or [],
        )


# Global linker instance
_semantic_scholar_linker: SemanticScholarLinker | None = None


def get_semantic_scholar_linker() -> SemanticScholarLinker:
    """Get Semantic Scholar linker instance."""
    global _semantic_scholar_linker
    if _semantic_scholar_linker is None:
        _semantic_scholar_linker = SemanticScholarLinker()
    return _semantic_scholar_linker
