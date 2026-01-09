"""
Citation Tree Service for reference chain visualization.
Builds hierarchical citation networks from academic papers.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

import httpx

from app.config import settings


class CitationType(str, Enum):
    """Types of citation relationships."""

    CITES = "cites"  # This paper cites the target
    CITED_BY = "cited_by"  # This paper is cited by target
    RELATED = "related"  # Papers share common citations


@dataclass
class CitationNode:
    """Represents a paper in the citation tree."""

    paper_id: str  # DOI or other identifier
    title: str
    authors: list[str]
    year: Optional[int] = None
    venue: Optional[str] = None  # Journal/Conference
    citation_count: int = 0
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    depth: int = 0  # Depth in citation tree (0 = root)


@dataclass
class CitationEdge:
    """Represents a citation relationship."""

    source_id: str
    target_id: str
    citation_type: CitationType
    context: Optional[str] = None  # Citation context/snippet


@dataclass
class CitationTree:
    """Hierarchical citation network."""

    root: CitationNode
    nodes: dict[str, CitationNode] = field(default_factory=dict)
    edges: list[CitationEdge] = field(default_factory=list)
    max_depth: int = 2
    built_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_edges(self) -> int:
        return len(self.edges)

    def get_references(self, paper_id: str) -> list[CitationNode]:
        """Get papers cited by the given paper."""
        ref_ids = [
            e.target_id
            for e in self.edges
            if e.source_id == paper_id and e.citation_type == CitationType.CITES
        ]
        return [self.nodes[rid] for rid in ref_ids if rid in self.nodes]

    def get_citations(self, paper_id: str) -> list[CitationNode]:
        """Get papers that cite the given paper."""
        cit_ids = [
            e.source_id
            for e in self.edges
            if e.target_id == paper_id and e.citation_type == CitationType.CITES
        ]
        return [self.nodes[cid] for cid in cit_ids if cid in self.nodes]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "root": {
                "paper_id": self.root.paper_id,
                "title": self.root.title,
                "authors": self.root.authors,
                "year": self.root.year,
                "citation_count": self.root.citation_count,
            },
            "nodes": [
                {
                    "paper_id": n.paper_id,
                    "title": n.title,
                    "authors": n.authors,
                    "year": n.year,
                    "venue": n.venue,
                    "citation_count": n.citation_count,
                    "depth": n.depth,
                    "doi": n.doi,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.citation_type.value,
                }
                for e in self.edges
            ],
            "stats": {
                "total_nodes": self.total_nodes,
                "total_edges": self.total_edges,
                "max_depth": self.max_depth,
            },
        }


class CitationService:
    """
    Service for building and querying citation trees.
    Uses Semantic Scholar API for citation data.
    """

    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

    def __init__(self):
        self.api_key = settings.semantic_scholar_api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["x-api-key"] = self.api_key

            self._client = httpx.AsyncClient(
                base_url=self.SEMANTIC_SCHOLAR_API,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    async def get_paper_by_doi(self, doi: str) -> Optional[CitationNode]:
        """
        Fetch paper metadata by DOI.

        Args:
            doi: Digital Object Identifier

        Returns:
            CitationNode or None if not found
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"/paper/DOI:{doi}",
                params={
                    "fields": "paperId,title,authors,year,venue,citationCount,abstract,externalIds,url",
                },
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            return CitationNode(
                paper_id=data.get("paperId", doi),
                title=data.get("title", "Unknown"),
                authors=[a.get("name", "") for a in data.get("authors", [])],
                year=data.get("year"),
                venue=data.get("venue"),
                citation_count=data.get("citationCount", 0),
                abstract=data.get("abstract"),
                doi=doi,
                url=data.get("url"),
            )

        except Exception as e:
            print(f"Error fetching paper {doi}: {e}")
            return None

    async def get_references(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[CitationNode]:
        """
        Get papers referenced by the given paper.

        Args:
            paper_id: Semantic Scholar paper ID or DOI
            limit: Maximum number of references to return

        Returns:
            List of referenced papers
        """
        client = await self._get_client()

        try:
            # Handle DOI format
            if paper_id.startswith("10."):
                paper_id = f"DOI:{paper_id}"

            response = await client.get(
                f"/paper/{paper_id}/references",
                params={
                    "fields": "paperId,title,authors,year,venue,citationCount,externalIds",
                    "limit": limit,
                },
            )
            response.raise_for_status()
            data = response.json()

            references = []
            for ref in data.get("data", []):
                cited_paper = ref.get("citedPaper", {})
                if not cited_paper or not cited_paper.get("paperId"):
                    continue

                ext_ids = cited_paper.get("externalIds", {})

                references.append(
                    CitationNode(
                        paper_id=cited_paper.get("paperId"),
                        title=cited_paper.get("title", "Unknown"),
                        authors=[
                            a.get("name", "") for a in cited_paper.get("authors", [])
                        ],
                        year=cited_paper.get("year"),
                        venue=cited_paper.get("venue"),
                        citation_count=cited_paper.get("citationCount", 0),
                        doi=ext_ids.get("DOI"),
                    )
                )

            return references

        except Exception as e:
            print(f"Error fetching references for {paper_id}: {e}")
            return []

    async def get_citations(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[CitationNode]:
        """
        Get papers that cite the given paper.

        Args:
            paper_id: Semantic Scholar paper ID or DOI
            limit: Maximum number of citations to return

        Returns:
            List of citing papers
        """
        client = await self._get_client()

        try:
            if paper_id.startswith("10."):
                paper_id = f"DOI:{paper_id}"

            response = await client.get(
                f"/paper/{paper_id}/citations",
                params={
                    "fields": "paperId,title,authors,year,venue,citationCount,externalIds",
                    "limit": limit,
                },
            )
            response.raise_for_status()
            data = response.json()

            citations = []
            for cit in data.get("data", []):
                citing_paper = cit.get("citingPaper", {})
                if not citing_paper or not citing_paper.get("paperId"):
                    continue

                ext_ids = citing_paper.get("externalIds", {})

                citations.append(
                    CitationNode(
                        paper_id=citing_paper.get("paperId"),
                        title=citing_paper.get("title", "Unknown"),
                        authors=[
                            a.get("name", "") for a in citing_paper.get("authors", [])
                        ],
                        year=citing_paper.get("year"),
                        venue=citing_paper.get("venue"),
                        citation_count=citing_paper.get("citationCount", 0),
                        doi=ext_ids.get("DOI"),
                    )
                )

            return citations

        except Exception as e:
            print(f"Error fetching citations for {paper_id}: {e}")
            return []

    async def build_citation_tree(
        self,
        root_doi: str,
        max_depth: int = 2,
        refs_per_level: int = 10,
        cites_per_level: int = 10,
    ) -> Optional[CitationTree]:
        """
        Build a citation tree starting from a root paper.

        Args:
            root_doi: DOI of the root paper
            max_depth: Maximum depth to traverse
            refs_per_level: Max references to fetch per paper
            cites_per_level: Max citations to fetch per paper

        Returns:
            CitationTree or None if root paper not found
        """
        # Get root paper
        root = await self.get_paper_by_doi(root_doi)
        if root is None:
            return None

        root.depth = 0
        tree = CitationTree(root=root, max_depth=max_depth)
        tree.nodes[root.paper_id] = root

        # BFS to build tree
        queue = [(root.paper_id, 0)]
        visited = {root.paper_id}

        while queue:
            paper_id, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            # Fetch references and citations concurrently
            refs_task = self.get_references(paper_id, limit=refs_per_level)
            cites_task = self.get_citations(paper_id, limit=cites_per_level)

            refs, cites = await asyncio.gather(refs_task, cites_task)

            # Process references (papers this one cites)
            for ref in refs:
                if ref.paper_id not in visited:
                    ref.depth = depth + 1
                    tree.nodes[ref.paper_id] = ref
                    visited.add(ref.paper_id)
                    queue.append((ref.paper_id, depth + 1))

                tree.edges.append(
                    CitationEdge(
                        source_id=paper_id,
                        target_id=ref.paper_id,
                        citation_type=CitationType.CITES,
                    )
                )

            # Process citations (papers that cite this one)
            for cit in cites:
                if cit.paper_id not in visited:
                    cit.depth = depth + 1
                    tree.nodes[cit.paper_id] = cit
                    visited.add(cit.paper_id)
                    queue.append((cit.paper_id, depth + 1))

                tree.edges.append(
                    CitationEdge(
                        source_id=cit.paper_id,
                        target_id=paper_id,
                        citation_type=CitationType.CITES,
                    )
                )

            # Rate limiting
            await asyncio.sleep(0.5)

        return tree

    async def find_citation_path(
        self,
        source_doi: str,
        target_doi: str,
        max_hops: int = 3,
    ) -> Optional[list[CitationNode]]:
        """
        Find a citation path between two papers.

        Args:
            source_doi: Starting paper DOI
            target_doi: Target paper DOI
            max_hops: Maximum path length

        Returns:
            List of papers forming the path, or None if not found
        """
        source = await self.get_paper_by_doi(source_doi)
        target = await self.get_paper_by_doi(target_doi)

        if not source or not target:
            return None

        # BFS to find path
        queue = [(source.paper_id, [source])]
        visited = {source.paper_id}

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            refs = await self.get_references(current_id, limit=20)

            for ref in refs:
                if ref.paper_id == target.paper_id:
                    return path + [target]

                if ref.paper_id not in visited:
                    visited.add(ref.paper_id)
                    queue.append((ref.paper_id, path + [ref]))

            await asyncio.sleep(0.3)

        return None

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_citation_service: Optional[CitationService] = None


def get_citation_service() -> CitationService:
    """Get or create singleton CitationService instance."""
    global _citation_service
    if _citation_service is None:
        _citation_service = CitationService()
    return _citation_service
