"""
Credibility Scoring Service using CrossRef and Altmetrics.
Evaluates paper and source credibility for academic content.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

import httpx

from app.config import settings


class CredibilityLevel(str, Enum):
    """Credibility assessment levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class CredibilityScore:
    """Comprehensive credibility assessment."""

    overall_score: float  # 0-100
    level: CredibilityLevel

    # Component scores
    citation_score: float = 0.0  # Based on citation count and impact
    venue_score: float = 0.0  # Journal/conference reputation
    author_score: float = 0.0  # Author h-index and reputation
    recency_score: float = 0.0  # How recent is the work
    altmetric_score: float = 0.0  # Social/news attention

    # Metadata
    citation_count: int = 0
    journal_impact_factor: Optional[float] = None
    altmetric_attention: Optional[int] = None
    is_peer_reviewed: bool = False
    is_retracted: bool = False

    # Warnings and notes
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "level": self.level.value,
            "components": {
                "citation_score": round(self.citation_score, 1),
                "venue_score": round(self.venue_score, 1),
                "author_score": round(self.author_score, 1),
                "recency_score": round(self.recency_score, 1),
                "altmetric_score": round(self.altmetric_score, 1),
            },
            "metadata": {
                "citation_count": self.citation_count,
                "journal_impact_factor": self.journal_impact_factor,
                "altmetric_attention": self.altmetric_attention,
                "is_peer_reviewed": self.is_peer_reviewed,
                "is_retracted": self.is_retracted,
            },
            "warnings": self.warnings,
            "notes": self.notes,
        }


class CredibilityScorer:
    """
    Evaluates credibility of academic papers using multiple signals.
    """

    CROSSREF_API = "https://api.crossref.org/works"
    ALTMETRIC_API = "https://api.altmetric.com/v1"

    def __init__(self):
        self.crossref_email = settings.crossref_email
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.crossref_email:
                headers["User-Agent"] = f"Unfold/1.0 (mailto:{self.crossref_email})"

            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
            )
        return self._client

    async def get_crossref_metadata(self, doi: str) -> Optional[dict]:
        """
        Fetch paper metadata from CrossRef.

        Args:
            doi: Digital Object Identifier

        Returns:
            CrossRef metadata dict or None
        """
        client = await self._get_client()

        try:
            response = await client.get(f"{self.CROSSREF_API}/{doi}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json().get("message", {})
        except Exception as e:
            print(f"CrossRef API error for {doi}: {e}")
            return None

    async def get_altmetric_data(self, doi: str) -> Optional[dict]:
        """
        Fetch Altmetric attention data.

        Args:
            doi: Digital Object Identifier

        Returns:
            Altmetric data dict or None
        """
        client = await self._get_client()

        try:
            response = await client.get(f"{self.ALTMETRIC_API}/doi/{doi}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Altmetric API error for {doi}: {e}")
            return None

    def _calculate_citation_score(
        self,
        citation_count: int,
        year: Optional[int],
    ) -> float:
        """Calculate citation score (0-100)."""
        if citation_count == 0:
            return 20.0

        # Normalize by age (citations per year)
        if year:
            age = max(1, datetime.utcnow().year - year)
            citations_per_year = citation_count / age
        else:
            citations_per_year = citation_count / 5  # Assume 5 years

        # Score based on citations per year
        # 0-1: 30, 1-5: 50, 5-20: 70, 20-50: 85, 50+: 95
        if citations_per_year < 1:
            return 30.0
        elif citations_per_year < 5:
            return 50.0
        elif citations_per_year < 20:
            return 70.0
        elif citations_per_year < 50:
            return 85.0
        else:
            return 95.0

    def _calculate_venue_score(
        self,
        venue: Optional[str],
        impact_factor: Optional[float],
        is_peer_reviewed: bool,
    ) -> float:
        """Calculate venue/journal score (0-100)."""
        score = 30.0  # Base score

        if is_peer_reviewed:
            score += 20.0

        if impact_factor:
            if impact_factor > 10:
                score += 40.0
            elif impact_factor > 5:
                score += 30.0
            elif impact_factor > 2:
                score += 20.0
            elif impact_factor > 1:
                score += 10.0

        # Known high-impact venues
        high_impact_venues = [
            "nature",
            "science",
            "cell",
            "lancet",
            "nejm",
            "jama",
            "pnas",
            "acm",
            "ieee",
            "neurips",
            "icml",
        ]
        if venue:
            venue_lower = venue.lower()
            if any(v in venue_lower for v in high_impact_venues):
                score = min(95.0, score + 20.0)

        return min(100.0, score)

    def _calculate_recency_score(self, year: Optional[int]) -> float:
        """Calculate recency score (0-100)."""
        if not year:
            return 50.0

        age = datetime.utcnow().year - year

        if age <= 1:
            return 95.0
        elif age <= 3:
            return 85.0
        elif age <= 5:
            return 75.0
        elif age <= 10:
            return 60.0
        elif age <= 20:
            return 45.0
        else:
            return 30.0

    def _calculate_altmetric_score(self, attention: Optional[int]) -> float:
        """Calculate Altmetric attention score (0-100)."""
        if not attention:
            return 0.0

        if attention >= 1000:
            return 95.0
        elif attention >= 500:
            return 85.0
        elif attention >= 100:
            return 70.0
        elif attention >= 50:
            return 60.0
        elif attention >= 20:
            return 50.0
        elif attention >= 5:
            return 35.0
        else:
            return 20.0

    def _determine_level(self, score: float) -> CredibilityLevel:
        """Determine credibility level from overall score."""
        if score >= 70:
            return CredibilityLevel.HIGH
        elif score >= 45:
            return CredibilityLevel.MEDIUM
        elif score >= 20:
            return CredibilityLevel.LOW
        else:
            return CredibilityLevel.UNKNOWN

    async def score_paper(self, doi: str) -> CredibilityScore:
        """
        Calculate comprehensive credibility score for a paper.

        Args:
            doi: Digital Object Identifier

        Returns:
            CredibilityScore with all components
        """
        warnings = []
        notes = []

        # Fetch data from APIs
        crossref_data = await self.get_crossref_metadata(doi)
        altmetric_data = await self.get_altmetric_data(doi)

        # Extract CrossRef metadata
        citation_count = 0
        year = None
        venue = None
        is_peer_reviewed = False
        is_retracted = False
        impact_factor = None

        if crossref_data:
            citation_count = crossref_data.get("is-referenced-by-count", 0)

            # Get publication year
            published = crossref_data.get("published", {})
            date_parts = published.get("date-parts", [[None]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]

            # Get venue
            container = crossref_data.get("container-title", [])
            if container:
                venue = container[0]

            # Check peer review status
            if crossref_data.get("type") in ["journal-article", "proceedings-article"]:
                is_peer_reviewed = True

            # Check retraction status
            if crossref_data.get("update-to"):
                for update in crossref_data.get("update-to", []):
                    if update.get("type") == "retraction":
                        is_retracted = True
                        warnings.append("This paper has been retracted")

            # Check for preprint
            if crossref_data.get("subtype") == "preprint":
                notes.append("This is a preprint (not peer-reviewed)")
                is_peer_reviewed = False

        else:
            warnings.append("Could not verify paper in CrossRef database")

        # Extract Altmetric data
        altmetric_attention = None
        if altmetric_data:
            altmetric_attention = altmetric_data.get("score")

            # Check for news mentions
            if altmetric_data.get("cited_by_msm_count", 0) > 0:
                notes.append(
                    f"Mentioned in {altmetric_data['cited_by_msm_count']} news outlets"
                )

        # Calculate component scores
        citation_score = self._calculate_citation_score(citation_count, year)
        venue_score = self._calculate_venue_score(
            venue, impact_factor, is_peer_reviewed
        )
        recency_score = self._calculate_recency_score(year)
        altmetric_score = self._calculate_altmetric_score(altmetric_attention)
        author_score = 50.0  # Placeholder - would need author lookup

        # Calculate overall score (weighted average)
        weights = {
            "citation": 0.30,
            "venue": 0.25,
            "author": 0.15,
            "recency": 0.15,
            "altmetric": 0.15,
        }

        overall_score = (
            weights["citation"] * citation_score
            + weights["venue"] * venue_score
            + weights["author"] * author_score
            + weights["recency"] * recency_score
            + weights["altmetric"] * altmetric_score
        )

        # Apply penalties
        if is_retracted:
            overall_score = 0.0

        level = self._determine_level(overall_score)

        return CredibilityScore(
            overall_score=overall_score,
            level=level,
            citation_score=citation_score,
            venue_score=venue_score,
            author_score=author_score,
            recency_score=recency_score,
            altmetric_score=altmetric_score,
            citation_count=citation_count,
            journal_impact_factor=impact_factor,
            altmetric_attention=altmetric_attention,
            is_peer_reviewed=is_peer_reviewed,
            is_retracted=is_retracted,
            warnings=warnings,
            notes=notes,
        )

    async def compare_papers(
        self,
        dois: list[str],
    ) -> list[tuple[str, CredibilityScore]]:
        """
        Compare credibility of multiple papers.

        Args:
            dois: List of DOIs to compare

        Returns:
            List of (doi, score) tuples sorted by credibility
        """
        results = []
        for doi in dois:
            score = await self.score_paper(doi)
            results.append((doi, score))

        # Sort by overall score descending
        results.sort(key=lambda x: x[1].overall_score, reverse=True)
        return results

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_credibility_scorer: Optional[CredibilityScorer] = None


def get_credibility_scorer() -> CredibilityScorer:
    """Get or create singleton CredibilityScorer instance."""
    global _credibility_scorer
    if _credibility_scorer is None:
        _credibility_scorer = CredibilityScorer()
    return _credibility_scorer
