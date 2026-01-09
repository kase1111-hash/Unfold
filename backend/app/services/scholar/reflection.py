"""
Reflection Engine for time-based reading snapshots and diffs.
Tracks how understanding evolves over multiple reading sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import hashlib
import difflib


class ReflectionType(str, Enum):
    """Types of reflection entries."""

    INITIAL_READING = "initial_reading"
    RE_READING = "re_reading"
    REVIEW = "review"
    INSIGHT = "insight"
    QUESTION = "question"
    CONNECTION = "connection"


@dataclass
class ReadingSnapshot:
    """Captures user's understanding at a point in time."""

    snapshot_id: str
    user_id: str
    document_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    reflection_type: ReflectionType = ReflectionType.INITIAL_READING

    # Understanding metrics
    complexity_level: int = 50  # 0-100
    comprehension_score: float = 0.5  # 0-1 estimated
    time_spent_minutes: float = 0.0

    # User reflections
    summary: Optional[str] = None  # User's summary of understanding
    key_takeaways: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)  # Unanswered questions
    connections: list[str] = field(
        default_factory=list
    )  # Connections to other knowledge

    # Highlighted sections
    highlights: list[dict] = field(default_factory=list)  # [{text, note, position}]
    annotations: list[dict] = field(default_factory=list)

    # Reading position
    sections_read: list[str] = field(default_factory=list)
    scroll_depth: float = 0.0  # 0-1

    def content_hash(self) -> str:
        """Generate hash of snapshot content for comparison."""
        content = f"{self.summary or ''}{self.key_takeaways}{self.questions}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "document_id": self.document_id,
            "created_at": self.created_at.isoformat(),
            "reflection_type": self.reflection_type.value,
            "complexity_level": self.complexity_level,
            "comprehension_score": self.comprehension_score,
            "time_spent_minutes": self.time_spent_minutes,
            "summary": self.summary,
            "key_takeaways": self.key_takeaways,
            "questions": self.questions,
            "connections": self.connections,
            "highlights_count": len(self.highlights),
            "sections_read": len(self.sections_read),
            "scroll_depth": self.scroll_depth,
        }


@dataclass
class ReflectionDiff:
    """Represents changes between two snapshots."""

    from_snapshot_id: str
    to_snapshot_id: str
    time_between: float  # Hours between snapshots

    # Understanding changes
    complexity_change: int = 0  # Positive = reading at higher complexity
    comprehension_change: float = 0.0
    total_time_added: float = 0.0

    # Content changes
    summary_diff: Optional[str] = None
    new_takeaways: list[str] = field(default_factory=list)
    resolved_questions: list[str] = field(default_factory=list)
    new_questions: list[str] = field(default_factory=list)
    new_connections: list[str] = field(default_factory=list)
    new_highlights: int = 0

    # Progress
    new_sections_read: list[str] = field(default_factory=list)
    scroll_depth_change: float = 0.0

    # Insights
    learning_velocity: float = 0.0  # Comprehension gain per hour
    engagement_trend: str = "stable"  # increasing, stable, decreasing

    def to_dict(self) -> dict:
        return {
            "from_snapshot": self.from_snapshot_id,
            "to_snapshot": self.to_snapshot_id,
            "time_between_hours": round(self.time_between, 1),
            "changes": {
                "complexity_change": self.complexity_change,
                "comprehension_change": round(self.comprehension_change, 2),
                "total_time_added": round(self.total_time_added, 1),
            },
            "content": {
                "new_takeaways": self.new_takeaways,
                "resolved_questions": self.resolved_questions,
                "new_questions": self.new_questions,
                "new_connections": self.new_connections,
                "new_highlights": self.new_highlights,
            },
            "progress": {
                "new_sections": len(self.new_sections_read),
                "scroll_depth_change": round(self.scroll_depth_change, 2),
            },
            "insights": {
                "learning_velocity": round(self.learning_velocity, 3),
                "engagement_trend": self.engagement_trend,
            },
        }


class ReflectionEngine:
    """
    Manages reading snapshots and tracks learning evolution over time.
    """

    def __init__(self):
        self._snapshots: dict[str, list[ReadingSnapshot]] = (
            {}
        )  # user_id+doc_id -> snapshots

    def _get_key(self, user_id: str, document_id: str) -> str:
        """Generate storage key for user-document pair."""
        return f"{user_id}:{document_id}"

    def create_snapshot(
        self,
        user_id: str,
        document_id: str,
        reflection_type: ReflectionType = ReflectionType.INITIAL_READING,
        complexity_level: int = 50,
        time_spent_minutes: float = 0.0,
        summary: Optional[str] = None,
        key_takeaways: Optional[list[str]] = None,
        questions: Optional[list[str]] = None,
        connections: Optional[list[str]] = None,
        highlights: Optional[list[dict]] = None,
        sections_read: Optional[list[str]] = None,
        scroll_depth: float = 0.0,
    ) -> ReadingSnapshot:
        """
        Create a new reading snapshot.

        Args:
            user_id: User identifier
            document_id: Document identifier
            reflection_type: Type of reading session
            complexity_level: Complexity setting used
            time_spent_minutes: Time spent in this session
            summary: User's summary of understanding
            key_takeaways: Key points learned
            questions: Remaining questions
            connections: Connections made
            highlights: Highlighted passages
            sections_read: Sections covered
            scroll_depth: How far scrolled (0-1)

        Returns:
            New ReadingSnapshot
        """
        key = self._get_key(user_id, document_id)

        # Generate snapshot ID
        snapshot_id = (
            f"snap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_id[:6]}"
        )

        # Estimate comprehension based on engagement signals
        comprehension_score = self._estimate_comprehension(
            time_spent_minutes=time_spent_minutes,
            scroll_depth=scroll_depth,
            highlights_count=len(highlights or []),
            takeaways_count=len(key_takeaways or []),
            questions_count=len(questions or []),
        )

        snapshot = ReadingSnapshot(
            snapshot_id=snapshot_id,
            user_id=user_id,
            document_id=document_id,
            reflection_type=reflection_type,
            complexity_level=complexity_level,
            comprehension_score=comprehension_score,
            time_spent_minutes=time_spent_minutes,
            summary=summary,
            key_takeaways=key_takeaways or [],
            questions=questions or [],
            connections=connections or [],
            highlights=highlights or [],
            sections_read=sections_read or [],
            scroll_depth=scroll_depth,
        )

        if key not in self._snapshots:
            self._snapshots[key] = []

        self._snapshots[key].append(snapshot)
        return snapshot

    def _estimate_comprehension(
        self,
        time_spent_minutes: float,
        scroll_depth: float,
        highlights_count: int,
        takeaways_count: int,
        questions_count: int,
    ) -> float:
        """Estimate comprehension score from engagement signals."""
        score = 0.3  # Base score

        # Time factor (optimal 10-30 min)
        if 10 <= time_spent_minutes <= 30:
            score += 0.15
        elif time_spent_minutes > 5:
            score += 0.10

        # Scroll depth factor
        score += scroll_depth * 0.2

        # Engagement factors
        score += min(0.15, highlights_count * 0.03)
        score += min(0.15, takeaways_count * 0.05)
        score += min(0.05, questions_count * 0.01)

        return min(1.0, score)

    def get_snapshots(
        self,
        user_id: str,
        document_id: str,
    ) -> list[ReadingSnapshot]:
        """Get all snapshots for a user-document pair."""
        key = self._get_key(user_id, document_id)
        return self._snapshots.get(key, [])

    def get_latest_snapshot(
        self,
        user_id: str,
        document_id: str,
    ) -> Optional[ReadingSnapshot]:
        """Get the most recent snapshot."""
        snapshots = self.get_snapshots(user_id, document_id)
        if not snapshots:
            return None
        return snapshots[-1]

    def compare_snapshots(
        self,
        snapshot1: ReadingSnapshot,
        snapshot2: ReadingSnapshot,
    ) -> ReflectionDiff:
        """
        Compare two snapshots and generate a diff.

        Args:
            snapshot1: Earlier snapshot
            snapshot2: Later snapshot

        Returns:
            ReflectionDiff with changes between snapshots
        """
        time_between = (
            snapshot2.created_at - snapshot1.created_at
        ).total_seconds() / 3600  # Hours

        # Calculate changes
        complexity_change = snapshot2.complexity_level - snapshot1.complexity_level
        comprehension_change = (
            snapshot2.comprehension_score - snapshot1.comprehension_score
        )
        total_time_added = snapshot2.time_spent_minutes

        # Compare summaries
        summary_diff = None
        if snapshot1.summary and snapshot2.summary:
            diff = difflib.unified_diff(
                snapshot1.summary.split(),
                snapshot2.summary.split(),
                lineterm="",
            )
            summary_diff = " ".join(diff)

        # Compare takeaways
        takeaways1 = set(snapshot1.key_takeaways)
        takeaways2 = set(snapshot2.key_takeaways)
        new_takeaways = list(takeaways2 - takeaways1)

        # Compare questions
        questions1 = set(snapshot1.questions)
        questions2 = set(snapshot2.questions)
        resolved_questions = list(questions1 - questions2)
        new_questions = list(questions2 - questions1)

        # Compare connections
        connections1 = set(snapshot1.connections)
        connections2 = set(snapshot2.connections)
        new_connections = list(connections2 - connections1)

        # Compare highlights
        new_highlights = len(snapshot2.highlights) - len(snapshot1.highlights)

        # Compare sections
        sections1 = set(snapshot1.sections_read)
        sections2 = set(snapshot2.sections_read)
        new_sections = list(sections2 - sections1)

        scroll_depth_change = snapshot2.scroll_depth - snapshot1.scroll_depth

        # Calculate learning velocity
        learning_velocity = 0.0
        if time_between > 0:
            learning_velocity = comprehension_change / time_between

        # Determine engagement trend
        if comprehension_change > 0.1:
            engagement_trend = "increasing"
        elif comprehension_change < -0.1:
            engagement_trend = "decreasing"
        else:
            engagement_trend = "stable"

        return ReflectionDiff(
            from_snapshot_id=snapshot1.snapshot_id,
            to_snapshot_id=snapshot2.snapshot_id,
            time_between=time_between,
            complexity_change=complexity_change,
            comprehension_change=comprehension_change,
            total_time_added=total_time_added,
            summary_diff=summary_diff,
            new_takeaways=new_takeaways,
            resolved_questions=resolved_questions,
            new_questions=new_questions,
            new_connections=new_connections,
            new_highlights=new_highlights,
            new_sections_read=new_sections,
            scroll_depth_change=scroll_depth_change,
            learning_velocity=learning_velocity,
            engagement_trend=engagement_trend,
        )

    def get_learning_journey(
        self,
        user_id: str,
        document_id: str,
    ) -> dict:
        """
        Get the complete learning journey for a document.

        Returns:
            Dictionary with journey summary and all diffs
        """
        snapshots = self.get_snapshots(user_id, document_id)

        if not snapshots:
            return {"error": "No snapshots found"}

        if len(snapshots) == 1:
            return {
                "snapshot_count": 1,
                "first_read": snapshots[0].to_dict(),
                "diffs": [],
                "summary": {
                    "total_time_minutes": snapshots[0].time_spent_minutes,
                    "current_comprehension": snapshots[0].comprehension_score,
                    "questions_remaining": len(snapshots[0].questions),
                    "takeaways_count": len(snapshots[0].key_takeaways),
                },
            }

        # Generate all diffs
        diffs = []
        for i in range(len(snapshots) - 1):
            diff = self.compare_snapshots(snapshots[i], snapshots[i + 1])
            diffs.append(diff.to_dict())

        # Calculate journey summary
        total_time = sum(s.time_spent_minutes for s in snapshots)
        comprehension_growth = (
            snapshots[-1].comprehension_score - snapshots[0].comprehension_score
        )
        time_span = (
            snapshots[-1].created_at - snapshots[0].created_at
        ).total_seconds() / 86400  # Days

        return {
            "snapshot_count": len(snapshots),
            "first_read": snapshots[0].to_dict(),
            "latest_read": snapshots[-1].to_dict(),
            "diffs": diffs,
            "summary": {
                "total_time_minutes": round(total_time, 1),
                "time_span_days": round(time_span, 1),
                "comprehension_growth": round(comprehension_growth, 2),
                "current_comprehension": round(snapshots[-1].comprehension_score, 2),
                "questions_remaining": len(snapshots[-1].questions),
                "total_takeaways": len(snapshots[-1].key_takeaways),
                "total_connections": len(snapshots[-1].connections),
                "total_highlights": len(snapshots[-1].highlights),
            },
        }

    def get_reflection_prompts(
        self,
        user_id: str,
        document_id: str,
    ) -> list[str]:
        """
        Generate reflection prompts based on reading history.

        Returns:
            List of personalized reflection prompts
        """
        prompts = []
        snapshots = self.get_snapshots(user_id, document_id)

        if not snapshots:
            prompts.append("What do you hope to learn from this document?")
            prompts.append("What prior knowledge do you bring to this topic?")
            return prompts

        latest = snapshots[-1]

        # Based on questions
        if latest.questions:
            prompts.append(
                f"You previously asked: '{latest.questions[0]}'. "
                "Have you found an answer?"
            )

        # Based on comprehension
        if latest.comprehension_score < 0.5:
            prompts.append(
                "What parts are still unclear? Try summarizing in your own words."
            )
        else:
            prompts.append("How does this connect to what you already knew?")

        # Based on time between reads
        if len(snapshots) >= 2:
            time_since = (
                datetime.utcnow() - latest.created_at
            ).total_seconds() / 86400  # Days

            if time_since > 7:
                prompts.append(
                    "It's been a while since your last read. "
                    "What do you remember without looking?"
                )

        # Generic prompts
        prompts.append("What's the most important thing you learned?")
        prompts.append("How might you apply this knowledge?")

        return prompts[:4]  # Return top 4 prompts


# Singleton instance
_reflection_engine: Optional[ReflectionEngine] = None


def get_reflection_engine() -> ReflectionEngine:
    """Get or create singleton ReflectionEngine instance."""
    global _reflection_engine
    if _reflection_engine is None:
        _reflection_engine = ReflectionEngine()
    return _reflection_engine
