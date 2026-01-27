"""
Engagement Tracker for measuring user reading behavior.
Tracks dwell time, scroll depth, interactions, and comprehension signals.
"""

from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class InteractionType(str, Enum):
    """Types of user interactions."""

    SCROLL = "scroll"
    HIGHLIGHT = "highlight"
    CLICK_NODE = "click_node"
    EXPAND_SECTION = "expand_section"
    COLLAPSE_SECTION = "collapse_section"
    ADJUST_COMPLEXITY = "adjust_complexity"
    VIEW_PARAPHRASE = "view_paraphrase"
    ASK_QUESTION = "ask_question"
    CREATE_FLASHCARD = "create_flashcard"
    REVIEW_FLASHCARD = "review_flashcard"


@dataclass
class ReadingSession:
    """Represents a single reading session."""

    session_id: str
    user_id: str
    document_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    total_dwell_time_ms: int = 0
    max_scroll_depth: float = 0.0  # 0-1 representing % of document
    sections_viewed: set = field(default_factory=set)
    sections_expanded: set = field(default_factory=set)
    complexity_adjustments: list = field(default_factory=list)
    interactions: list = field(default_factory=list)
    highlights_created: int = 0
    questions_asked: int = 0
    flashcards_created: int = 0

    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes."""
        end = self.ended_at or datetime.now(timezone.utc)
        delta = end - self.started_at
        return delta.total_seconds() / 60

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None


@dataclass
class SectionEngagement:
    """Engagement metrics for a document section."""

    section_id: str
    total_dwell_time_ms: int = 0
    view_count: int = 0
    expand_count: int = 0
    scroll_passes: int = 0
    highlights: int = 0
    complexity_views: dict = field(default_factory=dict)  # complexity -> view count


@dataclass
class UserEngagementProfile:
    """Aggregated engagement profile for a user."""

    user_id: str
    total_reading_time_minutes: float = 0
    documents_read: int = 0
    avg_session_duration_minutes: float = 0
    avg_scroll_depth: float = 0
    preferred_complexity: float = 50.0
    total_highlights: int = 0
    total_flashcards: int = 0
    comprehension_score: float = 0.5  # Estimated 0-1


class EngagementTracker:
    """
    Tracks and analyzes user engagement with documents.
    """

    def __init__(self):
        self._sessions: dict[str, ReadingSession] = {}
        self._section_engagement: dict[str, dict[str, SectionEngagement]] = defaultdict(
            dict
        )
        self._user_profiles: dict[str, UserEngagementProfile] = {}

    def start_session(
        self,
        session_id: str,
        user_id: str,
        document_id: str,
    ) -> ReadingSession:
        """
        Start a new reading session.

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            document_id: Document being read

        Returns:
            New ReadingSession
        """
        session = ReadingSession(
            session_id=session_id,
            user_id=user_id,
            document_id=document_id,
        )
        self._sessions[session_id] = session
        return session

    def end_session(self, session_id: str) -> Optional[ReadingSession]:
        """
        End a reading session and update user profile.

        Args:
            session_id: Session to end

        Returns:
            Completed session or None if not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        session.ended_at = datetime.now(timezone.utc)

        # Update user profile
        self._update_user_profile(session)

        return session

    def record_dwell_time(
        self,
        session_id: str,
        section_id: str,
        dwell_time_ms: int,
    ) -> None:
        """
        Record dwell time for a section.

        Args:
            session_id: Current session
            section_id: Section being viewed
            dwell_time_ms: Time spent in milliseconds
        """
        session = self._sessions.get(session_id)
        if session is None:
            return

        session.total_dwell_time_ms += dwell_time_ms
        session.sections_viewed.add(section_id)

        # Update section engagement
        doc_id = session.document_id
        if section_id not in self._section_engagement[doc_id]:
            self._section_engagement[doc_id][section_id] = SectionEngagement(
                section_id=section_id
            )

        section_eng = self._section_engagement[doc_id][section_id]
        section_eng.total_dwell_time_ms += dwell_time_ms
        section_eng.view_count += 1

    def record_scroll(
        self,
        session_id: str,
        scroll_depth: float,
        section_id: Optional[str] = None,
    ) -> None:
        """
        Record scroll position/depth.

        Args:
            session_id: Current session
            scroll_depth: Current scroll depth (0-1)
            section_id: Section being scrolled past
        """
        session = self._sessions.get(session_id)
        if session is None:
            return

        session.max_scroll_depth = max(session.max_scroll_depth, scroll_depth)

        if section_id:
            doc_id = session.document_id
            if section_id in self._section_engagement[doc_id]:
                self._section_engagement[doc_id][section_id].scroll_passes += 1

        self._record_interaction(
            session_id,
            InteractionType.SCROLL,
            {
                "depth": scroll_depth,
                "section_id": section_id,
            },
        )

    def record_interaction(
        self,
        session_id: str,
        interaction_type: InteractionType,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Record a user interaction.

        Args:
            session_id: Current session
            interaction_type: Type of interaction
            metadata: Additional interaction data
        """
        self._record_interaction(session_id, interaction_type, metadata)

    def _record_interaction(
        self,
        session_id: str,
        interaction_type: InteractionType,
        metadata: Optional[dict] = None,
    ) -> None:
        """Internal method to record interactions."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        interaction = {
            "type": interaction_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        session.interactions.append(interaction)

        # Update counters
        if interaction_type == InteractionType.HIGHLIGHT:
            session.highlights_created += 1
        elif interaction_type == InteractionType.ASK_QUESTION:
            session.questions_asked += 1
        elif interaction_type == InteractionType.CREATE_FLASHCARD:
            session.flashcards_created += 1
        elif interaction_type == InteractionType.EXPAND_SECTION:
            section_id = metadata.get("section_id") if metadata else None
            if section_id:
                session.sections_expanded.add(section_id)
        elif interaction_type == InteractionType.ADJUST_COMPLEXITY:
            complexity = metadata.get("complexity") if metadata else None
            if complexity is not None:
                session.complexity_adjustments.append(complexity)

    def _update_user_profile(self, session: ReadingSession) -> None:
        """Update user profile based on completed session."""
        user_id = session.user_id

        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = UserEngagementProfile(user_id=user_id)

        profile = self._user_profiles[user_id]

        # Update aggregates
        profile.total_reading_time_minutes += session.duration_minutes
        profile.documents_read += 1
        profile.total_highlights += session.highlights_created
        profile.total_flashcards += session.flashcards_created

        # Update averages (running average)
        n = profile.documents_read
        profile.avg_session_duration_minutes = (
            profile.avg_session_duration_minutes * (n - 1) + session.duration_minutes
        ) / n
        profile.avg_scroll_depth = (
            profile.avg_scroll_depth * (n - 1) + session.max_scroll_depth
        ) / n

        # Update preferred complexity
        if session.complexity_adjustments:
            avg_complexity = sum(session.complexity_adjustments) / len(
                session.complexity_adjustments
            )
            profile.preferred_complexity = (
                profile.preferred_complexity * (n - 1) + avg_complexity
            ) / n

        # Estimate comprehension score based on engagement signals
        profile.comprehension_score = self._estimate_comprehension(session, profile)

    def _estimate_comprehension(
        self,
        session: ReadingSession,
        profile: UserEngagementProfile,
    ) -> float:
        """
        Estimate comprehension score based on engagement signals.

        Higher scores indicate better comprehension based on:
        - Session duration (not too short, not too long)
        - Scroll depth (completed reading)
        - Interactions (highlights, questions, flashcards)
        - Complexity adjustments (finding right level)
        """
        score = 0.5  # Base score

        # Duration score (optimal is 5-30 minutes)
        duration = session.duration_minutes
        if 5 <= duration <= 30:
            score += 0.1
        elif duration < 2:
            score -= 0.1
        elif duration > 60:
            score -= 0.05

        # Scroll depth score
        if session.max_scroll_depth >= 0.9:
            score += 0.15
        elif session.max_scroll_depth >= 0.7:
            score += 0.1
        elif session.max_scroll_depth >= 0.5:
            score += 0.05

        # Interaction score
        interaction_count = (
            session.highlights_created * 2
            + session.questions_asked * 3
            + session.flashcards_created * 2
        )
        score += min(0.2, interaction_count * 0.02)

        # Section expansion score
        if len(session.sections_expanded) > 0:
            score += min(0.1, len(session.sections_expanded) * 0.02)

        return max(0.0, min(1.0, score))

    def get_session(self, session_id: str) -> Optional[ReadingSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_user_profile(self, user_id: str) -> Optional[UserEngagementProfile]:
        """Get user engagement profile."""
        return self._user_profiles.get(user_id)

    def get_section_engagement(
        self,
        document_id: str,
    ) -> dict[str, SectionEngagement]:
        """Get engagement data for all sections of a document."""
        return dict(self._section_engagement.get(document_id, {}))

    def get_session_summary(self, session_id: str) -> Optional[dict]:
        """
        Get a summary of a reading session.

        Args:
            session_id: Session to summarize

        Returns:
            Summary dictionary or None
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        return {
            "session_id": session.session_id,
            "document_id": session.document_id,
            "duration_minutes": round(session.duration_minutes, 2),
            "total_dwell_time_seconds": session.total_dwell_time_ms / 1000,
            "max_scroll_depth_percent": round(session.max_scroll_depth * 100, 1),
            "sections_viewed": len(session.sections_viewed),
            "sections_expanded": len(session.sections_expanded),
            "highlights_created": session.highlights_created,
            "questions_asked": session.questions_asked,
            "flashcards_created": session.flashcards_created,
            "total_interactions": len(session.interactions),
            "is_active": session.is_active,
        }

    def get_reading_recommendations(
        self,
        user_id: str,
        document_id: str,
    ) -> dict:
        """
        Get personalized reading recommendations based on engagement.

        Args:
            user_id: User to get recommendations for
            document_id: Document being read

        Returns:
            Recommendations dictionary
        """
        profile = self._user_profiles.get(user_id)
        section_data = self._section_engagement.get(document_id, {})

        recommendations = {
            "suggested_complexity": 50,
            "focus_sections": [],
            "review_sections": [],
            "tips": [],
        }

        if profile:
            recommendations["suggested_complexity"] = round(
                profile.preferred_complexity
            )

            if profile.avg_scroll_depth < 0.5:
                recommendations["tips"].append(
                    "Try reading sections fully before moving on"
                )

            if profile.total_highlights < profile.documents_read:
                recommendations["tips"].append(
                    "Highlighting key concepts can improve retention"
                )

            if profile.total_flashcards < profile.documents_read:
                recommendations["tips"].append(
                    "Creating flashcards helps with long-term memory"
                )

        # Find sections that need more attention
        for section_id, engagement in section_data.items():
            if engagement.view_count > 0 and engagement.total_dwell_time_ms < 5000:
                recommendations["review_sections"].append(section_id)
            elif engagement.expand_count > 2:
                recommendations["focus_sections"].append(section_id)

        return recommendations


# Singleton instance
_engagement_tracker: Optional[EngagementTracker] = None


def get_engagement_tracker() -> EngagementTracker:
    """Get or create singleton EngagementTracker instance."""
    global _engagement_tracker
    if _engagement_tracker is None:
        _engagement_tracker = EngagementTracker()
    return _engagement_tracker
