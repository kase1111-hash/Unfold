"""Adaptive learning data models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.models.base import TimestampMixin


class FlashcardDifficulty(str, Enum):
    """Flashcard difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReviewQuality(int, Enum):
    """SM2 review quality ratings."""

    COMPLETE_BLACKOUT = 0
    INCORRECT_REMEMBERED = 1
    INCORRECT_EASY_RECALL = 2
    CORRECT_DIFFICULT = 3
    CORRECT_HESITATION = 4
    CORRECT_PERFECT = 5


class FlashcardBase(BaseModel):
    """Base flashcard model."""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="Question text"
    )
    answer: str = Field(..., min_length=1, max_length=2000, description="Answer text")
    source_node_id: str = Field(..., description="Source knowledge graph node")
    difficulty: FlashcardDifficulty = Field(
        FlashcardDifficulty.MEDIUM, description="Difficulty level"
    )


class FlashcardCreate(FlashcardBase):
    """Model for creating a flashcard."""

    user_id: str = Field(..., description="User ID")
    document_id: str = Field(..., description="Source document ID")


class Flashcard(FlashcardBase, TimestampMixin):
    """Full flashcard model with SM2 scheduling data."""

    card_id: str = Field(..., description="Unique flashcard ID")
    user_id: str = Field(..., description="User ID")
    document_id: str = Field(..., description="Source document ID")

    # SM2 algorithm fields
    easiness: float = Field(2.5, ge=1.3, description="Easiness factor")
    interval: int = Field(1, ge=1, description="Current interval in days")
    repetitions: int = Field(0, ge=0, description="Number of successful reviews")
    next_review: datetime = Field(..., description="Next scheduled review date")
    last_reviewed: datetime | None = Field(None, description="Last review timestamp")

    class Config:
        """Pydantic config."""

        from_attributes = True


class FlashcardReview(BaseModel):
    """Model for submitting a flashcard review."""

    card_id: str = Field(..., description="Flashcard ID")
    quality: ReviewQuality = Field(..., description="Review quality rating (0-5)")
    time_spent_seconds: int = Field(0, ge=0, description="Time spent on review")


class EngagementMetrics(BaseModel):
    """User engagement metrics for a document."""

    user_id: str
    document_id: str
    dwell_time_seconds: int = Field(0, ge=0, description="Total time spent reading")
    scroll_depth: float = Field(0.0, ge=0.0, le=1.0, description="Maximum scroll depth")
    scroll_velocity: float = Field(0.0, ge=0.0, description="Average scroll velocity")
    annotations_count: int = Field(0, ge=0, description="Number of annotations")
    highlights_count: int = Field(0, ge=0, description="Number of highlights")
    chat_queries_count: int = Field(0, ge=0, description="Number of chat queries")
    last_accessed: datetime = Field(..., description="Last access timestamp")


class LearningProgress(BaseModel):
    """User learning progress for a document."""

    user_id: str
    document_id: str
    total_flashcards: int = Field(0, ge=0)
    cards_due: int = Field(0, ge=0)
    cards_learned: int = Field(0, ge=0)
    average_easiness: float = Field(2.5, ge=1.3)
    retention_rate: float = Field(0.0, ge=0.0, le=1.0)
    streak_days: int = Field(0, ge=0)
    last_study_date: datetime | None = None


class ReflectionSnapshot(TimestampMixin):
    """User reflection snapshot for tracking comprehension evolution."""

    snapshot_id: str = Field(..., description="Unique snapshot ID")
    user_id: str = Field(..., description="User ID")
    document_id: str | None = Field(None, description="Document ID (optional)")
    nodes_added: int = Field(0, ge=0, description="Graph nodes added")
    nodes_deleted: int = Field(0, ge=0, description="Graph nodes removed")
    connections_made: int = Field(0, ge=0, description="New connections created")
    notes: str | None = Field(
        None, max_length=2000, description="User reflection notes"
    )
    insights: list[str] = Field(
        default_factory=list, description="AI-generated insights"
    )


class ExportFormat(str, Enum):
    """Export format options."""

    ANKI = "anki"
    MARKDOWN = "markdown"
    OBSIDIAN = "obsidian"
    CSV = "csv"


class ExportRequest(BaseModel):
    """Request for exporting learning data."""

    user_id: str
    document_id: str | None = None
    format: ExportFormat = Field(ExportFormat.MARKDOWN)
    include_flashcards: bool = True
    include_annotations: bool = True
    include_progress: bool = True
