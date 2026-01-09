"""Learning and engagement database models."""

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.postgres import Base
from app.models.learning import FlashcardDifficulty


class FlashcardORM(Base):
    """SQLAlchemy model for flashcards table."""

    __tablename__ = "flashcards"

    card_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_node_id: Mapped[str] = mapped_column(String(100), nullable=False)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[FlashcardDifficulty] = mapped_column(
        Enum(FlashcardDifficulty, name="flashcard_difficulty"),
        default=FlashcardDifficulty.MEDIUM,
        nullable=False,
    )

    # SM2 algorithm fields
    easiness: Mapped[float] = mapped_column(Float, default=2.5, nullable=False)
    interval: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    repetitions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_review: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_reviewed: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped["UserORM"] = relationship("UserORM", back_populates="flashcards")

    def __repr__(self) -> str:
        return f"<Flashcard(card_id={self.card_id}, user_id={self.user_id})>"


class EngagementMetricsORM(Base):
    """SQLAlchemy model for user engagement metrics."""

    __tablename__ = "engagement_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    dwell_time_seconds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    scroll_depth: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    scroll_velocity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    annotations_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    highlights_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chat_queries_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<EngagementMetrics(user_id={self.user_id}, document_id={self.document_id})>"


class ReflectionSnapshotORM(Base):
    """SQLAlchemy model for user reflection snapshots."""

    __tablename__ = "reflection_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str | None] = mapped_column(
        String(100),
        ForeignKey("documents.doc_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    nodes_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    nodes_deleted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    connections_made: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    insights: Mapped[list[str]] = mapped_column(
        ARRAY(String), default=list, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ReflectionSnapshot(snapshot_id={self.snapshot_id}, user_id={self.user_id})>"


# Import for relationship type hints
from app.db.models.user import UserORM  # noqa: E402
