"""Flashcard repository for database operations.

Every query is scoped to one user: a card that belongs to someone else is
treated exactly like a missing one.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.learning import FlashcardORM
from app.models.learning import Flashcard, FlashcardDifficulty

if TYPE_CHECKING:
    from app.services.learning.sm2 import CardReviewState


class FlashcardRepository:
    """Repository for flashcard database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create_many(
        self, user_id: str, document_id: str, cards: list[dict]
    ) -> list[Flashcard]:
        """Store new cards for a user. New cards are due immediately.

        Args:
            user_id: Owner of the cards
            document_id: Source document
            cards: Dicts with "question" and "answer", and optionally
                "hint", "card_type", "difficulty" and "source_node_id"

        Returns:
            Created flashcards, in the order given
        """
        now = datetime.now(timezone.utc)
        cards_orm = [
            FlashcardORM(
                card_id=str(uuid4()),
                user_id=user_id,
                document_id=document_id,
                source_node_id=card.get("source_node_id"),
                question=card["question"],
                answer=card["answer"],
                hint=card.get("hint"),
                card_type=card.get("card_type") or "recall",
                difficulty=card.get("difficulty") or FlashcardDifficulty.MEDIUM,
                easiness=2.5,
                interval=0,
                repetitions=0,
                total_reviews=0,
                correct_reviews=0,
                next_review=now,
                created_at=now,
                updated_at=now,
            )
            for card in cards
        ]
        self.session.add_all(cards_orm)
        await self.session.flush()
        return [self._to_model(card_orm) for card_orm in cards_orm]

    async def get(
        self, card_id: str, user_id: str, for_update: bool = False
    ) -> Flashcard | None:
        """Get one of a user's cards.

        Args:
            card_id: Flashcard identifier
            user_id: Owner of the card
            for_update: Lock the row until the transaction ends

        Returns:
            Flashcard if found and owned by the user, None otherwise
        """
        query = select(FlashcardORM).where(
            FlashcardORM.card_id == card_id, FlashcardORM.user_id == user_id
        )
        if for_update:
            query = query.with_for_update()
        result = await self.session.execute(query)
        card_orm = result.scalar_one_or_none()
        return self._to_model(card_orm) if card_orm is not None else None

    async def list_cards(
        self, user_id: str, document_id: str | None = None
    ) -> list[Flashcard]:
        """List a user's cards, oldest first.

        Args:
            user_id: Owner of the cards
            document_id: Only cards from this document (optional)

        Returns:
            List of flashcards
        """
        query = select(FlashcardORM).where(FlashcardORM.user_id == user_id)
        if document_id is not None:
            query = query.where(FlashcardORM.document_id == document_id)
        query = query.order_by(FlashcardORM.created_at, FlashcardORM.card_id)
        result = await self.session.execute(query)
        return [self._to_model(card_orm) for card_orm in result.scalars().all()]

    async def list_due(
        self, user_id: str, limit: int, document_id: str | None = None
    ) -> tuple[list[Flashcard], int]:
        """List a user's cards that are due now, most overdue first.

        Args:
            user_id: Owner of the cards
            limit: Maximum number of cards to return
            document_id: Only cards from this document (optional)

        Returns:
            Tuple of (due cards, up to ``limit``; total number of due cards)
        """
        conditions = [
            FlashcardORM.user_id == user_id,
            FlashcardORM.next_review <= datetime.now(timezone.utc),
        ]
        if document_id is not None:
            conditions.append(FlashcardORM.document_id == document_id)

        count_result = await self.session.execute(
            select(func.count(FlashcardORM.card_id)).where(*conditions)
        )
        total = count_result.scalar() or 0

        result = await self.session.execute(
            select(FlashcardORM)
            .where(*conditions)
            .order_by(FlashcardORM.next_review, FlashcardORM.created_at)
            .limit(limit)
        )
        cards = [self._to_model(card_orm) for card_orm in result.scalars().all()]
        return cards, total

    async def list_upcoming(self, user_id: str, days: int) -> list[Flashcard]:
        """List a user's cards that become due within the next ``days`` days.

        Args:
            user_id: Owner of the cards
            days: Number of days to look ahead

        Returns:
            Cards that are not due yet, soonest first
        """
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(FlashcardORM)
            .where(
                FlashcardORM.user_id == user_id,
                FlashcardORM.next_review > now,
                FlashcardORM.next_review <= now + timedelta(days=days),
            )
            .order_by(FlashcardORM.next_review)
        )
        return [self._to_model(card_orm) for card_orm in result.scalars().all()]

    async def update_schedule(
        self, card_id: str, user_id: str, state: "CardReviewState"
    ) -> bool:
        """Save a card's SM2 state after a review.

        Args:
            card_id: Flashcard identifier
            user_id: Owner of the card
            state: Review state returned by sm2.apply_review

        Returns:
            True if updated, False if the user has no such card
        """
        result = await self.session.execute(
            update(FlashcardORM)
            .where(FlashcardORM.card_id == card_id, FlashcardORM.user_id == user_id)
            .values(
                easiness=state.easiness_factor,
                interval=state.interval,
                repetitions=state.repetitions,
                total_reviews=state.total_reviews,
                correct_reviews=state.correct_reviews,
                next_review=state.next_review,
                last_reviewed=state.last_review,
                updated_at=datetime.now(timezone.utc),
            )
        )
        return result.rowcount > 0

    async def delete(self, card_id: str, user_id: str) -> bool:
        """Delete one of a user's cards.

        Args:
            card_id: Flashcard identifier
            user_id: Owner of the card

        Returns:
            True if deleted, False if the user has no such card
        """
        result = await self.session.execute(
            delete(FlashcardORM).where(
                FlashcardORM.card_id == card_id, FlashcardORM.user_id == user_id
            )
        )
        return result.rowcount > 0

    def _to_model(self, card_orm: FlashcardORM) -> Flashcard:
        """Convert ORM model to Pydantic model.

        Args:
            card_orm: SQLAlchemy ORM model

        Returns:
            Pydantic Flashcard model
        """
        return Flashcard(
            card_id=card_orm.card_id,
            user_id=card_orm.user_id,
            document_id=card_orm.document_id,
            source_node_id=card_orm.source_node_id,
            question=card_orm.question,
            answer=card_orm.answer,
            hint=card_orm.hint,
            card_type=card_orm.card_type,
            difficulty=card_orm.difficulty,
            easiness=card_orm.easiness,
            interval=card_orm.interval,
            repetitions=card_orm.repetitions,
            total_reviews=card_orm.total_reviews,
            correct_reviews=card_orm.correct_reviews,
            next_review=card_orm.next_review,
            last_reviewed=card_orm.last_reviewed,
            created_at=card_orm.created_at,
            updated_at=card_orm.updated_at,
        )
