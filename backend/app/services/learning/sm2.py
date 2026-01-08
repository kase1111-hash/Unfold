"""
SM2 Spaced Repetition Algorithm Implementation.
Based on the SuperMemo 2 algorithm by Piotr Wozniak.
"""

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from enum import IntEnum


class ResponseQuality(IntEnum):
    """
    Response quality ratings for SM2 algorithm.
    0-2: Incorrect responses (card needs to be relearned)
    3-5: Correct responses with varying recall difficulty
    """
    BLACKOUT = 0  # Complete blackout, no recall
    INCORRECT = 1  # Incorrect, but recognized after seeing answer
    DIFFICULT_INCORRECT = 2  # Incorrect, but seemed easy after seeing answer
    DIFFICULT_CORRECT = 3  # Correct, but with serious difficulty
    CORRECT_HESITATION = 4  # Correct, with some hesitation
    PERFECT = 5  # Perfect response, instant recall


@dataclass
class CardReviewState:
    """State of a flashcard in the spaced repetition system."""
    card_id: str
    easiness_factor: float = 2.5  # EF starts at 2.5
    interval: int = 0  # Days until next review
    repetitions: int = 0  # Number of successful reviews
    next_review: datetime = field(default_factory=datetime.utcnow)
    last_review: Optional[datetime] = None
    total_reviews: int = 0
    correct_reviews: int = 0

    @property
    def retention_rate(self) -> float:
        """Calculate the user's retention rate for this card."""
        if self.total_reviews == 0:
            return 0.0
        return self.correct_reviews / self.total_reviews

    @property
    def is_due(self) -> bool:
        """Check if the card is due for review."""
        return datetime.utcnow() >= self.next_review

    @property
    def days_until_due(self) -> int:
        """Days until card is due (negative if overdue)."""
        delta = self.next_review - datetime.utcnow()
        return delta.days


class SM2Scheduler:
    """
    SuperMemo 2 (SM2) spaced repetition scheduler.

    The algorithm adjusts review intervals based on how well
    the user recalls each flashcard, optimizing for long-term retention.
    """

    # Minimum easiness factor (prevents intervals from becoming too short)
    MIN_EF = 1.3

    # Initial intervals for new cards (in days)
    INITIAL_INTERVALS = [1, 6]  # First review after 1 day, second after 6 days

    def __init__(self):
        self._cards: dict[str, CardReviewState] = {}

    def add_card(self, card_id: str) -> CardReviewState:
        """
        Add a new card to the scheduler.

        Args:
            card_id: Unique identifier for the card

        Returns:
            New CardReviewState for the card
        """
        state = CardReviewState(card_id=card_id)
        self._cards[card_id] = state
        return state

    def get_card(self, card_id: str) -> Optional[CardReviewState]:
        """Get the current state of a card."""
        return self._cards.get(card_id)

    def review_card(
        self,
        card_id: str,
        quality: ResponseQuality,
    ) -> CardReviewState:
        """
        Process a review for a card and update its schedule.

        This implements the core SM2 algorithm:
        1. Update easiness factor based on response quality
        2. Calculate new interval
        3. Schedule next review

        Args:
            card_id: The card being reviewed
            quality: Quality of the response (0-5)

        Returns:
            Updated CardReviewState
        """
        state = self._cards.get(card_id)
        if state is None:
            state = self.add_card(card_id)

        state.total_reviews += 1
        state.last_review = datetime.utcnow()

        # Quality < 3 means incorrect response - reset repetitions
        if quality < ResponseQuality.DIFFICULT_CORRECT:
            state.repetitions = 0
            state.interval = 1  # Review again tomorrow
        else:
            state.correct_reviews += 1

            # Update easiness factor using SM2 formula
            # EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            q = int(quality)
            ef_delta = 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
            state.easiness_factor = max(self.MIN_EF, state.easiness_factor + ef_delta)

            # Calculate new interval
            if state.repetitions == 0:
                state.interval = self.INITIAL_INTERVALS[0]
            elif state.repetitions == 1:
                state.interval = self.INITIAL_INTERVALS[1]
            else:
                # I(n) = I(n-1) * EF
                state.interval = round(state.interval * state.easiness_factor)

            state.repetitions += 1

        # Schedule next review
        state.next_review = datetime.utcnow() + timedelta(days=state.interval)

        return state

    def get_due_cards(
        self,
        limit: Optional[int] = None,
    ) -> list[CardReviewState]:
        """
        Get all cards that are due for review.

        Args:
            limit: Maximum number of cards to return

        Returns:
            List of due cards, sorted by overdue time
        """
        now = datetime.utcnow()
        due_cards = [
            card for card in self._cards.values()
            if card.next_review <= now
        ]

        # Sort by how overdue they are (most overdue first)
        due_cards.sort(key=lambda c: c.next_review)

        if limit is not None:
            due_cards = due_cards[:limit]

        return due_cards

    def get_upcoming_cards(
        self,
        days: int = 7,
    ) -> list[CardReviewState]:
        """
        Get cards scheduled for the next N days.

        Args:
            days: Number of days to look ahead

        Returns:
            List of upcoming cards with their scheduled dates
        """
        now = datetime.utcnow()
        cutoff = now + timedelta(days=days)

        upcoming = [
            card for card in self._cards.values()
            if now < card.next_review <= cutoff
        ]

        upcoming.sort(key=lambda c: c.next_review)
        return upcoming

    def get_study_stats(self) -> dict:
        """
        Get overall study statistics.

        Returns:
            Dictionary with study metrics
        """
        if not self._cards:
            return {
                "total_cards": 0,
                "due_now": 0,
                "due_today": 0,
                "average_ef": 0,
                "average_retention": 0,
                "mature_cards": 0,
                "learning_cards": 0,
            }

        now = datetime.utcnow()
        today_end = now.replace(hour=23, minute=59, second=59)

        due_now = sum(1 for c in self._cards.values() if c.next_review <= now)
        due_today = sum(1 for c in self._cards.values() if c.next_review <= today_end)

        # Cards with interval > 21 days are considered "mature"
        mature = sum(1 for c in self._cards.values() if c.interval > 21)
        learning = len(self._cards) - mature

        avg_ef = sum(c.easiness_factor for c in self._cards.values()) / len(self._cards)

        cards_with_reviews = [c for c in self._cards.values() if c.total_reviews > 0]
        avg_retention = (
            sum(c.retention_rate for c in cards_with_reviews) / len(cards_with_reviews)
            if cards_with_reviews else 0
        )

        return {
            "total_cards": len(self._cards),
            "due_now": due_now,
            "due_today": due_today,
            "average_ef": round(avg_ef, 2),
            "average_retention": round(avg_retention * 100, 1),
            "mature_cards": mature,
            "learning_cards": learning,
        }

    def get_optimal_review_count(
        self,
        available_time_minutes: int,
        avg_seconds_per_card: int = 30,
    ) -> int:
        """
        Calculate optimal number of cards to review given available time.

        Args:
            available_time_minutes: Time available for study
            avg_seconds_per_card: Average review time per card

        Returns:
            Recommended number of cards to review
        """
        max_cards = (available_time_minutes * 60) // avg_seconds_per_card
        due_cards = len(self.get_due_cards())

        # Don't recommend more than what's due
        return min(max_cards, due_cards)

    def predict_retention(
        self,
        card_id: str,
        days_from_now: int,
    ) -> float:
        """
        Predict retention probability for a card at a future date.
        Uses exponential forgetting curve.

        Args:
            card_id: The card to predict for
            days_from_now: Days in the future

        Returns:
            Estimated retention probability (0-1)
        """
        state = self._cards.get(card_id)
        if state is None:
            return 0.0

        # Simplified forgetting curve: R = e^(-t/S)
        # where S is stability (related to interval and EF)
        stability = state.interval * state.easiness_factor
        if stability <= 0:
            return 0.0

        import math
        retention = math.exp(-days_from_now / stability)
        return round(retention, 3)

    def export_state(self) -> dict:
        """Export scheduler state for persistence."""
        return {
            card_id: {
                "easiness_factor": state.easiness_factor,
                "interval": state.interval,
                "repetitions": state.repetitions,
                "next_review": state.next_review.isoformat(),
                "last_review": state.last_review.isoformat() if state.last_review else None,
                "total_reviews": state.total_reviews,
                "correct_reviews": state.correct_reviews,
            }
            for card_id, state in self._cards.items()
        }

    def import_state(self, data: dict) -> None:
        """Import scheduler state from persistence."""
        for card_id, card_data in data.items():
            state = CardReviewState(
                card_id=card_id,
                easiness_factor=card_data["easiness_factor"],
                interval=card_data["interval"],
                repetitions=card_data["repetitions"],
                next_review=datetime.fromisoformat(card_data["next_review"]),
                last_review=(
                    datetime.fromisoformat(card_data["last_review"])
                    if card_data["last_review"] else None
                ),
                total_reviews=card_data["total_reviews"],
                correct_reviews=card_data["correct_reviews"],
            )
            self._cards[card_id] = state


# Singleton instance
_sm2_scheduler: Optional[SM2Scheduler] = None


def get_sm2_scheduler() -> SM2Scheduler:
    """Get or create singleton SM2Scheduler instance."""
    global _sm2_scheduler
    if _sm2_scheduler is None:
        _sm2_scheduler = SM2Scheduler()
    return _sm2_scheduler
