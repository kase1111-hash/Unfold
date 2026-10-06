"""
Learning API routes for Phase 4 features.
Includes flashcards, spaced repetition, engagement tracking, and exports.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, ValidationError

from app.api.v1.dependencies import CurrentUser, DBSession, get_owned_document
from app.models.learning import Flashcard, FlashcardBase, FlashcardCreate
from app.models.user import User
from app.repositories.document import DocumentRepository
from app.repositories.flashcard import FlashcardRepository
from app.services.learning import (
    get_relevance_scorer,
    get_flashcard_generator,
    get_export_service,
    get_engagement_tracker,
    apply_review,
    compute_study_stats,
    CardReviewState,
    ResponseQuality,
    InteractionType,
    FlashcardData,
    ReadingSession,
    UserEngagementProfile,
)
from app.services.learning.flashcards import to_card_difficulty

router = APIRouter(prefix="/learning", tags=["learning"])


# ============== Pydantic Models ==============


class RelevanceRequest(BaseModel):
    """Request to score text relevance."""

    text: str = Field(..., min_length=10)
    query: str = Field(..., min_length=3)
    tfidf_weight: float = Field(default=0.3, ge=0, le=1)
    semantic_weight: float = Field(default=0.7, ge=0, le=1)


class RankPassagesRequest(BaseModel):
    """Request to rank multiple passages."""

    passages: list[str] = Field(..., min_length=1)
    query: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(default=None, ge=1)


class FocusOrderRequest(BaseModel):
    """Request to compute focus reading order."""

    sections: list[dict] = Field(..., min_length=1)
    learning_goal: str = Field(..., min_length=3)


class GenerateFlashcardsRequest(BaseModel):
    """Request to generate flashcards from one of the caller's documents."""

    document_id: str = Field(..., min_length=1)
    num_cards: int = Field(default=10, ge=1, le=50)
    difficulty: str = Field(default="intermediate")
    context: Optional[str] = None


class ReviewCardRequest(BaseModel):
    """Request to review a flashcard."""

    card_id: str
    quality: int = Field(..., ge=0, le=5)


class StartSessionRequest(BaseModel):
    """Request to start a reading session."""

    document_id: str


class RecordDwellTimeRequest(BaseModel):
    """Request to record dwell time."""

    session_id: str
    section_id: str
    dwell_time_ms: int = Field(..., ge=0)


class RecordScrollRequest(BaseModel):
    """Request to record scroll position."""

    session_id: str
    scroll_depth: float = Field(..., ge=0, le=1)
    section_id: Optional[str] = None


class RecordInteractionRequest(BaseModel):
    """Request to record an interaction."""

    session_id: str
    interaction_type: str
    metadata: Optional[dict] = None


class ExportFlashcardItem(BaseModel):
    """One flashcard to export."""

    card_id: Optional[str] = None
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    hint: Optional[str] = None
    source: Optional[str] = None


class ExportFlashcardsRequest(BaseModel):
    """Request to export flashcards."""

    flashcards: list[ExportFlashcardItem]
    format: str = Field(default="json")
    title: str = Field(default="Unfold Flashcards")


# ============== Relevance Scoring ==============


@router.post("/relevance/score")
async def score_relevance(
    request: RelevanceRequest,
    current_user: CurrentUser,
):
    """
    Score text relevance to a query using TF-IDF and semantic similarity.
    """
    scorer = get_relevance_scorer()
    result = scorer.score_relevance(
        text=request.text,
        query=request.query,
        tfidf_weight=request.tfidf_weight,
        semantic_weight=request.semantic_weight,
    )
    return result


@router.post("/relevance/rank")
async def rank_passages(
    request: RankPassagesRequest,
    current_user: CurrentUser,
):
    """
    Rank multiple passages by relevance to a query.
    """
    scorer = get_relevance_scorer()
    results = scorer.rank_passages(
        passages=request.passages,
        query=request.query,
        top_k=request.top_k,
    )
    return {"ranked_passages": results}


@router.post("/relevance/focus-order")
async def compute_focus_order(
    request: FocusOrderRequest,
    current_user: CurrentUser,
):
    """
    Compute optimal reading order for focus mode based on learning goals.
    """
    scorer = get_relevance_scorer()
    ordered_sections = scorer.compute_focus_order(
        sections=request.sections,
        learning_goal=request.learning_goal,
    )
    return {"sections": ordered_sections}


# ============== Flashcard Helpers ==============


def _card_not_found(card_id: str) -> HTTPException:
    """404 for a card that is missing or belongs to another user."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "NOT_FOUND", "message": f"Flashcard {card_id} not found"},
    )


def _review_state(card: Flashcard) -> CardReviewState:
    """SM2 state of a stored card."""
    return CardReviewState(
        card_id=card.card_id,
        easiness_factor=card.easiness,
        interval=card.interval,
        repetitions=card.repetitions,
        next_review=card.next_review,
        last_review=card.last_reviewed,
        total_reviews=card.total_reviews,
        correct_reviews=card.correct_reviews,
    )


def _card_response(card: Flashcard) -> dict:
    """API representation of a stored card."""
    return {
        "card_id": card.card_id,
        "document_id": card.document_id,
        "question": card.question,
        "answer": card.answer,
        "hint": card.hint,
        "card_type": card.card_type,
        "difficulty": card.difficulty.value,
        "interval_days": card.interval,
        "repetitions": card.repetitions,
        "easiness_factor": round(card.easiness, 2),
        "next_review": card.next_review.isoformat(),
    }


def _generated_card_fields(card: dict) -> Optional[dict]:
    """Fields to store for a generated card, or None to skip it.

    LLM output may lack keys or exceed the limits a hand-made card must
    meet, so it goes through the same validation.
    """
    question, answer, hint = card.get("question"), card.get("answer"), card.get("hint")
    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    if not isinstance(hint, str) or not hint.strip():
        hint = None
    try:
        fields = FlashcardBase(
            question=question.strip(),
            answer=answer.strip(),
            hint=hint,
            difficulty=to_card_difficulty(card.get("difficulty")),
        )
    except ValidationError:
        return None
    return {**fields.model_dump(), "card_type": str(card.get("type") or "recall")[:32]}


# ============== Flashcard Generation ==============


@router.post("/flashcards/generate")
async def generate_flashcards(
    request: GenerateFlashcardsRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Generate flashcards from one of the caller's documents and save them.

    Uses LLM-based question synthesis when an API key is configured and
    rule-based extraction otherwise. New cards are due for review at once.
    """
    await get_owned_document(db, request.document_id, current_user)
    content = await DocumentRepository(db).get_content(
        request.document_id, owner_id=current_user.user_id
    )
    if not content or not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "NO_CONTENT",
                "message": f"Document {request.document_id} has no text to generate flashcards from",
            },
        )

    generator = get_flashcard_generator()
    generated = await generator.generate_flashcards(
        text=content,
        num_cards=request.num_cards,
        difficulty=request.difficulty,
        context=request.context,
    )
    new_cards = [
        fields
        for fields in map(_generated_card_fields, generated[: request.num_cards])
        if fields is not None
    ]
    cards = await FlashcardRepository(db).create_many(
        current_user.user_id, request.document_id, new_cards
    )
    return {
        "document_id": request.document_id,
        "flashcards": [_card_response(card) for card in cards],
        "count": len(cards),
    }


@router.post("/flashcards/cloze")
async def generate_cloze_deletions(
    text: str = Query(..., min_length=50),
    num_deletions: int = Query(default=3, ge=1, le=10),
    current_user: CurrentUser = None,
):
    """
    Generate cloze deletion (fill-in-the-blank) flashcards.
    """
    generator = get_flashcard_generator()
    cards = await generator.generate_cloze_deletions(
        text=text,
        num_deletions=num_deletions,
    )
    return {"cloze_cards": cards, "count": len(cards)}


# ============== Flashcard Management ==============


@router.get("/flashcards")
async def list_flashcards(
    current_user: CurrentUser,
    db: DBSession,
    document_id: Optional[str] = None,
):
    """
    List the caller's flashcards, optionally for one document.
    """
    cards = await FlashcardRepository(db).list_cards(
        current_user.user_id, document_id=document_id
    )
    return {"flashcards": [_card_response(card) for card in cards], "total": len(cards)}


@router.post("/flashcards", status_code=status.HTTP_201_CREATED)
async def create_flashcard(
    request: FlashcardCreate,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Create a flashcard by hand for one of the caller's documents.

    The card is due for review at once.
    """
    await get_owned_document(db, request.document_id, current_user)
    [card] = await FlashcardRepository(db).create_many(
        current_user.user_id,
        request.document_id,
        [{**request.model_dump(exclude={"document_id"}), "card_type": "manual"}],
    )
    return _card_response(card)


@router.delete("/flashcards/{card_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_flashcard(
    card_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    """
    Delete one of the caller's flashcards.
    """
    if not await FlashcardRepository(db).delete(card_id, current_user.user_id):
        raise _card_not_found(card_id)


# ============== Spaced Repetition ==============


@router.post("/flashcards/review")
async def review_flashcard(
    request: ReviewCardRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Record a flashcard review and update its schedule.

    Quality ratings:
    - 0: Complete blackout
    - 1: Incorrect, recognized after
    - 2: Incorrect, seemed easy after
    - 3: Correct with difficulty
    - 4: Correct with hesitation
    - 5: Perfect recall
    """
    try:
        quality = ResponseQuality(request.quality)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid quality rating")

    repo = FlashcardRepository(db)
    # Lock the row so two reviews of one card can't overwrite each other
    card = await repo.get(request.card_id, current_user.user_id, for_update=True)
    if card is None:
        raise _card_not_found(request.card_id)

    state = apply_review(_review_state(card), quality)
    await repo.update_schedule(card.card_id, current_user.user_id, state)

    return {
        "card_id": card.card_id,
        "quality": int(quality),
        "next_review": state.next_review.isoformat(),
        "interval_days": state.interval,
        "easiness_factor": round(state.easiness_factor, 2),
        "repetitions": state.repetitions,
        "retention_rate": round(state.retention_rate * 100, 1),
    }


@router.get("/flashcards/due")
async def get_due_flashcards(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(default=20, ge=1, le=100),
    document_id: Optional[str] = None,
):
    """
    Get the caller's flashcards that are due for review, most overdue first.
    """
    cards, total = await FlashcardRepository(db).list_due(
        current_user.user_id, limit, document_id=document_id
    )
    return {
        "due_cards": [
            {
                **_card_response(card),
                "days_overdue": -_review_state(card).days_until_due,
            }
            for card in cards
        ],
        "total_due": total,
    }


@router.get("/flashcards/upcoming")
async def get_upcoming_flashcards(
    current_user: CurrentUser,
    db: DBSession,
    days: int = Query(default=7, ge=1, le=30),
):
    """
    Get the caller's flashcards scheduled for the next N days.
    """
    cards = await FlashcardRepository(db).list_upcoming(current_user.user_id, days)

    return {
        "upcoming_cards": [
            {
                "card_id": card.card_id,
                "scheduled_date": card.next_review.isoformat(),
                "days_until_due": _review_state(card).days_until_due,
            }
            for card in cards
        ],
        "count": len(cards),
    }


@router.get("/flashcards/stats")
async def get_study_stats(current_user: CurrentUser, db: DBSession):
    """
    Get study statistics for the caller's flashcards.
    """
    cards = await FlashcardRepository(db).list_cards(current_user.user_id)
    return compute_study_stats(_review_state(card) for card in cards)


# ============== Engagement Tracking ==============


def _get_owned_session(session_id: str, user: User) -> ReadingSession:
    """Load one of the user's reading sessions, or raise 404.

    Another user's session is reported exactly like a missing one.
    """
    session = get_engagement_tracker().get_session(session_id)
    if session is None or session.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/engagement/session/start")
async def start_reading_session(
    request: StartSessionRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Start a new reading session on one of the caller's documents.
    """
    await get_owned_document(db, request.document_id, current_user)
    tracker = get_engagement_tracker()
    session_id = f"session_{uuid.uuid4().hex[:12]}"

    session = tracker.start_session(
        session_id=session_id,
        user_id=current_user.user_id,
        document_id=request.document_id,
    )

    return {
        "session_id": session.session_id,
        "started_at": session.started_at.isoformat(),
    }


@router.post("/engagement/session/{session_id}/end")
async def end_reading_session(
    session_id: str,
    current_user: CurrentUser,
):
    """
    End a reading session.
    """
    _get_owned_session(session_id, current_user)
    tracker = get_engagement_tracker()
    tracker.end_session(session_id)

    summary = tracker.get_session_summary(session_id)
    return summary


@router.post("/engagement/dwell-time")
async def record_dwell_time(
    request: RecordDwellTimeRequest,
    current_user: CurrentUser,
):
    """
    Record dwell time for a section.
    """
    _get_owned_session(request.session_id, current_user)
    tracker = get_engagement_tracker()
    tracker.record_dwell_time(
        session_id=request.session_id,
        section_id=request.section_id,
        dwell_time_ms=request.dwell_time_ms,
    )
    return {"status": "recorded"}


@router.post("/engagement/scroll")
async def record_scroll(
    request: RecordScrollRequest,
    current_user: CurrentUser,
):
    """
    Record scroll position.
    """
    _get_owned_session(request.session_id, current_user)
    tracker = get_engagement_tracker()
    tracker.record_scroll(
        session_id=request.session_id,
        scroll_depth=request.scroll_depth,
        section_id=request.section_id,
    )
    return {"status": "recorded"}


@router.post("/engagement/interaction")
async def record_interaction(
    request: RecordInteractionRequest,
    current_user: CurrentUser,
):
    """
    Record a user interaction.
    """
    _get_owned_session(request.session_id, current_user)
    tracker = get_engagement_tracker()

    try:
        interaction_type = InteractionType(request.interaction_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid interaction type")

    tracker.record_interaction(
        session_id=request.session_id,
        interaction_type=interaction_type,
        metadata=request.metadata,
    )
    return {"status": "recorded"}


@router.get("/engagement/session/{session_id}")
async def get_session_summary(
    session_id: str,
    current_user: CurrentUser,
):
    """
    Get summary of a reading session.
    """
    _get_owned_session(session_id, current_user)
    tracker = get_engagement_tracker()
    return tracker.get_session_summary(session_id)


@router.get("/engagement/profile")
async def get_user_profile(current_user: CurrentUser):
    """
    Get the current user's engagement profile.

    Percentages (avg_scroll_depth, comprehension_score) are on a 0-100 scale.
    """
    tracker = get_engagement_tracker()
    # A user with no finished sessions gets the defaults, in the same units
    profile = tracker.get_user_profile(
        current_user.user_id
    ) or UserEngagementProfile(user_id=current_user.user_id)

    return {
        "user_id": profile.user_id,
        "total_reading_time_minutes": round(profile.total_reading_time_minutes, 2),
        "documents_read": profile.documents_read,
        "avg_session_duration_minutes": round(profile.avg_session_duration_minutes, 2),
        "avg_scroll_depth": round(profile.avg_scroll_depth * 100, 1),
        "preferred_complexity": round(profile.preferred_complexity),
        "total_highlights": profile.total_highlights,
        "total_flashcards": profile.total_flashcards,
        "comprehension_score": round(profile.comprehension_score * 100, 1),
    }


@router.get("/engagement/recommendations")
async def get_reading_recommendations(
    document_id: str,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Get personalized reading recommendations based on engagement.
    """
    await get_owned_document(db, document_id, current_user)
    tracker = get_engagement_tracker()
    recommendations = tracker.get_reading_recommendations(
        user_id=current_user.user_id,
        document_id=document_id,
    )
    return recommendations


# ============== Export ==============


def _to_export_data(items: list[ExportFlashcardItem]) -> list[FlashcardData]:
    """Convert validated request items to FlashcardData objects."""
    return [
        FlashcardData(
            card_id=item.card_id or f"fc_{i}",
            question=item.question,
            answer=item.answer,
            tags=item.tags,
            hint=item.hint,
            source=item.source,
        )
        for i, item in enumerate(items)
    ]


@router.post("/export/flashcards")
async def export_flashcards(
    request: ExportFlashcardsRequest,
    current_user: CurrentUser,
):
    """
    Export flashcards to various formats.

    Supported formats: json, anki_csv, anki_txt, anki_json,
    obsidian_sr, obsidian_callout, markdown_table
    """
    export_service = get_export_service()

    # Convert to FlashcardData objects
    flashcards = _to_export_data(request.flashcards)

    format_handlers = {
        "json": export_service.export_to_json,
        "anki_csv": export_service.export_to_anki_csv,
        "anki_txt": export_service.export_to_anki_txt,
        "anki_json": lambda fc: export_service.export_to_anki_json(fc, request.title),
        "obsidian_sr": lambda fc: export_service.export_to_obsidian_markdown(
            fc, request.title
        ),
        "obsidian_callout": lambda fc: export_service.export_to_obsidian_callout(
            fc, request.title
        ),
        "markdown_table": lambda fc: export_service.export_to_markdown_table(
            fc, request.title
        ),
    }

    if request.format not in format_handlers:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {list(format_handlers.keys())}",
        )

    content = format_handlers[request.format](flashcards)

    return {
        "format": request.format,
        "content": content,
        "count": len(flashcards),
    }


@router.post("/export/bundle")
async def export_flashcard_bundle(
    request: ExportFlashcardsRequest,
    current_user: CurrentUser,
):
    """
    Create a ZIP bundle with all export formats.
    """
    export_service = get_export_service()

    flashcards = _to_export_data(request.flashcards)

    bundle =export_service.create_export_bundle(flashcards, title=request.title)

    return Response(
        content=bundle,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{request.title.replace(" ", "_")}_export.zip"',
        },
    )
