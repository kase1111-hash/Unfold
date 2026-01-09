"""
Learning API routes for Phase 4 features.
Includes flashcards, spaced repetition, engagement tracking, and exports.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from app.api.v1.dependencies import CurrentUser
from app.services.learning import (
    get_relevance_scorer,
    get_flashcard_generator,
    get_sm2_scheduler,
    get_export_service,
    get_engagement_tracker,
    ResponseQuality,
    InteractionType,
    FlashcardData,
)

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
    """Request to generate flashcards."""

    text: str = Field(..., min_length=50)
    num_cards: int = Field(default=5, ge=1, le=20)
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


class ExportFlashcardsRequest(BaseModel):
    """Request to export flashcards."""

    flashcards: list[dict]
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


# ============== Flashcard Generation ==============


@router.post("/flashcards/generate")
async def generate_flashcards(
    request: GenerateFlashcardsRequest,
    current_user: CurrentUser,
):
    """
    Generate flashcards from text using LLM-based question synthesis.
    """
    generator = get_flashcard_generator()
    flashcards = await generator.generate_flashcards(
        text=request.text,
        num_cards=request.num_cards,
        difficulty=request.difficulty,
        context=request.context,
    )
    return {"flashcards": flashcards, "count": len(flashcards)}


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


# ============== Spaced Repetition ==============


@router.post("/flashcards/add")
async def add_flashcard_to_scheduler(
    card_id: str,
    current_user: CurrentUser,
):
    """
    Add a flashcard to the spaced repetition scheduler.
    """
    scheduler = get_sm2_scheduler()
    state = scheduler.add_card(card_id)
    return {
        "card_id": state.card_id,
        "next_review": state.next_review.isoformat(),
        "interval_days": state.interval,
    }


@router.post("/flashcards/review")
async def review_flashcard(
    request: ReviewCardRequest,
    current_user: CurrentUser,
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
    scheduler = get_sm2_scheduler()

    try:
        quality = ResponseQuality(request.quality)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid quality rating")

    state = scheduler.review_card(request.card_id, quality)

    return {
        "card_id": state.card_id,
        "next_review": state.next_review.isoformat(),
        "interval_days": state.interval,
        "easiness_factor": round(state.easiness_factor, 2),
        "repetitions": state.repetitions,
        "retention_rate": round(state.retention_rate * 100, 1),
    }


@router.get("/flashcards/due")
async def get_due_flashcards(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: CurrentUser = None,
):
    """
    Get flashcards that are due for review.
    """
    scheduler = get_sm2_scheduler()
    due_cards = scheduler.get_due_cards(limit=limit)

    return {
        "due_cards": [
            {
                "card_id": card.card_id,
                "days_overdue": -card.days_until_due,
                "repetitions": card.repetitions,
                "easiness_factor": round(card.easiness_factor, 2),
            }
            for card in due_cards
        ],
        "total_due": len(due_cards),
    }


@router.get("/flashcards/upcoming")
async def get_upcoming_flashcards(
    days: int = Query(default=7, ge=1, le=30),
    current_user: CurrentUser = None,
):
    """
    Get flashcards scheduled for the next N days.
    """
    scheduler = get_sm2_scheduler()
    upcoming = scheduler.get_upcoming_cards(days=days)

    return {
        "upcoming_cards": [
            {
                "card_id": card.card_id,
                "scheduled_date": card.next_review.isoformat(),
                "days_until_due": card.days_until_due,
            }
            for card in upcoming
        ],
        "count": len(upcoming),
    }


@router.get("/flashcards/stats")
async def get_study_stats(current_user: CurrentUser):
    """
    Get overall study statistics.
    """
    scheduler = get_sm2_scheduler()
    return scheduler.get_study_stats()


# ============== Engagement Tracking ==============


@router.post("/engagement/session/start")
async def start_reading_session(
    request: StartSessionRequest,
    current_user: CurrentUser,
):
    """
    Start a new reading session.
    """
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
    tracker = get_engagement_tracker()
    session = tracker.end_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

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
    tracker = get_engagement_tracker()
    summary = tracker.get_session_summary(session_id)

    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return summary


@router.get("/engagement/profile")
async def get_user_profile(current_user: CurrentUser):
    """
    Get the current user's engagement profile.
    """
    tracker = get_engagement_tracker()
    profile = tracker.get_user_profile(current_user.user_id)

    if profile is None:
        return {
            "user_id": current_user.user_id,
            "total_reading_time_minutes": 0,
            "documents_read": 0,
            "avg_session_duration_minutes": 0,
            "avg_scroll_depth": 0,
            "preferred_complexity": 50,
            "total_highlights": 0,
            "total_flashcards": 0,
            "comprehension_score": 0.5,
        }

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
):
    """
    Get personalized reading recommendations based on engagement.
    """
    tracker = get_engagement_tracker()
    recommendations = tracker.get_reading_recommendations(
        user_id=current_user.user_id,
        document_id=document_id,
    )
    return recommendations


# ============== Export ==============


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
    flashcards = [
        FlashcardData(
            card_id=fc.get("card_id", f"fc_{i}"),
            question=fc["question"],
            answer=fc["answer"],
            tags=fc.get("tags", []),
            hint=fc.get("hint"),
            source=fc.get("source"),
        )
        for i, fc in enumerate(request.flashcards)
    ]

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

    flashcards = [
        FlashcardData(
            card_id=fc.get("card_id", f"fc_{i}"),
            question=fc["question"],
            answer=fc["answer"],
            tags=fc.get("tags", []),
            hint=fc.get("hint"),
            source=fc.get("source"),
        )
        for i, fc in enumerate(request.flashcards)
    ]

    bundle = export_service.create_export_bundle(flashcards, title=request.title)

    return Response(
        content=bundle,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{request.title.replace(" ", "_")}_export.zip"',
        },
    )
