"""
Scholar API routes for Phase 5 features.
Includes citations, credibility scoring, Zotero export, reflection, and annotations.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from app.api.v1.dependencies import CurrentUser
from app.services.scholar import (
    get_citation_service,
    get_credibility_scorer,
    get_zotero_exporter,
    get_reflection_engine,
    get_annotation_service,
    AnnotationType,
    AnnotationVisibility,
    ReflectionType,
    ZoteroItem,
)

router = APIRouter(prefix="/scholar", tags=["scholar"])


# ============== Pydantic Models ==============

class CitationTreeRequest(BaseModel):
    """Request to build a citation tree."""
    doi: str = Field(..., description="DOI of the root paper")
    max_depth: int = Field(default=2, ge=1, le=3)
    refs_per_level: int = Field(default=10, ge=1, le=50)
    cites_per_level: int = Field(default=10, ge=1, le=50)


class CitationPathRequest(BaseModel):
    """Request to find citation path between papers."""
    source_doi: str
    target_doi: str
    max_hops: int = Field(default=3, ge=1, le=5)


class CredibilityRequest(BaseModel):
    """Request to score paper credibility."""
    doi: str


class CredibilityCompareRequest(BaseModel):
    """Request to compare multiple papers."""
    dois: list[str] = Field(..., min_length=2, max_length=10)


class ZoteroExportRequest(BaseModel):
    """Request to export citations to Zotero format."""
    items: list[dict]
    format: str = Field(default="ris", pattern="^(ris|bibtex|csl-json)$")


class ReflectionSnapshotRequest(BaseModel):
    """Request to create a reading snapshot."""
    document_id: str
    reflection_type: str = Field(default="initial_reading")
    complexity_level: int = Field(default=50, ge=0, le=100)
    time_spent_minutes: float = Field(default=0, ge=0)
    summary: Optional[str] = None
    key_takeaways: Optional[list[str]] = None
    questions: Optional[list[str]] = None
    connections: Optional[list[str]] = None
    highlights: Optional[list[dict]] = None
    sections_read: Optional[list[str]] = None
    scroll_depth: float = Field(default=0, ge=0, le=1)


class AnnotationCreateRequest(BaseModel):
    """Request to create an annotation."""
    document_id: str
    annotation_type: str = Field(default="highlight")
    content: str = ""
    selected_text: Optional[str] = None
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)
    section_id: Optional[str] = None
    visibility: str = Field(default="private")
    parent_id: Optional[str] = None
    tags: Optional[list[str]] = None


class AnnotationUpdateRequest(BaseModel):
    """Request to update an annotation."""
    content: Optional[str] = None
    tags: Optional[list[str]] = None
    visibility: Optional[str] = None


class ReactionRequest(BaseModel):
    """Request to add a reaction."""
    emoji: str = Field(..., min_length=1, max_length=10)


# ============== Citation Tree ==============

@router.post("/citations/tree")
async def build_citation_tree(
    request: CitationTreeRequest,
    current_user: CurrentUser,
):
    """
    Build a citation tree starting from a root paper.
    Shows references (papers cited) and citations (papers citing).
    """
    service = get_citation_service()

    tree = await service.build_citation_tree(
        root_doi=request.doi,
        max_depth=request.max_depth,
        refs_per_level=request.refs_per_level,
        cites_per_level=request.cites_per_level,
    )

    if tree is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    return tree.to_dict()


@router.get("/citations/paper/{doi:path}")
async def get_paper_metadata(
    doi: str,
    current_user: CurrentUser,
):
    """
    Get metadata for a single paper by DOI.
    """
    service = get_citation_service()
    paper = await service.get_paper_by_doi(doi)

    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "abstract": paper.abstract,
        "doi": paper.doi,
        "url": paper.url,
    }


@router.get("/citations/references/{doi:path}")
async def get_references(
    doi: str,
    limit: int = Query(default=20, ge=1, le=100),
    current_user: CurrentUser = None,
):
    """
    Get papers referenced by a given paper.
    """
    service = get_citation_service()
    references = await service.get_references(doi, limit=limit)

    return {
        "doi": doi,
        "references": [
            {
                "paper_id": ref.paper_id,
                "title": ref.title,
                "authors": ref.authors,
                "year": ref.year,
                "venue": ref.venue,
                "citation_count": ref.citation_count,
            }
            for ref in references
        ],
        "count": len(references),
    }


@router.get("/citations/citing/{doi:path}")
async def get_citations(
    doi: str,
    limit: int = Query(default=20, ge=1, le=100),
    current_user: CurrentUser = None,
):
    """
    Get papers that cite a given paper.
    """
    service = get_citation_service()
    citations = await service.get_citations(doi, limit=limit)

    return {
        "doi": doi,
        "citations": [
            {
                "paper_id": cit.paper_id,
                "title": cit.title,
                "authors": cit.authors,
                "year": cit.year,
                "venue": cit.venue,
                "citation_count": cit.citation_count,
            }
            for cit in citations
        ],
        "count": len(citations),
    }


@router.post("/citations/path")
async def find_citation_path(
    request: CitationPathRequest,
    current_user: CurrentUser,
):
    """
    Find a citation path between two papers.
    """
    service = get_citation_service()

    path = await service.find_citation_path(
        source_doi=request.source_doi,
        target_doi=request.target_doi,
        max_hops=request.max_hops,
    )

    if path is None:
        return {
            "found": False,
            "message": f"No path found within {request.max_hops} hops",
        }

    return {
        "found": True,
        "path_length": len(path),
        "path": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
            }
            for paper in path
        ],
    }


# ============== Credibility Scoring ==============

@router.post("/credibility/score")
async def score_credibility(
    request: CredibilityRequest,
    current_user: CurrentUser,
):
    """
    Calculate credibility score for a paper using CrossRef and Altmetrics.
    """
    scorer = get_credibility_scorer()
    score = await scorer.score_paper(request.doi)
    return score.to_dict()


@router.post("/credibility/compare")
async def compare_credibility(
    request: CredibilityCompareRequest,
    current_user: CurrentUser,
):
    """
    Compare credibility of multiple papers.
    Returns papers ranked by credibility score.
    """
    scorer = get_credibility_scorer()
    results = await scorer.compare_papers(request.dois)

    return {
        "comparisons": [
            {
                "doi": doi,
                "score": score.to_dict(),
            }
            for doi, score in results
        ],
        "count": len(results),
    }


# ============== Zotero Export ==============

@router.post("/zotero/export")
async def export_to_zotero(
    request: ZoteroExportRequest,
    current_user: CurrentUser,
):
    """
    Export citations to Zotero-compatible format.
    Supported formats: ris, bibtex, csl-json
    """
    exporter = get_zotero_exporter()

    # Convert dicts to ZoteroItems
    items = [exporter.create_item_from_dict(item) for item in request.items]

    if request.format == "ris":
        content = exporter.export_to_ris(items)
        media_type = "application/x-research-info-systems"
        ext = "ris"
    elif request.format == "bibtex":
        content = exporter.export_to_bibtex(items)
        media_type = "application/x-bibtex"
        ext = "bib"
    else:  # csl-json
        content = exporter.export_to_csl_json(items)
        media_type = "application/json"
        ext = "json"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="citations.{ext}"',
        },
    )


@router.post("/zotero/preview")
async def preview_zotero_export(
    request: ZoteroExportRequest,
    current_user: CurrentUser,
):
    """
    Preview citation export without downloading.
    """
    exporter = get_zotero_exporter()
    items = [exporter.create_item_from_dict(item) for item in request.items]

    if request.format == "ris":
        content = exporter.export_to_ris(items)
    elif request.format == "bibtex":
        content = exporter.export_to_bibtex(items)
    else:
        content = exporter.export_to_csl_json(items)

    return {
        "format": request.format,
        "content": content,
        "item_count": len(items),
    }


# ============== Reflection Engine ==============

@router.post("/reflection/snapshot")
async def create_snapshot(
    request: ReflectionSnapshotRequest,
    current_user: CurrentUser,
):
    """
    Create a reading snapshot to capture current understanding.
    """
    engine = get_reflection_engine()

    try:
        reflection_type = ReflectionType(request.reflection_type)
    except ValueError:
        reflection_type = ReflectionType.INITIAL_READING

    snapshot = engine.create_snapshot(
        user_id=current_user.user_id,
        document_id=request.document_id,
        reflection_type=reflection_type,
        complexity_level=request.complexity_level,
        time_spent_minutes=request.time_spent_minutes,
        summary=request.summary,
        key_takeaways=request.key_takeaways,
        questions=request.questions,
        connections=request.connections,
        highlights=request.highlights,
        sections_read=request.sections_read,
        scroll_depth=request.scroll_depth,
    )

    return snapshot.to_dict()


@router.get("/reflection/snapshots/{document_id}")
async def get_snapshots(
    document_id: str,
    current_user: CurrentUser,
):
    """
    Get all reading snapshots for a document.
    """
    engine = get_reflection_engine()
    snapshots = engine.get_snapshots(current_user.user_id, document_id)

    return {
        "document_id": document_id,
        "snapshots": [s.to_dict() for s in snapshots],
        "count": len(snapshots),
    }


@router.get("/reflection/journey/{document_id}")
async def get_learning_journey(
    document_id: str,
    current_user: CurrentUser,
):
    """
    Get the complete learning journey for a document.
    Shows how understanding evolved over multiple reading sessions.
    """
    engine = get_reflection_engine()
    journey = engine.get_learning_journey(current_user.user_id, document_id)
    return journey


@router.get("/reflection/prompts/{document_id}")
async def get_reflection_prompts(
    document_id: str,
    current_user: CurrentUser,
):
    """
    Get personalized reflection prompts based on reading history.
    """
    engine = get_reflection_engine()
    prompts = engine.get_reflection_prompts(current_user.user_id, document_id)

    return {
        "document_id": document_id,
        "prompts": prompts,
    }


# ============== Collaborative Annotations ==============

@router.post("/annotations")
async def create_annotation(
    request: AnnotationCreateRequest,
    current_user: CurrentUser,
):
    """
    Create a new annotation on a document.
    """
    service = get_annotation_service()

    try:
        annotation_type = AnnotationType(request.annotation_type)
    except ValueError:
        annotation_type = AnnotationType.HIGHLIGHT

    try:
        visibility = AnnotationVisibility(request.visibility)
    except ValueError:
        visibility = AnnotationVisibility.PRIVATE

    annotation = service.create_annotation(
        document_id=request.document_id,
        user_id=current_user.user_id,
        user_name=current_user.username,
        annotation_type=annotation_type,
        content=request.content,
        selected_text=request.selected_text,
        start_offset=request.start_offset,
        end_offset=request.end_offset,
        section_id=request.section_id,
        visibility=visibility,
        parent_id=request.parent_id,
        tags=request.tags,
    )

    return annotation.to_dict()


@router.get("/annotations/{document_id}")
async def get_annotations(
    document_id: str,
    section_id: Optional[str] = None,
    annotation_type: Optional[str] = None,
    visibility: Optional[str] = None,
    current_user: CurrentUser = None,
):
    """
    Get annotations for a document with optional filters.
    """
    service = get_annotation_service()

    # Parse filters
    type_filter = None
    if annotation_type:
        try:
            type_filter = [AnnotationType(annotation_type)]
        except ValueError:
            pass

    visibility_filter = None
    if visibility:
        try:
            visibility_filter = [AnnotationVisibility(visibility)]
        except ValueError:
            pass

    annotations = service.get_annotations(
        document_id=document_id,
        user_id=current_user.user_id,
        visibility_filter=visibility_filter,
        type_filter=type_filter,
        section_id=section_id,
    )

    return {
        "document_id": document_id,
        "annotations": [a.to_dict() for a in annotations],
        "count": len(annotations),
    }


@router.put("/annotations/{document_id}/{annotation_id}")
async def update_annotation(
    document_id: str,
    annotation_id: str,
    request: AnnotationUpdateRequest,
    current_user: CurrentUser,
):
    """
    Update an existing annotation.
    """
    service = get_annotation_service()

    updates = {}
    if request.content is not None:
        updates["content"] = request.content
    if request.tags is not None:
        updates["tags"] = request.tags
    if request.visibility is not None:
        try:
            updates["visibility"] = AnnotationVisibility(request.visibility)
        except ValueError:
            pass

    annotation = service.update_annotation(
        document_id=document_id,
        annotation_id=annotation_id,
        user_id=current_user.user_id,
        updates=updates,
    )

    if annotation is None:
        raise HTTPException(status_code=404, detail="Annotation not found")

    return annotation.to_dict()


@router.delete("/annotations/{document_id}/{annotation_id}")
async def delete_annotation(
    document_id: str,
    annotation_id: str,
    current_user: CurrentUser,
):
    """
    Delete an annotation (soft delete).
    """
    service = get_annotation_service()

    success = service.delete_annotation(
        document_id=document_id,
        annotation_id=annotation_id,
        user_id=current_user.user_id,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Annotation not found")

    return {"status": "deleted", "annotation_id": annotation_id}


@router.post("/annotations/{document_id}/{annotation_id}/reaction")
async def add_reaction(
    document_id: str,
    annotation_id: str,
    request: ReactionRequest,
    current_user: CurrentUser,
):
    """
    Add a reaction (emoji) to an annotation.
    """
    service = get_annotation_service()

    annotation = service.add_reaction(
        document_id=document_id,
        annotation_id=annotation_id,
        user_id=current_user.user_id,
        emoji=request.emoji,
    )

    if annotation is None:
        raise HTTPException(status_code=404, detail="Annotation not found")

    return annotation.to_dict()


@router.get("/annotations/{document_id}/thread/{parent_id}")
async def get_annotation_thread(
    document_id: str,
    parent_id: str,
    current_user: CurrentUser,
):
    """
    Get all replies in an annotation thread.
    """
    service = get_annotation_service()

    replies = service.get_thread(
        document_id=document_id,
        parent_id=parent_id,
        user_id=current_user.user_id,
    )

    return {
        "parent_id": parent_id,
        "replies": [r.to_dict() for r in replies],
        "count": len(replies),
    }


@router.get("/annotations/{document_id}/stats")
async def get_annotation_stats(
    document_id: str,
    current_user: CurrentUser,
):
    """
    Get annotation statistics for a document.
    """
    service = get_annotation_service()
    stats = service.get_annotation_stats(document_id, current_user.user_id)
    return stats
