"""Knowledge graph API endpoints.

Every route requires a bearer token. Graph data is scoped to the caller's
documents: a node or document that belongs to someone else is reported as
not found (404), exactly like a missing one. When Neo4j can't be reached
the driver's connectivity errors propagate and main.py turns them into
503 GRAPH_UNAVAILABLE.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies import CurrentUser, DBSession, get_owned_document
from app.db import GRAPH_UNAVAILABLE_ERRORS
from app.db.models.document import DocumentORM
from app.models.document import DocumentStatus
from app.models.graph import (
    GraphNode,
    GraphNodeCreate,
    GraphRelation,
    NodeType,
    RelationType,
)
from app.models.user import User
from app.repositories.document import DocumentRepository
from app.services.graph import get_graph_builder, get_embedding_service
from app.services.graph.document_graph import build_document_graph
from app.services.external import get_wikipedia_linker, get_semantic_scholar_linker

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class BuildGraphRequest(BaseModel):
    """Request to build graph from text."""

    text: str = Field(..., min_length=10, max_length=50000, description="Source text")
    source_doc_id: str = Field(..., description="Source document ID")
    extract_relations: bool = Field(True, description="Whether to extract relations")
    generate_embeddings: bool = Field(
        True, description="Whether to generate embeddings"
    )


class BuildGraphResponse(BaseModel):
    """Response from graph building."""

    nodes_created: int
    relations_created: int
    node_ids: list[str]
    errors: list[str]


class DocumentGraphBuildResponse(BaseModel):
    """Response from (re)building a stored document's graph."""

    doc_id: str
    nodes_created: int
    relations_created: int
    errors: list[str]


class CreateNodeRequest(GraphNodeCreate):
    """Request to create a single node."""

    pass


class CreateRelationRequest(BaseModel):
    """Request to create a relation."""

    source_node_id: str = Field(..., description="Source node ID (node_id)")
    target_node_id: str = Field(..., description="Target node ID (node_id)")
    relation_type: RelationType = Field(..., description="Type of relation")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Relation strength")
    metadata: dict | None = Field(None, description="Optional metadata")


class NodeListResponse(BaseModel):
    """Response with list of nodes."""

    nodes: list[GraphNode]
    total: int


class WikipediaLinkResponse(BaseModel):
    """Wikipedia link result."""

    entity: str
    title: str | None
    url: str | None
    extract: str | None
    found: bool


class PaperSearchResponse(BaseModel):
    """Semantic Scholar paper search result."""

    paper_id: str
    title: str
    abstract: str | None
    year: int | None
    citation_count: int | None
    authors: list[str]
    url: str | None


# Ownership helpers
async def _owned_doc_ids(db: AsyncSession, user: User) -> set[str]:
    """IDs of the documents ``user`` owns; graph nodes are scoped by these."""
    result = await db.execute(
        select(DocumentORM.doc_id).where(DocumentORM.owner_id == user.user_id)
    )
    return set(result.scalars().all())


async def _get_owned_node(node_id: str, owned_doc_ids: set[str]) -> GraphNode:
    """Load a node from one of the caller's documents, or raise 404."""
    node = await get_graph_builder().get_node(node_id)

    if node is None or node.source_doc_id not in owned_doc_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NODE_NOT_FOUND", "message": f"Node {node_id} not found"},
        )

    return node


# Endpoints
@router.post(
    "/build",
    response_model=BuildGraphResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Build knowledge graph from text",
)
async def build_graph(
    request: BuildGraphRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> BuildGraphResponse:
    """Build a knowledge graph from source text.

    Extracts entities, relations, and optionally generates embeddings.
    The source document must belong to the caller.
    """
    await get_owned_document(db, request.source_doc_id, current_user)

    builder = get_graph_builder()

    # Generate embeddings if requested
    embeddings = None
    if request.generate_embeddings:
        try:
            embedding_service = get_embedding_service(use_openai=True)
            # Extract entities first to get texts for embedding (spaCy is
            # CPU-bound, so keep it off the event loop)
            entities = await asyncio.to_thread(
                builder.entity_extractor.extract_entities, request.text
            )
            entity_texts = [e.text for e in entities]

            if entity_texts:
                embeddings = await embedding_service.embed_texts(entity_texts)
        except Exception as e:
            # Continue without embeddings if service unavailable
            logger.warning(f"Embedding generation failed: {e}")

    result = await builder.build_from_text(
        text=request.text,
        source_doc_id=request.source_doc_id,
        extract_relations=request.extract_relations,
        embeddings=embeddings,
    )

    return BuildGraphResponse(
        nodes_created=result.nodes_created,
        relations_created=result.relations_created,
        node_ids=result.node_ids,
        errors=result.errors,
    )


@router.post(
    "/documents/{doc_id}/build",
    response_model=DocumentGraphBuildResponse,
    summary="(Re)build the knowledge graph for a stored document",
)
async def build_graph_for_document(
    doc_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> DocumentGraphBuildResponse:
    """Rebuild a document's graph from its stored text.

    Existing nodes for the document are replaced, so this is safe to call
    again. Marks the document as indexed when nodes were created.
    """
    await get_owned_document(db, doc_id, current_user)

    repo = DocumentRepository(db)
    content = await repo.get_content(doc_id, owner_id=current_user.user_id)
    if not content or not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "NO_CONTENT",
                "message": "Document has no extracted text to build a graph from",
            },
        )

    result = await build_document_graph(doc_id, content)

    # The rebuild replaced every node, so replace the stored IDs too.
    await repo.set_graph_nodes(doc_id, result.node_ids)
    if result.nodes_created:
        await repo.update_status(doc_id, DocumentStatus.INDEXED)

    return DocumentGraphBuildResponse(
        doc_id=doc_id,
        nodes_created=result.nodes_created,
        relations_created=result.relations_created,
        errors=result.errors,
    )


@router.post(
    "/nodes",
    response_model=GraphNode,
    status_code=status.HTTP_201_CREATED,
    summary="Create a single node",
)
async def create_node(
    request: CreateNodeRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> GraphNode:
    """Create a single node in the knowledge graph.

    The node's source document must belong to the caller.
    """
    await get_owned_document(db, request.source_doc_id, current_user)

    builder = get_graph_builder()

    try:
        node = await builder.add_node(request)
        return node
    except GRAPH_UNAVAILABLE_ERRORS:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "NODE_CREATION_FAILED", "message": str(e)},
        )


@router.post(
    "/relations",
    response_model=GraphRelation,
    status_code=status.HTTP_201_CREATED,
    summary="Create a relation between nodes",
)
async def create_relation(
    request: CreateRelationRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> GraphRelation:
    """Create a relation between two existing nodes.

    Both nodes must come from the caller's documents.
    """
    owned_doc_ids = await _owned_doc_ids(db, current_user)
    await _get_owned_node(request.source_node_id, owned_doc_ids)
    await _get_owned_node(request.target_node_id, owned_doc_ids)

    builder = get_graph_builder()

    try:
        relation = await builder.add_relation(
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            relation_type=request.relation_type,
            weight=request.weight,
            metadata=request.metadata,
        )
        return relation
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NODE_NOT_FOUND", "message": str(e)},
        )
    except GRAPH_UNAVAILABLE_ERRORS:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "RELATION_CREATION_FAILED", "message": str(e)},
        )


@router.get(
    "/nodes/{node_id}",
    response_model=GraphNode,
    summary="Get node by ID",
)
async def get_node(
    node_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> GraphNode:
    """Get a node by its node_id (as returned by the list endpoints)."""
    return await _get_owned_node(node_id, await _owned_doc_ids(db, current_user))


@router.get(
    "/nodes",
    response_model=NodeListResponse,
    summary="Search nodes",
)
async def search_nodes(
    current_user: CurrentUser,
    db: DBSession,
    query: str | None = Query(None, description="Text query to match labels"),
    node_type: NodeType | None = Query(None, description="Filter by node type"),
    source_doc_id: str | None = Query(None, description="Filter by source document"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> NodeListResponse:
    """Search for nodes in the caller's documents."""
    builder = get_graph_builder()

    if source_doc_id:
        await get_owned_document(db, source_doc_id, current_user)
        nodes = await builder.search_nodes(
            query=query,
            node_type=node_type,
            source_doc_id=source_doc_id,
            limit=limit,
        )
    else:
        nodes = await builder.search_nodes(
            query=query,
            node_type=node_type,
            source_doc_ids=sorted(await _owned_doc_ids(db, current_user)),
            limit=limit,
        )

    return NodeListResponse(
        nodes=nodes,
        total=len(nodes),
    )


@router.get(
    "/nodes/{node_id}/related",
    response_model=NodeListResponse,
    summary="Get related nodes",
)
async def get_related_nodes(
    node_id: str,
    current_user: CurrentUser,
    db: DBSession,
    relation_types: list[RelationType] | None = Query(
        None, description="Filter by relation types"
    ),
    max_depth: int = Query(2, ge=1, le=5, description="Maximum traversal depth"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> NodeListResponse:
    """Get nodes related to a given node through graph traversal."""
    owned_doc_ids = await _owned_doc_ids(db, current_user)
    await _get_owned_node(node_id, owned_doc_ids)

    builder = get_graph_builder()

    nodes = await builder.get_related_nodes(
        node_id=node_id,
        relation_types=relation_types,
        max_depth=max_depth,
        limit=limit,
    )
    nodes = [n for n in nodes if n.source_doc_id in owned_doc_ids]

    return NodeListResponse(
        nodes=nodes,
        total=len(nodes),
    )


@router.get(
    "/documents/{doc_id}/relations",
    summary="Get all relations for a document",
)
async def get_document_relations(
    doc_id: str,
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(500, ge=1, le=1000, description="Maximum relations"),
) -> dict:
    """Get all relations between nodes belonging to a document."""
    await get_owned_document(db, doc_id, current_user)

    builder = get_graph_builder()

    relations = await builder.get_document_relations(doc_id=doc_id, limit=limit)

    return {
        "relations": [
            {
                "relation_id": r.relation_id,
                "source_node_id": r.source_node_id,
                "target_node_id": r.target_node_id,
                "type": r.type.value,
                "weight": r.weight,
            }
            for r in relations
        ],
        "total": len(relations),
    }


@router.delete(
    "/documents/{doc_id}/nodes",
    status_code=status.HTTP_200_OK,
    summary="Delete all nodes for a document",
)
async def delete_document_nodes(
    doc_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    """Delete all knowledge graph nodes associated with a document.

    The document must belong to the caller.
    """
    await get_owned_document(db, doc_id, current_user)

    builder = get_graph_builder()

    deleted = await builder.delete_document_nodes(doc_id)

    return {
        "status": "success",
        "message": f"Deleted {deleted} nodes",
        "deleted_count": deleted,
    }


# External linking endpoints
@router.get(
    # :path so labels containing "/" (e.g. "TCP/IP") still match
    "/link/wikipedia/{entity:path}",
    response_model=WikipediaLinkResponse,
    summary="Link entity to Wikipedia",
)
async def link_to_wikipedia(
    entity: str,
    current_user: CurrentUser,
) -> WikipediaLinkResponse:
    """Find the best matching Wikipedia article for an entity."""
    linker = get_wikipedia_linker()

    result = await linker.find_best_match(entity)

    if result is None:
        return WikipediaLinkResponse(
            entity=entity,
            title=None,
            url=None,
            extract=None,
            found=False,
        )

    return WikipediaLinkResponse(
        entity=entity,
        title=result.title,
        url=result.url,
        extract=result.extract,
        found=True,
    )


@router.get(
    "/link/papers",
    response_model=list[PaperSearchResponse],
    summary="Search academic papers",
)
async def search_papers(
    current_user: CurrentUser,
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
) -> list[PaperSearchResponse]:
    """Search for academic papers on Semantic Scholar."""
    linker = get_semantic_scholar_linker()

    papers = await linker.search_papers(query, limit=limit)

    return [
        PaperSearchResponse(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            citation_count=paper.citation_count,
            authors=[a.name for a in paper.authors],
            url=paper.url,
        )
        for paper in papers
    ]


@router.get(
    # :path so DOI-style IDs ("DOI:10.1000/xyz") still match
    "/link/papers/{paper_id:path}",
    response_model=PaperSearchResponse,
    summary="Get paper by ID",
)
async def get_paper(
    paper_id: str,
    current_user: CurrentUser,
) -> PaperSearchResponse:
    """Get paper details from Semantic Scholar.

    Supports Semantic Scholar IDs, DOI (DOI:xxx), or arXiv IDs (ARXIV:xxx).
    """
    linker = get_semantic_scholar_linker()

    paper = await linker.get_paper(paper_id)

    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PAPER_NOT_FOUND",
                "message": f"Paper {paper_id} not found",
            },
        )

    return PaperSearchResponse(
        paper_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        year=paper.year,
        citation_count=paper.citation_count,
        authors=[a.name for a in paper.authors],
        url=paper.url,
    )
