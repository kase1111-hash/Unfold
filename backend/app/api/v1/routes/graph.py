"""Knowledge graph API endpoints."""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.v1.dependencies import CurrentUser
from app.models.graph import (
    GraphNode,
    GraphNodeCreate,
    GraphRelation,
    NodeType,
    RelationType,
)
from app.services.graph import get_graph_builder, get_embedding_service
from app.services.external import get_wikipedia_linker, get_semantic_scholar_linker

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


class CreateNodeRequest(GraphNodeCreate):
    """Request to create a single node."""

    pass


class CreateRelationRequest(BaseModel):
    """Request to create a relation."""

    source_node_id: str = Field(..., description="Source node element ID")
    target_node_id: str = Field(..., description="Target node element ID")
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
) -> BuildGraphResponse:
    """Build a knowledge graph from source text.

    Extracts entities, relations, and optionally generates embeddings.
    Requires authentication.
    """
    builder = get_graph_builder()

    # Generate embeddings if requested
    embeddings = None
    if request.generate_embeddings:
        try:
            embedding_service = get_embedding_service(use_openai=True)
            # Extract entities first to get texts for embedding
            extractor = builder.entity_extractor
            entities = extractor.extract_entities(request.text)
            entity_texts = [e.text for e in entities]

            if entity_texts:
                embeddings = await embedding_service.embed_texts(entity_texts)
        except Exception as e:
            # Continue without embeddings if service unavailable
            print(f"Embedding generation failed: {e}")

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
    "/nodes",
    response_model=GraphNode,
    status_code=status.HTTP_201_CREATED,
    summary="Create a single node",
)
async def create_node(
    request: CreateNodeRequest,
    current_user: CurrentUser,
) -> GraphNode:
    """Create a single node in the knowledge graph.

    Requires authentication.
    """
    builder = get_graph_builder()

    try:
        node = await builder.add_node(request)
        return node
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
) -> GraphRelation:
    """Create a relation between two existing nodes.

    Requires authentication.
    """
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
async def get_node(node_id: str) -> GraphNode:
    """Get a node by its element ID."""
    builder = get_graph_builder()

    node = await builder.get_node(node_id)

    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NODE_NOT_FOUND", "message": f"Node {node_id} not found"},
        )

    return node


@router.get(
    "/nodes",
    response_model=NodeListResponse,
    summary="Search nodes",
)
async def search_nodes(
    query: str | None = Query(None, description="Text query to match labels"),
    node_type: NodeType | None = Query(None, description="Filter by node type"),
    source_doc_id: str | None = Query(None, description="Filter by source document"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> NodeListResponse:
    """Search for nodes in the knowledge graph."""
    builder = get_graph_builder()

    nodes = await builder.search_nodes(
        query=query,
        node_type=node_type,
        source_doc_id=source_doc_id,
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
    relation_types: list[RelationType] | None = Query(
        None, description="Filter by relation types"
    ),
    max_depth: int = Query(2, ge=1, le=5, description="Maximum traversal depth"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> NodeListResponse:
    """Get nodes related to a given node through graph traversal."""
    builder = get_graph_builder()

    nodes = await builder.get_related_nodes(
        node_id=node_id,
        relation_types=relation_types,
        max_depth=max_depth,
        limit=limit,
    )

    return NodeListResponse(
        nodes=nodes,
        total=len(nodes),
    )


@router.delete(
    "/documents/{doc_id}/nodes",
    status_code=status.HTTP_200_OK,
    summary="Delete all nodes for a document",
)
async def delete_document_nodes(
    doc_id: str,
    current_user: CurrentUser,
) -> dict:
    """Delete all knowledge graph nodes associated with a document.

    Requires authentication.
    """
    builder = get_graph_builder()

    deleted = await builder.delete_document_nodes(doc_id)

    return {
        "status": "success",
        "message": f"Deleted {deleted} nodes",
        "deleted_count": deleted,
    }


# External linking endpoints
@router.get(
    "/link/wikipedia/{entity}",
    response_model=WikipediaLinkResponse,
    summary="Link entity to Wikipedia",
)
async def link_to_wikipedia(entity: str) -> WikipediaLinkResponse:
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
    "/link/papers/{paper_id}",
    response_model=PaperSearchResponse,
    summary="Get paper by ID",
)
async def get_paper(paper_id: str) -> PaperSearchResponse:
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
