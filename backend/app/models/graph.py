"""Knowledge graph data models."""

from enum import Enum

from pydantic import BaseModel, Field

from app.models.base import TimestampMixin


class NodeType(str, Enum):
    """Knowledge graph node types."""

    CONCEPT = "Concept"
    AUTHOR = "Author"
    PAPER = "Paper"
    METHOD = "Method"
    DATASET = "Dataset"
    INSTITUTION = "Institution"
    TERM = "Term"


class RelationType(str, Enum):
    """Knowledge graph relationship types."""

    EXPLAINS = "EXPLAINS"
    CITES = "CITES"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    DERIVES_FROM = "DERIVES_FROM"
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    USES_METHOD = "USES_METHOD"
    USES_DATASET = "USES_DATASET"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"


class GraphNodeBase(BaseModel):
    """Base model for graph nodes."""

    label: str = Field(..., min_length=1, max_length=500, description="Node label/name")
    type: NodeType = Field(..., description="Node type")
    description: str | None = Field(None, max_length=2000, description="Node description")
    metadata: dict | None = Field(None, description="Additional metadata")


class GraphNodeCreate(GraphNodeBase):
    """Model for creating a graph node."""

    source_doc_id: str = Field(..., description="Source document ID")


class GraphNode(GraphNodeBase, TimestampMixin):
    """Full graph node model."""

    node_id: str = Field(..., description="Unique node identifier")
    embedding: list[float] | None = Field(None, description="Vector embedding")
    source_doc_id: str = Field(..., description="Source document ID")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    external_links: dict[str, str] = Field(
        default_factory=dict,
        description="External links (Wikipedia, Semantic Scholar, etc.)",
    )

    class Config:
        """Pydantic config."""

        from_attributes = True


class GraphRelation(BaseModel):
    """Relationship between graph nodes."""

    relation_id: str = Field(..., description="Unique relation identifier")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    type: RelationType = Field(..., description="Relationship type")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Relationship strength")
    metadata: dict | None = Field(None, description="Additional metadata")


class GraphTraversalRequest(BaseModel):
    """Request for graph traversal."""

    start_node_id: str = Field(..., description="Starting node ID")
    max_depth: int = Field(3, ge=1, le=5, description="Maximum traversal depth")
    relation_types: list[RelationType] | None = Field(
        None, description="Filter by relation types"
    )
    node_types: list[NodeType] | None = Field(None, description="Filter by node types")
    limit: int = Field(50, ge=1, le=200, description="Maximum nodes to return")


class GraphSearchRequest(BaseModel):
    """Request for semantic graph search."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    node_types: list[NodeType] | None = Field(None, description="Filter by node types")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    similarity_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )
