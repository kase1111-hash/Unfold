"""Knowledge graph builder service."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.db.neo4j import (
    create_node,
    create_relationship,
    delete_node,
    get_neo4j_session_context,
    get_node_by_id,
    search_nodes,
    traverse_graph,
)
from app.models.graph import (
    GraphNode,
    GraphNodeCreate,
    GraphRelation,
    NodeType,
    RelationType,
)
from app.services.graph.extractor import ExtractedEntity, get_entity_extractor
from app.services.graph.relations import ExtractedRelation, get_relation_extractor


@dataclass
class GraphBuildResult:
    """Result of building a knowledge graph from text."""

    nodes_created: int
    relations_created: int
    node_ids: list[str]
    relation_ids: list[str]
    errors: list[str]


class KnowledgeGraphBuilder:
    """Build and manage knowledge graphs from documents."""

    def __init__(self, use_llm_relations: bool = True):
        """Initialize knowledge graph builder.

        Args:
            use_llm_relations: Whether to use LLM for relation extraction
        """
        self.entity_extractor = get_entity_extractor()
        self.relation_extractor = get_relation_extractor(use_llm=use_llm_relations)
        self.use_llm_relations = use_llm_relations

    async def build_from_text(
        self,
        text: str,
        source_doc_id: str,
        extract_relations: bool = True,
        embeddings: list[list[float]] | None = None,
    ) -> GraphBuildResult:
        """Build knowledge graph from text.

        Args:
            text: Source text to process
            source_doc_id: ID of source document
            extract_relations: Whether to extract relations between entities
            embeddings: Optional pre-computed embeddings for entities

        Returns:
            GraphBuildResult with statistics and IDs
        """
        result = GraphBuildResult(
            nodes_created=0,
            relations_created=0,
            node_ids=[],
            relation_ids=[],
            errors=[],
        )

        # Step 1: Extract entities
        entities = self.entity_extractor.extract_entities(text)

        if not entities:
            result.errors.append("No entities extracted from text")
            return result

        # Step 2: Create nodes in Neo4j
        entity_to_node_id: dict[str, str] = {}

        async with get_neo4j_session_context() as session:
            for i, entity in enumerate(entities):
                try:
                    node_id = f"node_{uuid4().hex[:12]}"

                    # Get embedding if available
                    embedding = None
                    if embeddings and i < len(embeddings):
                        embedding = embeddings[i]

                    node_data = await create_node(
                        session,
                        node_type=entity.to_node_type().value,
                        properties={
                            "node_id": node_id,
                            "label": entity.text,
                            "type": entity.to_node_type().value,
                            "source_doc_id": source_doc_id,
                            "confidence": entity.confidence,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            **({"embedding": embedding} if embedding else {}),
                            **(entity.metadata or {}),
                        },
                    )

                    entity_to_node_id[entity.text.lower()] = node_data["id"]
                    result.node_ids.append(node_data["id"])
                    result.nodes_created += 1

                except Exception as e:
                    result.errors.append(f"Failed to create node for '{entity.text}': {e}")

        # Step 3: Extract and create relations
        if extract_relations and len(entities) >= 2:
            try:
                if self.use_llm_relations:
                    relations = await self.relation_extractor.extract_relations(
                        text, entities
                    )
                else:
                    relations = self.relation_extractor.extract_relations(text, entities)

                async with get_neo4j_session_context() as session:
                    for relation in relations:
                        try:
                            source_id = entity_to_node_id.get(relation.source_text.lower())
                            target_id = entity_to_node_id.get(relation.target_text.lower())

                            if not source_id or not target_id:
                                continue

                            rel_data = await create_relationship(
                                session,
                                source_id=source_id,
                                target_id=target_id,
                                rel_type=relation.relation_type.value,
                                properties={
                                    "confidence": relation.confidence,
                                    "context": relation.context,
                                    "created_at": datetime.now(timezone.utc).isoformat(),
                                },
                            )

                            result.relation_ids.append(rel_data["id"])
                            result.relations_created += 1

                        except Exception as e:
                            result.errors.append(
                                f"Failed to create relation {relation.source_text} -> {relation.target_text}: {e}"
                            )

            except Exception as e:
                result.errors.append(f"Relation extraction failed: {e}")

        return result

    async def add_node(
        self,
        node_data: GraphNodeCreate,
        embedding: list[float] | None = None,
    ) -> GraphNode:
        """Add a single node to the graph.

        Args:
            node_data: Node creation data
            embedding: Optional embedding vector

        Returns:
            Created GraphNode
        """
        node_id = f"node_{uuid4().hex[:12]}"

        async with get_neo4j_session_context() as session:
            result = await create_node(
                session,
                node_type=node_data.type.value,
                properties={
                    "node_id": node_id,
                    "label": node_data.label,
                    "type": node_data.type.value,
                    "description": node_data.description,
                    "source_doc_id": node_data.source_doc_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **({"embedding": embedding} if embedding else {}),
                    **(node_data.metadata or {}),
                },
            )

        return GraphNode(
            node_id=node_id,
            label=node_data.label,
            type=node_data.type,
            description=node_data.description,
            source_doc_id=node_data.source_doc_id,
            metadata=node_data.metadata,
            embedding=embedding,
            confidence=1.0,
            external_links={},
        )

    async def add_relation(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> GraphRelation:
        """Add a relation between two nodes.

        Args:
            source_node_id: Source node element ID
            target_node_id: Target node element ID
            relation_type: Type of relation
            weight: Relation strength (0-1)
            metadata: Optional metadata

        Returns:
            Created GraphRelation
        """
        relation_id = f"rel_{uuid4().hex[:12]}"

        async with get_neo4j_session_context() as session:
            await create_relationship(
                session,
                source_id=source_node_id,
                target_id=target_node_id,
                rel_type=relation_type.value,
                properties={
                    "relation_id": relation_id,
                    "weight": weight,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **(metadata or {}),
                },
            )

        return GraphRelation(
            relation_id=relation_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            type=relation_type,
            weight=weight,
            metadata=metadata,
        )

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID.

        Args:
            node_id: Node element ID

        Returns:
            GraphNode if found, None otherwise
        """
        async with get_neo4j_session_context() as session:
            result = await get_node_by_id(session, node_id)

        if result is None:
            return None

        props = result["properties"]

        return GraphNode(
            node_id=props.get("node_id", ""),
            label=props.get("label", ""),
            type=NodeType(props.get("type", "Concept")),
            description=props.get("description"),
            source_doc_id=props.get("source_doc_id", ""),
            embedding=props.get("embedding"),
            confidence=props.get("confidence", 1.0),
            external_links=props.get("external_links", {}),
            metadata={
                k: v for k, v in props.items()
                if k not in {"node_id", "label", "type", "description", "source_doc_id", "embedding", "confidence", "external_links"}
            },
        )

    async def search_nodes(
        self,
        query: str | None = None,
        node_type: NodeType | None = None,
        source_doc_id: str | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Search for nodes in the graph.

        Args:
            query: Optional text query to match label
            node_type: Optional node type filter
            source_doc_id: Optional source document filter
            limit: Maximum results

        Returns:
            List of matching nodes
        """
        properties = {}
        if source_doc_id:
            properties["source_doc_id"] = source_doc_id

        label = node_type.value if node_type else None

        async with get_neo4j_session_context() as session:
            results = await search_nodes(
                session,
                label=label,
                properties=properties if properties else None,
                limit=limit,
            )

        nodes = []
        for result in results:
            props = result["properties"]

            # Filter by query if provided
            if query and query.lower() not in props.get("label", "").lower():
                continue

            nodes.append(
                GraphNode(
                    node_id=props.get("node_id", ""),
                    label=props.get("label", ""),
                    type=NodeType(props.get("type", "Concept")),
                    description=props.get("description"),
                    source_doc_id=props.get("source_doc_id", ""),
                    embedding=props.get("embedding"),
                    confidence=props.get("confidence", 1.0),
                    external_links=props.get("external_links", {}),
                )
            )

        return nodes

    async def get_related_nodes(
        self,
        node_id: str,
        relation_types: list[RelationType] | None = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Get nodes related to a given node.

        Args:
            node_id: Starting node element ID
            relation_types: Optional filter for relation types
            max_depth: Maximum traversal depth
            limit: Maximum results

        Returns:
            List of related nodes
        """
        rel_type_strs = [rt.value for rt in relation_types] if relation_types else None

        async with get_neo4j_session_context() as session:
            results = await traverse_graph(
                session,
                start_node_id=node_id,
                relationship_types=rel_type_strs,
                direction="BOTH",
                max_depth=max_depth,
                limit=limit,
            )

        nodes = []
        for result in results:
            props = result["properties"]

            nodes.append(
                GraphNode(
                    node_id=props.get("node_id", ""),
                    label=props.get("label", ""),
                    type=NodeType(props.get("type", "Concept")),
                    description=props.get("description"),
                    source_doc_id=props.get("source_doc_id", ""),
                    embedding=props.get("embedding"),
                    confidence=props.get("confidence", 1.0),
                    external_links=props.get("external_links", {}),
                )
            )

        return nodes

    async def delete_document_nodes(self, doc_id: str) -> int:
        """Delete all nodes associated with a document.

        Args:
            doc_id: Document ID

        Returns:
            Number of nodes deleted
        """
        deleted = 0

        async with get_neo4j_session_context() as session:
            # Find all nodes for this document
            nodes = await search_nodes(
                session,
                properties={"source_doc_id": doc_id},
                limit=1000,
            )

            # Delete each node
            for node in nodes:
                if await delete_node(session, node["id"], detach=True):
                    deleted += 1

        return deleted


# Global builder instance
_builder: KnowledgeGraphBuilder | None = None


def get_graph_builder(use_llm: bool = True) -> KnowledgeGraphBuilder:
    """Get or create knowledge graph builder instance.

    Args:
        use_llm: Whether to use LLM for relation extraction

    Returns:
        KnowledgeGraphBuilder instance
    """
    global _builder
    if _builder is None:
        _builder = KnowledgeGraphBuilder(use_llm_relations=use_llm)
    return _builder
