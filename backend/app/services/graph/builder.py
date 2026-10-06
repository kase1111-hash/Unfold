"""Knowledge graph builder service.

Builds knowledge graphs from text using:
1. Entity extraction (spaCy NER + noun chunks)
2. Integrated relation extraction (coreference, dependency, LLM, patterns)

The integrated pipeline uses Ollama as the default LLM provider for local/offline
operation, with fallback to cloud APIs when configured.
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from app.db.neo4j import (
    GRAPH_UNAVAILABLE_ERRORS,
    create_node,
    create_relationship,
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
from app.services.graph.extractor import get_entity_extractor
from app.services.graph.relations import get_relation_extractor

logger = logging.getLogger(__name__)

# Node properties that are set by the builder and must not be overridden by
# caller-supplied metadata.
_NODE_FIELDS = {
    "node_id",
    "label",
    "type",
    "description",
    "source_doc_id",
    "embedding",
    "confidence",
    "external_links",
}


def _normalize_label(text: str) -> str:
    """Collapse whitespace (including PDF line breaks) inside an entity label."""
    return " ".join(text.split())


def _clamp_unit(value: float | None, default: float = 1.0) -> float:
    """Coerce a confidence/weight value into the 0..1 range."""
    try:
        number = float(value) if value is not None else default
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _node_from_properties(props: dict) -> GraphNode:
    """Build a GraphNode from stored Neo4j node properties."""
    try:
        node_type = NodeType(props.get("type", "Concept"))
    except ValueError:
        node_type = NodeType.CONCEPT

    return GraphNode(
        node_id=props.get("node_id", ""),
        label=props.get("label", ""),
        type=node_type,
        description=props.get("description"),
        source_doc_id=props.get("source_doc_id", ""),
        embedding=props.get("embedding"),
        confidence=_clamp_unit(props.get("confidence")),
        external_links=props.get("external_links", {}),
    )


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

    def __init__(
        self,
        use_llm_relations: bool = True,
        use_integrated: bool = True,
        llm_provider: str = "ollama",
    ):
        """Initialize knowledge graph builder.

        Args:
            use_llm_relations: Whether to use LLM for relation extraction
            use_integrated: Whether to use integrated pipeline (recommended)
            llm_provider: LLM provider for integrated pipeline ("ollama", "openai", etc.)
        """
        self.entity_extractor = get_entity_extractor()
        self.relation_extractor = get_relation_extractor(
            use_llm=use_llm_relations,
            use_integrated=use_integrated,
            llm_provider=llm_provider,
        )
        self.use_llm_relations = use_llm_relations
        self.use_integrated = use_integrated
        logger.info(
            f"[KnowledgeGraphBuilder] Initialized with integrated={use_integrated}, llm_provider={llm_provider}"
        )

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
            GraphBuildResult with statistics and IDs (node_ids are the
            nodes' node_id property, the public node identifier)

        Raises:
            GRAPH_UNAVAILABLE_ERRORS: If Neo4j cannot be reached
        """
        result = GraphBuildResult(
            nodes_created=0,
            relations_created=0,
            node_ids=[],
            relation_ids=[],
            errors=[],
        )

        # Step 1: Extract entities (spaCy is CPU-bound; keep it off the loop)
        entities = await asyncio.to_thread(self.entity_extractor.extract_entities, text)

        if not entities:
            result.errors.append("No entities extracted from text")
            return result

        # Step 2: Create nodes in Neo4j. entity_to_node_id maps the
        # normalized, lowercased label to the node_id property, which is what
        # create_relationship matches on in step 3.
        entity_to_node_id: dict[str, str] = {}

        async with get_neo4j_session_context() as session:
            for i, entity in enumerate(entities):
                label = _normalize_label(entity.text)
                key = label.lower()
                if not label or key in entity_to_node_id:
                    # Labels that only differed by line breaks collapse here
                    continue
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
                            **(entity.metadata or {}),
                            "node_id": node_id,
                            "label": label,
                            "type": entity.to_node_type().value,
                            "source_doc_id": source_doc_id,
                            "confidence": entity.confidence,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            **({"embedding": embedding} if embedding else {}),
                        },
                    )

                    if node_data is None:
                        result.errors.append(
                            f"Failed to create node for '{label}': graph database not available"
                        )
                        continue

                    entity_to_node_id[key] = node_id
                    result.node_ids.append(node_id)
                    result.nodes_created += 1

                except GRAPH_UNAVAILABLE_ERRORS:
                    raise
                except Exception as e:
                    result.errors.append(f"Failed to create node for '{label}': {e}")

        # Step 3: Extract and create relations
        if extract_relations and len(entities) >= 2:
            try:
                # Dispatch on the extractor actually in use (get_relation_extractor
                # may fall back to a different one than the config flags suggest):
                # await coroutine functions, run synchronous ones in a thread.
                extract = self.relation_extractor.extract_relations
                if inspect.iscoroutinefunction(extract):
                    relations = await extract(text, entities)
                else:
                    relations = await asyncio.to_thread(extract, text, entities)

                logger.info(f"Extracted {len(relations)} relations from text")

                async with get_neo4j_session_context() as session:
                    for relation in relations:
                        try:
                            source_text = _normalize_label(relation.source_text).lower()
                            target_text = _normalize_label(relation.target_text).lower()
                            source_id = entity_to_node_id.get(source_text)
                            target_id = entity_to_node_id.get(target_text)

                            if not source_id or not target_id:
                                # Try partial matching for multi-word entities
                                if not source_id:
                                    for key in entity_to_node_id:
                                        if source_text in key or key in source_text:
                                            source_id = entity_to_node_id[key]
                                            break
                                if not target_id:
                                    for key in entity_to_node_id:
                                        if target_text in key or key in target_text:
                                            target_id = entity_to_node_id[key]
                                            break

                            if not source_id or not target_id:
                                continue

                            # A node relating to itself carries no information
                            if source_id == target_id:
                                continue

                            # Handle relation_type as either enum or string
                            rel_type_value = (
                                relation.relation_type.value
                                if hasattr(relation.relation_type, "value")
                                else str(relation.relation_type)
                            )

                            rel_data = await create_relationship(
                                session,
                                source_id=source_id,
                                target_id=target_id,
                                rel_type=rel_type_value,
                                properties={
                                    "relation_id": f"rel_{uuid4().hex[:12]}",
                                    "weight": _clamp_unit(relation.confidence),
                                    "confidence": relation.confidence,
                                    "context": relation.context or "",
                                    "extraction_method": getattr(
                                        relation, "extraction_method", "unknown"
                                    ),
                                    "created_at": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                },
                            )

                            if rel_data is None:
                                continue

                            result.relation_ids.append(rel_data["properties"]["relation_id"])
                            result.relations_created += 1

                        except GRAPH_UNAVAILABLE_ERRORS:
                            raise
                        except Exception as e:
                            result.errors.append(
                                f"Failed to create relation {relation.source_text} -> {relation.target_text}: {e}"
                            )

            except GRAPH_UNAVAILABLE_ERRORS:
                raise
            except Exception as e:
                logger.error(f"Relation extraction failed: {e}")
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
            await create_node(
                session,
                node_type=node_data.type.value,
                properties={
                    # Metadata can't override node_id, type or source_doc_id
                    # (which decides who owns the node)
                    **{
                        k: v
                        for k, v in (node_data.metadata or {}).items()
                        if k not in _NODE_FIELDS
                    },
                    "node_id": node_id,
                    "label": node_data.label,
                    "type": node_data.type.value,
                    "description": node_data.description,
                    "source_doc_id": node_data.source_doc_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **({"embedding": embedding} if embedding else {}),
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
            source_node_id: Source node's node_id
            target_node_id: Target node's node_id
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
                    **(metadata or {}),
                    "relation_id": relation_id,
                    "weight": weight,
                    "created_at": datetime.now(timezone.utc).isoformat(),
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
            node_id: The node's node_id property

        Returns:
            GraphNode if found, None otherwise
        """
        async with get_neo4j_session_context() as session:
            result = await get_node_by_id(session, node_id)

        if result is None:
            return None

        props = result["properties"]
        node = _node_from_properties(props)
        node.metadata = {k: v for k, v in props.items() if k not in _NODE_FIELDS}
        return node

    async def search_nodes(
        self,
        query: str | None = None,
        node_type: NodeType | None = None,
        source_doc_id: str | None = None,
        limit: int = 50,
        source_doc_ids: list[str] | None = None,
    ) -> list[GraphNode]:
        """Search for nodes in the graph.

        Args:
            query: Optional text query to match label
            node_type: Optional node type filter
            source_doc_id: Optional source document filter
            limit: Maximum results
            source_doc_ids: Optional filter to nodes from any of these
                documents (an empty list matches nothing)

        Returns:
            List of matching nodes
        """
        properties: dict = {}
        if source_doc_id:
            properties["source_doc_id"] = source_doc_id
        elif source_doc_ids is not None:
            properties["source_doc_id"] = list(source_doc_ids)

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

            nodes.append(_node_from_properties(props))

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
            node_id: Starting node's node_id
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

        return [_node_from_properties(result["properties"]) for result in results]

    async def get_document_relations(
        self,
        doc_id: str,
        limit: int = 500,
    ) -> list[GraphRelation]:
        """Get all relations between nodes belonging to a document.

        Args:
            doc_id: Document ID to get relations for
            limit: Maximum relations to return

        Returns:
            List of GraphRelation objects
        """
        async with get_neo4j_session_context() as session:
            if session is None:
                return []

            query = """
            MATCH (a)-[r]->(b)
            WHERE a.source_doc_id = $doc_id AND b.source_doc_id = $doc_id
            RETURN a.node_id AS source_node_id,
                   b.node_id AS target_node_id,
                   type(r) AS relation_type,
                   coalesce(r.relation_id, elementId(r)) AS relation_id,
                   coalesce(r.weight, r.confidence, 1.0) AS weight
            LIMIT $limit
            """
            result = await session.run(query, doc_id=doc_id, limit=limit)
            records = await result.data()

        relations = []
        for record in records:
            rel_type_str = record.get("relation_type", "RELATED_TO")
            try:
                rel_type = RelationType(rel_type_str)
            except ValueError:
                rel_type = RelationType.RELATED_TO

            relations.append(
                GraphRelation(
                    relation_id=record["relation_id"],
                    source_node_id=record["source_node_id"],
                    target_node_id=record["target_node_id"],
                    type=rel_type,
                    weight=_clamp_unit(record["weight"]),
                )
            )

        return relations

    async def delete_document_nodes(self, doc_id: str) -> int:
        """Delete all nodes associated with a document.

        Args:
            doc_id: Document ID

        Returns:
            Number of nodes deleted
        """
        async with get_neo4j_session_context() as session:
            if session is None:
                return 0

            result = await session.run(
                """
                MATCH (n)
                WHERE n.source_doc_id = $doc_id
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                doc_id=doc_id,
            )
            record = await result.single()

        return record["deleted"] if record else 0


# Global builder instance
_builder: KnowledgeGraphBuilder | None = None


def get_graph_builder(
    use_llm: bool = True,
    use_integrated: bool = True,
    llm_provider: str = "ollama",
) -> KnowledgeGraphBuilder:
    """Get or create knowledge graph builder instance.

    Args:
        use_llm: Whether to use LLM for relation extraction
        use_integrated: Whether to use integrated pipeline (recommended)
        llm_provider: LLM provider for integrated pipeline ("ollama", "openai", etc.)

    Returns:
        KnowledgeGraphBuilder instance
    """
    global _builder
    if _builder is None:
        _builder = KnowledgeGraphBuilder(
            use_llm_relations=use_llm,
            use_integrated=use_integrated,
            llm_provider=llm_provider,
        )
    return _builder


def reset_builder():
    """Reset the global builder instance."""
    global _builder
    _builder = None
