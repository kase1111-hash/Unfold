"""Tests for knowledge graph services and API surface that need no Neo4j.

Builder tests replace the Neo4j calls with in-memory fakes; API tests cover
authentication, request validation and the external linkers (faked, so no
network is needed). Neo4j-backed behaviour lives in
tests/integration/test_knowledge_graph.py.
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from neo4j.exceptions import ServiceUnavailable, SessionExpired

import app.api.v1.routes.graph as graph_routes
import app.services.graph.builder as builder_module
import app.services.graph.document_graph as document_graph
from app.api.v1.routes import documents as documents_routes
from app.models.graph import (
    GraphNode,
    GraphNodeCreate,
    NodeType,
    RelationType,
    check_flat_metadata,
)
from app.services.external import Author, Paper, WikipediaResult
from app.services.graph.builder import KnowledgeGraphBuilder
from app.services.graph.extractor import (
    EntityExtractor,
    EntityType,
    ExtractedEntity,
)
from app.services.graph.relations import (
    ExtractedRelation,
    RelationExtractor,
    RuleBasedRelationExtractor,
)
from tests.pdf_utils import make_text_pdf


def _require_spacy_model() -> None:
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("spaCy model en_core_web_sm is not installed")


class TestEntityExtractor:
    """Tests for entity extraction service."""

    @pytest.fixture
    def extractor(self):
        """Create entity extractor instance."""
        _require_spacy_model()
        return EntityExtractor(model_name="en_core_web_sm")

    def test_extract_entities_basic(self, extractor):
        """Test basic entity extraction."""
        text = "Albert Einstein developed the theory of relativity at Princeton University."

        entities = extractor.extract_entities(text, min_confidence=0.0)

        # Should extract at least some entities
        assert len(entities) > 0

        # Check that entities have required fields
        for entity in entities:
            assert entity.text
            assert isinstance(entity.entity_type, EntityType)
            assert 0 <= entity.confidence <= 1

    def test_extract_entities_person(self, extractor):
        """Test extraction of person entities."""
        text = "Marie Curie won the Nobel Prize twice."

        entities = extractor.extract_entities(text, min_confidence=0.0)

        # Should find Marie Curie as a person
        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        names = [e.text.lower() for e in person_entities]

        assert any("marie" in name or "curie" in name for name in names)

    def test_extract_entities_organization(self, extractor):
        """Test extraction of organization entities."""
        text = "Google and Microsoft are major tech companies."

        entities = extractor.extract_entities(text, min_confidence=0.0)

        # Should find organizations
        org_entities = [e for e in entities if e.entity_type == EntityType.ORGANIZATION]

        assert len(org_entities) >= 1

    def test_extract_entities_empty_text(self, extractor):
        """Test extraction with empty text."""
        entities = extractor.extract_entities("")

        assert entities == []

    def test_extract_entities_no_noun_chunks(self, extractor):
        """Without noun chunks only named entities (spaCy NER labels) remain."""
        text = "Albert Einstein moved to the United States. The quick fox jumps."

        entities = extractor.extract_entities(text, include_noun_chunks=False)

        assert {e.text for e in entities} == {"Albert Einstein", "the United States"}
        assert all(e.metadata and "spacy_label" in e.metadata for e in entities)

    def test_extract_keywords(self, extractor):
        """Test keyword extraction."""
        text = """
        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks to process data.
        Neural networks are inspired by biological neurons.
        """

        keywords = extractor.extract_keywords(text, top_k=5)

        assert len(keywords) <= 5
        assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords)
        assert all(0 <= score <= 1 for _, score in keywords)

    def test_extract_sentences_with_entities(self, extractor):
        """Test sentence-level entity extraction."""
        text = (
            "Albert Einstein was born in Germany. He later moved to the United States."
        )

        results = extractor.extract_sentences_with_entities(text)

        assert len(results) == 2
        for result in results:
            assert "sentence" in result
            assert "entities" in result
            assert "start_char" in result
            assert "end_char" in result

    def test_entity_to_node_type_conversion(self):
        """Test EntityType to NodeType conversion."""
        entity = ExtractedEntity(
            text="Test",
            entity_type=EntityType.PERSON,
            start_char=0,
            end_char=4,
            confidence=0.9,
        )

        node_type = entity.to_node_type()

        assert node_type == NodeType.AUTHOR

    def test_entity_deduplication(self, extractor):
        """Test that duplicate entities are deduplicated."""
        text = "Python is a programming language. Python is used for data science. Python is popular."

        entities = extractor.extract_entities(text, min_confidence=0.0)

        # "Python" should appear only once (deduplicated)
        python_entities = [e for e in entities if e.text.lower() == "python"]
        assert len(python_entities) <= 1


class TestRuleBasedRelationExtractor:
    """Tests for rule-based relation extraction."""

    @pytest.fixture
    def extractor(self):
        """Create relation extractor instance."""
        _require_spacy_model()
        return RuleBasedRelationExtractor()

    def test_extract_relations_basic(self, extractor):
        """A subject-verb-object sentence yields a typed relation."""
        text = "Machine learning uses neural networks."

        entities = [
            ExtractedEntity(
                text="Machine learning",
                entity_type=EntityType.CONCEPT,
                start_char=0,
                end_char=16,
                confidence=0.9,
            ),
            ExtractedEntity(
                text="neural networks",
                entity_type=EntityType.CONCEPT,
                start_char=22,
                end_char=37,
                confidence=0.9,
            ),
        ]

        relations = extractor.extract_relations(text, entities)

        assert [
            (r.source_text, r.target_text, r.relation_type) for r in relations
        ] == [("Machine learning", "neural networks", RelationType.USES_METHOD)]

    def test_extract_relations_empty(self, extractor):
        """Test extraction with no entities."""
        relations = extractor.extract_relations("Some text", [])

        assert relations == []

    def test_infer_relation_type(self, extractor):
        """Test relation type inference from verbs."""
        assert extractor._infer_relation_type("explain") == RelationType.EXPLAINS
        assert extractor._infer_relation_type("cite") == RelationType.CITES
        assert extractor._infer_relation_type("use") == RelationType.USES_METHOD
        assert extractor._infer_relation_type("derive") == RelationType.DERIVES_FROM
        assert extractor._infer_relation_type("unknown_verb") == RelationType.RELATED_TO


class TestExtractedRelation:
    """Tests for ExtractedRelation model."""

    def test_extracted_relation_creation(self):
        """Test creating ExtractedRelation."""
        relation = ExtractedRelation(
            source_text="machine learning",
            target_text="neural networks",
            relation_type=RelationType.USES_METHOD,
            confidence=0.85,
            context="Machine learning uses neural networks",
        )

        assert relation.source_text == "machine learning"
        assert relation.target_text == "neural networks"
        assert relation.relation_type == RelationType.USES_METHOD
        assert relation.confidence == 0.85
        assert relation.context is not None


# ---------------------------------------------------------------------------
# KnowledgeGraphBuilder with Neo4j replaced by in-memory fakes
# ---------------------------------------------------------------------------


def _entity(text: str, entity_type: EntityType = EntityType.CONCEPT) -> ExtractedEntity:
    return ExtractedEntity(
        text=text, entity_type=entity_type, start_char=0, end_char=len(text), confidence=0.85
    )


def _relation(source: str, target: str, confidence: float = 0.8) -> ExtractedRelation:
    return ExtractedRelation(
        source_text=source,
        target_text=target,
        relation_type=RelationType.RELATED_TO,
        confidence=confidence,
        context=f"{source} ... {target}",
    )


class FakeGraph:
    """Records what the builder writes instead of talking to Neo4j."""

    def __init__(self):
        self.nodes: list[dict] = []
        self.relationships: list[dict] = []
        self.node_error: Exception | None = None
        self.relationship_error: Exception | None = None

    @asynccontextmanager
    async def session_context(self):
        yield object()

    async def create_node(self, session, node_type, properties):
        if self.node_error is not None:
            raise self.node_error
        self.nodes.append({"node_type": node_type, "properties": properties})
        # Like Neo4j, hand back an elementId that is NOT the node_id property
        return {"id": f"4:element:{len(self.nodes)}", "properties": properties}

    async def create_relationship(self, session, source_id, target_id, rel_type, properties):
        if self.relationship_error is not None:
            raise self.relationship_error
        known = {n["properties"]["node_id"] for n in self.nodes}
        if source_id not in known or target_id not in known:
            raise ValueError("Could not create relationship: nodes not found")
        self.relationships.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel_type,
                "properties": properties,
            }
        )
        return {"id": f"5:element:{len(self.relationships)}", "properties": properties}


class FakeEntityExtractor:
    def __init__(self, entities):
        self.entities = entities
        self.thread_ids: list[int] = []

    def extract_entities(self, text):
        self.thread_ids.append(threading.get_ident())
        return self.entities


class SyncRelationExtractor:
    def __init__(self, relations):
        self.relations = relations
        self.thread_ids: list[int] = []

    def extract_relations(self, text, entities):
        self.thread_ids.append(threading.get_ident())
        return self.relations


class AsyncRelationExtractor:
    def __init__(self, relations):
        self.relations = relations
        self.calls = 0

    async def extract_relations(self, text, entities):
        self.calls += 1
        return self.relations


@pytest.fixture
def fake_graph(monkeypatch) -> FakeGraph:
    graph = FakeGraph()
    monkeypatch.setattr(builder_module, "get_neo4j_session_context", graph.session_context)
    monkeypatch.setattr(builder_module, "create_node", graph.create_node)
    monkeypatch.setattr(builder_module, "create_relationship", graph.create_relationship)
    return graph


def _builder(entities, relation_extractor) -> KnowledgeGraphBuilder:
    # Config flags deliberately claim an LLM-only (async) extractor so the
    # tests prove dispatch follows the extractor actually in use.
    builder = KnowledgeGraphBuilder(use_llm_relations=True, use_integrated=False)
    builder.entity_extractor = FakeEntityExtractor(entities)
    builder.relation_extractor = relation_extractor
    return builder


class TestGraphBuilder:
    """build_from_text / add_node behaviour, independent of Neo4j."""

    def test_build_uses_node_id_property_everywhere(self, fake_graph):
        """node_ids, relation endpoints and relation ids are public ids."""
        entities = [_entity("Marie Curie", EntityType.PERSON), _entity("radium")]
        relations = [_relation("Marie Curie", "radium", confidence=0.7)]
        builder = _builder(entities, SyncRelationExtractor(relations))

        result = asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

        node_ids = [n["properties"]["node_id"] for n in fake_graph.nodes]
        assert result.errors == []
        assert result.nodes_created == 2
        assert result.node_ids == node_ids
        assert all(nid.startswith("node_") for nid in node_ids)
        assert [n["properties"]["source_doc_id"] for n in fake_graph.nodes] == ["doc_1"] * 2

        assert result.relations_created == 1
        (rel,) = fake_graph.relationships
        assert (rel["source_id"], rel["target_id"]) == (node_ids[0], node_ids[1])
        assert rel["properties"]["relation_id"].startswith("rel_")
        assert rel["properties"]["weight"] == 0.7
        assert result.relation_ids == [rel["properties"]["relation_id"]]

    def test_relation_weight_is_clamped_to_unit_range(self, fake_graph):
        entities = [_entity("alpha"), _entity("beta"), _entity("gamma")]
        relations = [_relation("alpha", "beta", 1.7), _relation("beta", "gamma", -0.2)]
        builder = _builder(entities, SyncRelationExtractor(relations))

        asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

        assert [r["properties"]["weight"] for r in fake_graph.relationships] == [1.0, 0.0]

    def test_self_loops_are_skipped(self, fake_graph):
        entities = [_entity("Marie Curie"), _entity("radium")]
        relations = [
            _relation("Marie Curie", "Marie Curie"),
            # Partial matching maps both ends onto the same node
            _relation("Marie Curie", "Curie"),
            _relation("Marie Curie", "radium"),
        ]
        builder = _builder(entities, SyncRelationExtractor(relations))

        result = asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

        assert result.relations_created == 1
        assert all(r["source_id"] != r["target_id"] for r in fake_graph.relationships)

    def test_labels_are_whitespace_normalized_and_deduplicated(self, fake_graph):
        """'Pierre\\nCurie' (a PDF line wrap) and 'Pierre Curie' become one node."""
        entities = [_entity("Pierre\nCurie"), _entity("Pierre  Curie"), _entity("polonium")]
        relations = [_relation("Pierre\nCurie", "polonium")]
        builder = _builder(entities, SyncRelationExtractor(relations))

        result = asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

        assert [n["properties"]["label"] for n in fake_graph.nodes] == [
            "Pierre Curie",
            "polonium",
        ]
        assert result.nodes_created == 2
        assert result.relations_created == 1

    def test_async_relation_extractor_is_awaited(self, fake_graph):
        """A coroutine extractor is awaited even if the config flags say otherwise."""
        entities = [_entity("alpha"), _entity("beta")]
        extractor = AsyncRelationExtractor([_relation("alpha", "beta")])
        builder = _builder(entities, extractor)
        builder.use_llm_relations, builder.use_integrated = False, True

        result = asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

        assert extractor.calls == 1
        assert result.errors == []
        assert result.relations_created == 1

    def test_sync_extraction_runs_off_the_event_loop(self, fake_graph):
        """spaCy entity extraction and sync relation extraction use worker threads."""
        entities = [_entity("alpha"), _entity("beta")]
        relation_extractor = SyncRelationExtractor([_relation("alpha", "beta")])
        builder = _builder(entities, relation_extractor)

        async def run():
            result = await builder.build_from_text("text", source_doc_id="doc_1")
            return result, threading.get_ident()

        result, loop_thread = asyncio.run(run())

        assert result.relations_created == 1
        assert len(builder.entity_extractor.thread_ids) == 1
        assert builder.entity_extractor.thread_ids[0] != loop_thread
        assert len(relation_extractor.thread_ids) == 1
        assert relation_extractor.thread_ids[0] != loop_thread

    def test_graph_outage_during_node_creation_propagates(self, fake_graph):
        """An unreachable Neo4j must not be reported as a build with errors."""
        fake_graph.node_error = ServiceUnavailable("Couldn't connect to localhost:7687")
        builder = _builder([_entity("alpha"), _entity("beta")], SyncRelationExtractor([]))

        with pytest.raises(ServiceUnavailable):
            asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

    def test_graph_outage_during_relation_creation_propagates(self, fake_graph):
        fake_graph.relationship_error = SessionExpired("session expired")
        builder = _builder(
            [_entity("alpha"), _entity("beta")],
            SyncRelationExtractor([_relation("alpha", "beta")]),
        )

        with pytest.raises(SessionExpired):
            asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

    def test_other_node_errors_are_reported_per_entity(self, fake_graph):
        builder = _builder([_entity("alpha"), _entity("beta")], SyncRelationExtractor([]))
        fake_graph.node_error = ValueError("bad property")

        result = asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

        assert result.nodes_created == 0
        assert result.errors == [
            "Failed to create node for 'alpha': bad property",
            "Failed to create node for 'beta': bad property",
        ]

    def test_add_node_metadata_cannot_override_builder_fields(self, fake_graph):
        """Metadata can't move a node into another document or retype it."""
        builder = _builder([], SyncRelationExtractor([]))
        request = GraphNodeCreate(
            label="Polonium",
            type=NodeType.CONCEPT,
            source_doc_id="my_doc",
            metadata={
                "source_doc_id": "someone_elses_doc",
                "node_id": "node_hijacked",
                "type": "Bogus",
                "note": "kept",
            },
        )

        node = asyncio.run(builder.add_node(request))

        (stored,) = fake_graph.nodes
        props = stored["properties"]
        assert props["source_doc_id"] == "my_doc"
        assert props["node_id"] == node.node_id
        assert node.node_id.startswith("node_") and node.node_id != "node_hijacked"
        assert props["type"] == "Concept"
        assert props["note"] == "kept"

    def test_ids_use_a_full_uuid(self, fake_graph):
        """node_id is the only match key, so 48 random bits were too few."""
        entities = [_entity("alpha"), _entity("beta")]
        builder = _builder(entities, SyncRelationExtractor([_relation("alpha", "beta")]))

        result = asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))
        node = asyncio.run(
            builder.add_node(GraphNodeCreate(label="x", type=NodeType.CONCEPT, source_doc_id="d"))
        )
        relation = asyncio.run(
            builder.add_relation(result.node_ids[0], result.node_ids[1], RelationType.EXPLAINS)
        )

        ids = result.node_ids + result.relation_ids + [node.node_id, relation.relation_id]
        for full_id in ids:
            prefix, _, hex_part = full_id.partition("_")
            assert prefix in ("node", "rel")
            assert len(hex_part) == 32
            int(hex_part, 16)

    def test_overlong_entity_labels_are_not_stored(self, fake_graph):
        """Run-on text (table rows...) extracted as one 'entity' is skipped."""
        long_label = "x" * (builder_module.MAX_ENTITY_LABEL_CHARS + 1)
        longest_kept = "y" * builder_module.MAX_ENTITY_LABEL_CHARS
        builder = _builder(
            [_entity(long_label), _entity(longest_kept), _entity("radium")],
            SyncRelationExtractor([_relation(long_label, "radium")]),
        )

        result = asyncio.run(builder.build_from_text("text", source_doc_id="doc_1"))

        assert [n["properties"]["label"] for n in fake_graph.nodes] == [longest_kept, "radium"]
        assert result.nodes_created == 2
        assert result.relations_created == 0
        assert result.errors == []

    def test_partial_source_match_stays_within_chunk(self, fake_graph):
        """A relation *source* that only partially matches an entity of an
        earlier chunk is not linked to it (partial matching is per chunk)."""
        known: dict[str, str] = {}
        first = _builder([_entity("Marie Curie"), _entity("Paris")], SyncRelationExtractor([]))
        asyncio.run(first.build_from_text("chunk 1", source_doc_id="doc_1", known_nodes=known))
        second = _builder(
            [_entity("radium"), _entity("polonium")],
            SyncRelationExtractor([_relation("Paris France", "radium")]),
        )

        asyncio.run(second.build_from_text("chunk 2", source_doc_id="doc_1", known_nodes=known))

        assert fake_graph.relationships == []

    def test_known_nodes_are_shared_across_calls(self, fake_graph):
        """One label -> node map per document build: an entity found again in
        a later chunk is not created twice, and relations can reach it."""
        known: dict[str, str] = {}
        first = _builder([_entity("Marie Curie"), _entity("Paris")], SyncRelationExtractor([]))
        first_result = asyncio.run(
            first.build_from_text("chunk 1", source_doc_id="doc_1", known_nodes=known)
        )
        second = _builder(
            [_entity("Marie\nCurie"), _entity("radium")],
            SyncRelationExtractor(
                [
                    _relation("Marie Curie", "radium"),
                    # 'Paris' is only an entity of the first chunk: exact
                    # matches may use it ...
                    _relation("radium", "Paris"),
                    # ... but partial matching stays within this chunk
                    _relation("radium", "Paris France"),
                ]
            ),
        )

        second_result = asyncio.run(
            second.build_from_text("chunk 2", source_doc_id="doc_1", known_nodes=known)
        )

        labels = [n["properties"]["label"] for n in fake_graph.nodes]
        assert labels == ["Marie Curie", "Paris", "radium"]
        assert first_result.nodes_created == 2
        assert second_result.nodes_created == 1
        assert second_result.node_ids == [known["radium"]]
        assert known == {
            "marie curie": first_result.node_ids[0],
            "paris": first_result.node_ids[1],
            "radium": second_result.node_ids[0],
        }
        assert [(r["source_id"], r["target_id"]) for r in fake_graph.relationships] == [
            (known["marie curie"], known["radium"]),
            (known["radium"], known["paris"]),
        ]

    def test_without_known_nodes_each_call_is_independent(self, fake_graph):
        """POST /graph/build (no shared map) keeps its old behaviour."""
        builder = _builder([_entity("Marie Curie")], SyncRelationExtractor([]))

        asyncio.run(builder.build_from_text("one", source_doc_id="doc_1"))
        asyncio.run(builder.build_from_text("two", source_doc_id="doc_1"))

        assert [n["properties"]["label"] for n in fake_graph.nodes] == ["Marie Curie"] * 2

    def test_overlong_stored_label_is_truncated_on_read(self):
        node = builder_module._node_from_properties(
            {"node_id": "node_1", "label": "z" * 900, "type": "Concept", "source_doc_id": "d"}
        )

        assert node.label == "z" * 500


class TestDocumentGraphBuild:
    """build_document_graph over several chunks (Neo4j faked)."""

    def test_entity_in_two_chunks_becomes_one_node(self, fake_graph, monkeypatch):
        builder = _builder(
            [_entity("Marie Curie", EntityType.PERSON), _entity("radium")],
            SyncRelationExtractor([_relation("Marie Curie", "radium")]),
        )
        deleted: list[str] = []

        async def delete_document_nodes(doc_id: str) -> int:
            deleted.append(doc_id)
            return 0

        monkeypatch.setattr(builder, "delete_document_nodes", delete_document_nodes)
        monkeypatch.setattr(document_graph, "get_graph_builder", lambda: builder)
        paragraph = "Marie Curie studied radium. " * 1200
        content = f"{paragraph}\n\n{paragraph}"
        assert len(document_graph.chunk_text(document_graph.normalize_text(content))) == 2

        result = asyncio.run(document_graph.build_document_graph("doc_1", content))

        assert deleted == ["doc_1"]
        assert len(builder.entity_extractor.thread_ids) == 2  # both chunks extracted
        assert [n["properties"]["label"] for n in fake_graph.nodes] == ["Marie Curie", "radium"]
        assert result.nodes_created == 2
        assert result.node_ids == [n["properties"]["node_id"] for n in fake_graph.nodes]
        # Each chunk's relation joins the same two nodes
        assert result.relations_created == 2
        assert {(r["source_id"], r["target_id"]) for r in fake_graph.relationships} == {
            tuple(result.node_ids)
        }


class TestOpenAIRelationExtractor:
    """The 'async' OpenAI relation call must not block the event loop."""

    def test_sync_openai_client_runs_in_a_worker_thread(self, monkeypatch):
        calls: list[int] = []

        def create(**kwargs):
            calls.append(threading.get_ident())
            message = SimpleNamespace(content="[]")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        extractor = RelationExtractor()
        monkeypatch.setattr(extractor, "_get_openai_client", lambda: client)

        async def run():
            return await extractor._call_openai("prompt"), threading.get_ident()

        content, loop_thread = asyncio.run(run())

        assert content == "[]"
        assert len(calls) == 1
        assert calls[0] != loop_thread


# ---------------------------------------------------------------------------
# API: authentication, validation, external linkers
# ---------------------------------------------------------------------------

GRAPH_ROUTES = [
    ("post", "/graph/build", {"text": "Marie Curie studied radium.", "source_doc_id": "doc_x"}),
    ("post", "/graph/documents/doc_x/build", None),
    ("post", "/graph/nodes", {"label": "Radium", "type": "Concept", "source_doc_id": "doc_x"}),
    (
        "post",
        "/graph/relations",
        {"source_node_id": "node_a", "target_node_id": "node_b", "relation_type": "EXPLAINS"},
    ),
    ("get", "/graph/nodes", None),
    ("get", "/graph/nodes/node_a", None),
    ("get", "/graph/nodes/node_a/related", None),
    ("get", "/graph/documents/doc_x/relations", None),
    ("delete", "/graph/documents/doc_x/nodes", None),
    ("get", "/graph/link/wikipedia/Python", None),
    ("get", "/graph/link/papers?query=machine+learning", None),
    ("get", "/graph/link/papers/abc123", None),
]


class TestGraphAPIAuth:
    """Every graph route requires a bearer token."""

    @pytest.mark.parametrize(("method", "path", "body"), GRAPH_ROUTES)
    def test_route_requires_auth(self, client, api_prefix, method, path, body):
        kwargs = {"json": body} if body is not None else {}

        response = getattr(client, method)(f"{api_prefix}{path}", **kwargs)

        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "MISSING_TOKEN"


class TestGraphAPIValidation:
    """Request validation happens before any graph-database access."""

    def test_traverse_excessive_depth_rejected(self, client, api_prefix, auth_headers):
        response = client.get(
            f"{api_prefix}/graph/nodes/node_a/related?max_depth=10", headers=auth_headers
        )

        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["loc"] == ["query", "max_depth"]
        assert error["type"] == "less_than_equal"

    def test_search_limit_zero_rejected(self, client, api_prefix, auth_headers):
        response = client.get(f"{api_prefix}/graph/nodes?limit=0", headers=auth_headers)

        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["loc"] == ["query", "limit"]
        assert error["type"] == "greater_than_equal"

    def test_search_unknown_node_type_rejected(self, client, api_prefix, auth_headers):
        response = client.get(f"{api_prefix}/graph/nodes?node_type=Bogus", headers=auth_headers)

        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["loc"] == ["query", "node_type"]
        assert error["type"] == "enum"

    def test_invalid_relation_type_rejected(self, client, api_prefix, auth_headers):
        response = client.post(
            f"{api_prefix}/graph/relations",
            json={
                "source_node_id": "node_a",
                "target_node_id": "node_b",
                "relation_type": "INVALID_TYPE",
                "weight": 0.8,
            },
            headers=auth_headers,
        )

        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["loc"] == ["body", "relation_type"]
        assert error["type"] == "enum"

    def test_build_short_text_rejected(self, client, api_prefix, auth_headers):
        response = client.post(
            f"{api_prefix}/graph/build",
            json={"text": "short", "source_doc_id": "doc_x"},
            headers=auth_headers,
        )

        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["loc"] == ["body", "text"]
        assert error["type"] == "string_too_short"

    def test_search_limit_up_to_1000(self, client, api_prefix, auth_headers, monkeypatch):
        limits: list[int] = []

        class FakeBuilder:
            async def search_nodes(self, **kwargs):
                limits.append(kwargs["limit"])
                return []

        monkeypatch.setattr(graph_routes, "get_graph_builder", lambda: FakeBuilder())

        accepted = client.get(f"{api_prefix}/graph/nodes?limit=1000", headers=auth_headers)
        rejected = client.get(f"{api_prefix}/graph/nodes?limit=1001", headers=auth_headers)

        assert accepted.status_code == 200
        assert accepted.json() == {"nodes": [], "total": 0}
        assert limits == [1000]
        assert rejected.status_code == 422
        (error,) = rejected.json()["detail"]
        assert error["loc"] == ["query", "limit"]
        assert error["type"] == "less_than_equal"

    @pytest.mark.parametrize(
        "metadata",
        [
            {"nested": {"a": 1}},
            {"list_of_objects": [{"a": 1}]},
            {"nested_list": [[1, 2]]},
            {"mixed_list": [1, "two"]},
            {"bool_and_number": [True, 1]},
            {"null_in_list": ["a", None]},
            {"too_big": 2**63},
            {"": "empty key"},
        ],
        ids=[
            "object", "list-of-objects", "nested-list", "mixed-list",
            "bool-and-number", "null-in-list", "int-over-64-bits", "empty-key",
        ],
    )
    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/graph/nodes", {"label": "Radium", "type": "Concept", "source_doc_id": "doc_x"}),
            (
                "/graph/relations",
                {"source_node_id": "node_a", "target_node_id": "node_b", "relation_type": "EXPLAINS"},
            ),
        ],
        ids=["node", "relation"],
    )
    def test_metadata_must_be_flat(self, client, api_prefix, auth_headers, path, body, metadata):
        """What Neo4j can't store as a property is a 422, not a 500 that
        echoes the database's error."""
        response = client.post(
            f"{api_prefix}{path}", json={**body, "metadata": metadata}, headers=auth_headers
        )

        assert response.status_code == 422, response.text
        (error,) = response.json()["detail"]
        assert error["loc"] == ["body", "metadata"]
        assert error["type"] == "value_error"

    def test_paper_search_requires_query(self, client, api_prefix, auth_headers):
        response = client.get(f"{api_prefix}/graph/link/papers", headers=auth_headers)

        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["loc"] == ["query", "query"]
        assert error["type"] == "missing"

    def test_paper_search_limit_capped(self, client, api_prefix, auth_headers):
        response = client.get(
            f"{api_prefix}/graph/link/papers?query=quantum&limit=51", headers=auth_headers
        )

        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["loc"] == ["query", "limit"]
        assert error["type"] == "less_than_equal"


class TestFlatMetadata:
    """check_flat_metadata accepts what Neo4j stores as properties."""

    def test_accepts_primitives_and_arrays_of_one_kind(self):
        metadata = {
            "name": "radium",
            "year": 1898,
            "score": 0.5,
            "verified": False,
            "missing": None,
            "aliases": ["Ra", "element 88"],
            "pages": [1, 2.5],
            "flags": [True, False],
            "empty": [],
            "spaced key": 1,
        }

        assert check_flat_metadata(metadata) == metadata
        assert check_flat_metadata(None) is None


@pytest.fixture
def owned_doc_id(client, api_prefix, auth_headers, monkeypatch) -> str:
    """A document owned by ``auth_headers``' user (no graph build)."""

    async def skip(doc_id: str) -> None:
        return None

    monkeypatch.setattr(documents_routes, "build_document_graph_task", skip)
    response = client.post(
        f"{api_prefix}/documents/upload",
        files={"file": ("curie.pdf", make_text_pdf(), "application/pdf")},
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["document"]["doc_id"]


DRIVER_ERROR = "{code: Neo.ClientError.Statement.TypeError} at graph-internal:7687"


class TestGraphWriteErrors:
    """Failed graph writes are reported without the driver's error text."""

    def test_create_node_failure_is_generic(
        self, client, api_prefix, auth_headers, owned_doc_id, monkeypatch
    ):
        class FailingBuilder:
            async def add_node(self, request):
                raise RuntimeError(DRIVER_ERROR)

        monkeypatch.setattr(graph_routes, "get_graph_builder", lambda: FailingBuilder())

        response = client.post(
            f"{api_prefix}/graph/nodes",
            json={"label": "Radium", "type": "Concept", "source_doc_id": owned_doc_id},
            headers=auth_headers,
        )

        assert response.status_code == 500
        assert response.json()["detail"] == {
            "code": "NODE_CREATION_FAILED",
            "message": "Failed to create the node",
        }

    @pytest.mark.parametrize(
        ("error", "status_code", "detail"),
        [
            (
                RuntimeError(DRIVER_ERROR),
                500,
                {"code": "RELATION_CREATION_FAILED", "message": "Failed to create the relation"},
            ),
            (
                ValueError(DRIVER_ERROR),
                404,
                {"code": "NODE_NOT_FOUND", "message": "Source or target node not found"},
            ),
        ],
        ids=["failure", "nodes-gone"],
    )
    def test_create_relation_failure_is_generic(
        self,
        client,
        api_prefix,
        auth_headers,
        owned_doc_id,
        monkeypatch,
        error,
        status_code,
        detail,
    ):
        class FailingBuilder:
            async def get_node(self, node_id):
                return GraphNode(
                    node_id=node_id, label=node_id, type=NodeType.CONCEPT, source_doc_id=owned_doc_id
                )

            async def add_relation(self, **kwargs):
                raise error

        monkeypatch.setattr(graph_routes, "get_graph_builder", lambda: FailingBuilder())

        response = client.post(
            f"{api_prefix}/graph/relations",
            json={"source_node_id": "node_a", "target_node_id": "node_b", "relation_type": "EXPLAINS"},
            headers=auth_headers,
        )

        assert response.status_code == status_code
        assert response.json()["detail"] == detail


class FakeWikipediaLinker:
    def __init__(self, result: WikipediaResult | None):
        self.result = result
        self.calls: list[str] = []

    async def find_best_match(self, entity):
        self.calls.append(entity)
        return self.result


class FakeSemanticScholarLinker:
    def __init__(self, papers: list[Paper]):
        self.papers = papers
        self.search_calls: list[tuple[str, int]] = []
        self.get_calls: list[str] = []

    async def search_papers(self, query, limit=10):
        self.search_calls.append((query, limit))
        return self.papers

    async def get_paper(self, paper_id):
        self.get_calls.append(paper_id)
        return next((p for p in self.papers if p.paper_id == paper_id), None)


SAMPLE_PAPER = Paper(
    paper_id="DOI:10.1000/xyz123",
    title="Radioactive Substances",
    abstract="A study of radium.",
    year=1903,
    citation_count=42,
    url="https://www.semanticscholar.org/paper/xyz123",
    authors=[Author(author_id="a1", name="Marie Curie"), Author(author_id="a2", name="Pierre Curie")],
)

SAMPLE_PAPER_JSON = {
    "paper_id": "DOI:10.1000/xyz123",
    "title": "Radioactive Substances",
    "abstract": "A study of radium.",
    "year": 1903,
    "citation_count": 42,
    "authors": ["Marie Curie", "Pierre Curie"],
    "url": "https://www.semanticscholar.org/paper/xyz123",
}


class TestExternalLinking:
    """Wikipedia / Semantic Scholar routes map linker results (linkers faked)."""

    def test_wikipedia_match(self, client, api_prefix, auth_headers, monkeypatch):
        fake = FakeWikipediaLinker(
            WikipediaResult(
                title="Radium",
                page_id=25601,
                url="https://en.wikipedia.org/wiki/Radium",
                extract="Radium is a chemical element.",
            )
        )
        monkeypatch.setattr(graph_routes, "get_wikipedia_linker", lambda: fake)

        response = client.get(f"{api_prefix}/graph/link/wikipedia/Radium", headers=auth_headers)

        assert response.status_code == 200
        assert response.json() == {
            "entity": "Radium",
            "title": "Radium",
            "url": "https://en.wikipedia.org/wiki/Radium",
            "extract": "Radium is a chemical element.",
            "found": True,
        }
        assert fake.calls == ["Radium"]

    def test_wikipedia_no_match_for_label_with_slash(
        self, client, api_prefix, auth_headers, monkeypatch
    ):
        """Labels containing '/' reach the linker instead of 404ing in routing."""
        fake = FakeWikipediaLinker(None)
        monkeypatch.setattr(graph_routes, "get_wikipedia_linker", lambda: fake)

        response = client.get(
            f"{api_prefix}/graph/link/wikipedia/TCP%2FIP", headers=auth_headers
        )

        assert response.status_code == 200
        assert response.json() == {
            "entity": "TCP/IP",
            "title": None,
            "url": None,
            "extract": None,
            "found": False,
        }
        assert fake.calls == ["TCP/IP"]

    def test_paper_search_maps_results(self, client, api_prefix, auth_headers, monkeypatch):
        fake = FakeSemanticScholarLinker([SAMPLE_PAPER])
        monkeypatch.setattr(graph_routes, "get_semantic_scholar_linker", lambda: fake)

        response = client.get(
            f"{api_prefix}/graph/link/papers",
            params={"query": "machine learning", "limit": 5},
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.json() == [SAMPLE_PAPER_JSON]
        assert fake.search_calls == [("machine learning", 5)]

    def test_get_paper_by_doi(self, client, api_prefix, auth_headers, monkeypatch):
        fake = FakeSemanticScholarLinker([SAMPLE_PAPER])
        monkeypatch.setattr(graph_routes, "get_semantic_scholar_linker", lambda: fake)

        response = client.get(
            f"{api_prefix}/graph/link/papers/DOI:10.1000/xyz123", headers=auth_headers
        )

        assert response.status_code == 200
        assert response.json() == SAMPLE_PAPER_JSON
        assert fake.get_calls == ["DOI:10.1000/xyz123"]

    def test_get_unknown_paper_returns_404(self, client, api_prefix, auth_headers, monkeypatch):
        fake = FakeSemanticScholarLinker([])
        monkeypatch.setattr(graph_routes, "get_semantic_scholar_linker", lambda: fake)

        response = client.get(f"{api_prefix}/graph/link/papers/nope", headers=auth_headers)

        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "PAPER_NOT_FOUND"
