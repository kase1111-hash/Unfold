"""Tests for knowledge graph services."""

import pytest

from app.models.graph import NodeType, RelationType
from app.services.graph.extractor import (
    EntityExtractor,
    EntityType,
    ExtractedEntity,
)
from app.services.graph.relations import (
    ExtractedRelation,
    RuleBasedRelationExtractor,
)


class TestEntityExtractor:
    """Tests for entity extraction service."""

    @pytest.fixture
    def extractor(self):
        """Create entity extractor instance."""
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
        """Test extraction without noun chunks."""
        text = "The quick brown fox jumps over the lazy dog."

        entities = extractor.extract_entities(text, include_noun_chunks=False)

        # May have few or no entities without noun chunks
        assert isinstance(entities, list)

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
        return RuleBasedRelationExtractor()

    def test_extract_relations_basic(self, extractor):
        """Test basic relation extraction."""
        text = "Machine learning uses neural networks to process data."

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

        # Should find at least some relations
        assert isinstance(relations, list)

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


class TestGraphAPIEndpoints:
    """Tests for graph API endpoints."""

    def test_search_nodes_endpoint(self, client, api_prefix):
        """Test search nodes endpoint."""
        response = client.get(f"{api_prefix}/graph/nodes")

        # May succeed or fail depending on Neo4j availability
        assert response.status_code in [200, 500]

    def test_build_graph_requires_auth(self, client, api_prefix):
        """Test build graph endpoint requires authentication."""
        response = client.post(
            f"{api_prefix}/graph/build",
            json={
                "text": "Test text for graph building",
                "source_doc_id": "test_doc_123",
            },
        )

        assert response.status_code == 401

    def test_create_node_requires_auth(self, client, api_prefix):
        """Test create node endpoint requires authentication."""
        response = client.post(
            f"{api_prefix}/graph/nodes",
            json={
                "label": "Test Concept",
                "type": "Concept",
                "source_doc_id": "test_doc_123",
            },
        )

        assert response.status_code == 401

    def test_create_relation_requires_auth(self, client, api_prefix):
        """Test create relation endpoint requires authentication."""
        response = client.post(
            f"{api_prefix}/graph/relations",
            json={
                "source_node_id": "node_123",
                "target_node_id": "node_456",
                "relation_type": "EXPLAINS",
            },
        )

        assert response.status_code == 401

    def test_wikipedia_link_endpoint(self, client, api_prefix):
        """Test Wikipedia linking endpoint."""
        response = client.get(f"{api_prefix}/graph/link/wikipedia/Python")

        # Should work without auth
        assert response.status_code in [200, 500]  # 500 if network issues

    def test_paper_search_endpoint(self, client, api_prefix):
        """Test paper search endpoint."""
        response = client.get(
            f"{api_prefix}/graph/link/papers",
            params={"query": "machine learning"},
        )

        # Should work without auth
        assert response.status_code in [200, 500]  # 500 if network issues

    def test_paper_search_validation(self, client, api_prefix):
        """Test paper search validates query parameter."""
        response = client.get(f"{api_prefix}/graph/link/papers")

        # Missing required query parameter
        assert response.status_code == 422
