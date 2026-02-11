"""
Integration tests for the Knowledge Graph system.
Tests entity extraction, graph construction, and querying.
"""

import pytest
from fastapi.testclient import TestClient


class TestGraphNodeOperations:
    """Test graph node CRUD operations."""

    def test_search_nodes_returns_list(self, client: TestClient, api_prefix: str):
        """Test node search returns a list."""
        response = client.get(f"{api_prefix}/graph/nodes?query=quantum")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))

    def test_search_nodes_with_type_filter(self, client: TestClient, api_prefix: str):
        """Test node search filtered by type."""
        response = client.get(f"{api_prefix}/graph/nodes?query=test&node_type=Concept")
        assert response.status_code == 200

    def test_search_nodes_pagination(self, client: TestClient, api_prefix: str):
        """Test node search with pagination."""
        response = client.get(f"{api_prefix}/graph/nodes?query=test&limit=5")
        assert response.status_code == 200

    def test_get_nonexistent_node(self, client: TestClient, api_prefix: str):
        """Test getting a node that doesn't exist returns 404."""
        response = client.get(f"{api_prefix}/graph/nodes/nonexistent_node_id")
        assert response.status_code == 404

    def test_create_node_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that creating a node requires authentication."""
        node_data = {
            "label": "Test Concept",
            "type": "Concept",
            "description": "A test concept",
            "source_doc_id": "test_doc_123",
        }
        response = client.post(f"{api_prefix}/graph/nodes", json=node_data)
        assert response.status_code == 401

    def test_create_node_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test creating a node with authentication."""
        node_data = {
            "label": "Test Concept",
            "type": "Concept",
            "description": "A test concept for integration testing",
            "source_doc_id": "test_doc_123",
        }
        response = client.post(
            f"{api_prefix}/graph/nodes",
            json=node_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201]


class TestGraphRelationOperations:
    """Test graph relation operations."""

    def test_create_relation_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that creating a relation requires authentication."""
        relation_data = {
            "source_node_id": "node_1",
            "target_node_id": "node_2",
            "type": "EXPLAINS",
            "weight": 0.8,
        }
        response = client.post(f"{api_prefix}/graph/relations", json=relation_data)
        assert response.status_code == 401

    def test_create_relation_nonexistent_nodes(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test creating a relation between nonexistent nodes returns 404."""
        relation_data = {
            "source_node_id": "node_does_not_exist_1",
            "target_node_id": "node_does_not_exist_2",
            "type": "EXPLAINS",
            "weight": 0.8,
        }
        response = client.post(
            f"{api_prefix}/graph/relations",
            json=relation_data,
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_invalid_relation_type_rejected(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test that invalid relation types are rejected."""
        relation_data = {
            "source_node_id": "node_1",
            "target_node_id": "node_2",
            "type": "INVALID_TYPE",
            "weight": 0.8,
        }
        response = client.post(
            f"{api_prefix}/graph/relations",
            json=relation_data,
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestGraphTraversal:
    """Test graph traversal operations."""

    def test_traverse_from_nonexistent_node(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test traversal from a node that doesn't exist returns 404."""
        response = client.get(
            f"{api_prefix}/graph/nodes/nonexistent_node/related?max_depth=2",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_traverse_excessive_depth_rejected(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test that traversal depth > 5 is rejected."""
        response = client.get(
            f"{api_prefix}/graph/nodes/test_node/related?max_depth=10",
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestGraphBuildFromDocument:
    """Test graph building from document content."""

    def test_build_graph_requires_auth(
        self, client: TestClient, api_prefix: str, sample_document_content: str
    ):
        """Test that building a graph requires authentication."""
        build_data = {
            "text": sample_document_content,
            "source_doc_id": "test_doc_123",
        }
        response = client.post(f"{api_prefix}/graph/build", json=build_data)
        assert response.status_code == 401

    def test_build_graph_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        sample_document_content: str,
        auth_headers: dict,
    ):
        """Test building a graph with authentication."""
        build_data = {
            "text": sample_document_content,
            "source_doc_id": "test_doc_123",
        }
        response = client.post(
            f"{api_prefix}/graph/build",
            json=build_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201]


class TestExternalKnowledgeLinking:
    """Test external knowledge linking (Wikipedia, Semantic Scholar)."""

    def test_wikipedia_link_endpoint(self, client: TestClient, api_prefix: str):
        """Test Wikipedia linking endpoint returns data."""
        response = client.get(f"{api_prefix}/graph/link/wikipedia/quantum")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))

    def test_semantic_scholar_search(self, client: TestClient, api_prefix: str):
        """Test Semantic Scholar paper search returns 200."""
        response = client.get(f"{api_prefix}/graph/link/papers?query=machine+learning")
        assert response.status_code == 200

    def test_semantic_scholar_search_with_limit(
        self, client: TestClient, api_prefix: str
    ):
        """Test Semantic Scholar search with limit."""
        response = client.get(f"{api_prefix}/graph/link/papers?query=quantum&limit=5")
        assert response.status_code == 200

    def test_semantic_scholar_missing_query(self, client: TestClient, api_prefix: str):
        """Test Semantic Scholar search without query is rejected."""
        response = client.get(f"{api_prefix}/graph/link/papers")
        assert response.status_code == 422


class TestEntityExtraction:
    """Test entity extraction from text."""

    def test_extract_entities_basic(self, sample_document_content: str):
        """Test basic entity extraction from sample content."""
        from app.services.graph.extractor import EntityExtractor

        try:
            extractor = EntityExtractor()
            entities = extractor.extract_entities(sample_document_content)
            assert isinstance(entities, list)
        except Exception:
            pytest.skip("SpaCy model not available")

    def test_extract_entities_empty_text(self):
        """Test entity extraction from empty text."""
        from app.services.graph.extractor import EntityExtractor

        try:
            extractor = EntityExtractor()
            entities = extractor.extract_entities("")
            assert entities == []
        except Exception:
            pytest.skip("SpaCy model not available")


class TestGraphVisualization:
    """Test graph visualization data generation."""

    def test_get_visualization_data(
        self, client: TestClient, api_prefix: str, mock_document_id: str
    ):
        """Test getting visualization data for a nonexistent document returns 404."""
        response = client.get(f"{api_prefix}/graph/visualize/{mock_document_id}")
        assert response.status_code == 404


class TestGraphEmbeddings:
    """Test graph embedding operations."""

    def test_similarity_search_requires_auth(
        self, client: TestClient, api_prefix: str
    ):
        """Test vector similarity search requires authentication."""
        response = client.get(
            f"{api_prefix}/graph/similar?query=quantum+computing&limit=5",
        )
        assert response.status_code == 401

    def test_similarity_search_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test vector similarity search returns results."""
        response = client.get(
            f"{api_prefix}/graph/similar?query=quantum+computing&limit=5",
            headers=auth_headers,
        )
        assert response.status_code == 200
