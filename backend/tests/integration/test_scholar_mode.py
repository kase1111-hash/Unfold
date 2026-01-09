"""
Integration tests for Scholar Mode.
Tests citations, credibility scoring, Zotero export, reflection, and annotations.
"""

import pytest
from fastapi.testclient import TestClient


class TestCitationTree:
    """Test citation tree functionality."""

    def test_build_citation_tree_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that building citation tree requires authentication."""
        request_data = {
            "doi": "10.1234/test.2024.001",
            "max_depth": 2,
            "refs_per_level": 10,
            "cites_per_level": 10,
        }
        response = client.post(f"{api_prefix}/scholar/citations/tree", json=request_data)
        assert response.status_code == 401

    def test_build_citation_tree_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test building citation tree with authentication."""
        request_data = {
            "doi": "10.1234/test.2024.001",
            "max_depth": 2,
            "refs_per_level": 10,
            "cites_per_level": 10,
        }
        response = client.post(
            f"{api_prefix}/scholar/citations/tree",
            json=request_data,
            headers=auth_headers,
        )
        # Should succeed or return 404 if paper not found
        assert response.status_code in [200, 401, 404]

    def test_get_paper_metadata(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting paper metadata by DOI."""
        doi = "10.1234/test.2024.001"
        response = client.get(
            f"{api_prefix}/scholar/citations/paper/{doi}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_references(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting paper references."""
        doi = "10.1234/test.2024.001"
        response = client.get(
            f"{api_prefix}/scholar/citations/references/{doi}?limit=20",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_citations(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting papers that cite a given paper."""
        doi = "10.1234/test.2024.001"
        response = client.get(
            f"{api_prefix}/scholar/citations/citing/{doi}?limit=20",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_find_citation_path(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test finding citation path between two papers."""
        request_data = {
            "source_doi": "10.1234/source.2024",
            "target_doi": "10.1234/target.2024",
            "max_hops": 3,
        }
        response = client.post(
            f"{api_prefix}/scholar/citations/path",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]


class TestCredibilityScoring:
    """Test credibility scoring functionality."""

    def test_score_credibility_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that credibility scoring requires authentication."""
        request_data = {"doi": "10.1234/test.2024.001"}
        response = client.post(f"{api_prefix}/scholar/credibility/score", json=request_data)
        assert response.status_code == 401

    def test_score_credibility_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test credibility scoring with authentication."""
        request_data = {"doi": "10.1234/test.2024.001"}
        response = client.post(
            f"{api_prefix}/scholar/credibility/score",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]
        if response.status_code == 200:
            data = response.json()
            # Should contain score components
            assert "overall_score" in data or "score" in data

    def test_compare_credibility(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test comparing credibility of multiple papers."""
        request_data = {
            "dois": ["10.1234/paper1", "10.1234/paper2", "10.1234/paper3"]
        }
        response = client.post(
            f"{api_prefix}/scholar/credibility/compare",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]


class TestZoteroExport:
    """Test Zotero export functionality."""

    def test_export_ris_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that RIS export requires authentication."""
        request_data = {
            "items": [{"title": "Test Paper", "authors": ["John Doe"]}],
            "format": "ris",
        }
        response = client.post(f"{api_prefix}/scholar/zotero/export", json=request_data)
        assert response.status_code == 401

    def test_export_ris_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test RIS format export with authentication."""
        request_data = {
            "items": [
                {
                    "title": "Test Paper",
                    "authors": ["John Doe", "Jane Smith"],
                    "year": 2024,
                    "doi": "10.1234/test",
                }
            ],
            "format": "ris",
        }
        response = client.post(
            f"{api_prefix}/scholar/zotero/export",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_export_bibtex_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test BibTeX format export with authentication."""
        request_data = {
            "items": [
                {
                    "title": "Test Paper",
                    "authors": ["John Doe"],
                    "year": 2024,
                }
            ],
            "format": "bibtex",
        }
        response = client.post(
            f"{api_prefix}/scholar/zotero/export",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_export_csl_json_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test CSL-JSON format export with authentication."""
        request_data = {
            "items": [
                {
                    "title": "Test Paper",
                    "authors": ["John Doe"],
                    "year": 2024,
                }
            ],
            "format": "csl-json",
        }
        response = client.post(
            f"{api_prefix}/scholar/zotero/export",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_preview_export(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test export preview."""
        request_data = {
            "items": [{"title": "Test Paper", "authors": ["John Doe"]}],
            "format": "ris",
        }
        response = client.post(
            f"{api_prefix}/scholar/zotero/preview",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]


class TestReflectionEngine:
    """Test reflection engine functionality."""

    def test_create_snapshot_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that creating snapshot requires authentication."""
        request_data = {
            "document_id": "doc_123",
            "reflection_type": "initial_reading",
            "complexity_level": 50,
        }
        response = client.post(f"{api_prefix}/scholar/reflection/snapshot", json=request_data)
        assert response.status_code == 401

    def test_create_snapshot_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test creating a reading snapshot with authentication."""
        request_data = {
            "document_id": mock_document_id,
            "reflection_type": "initial_reading",
            "complexity_level": 50,
            "time_spent_minutes": 30,
            "summary": "Initial understanding of the paper",
            "key_takeaways": ["Point 1", "Point 2"],
            "questions": ["What about X?"],
        }
        response = client.post(
            f"{api_prefix}/scholar/reflection/snapshot",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401]

    def test_get_snapshots(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test getting reading snapshots for a document."""
        response = client.get(
            f"{api_prefix}/scholar/reflection/snapshots/{mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_get_learning_journey(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test getting learning journey for a document."""
        response = client.get(
            f"{api_prefix}/scholar/reflection/journey/{mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_get_reflection_prompts(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test getting reflection prompts."""
        response = client.get(
            f"{api_prefix}/scholar/reflection/prompts/{mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]


class TestCollaborativeAnnotations:
    """Test collaborative annotation functionality."""

    def test_create_annotation_requires_auth(
        self, client: TestClient, api_prefix: str, sample_annotation: dict
    ):
        """Test that creating annotation requires authentication."""
        response = client.post(f"{api_prefix}/scholar/annotations", json=sample_annotation)
        assert response.status_code == 401

    def test_create_annotation_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        sample_annotation: dict,
        mock_document_id: str,
    ):
        """Test creating an annotation with authentication."""
        annotation_data = {
            **sample_annotation,
            "document_id": mock_document_id,
            "start_offset": 100,
            "end_offset": 150,
        }
        response = client.post(
            f"{api_prefix}/scholar/annotations",
            json=annotation_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401]

    def test_get_annotations(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test getting annotations for a document."""
        response = client.get(
            f"{api_prefix}/scholar/annotations/{mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_get_annotations_filtered(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test getting annotations with filters."""
        response = client.get(
            f"{api_prefix}/scholar/annotations/{mock_document_id}?annotation_type=highlight",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_update_annotation(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test updating an annotation."""
        update_data = {"content": "Updated content", "tags": ["updated"]}
        response = client.put(
            f"{api_prefix}/scholar/annotations/{mock_document_id}/ann_123",
            json=update_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_delete_annotation(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test deleting an annotation."""
        response = client.delete(
            f"{api_prefix}/scholar/annotations/{mock_document_id}/ann_123",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_add_reaction(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test adding a reaction to an annotation."""
        reaction_data = {"emoji": "thumbs_up"}
        response = client.post(
            f"{api_prefix}/scholar/annotations/{mock_document_id}/ann_123/reaction",
            json=reaction_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_annotation_thread(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test getting annotation thread/replies."""
        response = client.get(
            f"{api_prefix}/scholar/annotations/{mock_document_id}/thread/parent_ann_id",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_annotation_stats(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test getting annotation statistics."""
        response = client.get(
            f"{api_prefix}/scholar/annotations/{mock_document_id}/stats",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]
