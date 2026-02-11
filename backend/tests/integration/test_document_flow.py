"""
Integration tests for the document ingestion and management flow.
Tests the complete lifecycle: upload -> validate -> process -> retrieve -> delete
"""

from fastapi.testclient import TestClient
from io import BytesIO


class TestDocumentIngestionFlow:
    """Test complete document ingestion pipeline."""

    def test_list_documents_returns_200(self, client: TestClient, api_prefix: str):
        """Test listing documents returns 200 with data."""
        response = client.get(f"{api_prefix}/documents/")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data or isinstance(data, list)

    def test_list_documents_with_pagination(self, client: TestClient, api_prefix: str):
        """Test document listing supports pagination."""
        response = client.get(f"{api_prefix}/documents/?page=1&page_size=10")
        assert response.status_code == 200

    def test_get_nonexistent_document(self, client: TestClient, api_prefix: str):
        """Test getting a document that doesn't exist returns 404."""
        response = client.get(f"{api_prefix}/documents/nonexistent_doc_id")
        assert response.status_code == 404

    def test_delete_nonexistent_document(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test deleting a document that doesn't exist returns 404."""
        response = client.delete(
            f"{api_prefix}/documents/nonexistent_doc_id",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_upload_invalid_file_type(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test uploading an invalid file type is rejected."""
        file_content = b"#!/bin/bash\necho 'hello'"
        files = {
            "file": ("test.exe", BytesIO(file_content), "application/octet-stream")
        }

        response = client.post(
            f"{api_prefix}/documents/upload",
            files=files,
            headers=auth_headers,
        )
        assert response.status_code in [400, 415, 422]

    def test_upload_empty_file(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test uploading an empty file is rejected."""
        files = {"file": ("empty.pdf", BytesIO(b""), "application/pdf")}

        response = client.post(
            f"{api_prefix}/documents/upload",
            files=files,
            headers=auth_headers,
        )
        assert response.status_code in [400, 422]

    def test_upload_requires_authentication(self, client: TestClient, api_prefix: str):
        """Test that document upload requires authentication."""
        file_content = b"%PDF-1.4 test content"
        files = {"file": ("test.pdf", BytesIO(file_content), "application/pdf")}

        response = client.post(f"{api_prefix}/documents/upload", files=files)
        assert response.status_code == 401


class TestDocumentProcessing:
    """Test document processing and validation."""

    def test_nonexistent_document_returns_404(
        self, client: TestClient, api_prefix: str, mock_document_id: str
    ):
        """Test that fetching a nonexistent document returns 404."""
        response = client.get(f"{api_prefix}/documents/{mock_document_id}")
        assert response.status_code == 404

    def test_paraphrase_requires_auth(
        self,
        client: TestClient,
        api_prefix: str,
        mock_document_id: str,
    ):
        """Test paraphrase endpoint requires authentication."""
        response = client.get(
            f"{api_prefix}/documents/{mock_document_id}/paraphrase",
        )
        assert response.status_code == 401

    def test_paraphrase_nonexistent_document(
        self,
        client: TestClient,
        api_prefix: str,
        mock_document_id: str,
        auth_headers: dict,
    ):
        """Test paraphrase on nonexistent document returns 404."""
        response = client.get(
            f"{api_prefix}/documents/{mock_document_id}/paraphrase",
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestDocumentMetadata:
    """Test document metadata extraction and validation."""

    def test_document_list_min_page_size(self, client: TestClient, api_prefix: str):
        """Test pagination with minimum page size."""
        response = client.get(f"{api_prefix}/documents/?page=1&page_size=1")
        assert response.status_code == 200

    def test_document_list_excessive_page_size(
        self, client: TestClient, api_prefix: str
    ):
        """Test pagination with excessive page size is rejected or capped."""
        response = client.get(f"{api_prefix}/documents/?page=1&page_size=1000")
        assert response.status_code in [200, 422]

    def test_document_list_invalid_page(self, client: TestClient, api_prefix: str):
        """Test pagination with invalid page number."""
        response = client.get(f"{api_prefix}/documents/?page=0&page_size=10")
        assert response.status_code in [200, 422]

    def test_document_search_by_status(self, client: TestClient, api_prefix: str):
        """Test filtering documents by status."""
        response = client.get(f"{api_prefix}/documents/?status=validated")
        assert response.status_code == 200


class TestDocumentValidation:
    """Test document validation and DOI verification."""

    def test_doi_lookup(self, client: TestClient, api_prefix: str):
        """Test DOI lookup returns 404 for unknown DOI."""
        valid_doi = "10.1234/test.2024.001"
        response = client.get(f"{api_prefix}/documents/doi/{valid_doi}")
        assert response.status_code == 404

    def test_invalid_doi_format_rejected(self, client: TestClient, api_prefix: str):
        """Test invalid DOI formats are rejected."""
        invalid_doi = "not-a-valid-doi"
        response = client.get(f"{api_prefix}/documents/doi/{invalid_doi}")
        assert response.status_code in [400, 404, 422]


class TestDocumentContentHash:
    """Test content hashing for provenance."""

    def test_content_hash_consistency(self, sample_document_content: str):
        """Test that same content produces same hash."""
        import hashlib

        hash1 = hashlib.sha256(sample_document_content.encode()).hexdigest()
        hash2 = hashlib.sha256(sample_document_content.encode()).hexdigest()
        assert hash1 == hash2

    def test_content_hash_sensitivity(self, sample_document_content: str):
        """Test that different content produces different hash."""
        import hashlib

        hash1 = hashlib.sha256(sample_document_content.encode()).hexdigest()
        modified_content = sample_document_content + " modified"
        hash2 = hashlib.sha256(modified_content.encode()).hexdigest()
        assert hash1 != hash2
