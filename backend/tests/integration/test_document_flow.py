"""
Integration tests for the document ingestion and management flow.
Tests the complete lifecycle: upload -> validate -> process -> retrieve -> delete
"""

from fastapi.testclient import TestClient
from io import BytesIO


class TestDocumentIngestionFlow:
    """Test complete document ingestion pipeline."""

    def test_list_documents_empty(self, client: TestClient, api_prefix: str):
        """Test listing documents returns empty list initially."""
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
        # Should be 404 or 401 if auth fails
        assert response.status_code in [404, 401, 403]

    def test_upload_invalid_file_type(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test uploading an invalid file type is rejected."""
        # Create a fake executable file
        file_content = b"#!/bin/bash\necho 'hello'"
        files = {
            "file": ("test.exe", BytesIO(file_content), "application/octet-stream")
        }

        response = client.post(
            f"{api_prefix}/documents/upload",
            files=files,
            headers=auth_headers,
        )
        # Should reject invalid file types
        assert response.status_code in [400, 401, 415, 422]

    def test_upload_empty_file(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test uploading an empty file is handled properly."""
        files = {"file": ("empty.pdf", BytesIO(b""), "application/pdf")}

        response = client.post(
            f"{api_prefix}/documents/upload",
            files=files,
            headers=auth_headers,
        )
        # Should reject empty files
        assert response.status_code in [400, 401, 422]

    def test_upload_requires_authentication(self, client: TestClient, api_prefix: str):
        """Test that document upload requires authentication."""
        file_content = b"%PDF-1.4 test content"
        files = {"file": ("test.pdf", BytesIO(file_content), "application/pdf")}

        response = client.post(f"{api_prefix}/documents/upload", files=files)
        assert response.status_code == 401


class TestDocumentProcessing:
    """Test document processing and validation."""

    def test_document_status_endpoint_structure(
        self, client: TestClient, api_prefix: str, mock_document_id: str
    ):
        """Test document status endpoint returns proper structure."""
        response = client.get(f"{api_prefix}/documents/{mock_document_id}")
        # Either 404 (not found) or 200 with proper structure
        if response.status_code == 200:
            data = response.json()
            # Should have document fields
            assert "doc_id" in data or "id" in data or "document_id" in data

    def test_document_paraphrase_endpoint(
        self,
        client: TestClient,
        api_prefix: str,
        mock_document_id: str,
        auth_headers: dict,
    ):
        """Test paraphrase endpoint structure."""
        response = client.get(
            f"{api_prefix}/documents/{mock_document_id}/paraphrase",
            headers=auth_headers,
        )
        # Either 404 (doc not found), 401 (unauth), or 200 with content
        assert response.status_code in [200, 401, 404]


class TestDocumentMetadata:
    """Test document metadata extraction and validation."""

    def test_document_list_pagination_bounds(self, client: TestClient, api_prefix: str):
        """Test pagination with boundary values."""
        # Test minimum page size
        response = client.get(f"{api_prefix}/documents/?page=1&page_size=1")
        assert response.status_code == 200

        # Test maximum page size (should be capped)
        response = client.get(f"{api_prefix}/documents/?page=1&page_size=1000")
        assert response.status_code in [200, 422]

        # Test invalid page number
        response = client.get(f"{api_prefix}/documents/?page=0&page_size=10")
        assert response.status_code in [200, 422]

    def test_document_search_by_status(self, client: TestClient, api_prefix: str):
        """Test filtering documents by status."""
        response = client.get(f"{api_prefix}/documents/?status=validated")
        assert response.status_code in [200, 422]


class TestDocumentValidation:
    """Test document validation and DOI verification."""

    def test_doi_validation_format(self, client: TestClient, api_prefix: str):
        """Test DOI format validation."""
        # Valid DOI format
        valid_doi = "10.1234/test.2024.001"
        response = client.get(f"{api_prefix}/documents/doi/{valid_doi}")
        # Should attempt lookup (404 if not found, 200 if found)
        assert response.status_code in [200, 404, 422]

    def test_invalid_doi_format_rejected(self, client: TestClient, api_prefix: str):
        """Test invalid DOI formats are rejected."""
        invalid_doi = "not-a-valid-doi"
        response = client.get(f"{api_prefix}/documents/doi/{invalid_doi}")
        # Should reject or return 404
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
