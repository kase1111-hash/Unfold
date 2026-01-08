"""Tests for document endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestDocumentEndpoints:
    """Tests for document management endpoints."""

    def test_list_documents_empty(self, client: TestClient, api_prefix: str):
        """Test listing documents returns empty list initially."""
        response = client.get(f"{api_prefix}/documents/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] == []
        assert data["total"] == 0

    def test_list_documents_pagination(self, client: TestClient, api_prefix: str):
        """Test document listing respects pagination parameters."""
        response = client.get(f"{api_prefix}/documents/?page=2&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10

    def test_list_documents_max_page_size(self, client: TestClient, api_prefix: str):
        """Test document listing caps page size at 100."""
        response = client.get(f"{api_prefix}/documents/?page_size=200")
        assert response.status_code == 200
        data = response.json()
        assert data["page_size"] == 100

    def test_get_document_not_found(self, client: TestClient, api_prefix: str):
        """Test getting non-existent document returns 404."""
        response = client.get(f"{api_prefix}/documents/sha256:nonexistent")
        assert response.status_code == 404

    def test_delete_document_not_found(self, client: TestClient, api_prefix: str):
        """Test deleting non-existent document returns 404."""
        response = client.delete(f"{api_prefix}/documents/sha256:nonexistent")
        assert response.status_code == 404

    def test_upload_invalid_file_type(self, client: TestClient, api_prefix: str):
        """Test uploading unsupported file type returns 415."""
        response = client.post(
            f"{api_prefix}/documents/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
        )
        assert response.status_code == 415
