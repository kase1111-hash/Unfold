"""Tests for health check endpoints."""

from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data

    def test_health_check(self, client: TestClient, api_prefix: str):
        """Test basic health check returns healthy status."""
        response = client.get(f"{api_prefix}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data

    def test_detailed_health_check(self, client: TestClient, api_prefix: str):
        """Test detailed health check returns service status."""
        response = client.get(f"{api_prefix}/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "postgresql" in data["services"]
        assert "neo4j" in data["services"]
        assert "vector_store" in data["services"]

    def test_readiness_probe(self, client: TestClient, api_prefix: str):
        """Test Kubernetes readiness probe."""
        response = client.get(f"{api_prefix}/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_liveness_probe(self, client: TestClient, api_prefix: str):
        """Test Kubernetes liveness probe."""
        response = client.get(f"{api_prefix}/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
