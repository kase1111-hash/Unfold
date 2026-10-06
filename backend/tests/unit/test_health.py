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
        services = data["services"]
        assert services["postgresql"]["connected"] is True
        assert "neo4j" in services
        assert "vector_store" in services
        expected = (
            "healthy" if all(s["connected"] for s in services.values()) else "degraded"
        )
        assert data["status"] == expected

    def test_detailed_health_reports_vector_count(
        self, client: TestClient, api_prefix: str
    ):
        """vector_count is an int, even with several vectors in the index."""
        import numpy as np

        from app.db.vector import faiss_add_vectors, faiss_delete

        ids = ["health_vec_1", "health_vec_2"]
        vectors = np.random.rand(2, 3072).astype("float32")
        client.portal.call(faiss_add_vectors, vectors.tolist(), ids, [{}, {}])
        try:
            response = client.get(f"{api_prefix}/health/detailed")
            assert response.status_code == 200
            count = response.json()["services"]["vector_store"]["vector_count"]
            assert isinstance(count, int) and not isinstance(count, bool)
            assert count >= 2
        finally:
            client.portal.call(faiss_delete, ids)

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
