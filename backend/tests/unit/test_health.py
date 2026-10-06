"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes import health as health_routes


@pytest.fixture
def postgres_down(monkeypatch):
    """Make the routes see PostgreSQL as unreachable."""

    async def check_postgres_connection():
        return {
            "connected": False,
            "status": "error",
            "message": "PostgreSQL connection failed",
        }

    monkeypatch.setattr(
        health_routes, "check_postgres_connection", check_postgres_connection
    )


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

    def test_health_check_503_when_postgres_down(
        self, client: TestClient, api_prefix: str, postgres_down
    ):
        """A non-2xx status is what makes Docker's `curl -f` healthcheck fail."""
        response = client.get(f"{api_prefix}/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["environment"] == "test"

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

    def test_readiness_probe_503_when_postgres_down(
        self, client: TestClient, api_prefix: str, postgres_down
    ):
        response = client.get(f"{api_prefix}/health/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready"}

    def test_liveness_probe_ignores_postgres(
        self, client: TestClient, api_prefix: str, postgres_down
    ):
        """Liveness only says the process is up; restarting won't fix the DB."""
        response = client.get(f"{api_prefix}/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_liveness_probe(self, client: TestClient, api_prefix: str):
        """Test Kubernetes liveness probe."""
        response = client.get(f"{api_prefix}/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
