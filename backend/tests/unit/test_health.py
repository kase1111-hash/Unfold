"""Tests for health check endpoints."""

import logging
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes import health as health_routes
from app.config import get_settings

# Raw failure text of the kind the database checks produce: internal hosts,
# resolved addresses, users and paths.
RAW_ERRORS = {
    "postgresql": 'password authentication failed for user "unfold" (10.20.30.40:5432)',
    "neo4j": (
        "Service unavailable: Couldn't connect to neo4j-internal:7687 "
        "(resolved to ('10.20.30.41:7687',))"
    ),
    "vector_store": "/var/lib/unfold/faiss/index.bin: Permission denied",
}


@pytest.fixture
def services_down(monkeypatch):
    """Every check reports a failure with raw exception text."""

    def failing(status: str, message: str):
        async def check():
            return {"connected": False, "status": status, "message": message}

        return check

    monkeypatch.setattr(
        health_routes,
        "check_postgres_connection",
        failing("error", RAW_ERRORS["postgresql"]),
    )
    monkeypatch.setattr(
        health_routes,
        "check_neo4j_connection",
        failing("unavailable", RAW_ERRORS["neo4j"]),
    )
    monkeypatch.setattr(
        health_routes,
        "check_faiss_connection",
        failing("error", RAW_ERRORS["vector_store"]),
    )


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


class TestHealthErrorDetails:
    """/health/detailed is unauthenticated: it must not echo raw errors."""

    def test_detailed_health_hides_failure_details(
        self, client: TestClient, api_prefix: str, services_down, caplog
    ):
        caplog.set_level(logging.WARNING, logger="app.api.v1.routes.health")
        response = client.get(f"{api_prefix}/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["services"] == {
            "postgresql": {
                "connected": False,
                "status": "error",
                "message": "PostgreSQL is unavailable",
            },
            "neo4j": {
                "connected": False,
                "status": "unavailable",
                "message": "Neo4j is unavailable",
            },
            "vector_store": {
                "connected": False,
                "status": "error",
                "message": "Vector store is unavailable",
            },
        }
        for raw in RAW_ERRORS.values():
            assert raw not in response.text
        for fragment in ("10.20.30.4", "neo4j-internal", "/var/lib", "unfold"):
            assert fragment not in response.text

        # The details still reach the server log.
        for raw in RAW_ERRORS.values():
            assert raw in caplog.text

    def test_healthy_service_keeps_its_details(
        self, client: TestClient, api_prefix: str
    ):
        services = client.get(f"{api_prefix}/health/detailed").json()["services"]
        assert services["postgresql"] == {
            "connected": True,
            "status": "healthy",
            "message": "PostgreSQL connection successful",
        }

    async def test_postgres_check_returns_generic_message(self, monkeypatch, caplog):
        from app.db import postgres

        class BrokenEngine:
            def connect(self):
                raise OSError("could not connect to 10.20.30.40:5432 as unfold")

        caplog.set_level(logging.WARNING, logger="app.db.postgres")
        monkeypatch.setattr(postgres, "_engine", BrokenEngine())

        result = await postgres.check_postgres_connection()

        assert result == {
            "connected": False,
            "status": "error",
            "message": "PostgreSQL connection failed",
        }
        assert "could not connect to 10.20.30.40:5432" in caplog.text

    async def test_neo4j_check_returns_generic_message(self, monkeypatch, caplog):
        from neo4j.exceptions import ServiceUnavailable

        from app.db import neo4j as neo4j_db

        class BrokenSession:
            async def __aenter__(self):
                raise ServiceUnavailable(
                    "Couldn't connect to graph-internal.example:7687 (resolved to 10.20.30.40)"
                )

            async def __aexit__(self, *exc):
                return False

        class BrokenDriver:
            def session(self):
                return BrokenSession()

        caplog.set_level(logging.WARNING, logger="app.db.neo4j")
        monkeypatch.setattr(neo4j_db, "_driver", BrokenDriver())

        result = await neo4j_db.check_neo4j_connection()

        assert result == {
            "connected": False,
            "status": "unavailable",
            "message": "Neo4j service unavailable",
        }
        assert "10.20.30.40" in caplog.text

    @pytest.mark.requires_no_neo4j
    def test_neo4j_down_does_not_leak_its_address(
        self, client: TestClient, api_prefix: str
    ):
        response = client.get(f"{api_prefix}/health/detailed")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["services"]["neo4j"] == {
            "connected": False,
            "status": "unavailable",
            "message": "Neo4j is unavailable",
        }
        uri = urlparse(get_settings().neo4j_uri)
        assert f"{uri.hostname}:{uri.port}" not in response.text
        assert "Couldn't connect" not in response.text
        assert "resolved to" not in response.text
