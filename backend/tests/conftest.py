"""Pytest configuration and fixtures for Unfold tests."""

import os
import socket
import uuid
from urllib.parse import urlparse

# Test-environment defaults. These MUST be set before any `app.*` import,
# because Settings is cached (lru_cache) and many modules capture
# `settings = get_settings()` at import time. setdefault() lets CI or a
# developer override any of them explicitly.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("JWT_SECRET", "test-secret-key")
os.environ.setdefault(
    "DATABASE_URL", "postgresql://test:test@localhost:5432/unfold_test"
)
# The in-memory limiter (10 req/min on /auth/*) would 429 the suite, since
# every test shares the "testclient" IP.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import create_tables, drop_tables  # noqa: E402
from app.main import app  # noqa: E402
from tests.pdf_utils import make_text_pdf  # noqa: E402


def _neo4j_reachable() -> bool:
    uri = urlparse(get_settings().neo4j_uri)
    try:
        with socket.create_connection(
            (uri.hostname or "localhost", uri.port or 7687), timeout=1
        ):
            return True
    except OSError:
        return False


NEO4J_UP = _neo4j_reachable()


def pytest_collection_modifyitems(config, items):
    """Skip graph tests that need (or need the absence of) a live Neo4j."""
    skip_up = pytest.mark.skip(reason="needs a reachable Neo4j")
    skip_down = pytest.mark.skip(reason="checks behaviour when Neo4j is down")
    for item in items:
        if "requires_neo4j" in item.keywords and not NEO4J_UP:
            item.add_marker(skip_up)
        if "requires_no_neo4j" in item.keywords and NEO4J_UP:
            item.add_marker(skip_down)


@pytest.fixture(scope="session")
def _session_client():
    """One TestClient for the whole session, used as a context manager.

    Entering the context runs the app lifespan once and pins every request
    to one event loop; without it Starlette starts a new loop per request
    and pooled asyncpg connections fail with "Event loop is closed".
    """
    settings = get_settings()
    database = make_url(str(settings.database_url)).database or ""
    # The schema is dropped below, so never run against a non-test database.
    if settings.environment != "test" or not database.endswith("_test"):
        pytest.exit(
            "Refusing to drop/recreate tables: "
            f"ENVIRONMENT={settings.environment!r}, database={database!r}. "
            "Point DATABASE_URL at a database whose name ends in '_test'.",
            returncode=2,
        )
    with TestClient(app) as c:
        c.portal.call(drop_tables)
        c.portal.call(create_tables)
        yield c


@pytest.fixture
def client(_session_client: TestClient) -> TestClient:
    """Shared client with a clean cookie jar (no refresh-token leaks)."""
    _session_client.cookies.clear()
    return _session_client


@pytest.fixture
def api_prefix() -> str:
    """API version prefix."""
    return "/api/v1"


@pytest.fixture
def mock_document_id() -> str:
    """Generate a mock document ID."""
    return f"doc_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def sample_document_content() -> str:
    """Sample document content for testing."""
    return """
    # Understanding Quantum Computing

    Quantum computing represents a fundamental shift in how we process information.
    Unlike classical computers that use bits, quantum computers use quantum bits or qubits.

    ## Key Concepts

    Superposition allows qubits to exist in multiple states simultaneously.
    Entanglement creates correlations between qubits that persist across distances.

    The implications for cryptography and drug discovery are profound.
    Researchers at MIT and Google have made significant breakthroughs in this field.
    """


def _register_user(client: TestClient, api_prefix: str) -> dict:
    suffix = uuid.uuid4().hex[:8]
    response = client.post(
        f"{api_prefix}/auth/register",
        json={
            "email": f"test_{suffix}@example.com",
            "username": f"testuser_{suffix}",
            "password": "TestPassword123!",
            "full_name": "Test User",
        },
    )
    assert response.status_code == 201, (
        f"Auth setup failed: register returned {response.status_code}: {response.text}"
    )
    token = response.json().get("access_token")
    assert token, "Register succeeded but no access_token in response"
    # Registration sets a refresh cookie; don't let it leak into the test.
    client.cookies.clear()
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers(client: TestClient, api_prefix: str) -> dict:
    """Bearer headers for a freshly registered user."""
    return _register_user(client, api_prefix)


@pytest.fixture
def other_auth_headers(client: TestClient, api_prefix: str) -> dict:
    """Bearer headers for a second, unrelated user (ownership tests)."""
    return _register_user(client, api_prefix)


@pytest.fixture
def text_pdf() -> bytes:
    """A real PDF with extractable text containing named entities."""
    return make_text_pdf()


@pytest.fixture
def uploaded_document(
    client: TestClient, api_prefix: str, auth_headers: dict, text_pdf: bytes
) -> dict:
    """Upload ``text_pdf`` as the ``auth_headers`` user; return the document."""
    response = client.post(
        f"{api_prefix}/documents/upload",
        files={"file": ("curie.pdf", text_pdf, "application/pdf")},
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["document"]


@pytest.fixture
def sample_flashcard() -> dict:
    """Sample flashcard data for testing."""
    return {
        "question": "What is quantum superposition?",
        "answer": "The ability of a quantum system to exist in multiple states simultaneously until measured.",
        "difficulty": "medium",
        "tags": ["quantum", "physics", "fundamentals"],
    }
