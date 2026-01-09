"""Pytest configuration and fixtures for Unfold tests."""

import pytest
from fastapi.testclient import TestClient
from typing import Generator
import uuid
from datetime import datetime

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def api_prefix() -> str:
    """API version prefix."""
    return "/api/v1"


@pytest.fixture
def mock_user_id() -> str:
    """Generate a mock user ID."""
    return f"user_{uuid.uuid4().hex[:12]}"


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


@pytest.fixture
def sample_paper_metadata() -> dict:
    """Sample paper metadata for testing."""
    return {
        "title": "Advances in Quantum Error Correction",
        "authors": ["Alice Smith", "Bob Johnson"],
        "doi": "10.1234/quantum.2024.001",
        "abstract": "This paper presents novel approaches to quantum error correction.",
        "year": 2024,
        "venue": "Nature Physics",
        "citation_count": 42,
    }


@pytest.fixture
def auth_headers(client: TestClient, api_prefix: str) -> dict:
    """
    Get authentication headers for protected endpoints.
    Creates a test user and returns valid JWT tokens.
    """
    # Register a test user
    register_data = {
        "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
        "username": f"testuser_{uuid.uuid4().hex[:8]}",
        "password": "TestPassword123!",
        "full_name": "Test User",
    }

    response = client.post(f"{api_prefix}/auth/register", json=register_data)

    # If registration fails (user exists), try login
    if response.status_code != 201:
        login_data = {
            "username": register_data["email"],
            "password": register_data["password"],
        }
        response = client.post(
            f"{api_prefix}/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code in [200, 201]:
        data = response.json()
        token = data.get("tokens", {}).get("access_token") or data.get("access_token")
        if token:
            return {"Authorization": f"Bearer {token}"}

    # Return empty headers if auth fails (some tests may not need auth)
    return {}


@pytest.fixture
def sample_flashcard() -> dict:
    """Sample flashcard data for testing."""
    return {
        "question": "What is quantum superposition?",
        "answer": "The ability of a quantum system to exist in multiple states simultaneously until measured.",
        "difficulty": "medium",
        "tags": ["quantum", "physics", "fundamentals"],
    }


@pytest.fixture
def sample_annotation() -> dict:
    """Sample annotation data for testing."""
    return {
        "content": "This is a key insight about quantum computing",
        "selected_text": "quantum computers use quantum bits",
        "annotation_type": "highlight",
        "visibility": "private",
        "tags": ["important", "quantum"],
    }


@pytest.fixture
def sample_bias_content() -> str:
    """Sample content with potential bias for testing."""
    return """
    The chairman announced the new policy today.
    Mankind has always sought to understand the universe.
    The crazy deadline forced the team to work overtime.
    """


@pytest.fixture
def sample_consent_request() -> dict:
    """Sample consent request for testing."""
    return {
        "consent_type": "analytics",
        "granted": True,
        "ip_address": "192.168.1.1",
        "user_agent": "Mozilla/5.0 Test Browser",
    }


class MockExternalAPI:
    """Mock external API responses for testing."""

    @staticmethod
    def semantic_scholar_paper() -> dict:
        return {
            "paperId": "abc123",
            "title": "Test Paper",
            "authors": [{"name": "John Doe"}],
            "year": 2024,
            "citationCount": 10,
            "abstract": "Test abstract",
        }

    @staticmethod
    def crossref_work() -> dict:
        return {
            "DOI": "10.1234/test",
            "title": ["Test Paper Title"],
            "author": [{"given": "John", "family": "Doe"}],
            "is-referenced-by-count": 25,
        }

    @staticmethod
    def wikipedia_summary() -> dict:
        return {
            "title": "Quantum Computing",
            "extract": "Quantum computing is a type of computation...",
            "pageid": 12345,
        }


@pytest.fixture
def mock_external_api() -> MockExternalAPI:
    """Provide mock external API responses."""
    return MockExternalAPI()
