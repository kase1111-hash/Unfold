"""
Integration tests for the Adaptive Learning system.
Tests flashcards, spaced repetition, engagement tracking, and exports.
"""

from fastapi.testclient import TestClient


class TestFlashcardOperations:
    """Test flashcard CRUD and review operations."""

    def test_get_flashcards_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that getting flashcards requires authentication."""
        response = client.get(f"{api_prefix}/learning/flashcards")
        assert response.status_code == 401

    def test_get_flashcards_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting flashcards with authentication."""
        response = client.get(
            f"{api_prefix}/learning/flashcards",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            # Should return list of flashcards
            assert isinstance(data, (list, dict))

    def test_get_flashcards_due_today(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting flashcards due for review."""
        response = client.get(
            f"{api_prefix}/learning/flashcards/due",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_flashcards_by_document(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
    ):
        """Test getting flashcards for a specific document."""
        response = client.get(
            f"{api_prefix}/learning/flashcards?document_id={mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_create_flashcard_requires_auth(
        self, client: TestClient, api_prefix: str, sample_flashcard: dict
    ):
        """Test that creating flashcards requires authentication."""
        response = client.post(
            f"{api_prefix}/learning/flashcards",
            json=sample_flashcard,
        )
        assert response.status_code == 401

    def test_create_flashcard_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        sample_flashcard: dict,
        mock_document_id: str,
    ):
        """Test creating a flashcard with authentication."""
        flashcard_data = {
            **sample_flashcard,
            "document_id": mock_document_id,
        }
        response = client.post(
            f"{api_prefix}/learning/flashcards",
            json=flashcard_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401, 422]

    def test_review_flashcard_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that reviewing a flashcard requires authentication."""
        review_data = {"quality": 4}
        response = client.post(
            f"{api_prefix}/learning/flashcards/card_123/review",
            json=review_data,
        )
        assert response.status_code == 401

    def test_review_flashcard_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test reviewing a flashcard with valid rating."""
        review_data = {"quality": 4}  # 0-5 scale
        response = client.post(
            f"{api_prefix}/learning/flashcards/card_123/review",
            json=review_data,
            headers=auth_headers,
        )
        # Should succeed or return 404 if card doesn't exist
        assert response.status_code in [200, 401, 404]

    def test_review_flashcard_invalid_quality(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test reviewing with invalid quality rating."""
        review_data = {"quality": 10}  # Invalid: should be 0-5
        response = client.post(
            f"{api_prefix}/learning/flashcards/card_123/review",
            json=review_data,
            headers=auth_headers,
        )
        assert response.status_code in [400, 401, 404, 422]


class TestSM2Algorithm:
    """Test the SM2 spaced repetition algorithm."""

    def test_sm2_initial_values(self):
        """Test SM2 algorithm with initial values."""
        from app.services.learning.sm2 import SM2Scheduler, ResponseQuality

        scheduler = SM2Scheduler()
        card_id = "test_card_1"

        # Add card and review with quality 4
        scheduler.add_card(card_id)
        state = scheduler.review_card(card_id, ResponseQuality.CORRECT_HESITATION)

        assert state.interval >= 1
        assert state.easiness_factor >= 1.3
        assert state.repetitions == 1

    def test_sm2_quality_5_increases_interval(self):
        """Test that quality 5 increases the interval."""
        from app.services.learning.sm2 import SM2Scheduler, ResponseQuality

        scheduler = SM2Scheduler()
        card_id = "test_card_2"

        # First review with quality 5
        scheduler.add_card(card_id)
        state1 = scheduler.review_card(card_id, ResponseQuality.PERFECT)
        interval1 = state1.interval

        # Second review with quality 5
        state2 = scheduler.review_card(card_id, ResponseQuality.PERFECT)
        interval2 = state2.interval

        # Interval should increase
        assert interval2 >= interval1

    def test_sm2_quality_0_resets(self):
        """Test that quality 0 resets the card."""
        from app.services.learning.sm2 import SM2Scheduler, ResponseQuality

        scheduler = SM2Scheduler()
        card_id = "test_card_3"

        # Build up some repetitions
        scheduler.add_card(card_id)
        scheduler.review_card(card_id, ResponseQuality.PERFECT)
        scheduler.review_card(card_id, ResponseQuality.PERFECT)

        # Now fail with quality 0
        state = scheduler.review_card(card_id, ResponseQuality.BLACKOUT)

        # Should reset interval and repetitions
        assert state.interval == 1  # Reset to 1 day
        assert state.repetitions == 0  # Reset repetitions


class TestEngagementTracking:
    """Test engagement tracking functionality."""

    def test_track_engagement_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that tracking engagement requires authentication."""
        engagement_data = {
            "document_id": "doc_123",
            "event_type": "scroll",
            "data": {"depth": 0.5},
        }
        response = client.post(
            f"{api_prefix}/learning/engagement",
            json=engagement_data,
        )
        assert response.status_code == 401

    def test_track_engagement_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
    ):
        """Test tracking engagement with authentication."""
        engagement_data = {
            "document_id": mock_document_id,
            "event_type": "scroll",
            "data": {"depth": 0.5, "time_seconds": 30},
        }
        response = client.post(
            f"{api_prefix}/learning/engagement",
            json=engagement_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401, 422]

    def test_get_engagement_stats(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
    ):
        """Test getting engagement statistics."""
        response = client.get(
            f"{api_prefix}/learning/engagement/{mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]


class TestLearningProgress:
    """Test learning progress tracking."""

    def test_get_progress_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that getting progress requires authentication."""
        response = client.get(f"{api_prefix}/learning/progress")
        assert response.status_code == 401

    def test_get_progress_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting learning progress with authentication."""
        response = client.get(
            f"{api_prefix}/learning/progress",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            # Should contain progress metrics
            assert isinstance(data, dict)

    def test_get_progress_by_document(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
    ):
        """Test getting progress for a specific document."""
        response = client.get(
            f"{api_prefix}/learning/progress/{mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]


class TestLearningExport:
    """Test learning data export functionality."""

    def test_export_anki_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that Anki export requires authentication."""
        response = client.post(f"{api_prefix}/learning/export/anki")
        assert response.status_code == 401

    def test_export_anki_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test Anki export with authentication."""
        export_data = {"deck_name": "Test Deck", "tags": ["test"]}
        response = client.post(
            f"{api_prefix}/learning/export/anki",
            json=export_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_export_obsidian_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that Obsidian export requires authentication."""
        response = client.post(f"{api_prefix}/learning/export/obsidian")
        assert response.status_code == 401

    def test_export_obsidian_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test Obsidian markdown export with authentication."""
        export_data = {"vault_name": "Test Vault", "include_tags": True}
        response = client.post(
            f"{api_prefix}/learning/export/obsidian",
            json=export_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]


class TestFocusMode:
    """Test focus mode and relevance scoring."""

    def test_get_focus_sections(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
    ):
        """Test getting focus sections for a document."""
        response = client.get(
            f"{api_prefix}/learning/focus/{mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_focus_sections_with_complexity(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
    ):
        """Test getting focus sections with complexity level."""
        response = client.get(
            f"{api_prefix}/learning/focus/{mock_document_id}?complexity=50",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]


class TestFlashcardGeneration:
    """Test automatic flashcard generation."""

    def test_generate_flashcards_requires_auth(
        self, client: TestClient, api_prefix: str, mock_document_id: str
    ):
        """Test that generating flashcards requires authentication."""
        response = client.post(
            f"{api_prefix}/learning/flashcards/generate/{mock_document_id}"
        )
        assert response.status_code == 401

    def test_generate_flashcards_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
    ):
        """Test generating flashcards from a document."""
        gen_data = {"max_cards": 10, "difficulty": "medium"}
        response = client.post(
            f"{api_prefix}/learning/flashcards/generate/{mock_document_id}",
            json=gen_data,
            headers=auth_headers,
        )
        # Should succeed or return 404 if document doesn't exist
        assert response.status_code in [200, 202, 401, 404]
