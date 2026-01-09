"""
Integration tests for the Ethics system.
Tests provenance, bias auditing, privacy compliance, and analytics.
"""

import pytest
from fastapi.testclient import TestClient


class TestProvenanceTracking:
    """Test content provenance and C2PA manifest functionality."""

    def test_create_provenance_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that creating provenance requires authentication."""
        request_data = {
            "document_id": "doc_123",
            "content": "Test document content",
        }
        response = client.post(f"{api_prefix}/ethics/provenance/create", json=request_data)
        assert response.status_code == 401

    def test_create_provenance_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
        sample_document_content: str,
    ):
        """Test creating content provenance with authentication."""
        request_data = {
            "document_id": mock_document_id,
            "content": sample_document_content,
        }
        response = client.post(
            f"{api_prefix}/ethics/provenance/create",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401]
        if response.status_code in [200, 201]:
            data = response.json()
            # Should return credential with hash
            assert "credential_id" in data or "content_hash" in data

    def test_verify_provenance_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that verifying provenance requires authentication."""
        request_data = {
            "credential_id": "cred_123",
            "content": "Test content",
        }
        response = client.post(f"{api_prefix}/ethics/provenance/verify", json=request_data)
        assert response.status_code == 401

    def test_verify_provenance_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test verifying content provenance with authentication."""
        request_data = {
            "credential_id": "cred_123",
            "content": "Test content to verify",
        }
        response = client.post(
            f"{api_prefix}/ethics/provenance/verify",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_add_assertion_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test adding an assertion to a credential."""
        request_data = {
            "credential_id": "cred_123",
            "assertion_type": "modified",
            "actor": "test_user",
            "description": "Document was processed by AI",
        }
        response = client.post(
            f"{api_prefix}/ethics/provenance/assertion",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_provenance(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting provenance information."""
        response = client.get(
            f"{api_prefix}/ethics/provenance/cred_123",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_document_provenance(
        self, client: TestClient, api_prefix: str, auth_headers: dict, mock_document_id: str
    ):
        """Test getting all provenance for a document."""
        response = client.get(
            f"{api_prefix}/ethics/provenance/document/{mock_document_id}",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_create_manifest_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
        sample_document_content: str,
    ):
        """Test creating a provenance manifest."""
        request_data = {
            "document_id": mock_document_id,
            "title": "Test Document",
            "content": sample_document_content,
            "authors": ["John Doe"],
            "doi": "10.1234/test",
        }
        response = client.post(
            f"{api_prefix}/ethics/provenance/manifest",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401]


class TestBiasAuditing:
    """Test bias detection and audit functionality."""

    def test_audit_document_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that bias audit requires authentication."""
        request_data = {
            "document_id": "doc_123",
            "content": "Test content",
        }
        response = client.post(f"{api_prefix}/ethics/bias/audit", json=request_data)
        assert response.status_code == 401

    def test_audit_document_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
        sample_bias_content: str,
    ):
        """Test performing bias audit with authentication."""
        request_data = {
            "document_id": mock_document_id,
            "content": sample_bias_content,
        }
        response = client.post(
            f"{api_prefix}/ethics/bias/audit",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            # Should contain findings and metrics
            assert "findings" in data or "report_id" in data

    def test_audit_document_with_sections(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        mock_document_id: str,
        sample_bias_content: str,
    ):
        """Test bias audit with section breakdown."""
        request_data = {
            "document_id": mock_document_id,
            "content": sample_bias_content,
            "sections": ["Introduction", "Methods"],
        }
        response = client.post(
            f"{api_prefix}/ethics/bias/audit",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_analyze_sentiment(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test sentiment analysis endpoint."""
        response = client.post(
            f"{api_prefix}/ethics/bias/sentiment?text=This%20is%20great%20news!",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 422]

    def test_get_bias_categories(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting available bias categories."""
        response = client.get(
            f"{api_prefix}/ethics/bias/categories",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]


class TestPrivacyCompliance:
    """Test GDPR compliance and privacy functionality."""

    def test_record_consent_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that recording consent requires authentication."""
        request_data = {
            "consent_type": "analytics",
            "granted": True,
        }
        response = client.post(f"{api_prefix}/ethics/privacy/consent", json=request_data)
        assert response.status_code == 401

    def test_record_consent_with_auth(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        sample_consent_request: dict,
    ):
        """Test recording consent with authentication."""
        response = client.post(
            f"{api_prefix}/ethics/privacy/consent",
            json=sample_consent_request,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401]

    def test_withdraw_consent_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test withdrawing consent."""
        request_data = {"consent_type": "analytics"}
        response = client.post(
            f"{api_prefix}/ethics/privacy/consent/withdraw",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401, 404]

    def test_get_consents(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting all consent records."""
        response = client.get(
            f"{api_prefix}/ethics/privacy/consents",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_get_privacy_report(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test generating privacy report (DSAR)."""
        response = client.get(
            f"{api_prefix}/ethics/privacy/report",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_request_deletion_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that deletion request requires authentication."""
        response = client.post(f"{api_prefix}/ethics/privacy/delete")
        assert response.status_code == 401

    def test_request_deletion_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test requesting data deletion (right to be forgotten)."""
        response = client.post(
            f"{api_prefix}/ethics/privacy/delete",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_export_data_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that data export requires authentication."""
        response = client.get(f"{api_prefix}/ethics/privacy/export")
        assert response.status_code == 401

    def test_export_data_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test exporting user data (data portability)."""
        response = client.get(
            f"{api_prefix}/ethics/privacy/export",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_get_retention_policy(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting data retention policy."""
        response = client.get(
            f"{api_prefix}/ethics/privacy/retention/personal",
            headers=auth_headers,
        )
        assert response.status_code in [200, 400, 401]


class TestEthicsAnalytics:
    """Test ethics analytics and transparency dashboard."""

    def test_record_operation_requires_auth(self, client: TestClient, api_prefix: str):
        """Test that recording operations requires authentication."""
        request_data = {
            "operation_type": "ai_summary",
            "purpose": "Generated document summary",
        }
        response = client.post(f"{api_prefix}/ethics/analytics/operation", json=request_data)
        assert response.status_code == 401

    def test_record_operation_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test recording AI operation with authentication."""
        request_data = {
            "operation_type": "ai_summary",
            "purpose": "Generated document summary",
            "model_used": "claude-3",
            "input_tokens": 1500,
            "output_tokens": 500,
            "confidence_score": 0.95,
        }
        response = client.post(
            f"{api_prefix}/ethics/analytics/operation",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401]

    def test_record_metric_with_auth(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test recording ethics metric."""
        request_data = {
            "metric_type": "ai_usage",
            "name": "token_count",
            "value": 2000,
            "unit": "tokens",
        }
        response = client.post(
            f"{api_prefix}/ethics/analytics/metric",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201, 401]

    def test_get_operations(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting operation history."""
        response = client.get(
            f"{api_prefix}/ethics/analytics/operations?limit=50",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_get_operations_filtered(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting operations filtered by type."""
        response = client.get(
            f"{api_prefix}/ethics/analytics/operations?operation_type=ai_summary",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_get_dashboard(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting ethics dashboard."""
        response = client.get(
            f"{api_prefix}/ethics/analytics/dashboard?period_days=30",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            # Should contain summary metrics
            assert "summary" in data or "ai_operations_count" in data

    def test_get_ethics_profile(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting user ethics profile."""
        response = client.get(
            f"{api_prefix}/ethics/analytics/profile",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_update_preferences(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test updating ethics preferences."""
        request_data = {
            "transparency_level": "full",
            "receive_reports": True,
            "allow_aggregated": True,
        }
        response = client.put(
            f"{api_prefix}/ethics/analytics/preferences",
            json=request_data,
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]

    def test_get_aggregated_report(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Test getting aggregated platform ethics report."""
        response = client.get(
            f"{api_prefix}/ethics/analytics/aggregated?period_days=30",
            headers=auth_headers,
        )
        assert response.status_code in [200, 401]


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms."""

    def test_privacy_noise_mechanism(self):
        """Test that differential privacy adds appropriate noise."""
        from app.services.ethics.privacy import DifferentialPrivacy

        dp = DifferentialPrivacy(epsilon=1.0)

        # Add noise to a value
        original = 100.0
        noisy_values = [dp.add_laplace_noise(original, sensitivity=1.0) for _ in range(100)]

        # Values should be distributed around the original
        mean_noisy = sum(noisy_values) / len(noisy_values)
        assert abs(mean_noisy - original) < 10  # Should be close to original

        # But individual values should differ
        assert noisy_values[0] != noisy_values[1]

    def test_randomized_response(self):
        """Test randomized response mechanism for booleans."""
        from app.services.ethics.privacy import DifferentialPrivacy

        dp = DifferentialPrivacy(epsilon=1.0)

        # Run many trials
        true_count = sum(dp.randomized_response(True) for _ in range(1000))
        false_count = sum(dp.randomized_response(False) for _ in range(1000))

        # True values should produce more true responses than false values
        assert true_count > false_count


class TestProvenanceService:
    """Test provenance service directly."""

    def test_compute_content_hash(self, sample_document_content: str):
        """Test content hash computation."""
        from app.services.ethics.provenance import ProvenanceService

        service = ProvenanceService()
        hash1 = service.compute_content_hash(sample_document_content)
        hash2 = service.compute_content_hash(sample_document_content)

        # Same content should produce same hash
        assert hash1 == hash2
        # Hash should be 64 characters (SHA-256 hex)
        assert len(hash1) == 64

    def test_create_and_verify_credential(self, sample_document_content: str):
        """Test creating and verifying a credential."""
        from app.services.ethics.provenance import ProvenanceService

        service = ProvenanceService()

        # Create credential
        credential = service.create_credential(
            document_id="test_doc",
            content=sample_document_content,
            actor="test_user",
        )

        assert credential.document_id == "test_doc"
        assert credential.content_hash is not None

        # Verify with same content
        valid, message = service.verify_content_integrity(
            credential.credential_id,
            sample_document_content,
        )
        assert valid is True

        # Verify with modified content should fail
        valid, message = service.verify_content_integrity(
            credential.credential_id,
            sample_document_content + " modified",
        )
        assert valid is False
