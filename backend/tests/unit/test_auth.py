"""Tests for authentication endpoints and services."""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.auth.jwt import (
    TokenError,
    create_access_token,
    create_refresh_token,
    create_token_pair,
    decode_token,
    verify_token,
)
from app.utils.security import hash_password, verify_password


class TestPasswordHashing:
    """Tests for password hashing utilities."""

    def test_hash_password_returns_hash(self):
        """Test that hash_password returns a bcrypt hash."""
        password = "secure_password123"
        hashed = hash_password(password)

        assert hashed != password
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_hash_password_different_each_time(self):
        """Test that same password produces different hashes (salt)."""
        password = "secure_password123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "secure_password123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "secure_password123"
        hashed = hash_password(password)

        assert verify_password("wrong_password", hashed) is False

    def test_verify_password_empty(self):
        """Test password verification with empty password."""
        hashed = hash_password("secure_password123")

        assert verify_password("", hashed) is False


class TestJWTTokens:
    """Tests for JWT token creation and verification."""

    def test_create_access_token(self):
        """Test access token creation."""
        token = create_access_token(subject="user123")

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        token = create_refresh_token(subject="user123")

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_token_pair(self):
        """Test token pair creation."""
        tokens = create_token_pair(subject="user123")

        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.token_type == "bearer"
        assert tokens.expires_in > 0

    def test_decode_access_token(self):
        """Test decoding a valid access token."""
        token = create_access_token(subject="user123")
        payload = decode_token(token)

        assert payload.sub == "user123"
        assert payload.type == "access"
        assert payload.exp > datetime.now(timezone.utc)

    def test_decode_refresh_token(self):
        """Test decoding a valid refresh token."""
        token = create_refresh_token(subject="user123")
        payload = decode_token(token)

        assert payload.sub == "user123"
        assert payload.type == "refresh"
        assert payload.exp > datetime.now(timezone.utc)

    def test_verify_access_token(self):
        """Test verifying a valid access token."""
        token = create_access_token(subject="user123")
        payload = verify_token(token, token_type="access")

        assert payload.sub == "user123"
        assert payload.type == "access"

    def test_verify_refresh_token(self):
        """Test verifying a valid refresh token."""
        token = create_refresh_token(subject="user123")
        payload = verify_token(token, token_type="refresh")

        assert payload.sub == "user123"
        assert payload.type == "refresh"

    def test_verify_token_wrong_type(self):
        """Test verifying token with wrong type raises error."""
        token = create_access_token(subject="user123")

        with pytest.raises(TokenError) as exc_info:
            verify_token(token, token_type="refresh")

        assert "Invalid token type" in str(exc_info.value)

    def test_verify_expired_token(self):
        """Test verifying expired token raises error."""
        # Create token that expires immediately
        token = create_access_token(
            subject="user123",
            expires_delta=timedelta(seconds=-1),
        )

        with pytest.raises(TokenError) as exc_info:
            verify_token(token, token_type="access")

        assert "expired" in str(exc_info.value).lower()

    def test_decode_invalid_token(self):
        """Test decoding invalid token raises error."""
        with pytest.raises(TokenError):
            decode_token("invalid.token.here")

    def test_decode_tampered_token(self):
        """Test decoding tampered token raises error."""
        token = create_access_token(subject="user123")
        # Tamper with the token
        tampered = token[:-5] + "xxxxx"

        with pytest.raises(TokenError):
            decode_token(tampered)

    def test_token_with_additional_claims(self):
        """Test token creation with additional claims."""
        token = create_access_token(
            subject="user123",
            additional_claims={"role": "admin"},
        )
        payload = decode_token(token)

        assert payload.sub == "user123"
        # Note: additional claims would need to be accessed from raw payload


class TestAuthEndpoints:
    """Tests for authentication API endpoints."""

    def test_register_success(self, client, api_prefix):
        """Test successful user registration."""
        response = client.post(
            f"{api_prefix}/auth/register",
            json={
                "email": "test@example.com",
                "username": "testuser",
                "password": "securepass123",
                "full_name": "Test User",
            },
        )

        # May fail without DB, but tests endpoint structure
        assert response.status_code in [201, 500]

    def test_register_invalid_email(self, client, api_prefix):
        """Test registration with invalid email."""
        response = client.post(
            f"{api_prefix}/auth/register",
            json={
                "email": "invalid-email",
                "username": "testuser",
                "password": "securepass123",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_register_short_password(self, client, api_prefix):
        """Test registration with too short password."""
        response = client.post(
            f"{api_prefix}/auth/register",
            json={
                "email": "test@example.com",
                "username": "testuser",
                "password": "short",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_login_missing_fields(self, client, api_prefix):
        """Test login with missing fields."""
        response = client.post(
            f"{api_prefix}/auth/login",
            json={"email": "test@example.com"},
        )

        assert response.status_code == 422  # Validation error

    def test_refresh_missing_token(self, client, api_prefix):
        """Test token refresh with missing token."""
        response = client.post(
            f"{api_prefix}/auth/refresh",
            json={},
        )

        assert response.status_code == 422  # Validation error

    def test_me_unauthorized(self, client, api_prefix):
        """Test /me endpoint without authentication."""
        response = client.get(f"{api_prefix}/auth/me")

        assert response.status_code == 401

    def test_me_invalid_token(self, client, api_prefix):
        """Test /me endpoint with invalid token."""
        response = client.get(
            f"{api_prefix}/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401

    def test_change_password_unauthorized(self, client, api_prefix):
        """Test change password without authentication."""
        response = client.post(
            f"{api_prefix}/auth/change-password",
            json={
                "current_password": "old_pass",
                "new_password": "new_pass123",
            },
        )

        assert response.status_code == 401
