"""JWT token creation and verification."""

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.config import get_settings
from app.models.user import Token, TokenPayload

settings = get_settings()


class TokenError(Exception):
    """Exception raised for token-related errors."""

    pass


def create_access_token(
    subject: str,
    expires_delta: timedelta | None = None,
    additional_claims: dict | None = None,
) -> str:
    """Create a JWT access token.

    Args:
        subject: Token subject (usually user ID)
        expires_delta: Optional custom expiration time
        additional_claims: Optional additional JWT claims

    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_expiration_minutes)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    to_encode = {
        "sub": subject,
        "exp": expire,
        "iat": now,
        "type": "access",
    }

    if additional_claims:
        to_encode.update(additional_claims)

    return jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(
    subject: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT refresh token.

    Args:
        subject: Token subject (usually user ID)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT refresh token string
    """
    if expires_delta is None:
        expires_delta = timedelta(days=settings.jwt_refresh_expiration_days)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    to_encode = {
        "sub": subject,
        "exp": expire,
        "iat": now,
        "type": "refresh",
    }

    return jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def create_token_pair(subject: str) -> Token:
    """Create both access and refresh tokens.

    Args:
        subject: Token subject (usually user ID)

    Returns:
        Token object with access_token, refresh_token, and metadata
    """
    access_token = create_access_token(subject)
    refresh_token = create_refresh_token(subject)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.jwt_expiration_minutes * 60,
    )


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token.

    Args:
        token: JWT token string to decode

    Returns:
        TokenPayload with decoded claims

    Raises:
        TokenError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )

        # Extract claims
        subject = payload.get("sub")
        exp = payload.get("exp")
        iat = payload.get("iat")
        token_type = payload.get("type", "access")

        if subject is None:
            raise TokenError("Token missing subject claim")

        # Convert timestamps to datetime
        exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None
        iat_dt = datetime.fromtimestamp(iat, tz=timezone.utc) if iat else None

        if exp_dt is None or iat_dt is None:
            raise TokenError("Token missing required timestamp claims")

        return TokenPayload(
            sub=subject,
            exp=exp_dt,
            iat=iat_dt,
            type=token_type,
        )

    except JWTError as e:
        raise TokenError(f"Invalid token: {e}")


def verify_token(token: str, token_type: str = "access") -> TokenPayload:
    """Verify a JWT token and check its type.

    Args:
        token: JWT token string to verify
        token_type: Expected token type ("access" or "refresh")

    Returns:
        TokenPayload if valid

    Raises:
        TokenError: If token is invalid, expired, or wrong type
    """
    payload = decode_token(token)

    if payload.type != token_type:
        raise TokenError(f"Invalid token type: expected {token_type}, got {payload.type}")

    # Check expiration
    if payload.exp < datetime.now(timezone.utc):
        raise TokenError("Token has expired")

    return payload
