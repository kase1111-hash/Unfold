"""Authentication endpoints."""

from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies import CurrentUser, get_db
from app.config import get_settings
from app.models.user import Token, User, UserCreate
from app.services.auth.service import AuthError, AuthService

router = APIRouter()
settings = get_settings()

# Cookie settings for refresh token
REFRESH_TOKEN_COOKIE_NAME = "refresh_token"
REFRESH_TOKEN_MAX_AGE = settings.jwt_refresh_expiration_days * 24 * 60 * 60  # in seconds


def _set_refresh_token_cookie(response: Response, refresh_token: str) -> None:
    """Set the refresh token as an httpOnly cookie."""
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=refresh_token,
        max_age=REFRESH_TOKEN_MAX_AGE,
        httponly=True,
        secure=settings.environment in ("production", "staging"),  # HTTPS only in prod
        samesite="lax",
        path="/api/v1/auth",  # Only send to auth endpoints
    )


def _clear_refresh_token_cookie(response: Response) -> None:
    """Clear the refresh token cookie."""
    response.delete_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        path="/api/v1/auth",
    )


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request body."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")


class RegisterRequest(UserCreate):
    """Registration request body (extends UserCreate)."""

    pass


class RefreshRequest(BaseModel):
    """Token refresh request body (optional if using cookie)."""

    refresh_token: str | None = Field(None, description="Valid refresh token (optional if using cookie)")


class AccessTokenResponse(BaseModel):
    """Response with access token only (refresh token in cookie)."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthResponse(BaseModel):
    """Authentication response with user and access token."""

    user: User
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LegacyAuthResponse(BaseModel):
    """Legacy authentication response with both tokens (for backward compatibility)."""

    user: User
    tokens: Token


class ChangePasswordRequest(BaseModel):
    """Change password request body."""

    current_password: str = Field(..., min_length=1, description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class MessageResponse(BaseModel):
    """Simple message response."""

    message: str


# Helper to get auth service
async def get_auth_service(db: Annotated[AsyncSession, Depends(get_db)]) -> AuthService:
    """Get auth service instance."""
    return AuthService(db)


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    request: RegisterRequest,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthResponse:
    """Register a new user account.

    Creates a new user with the provided credentials and returns
    authentication tokens. The refresh token is set as an httpOnly cookie.
    """
    try:
        user, tokens = await auth_service.register(request)

        # Set refresh token as httpOnly cookie
        _set_refresh_token_cookie(response, tokens.refresh_token)

        return AuthResponse(
            user=user,
            access_token=tokens.access_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
        )
    except AuthError as e:
        if e.code == "EMAIL_EXISTS":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": e.code, "message": e.message},
            )
        elif e.code == "USERNAME_EXISTS":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": e.code, "message": e.message},
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login with email and password",
)
async def login(
    request: LoginRequest,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthResponse:
    """Authenticate user with email and password.

    Returns user information and access token. The refresh token is set as an httpOnly cookie.
    """
    try:
        user, tokens = await auth_service.login(request.email, request.password)

        # Set refresh token as httpOnly cookie
        _set_refresh_token_cookie(response, tokens.refresh_token)

        return AuthResponse(
            user=user,
            access_token=tokens.access_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
        )
    except AuthError as e:
        if e.code == "INVALID_CREDENTIALS":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": e.code, "message": e.message},
                headers={"WWW-Authenticate": "Bearer"},
            )
        elif e.code == "ACCOUNT_INACTIVE":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": e.code, "message": e.message},
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
    summary="Refresh access token",
)
async def refresh_token(
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    request: RefreshRequest | None = None,
    refresh_token_cookie: str | None = Cookie(None, alias=REFRESH_TOKEN_COOKIE_NAME),
) -> AccessTokenResponse:
    """Refresh access token using a valid refresh token.

    The refresh token can be provided either:
    1. In the httpOnly cookie (preferred, more secure)
    2. In the request body (for backward compatibility)

    Returns a new access token. A new refresh token is set as an httpOnly cookie.
    """
    # Try to get refresh token from cookie first, then from request body
    token = refresh_token_cookie
    if not token and request and request.refresh_token:
        token = request.refresh_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "MISSING_REFRESH_TOKEN", "message": "Refresh token required"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        new_tokens = await auth_service.refresh_tokens(token)

        # Set new refresh token as httpOnly cookie
        _set_refresh_token_cookie(response, new_tokens.refresh_token)

        return AccessTokenResponse(
            access_token=new_tokens.access_token,
            token_type=new_tokens.token_type,
            expires_in=new_tokens.expires_in,
        )
    except AuthError as e:
        # Clear invalid refresh token cookie
        _clear_refresh_token_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": e.code, "message": e.message},
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout user",
)
async def logout(response: Response) -> MessageResponse:
    """Logout the current user.

    Clears the refresh token cookie. The client should also discard the access token.
    """
    _clear_refresh_token_cookie(response)
    return MessageResponse(message="Logged out successfully")


@router.get(
    "/me",
    response_model=User,
    summary="Get current user",
)
async def get_me(
    current_user: CurrentUser,
) -> User:
    """Get the currently authenticated user's information.

    Requires a valid access token in the Authorization header.
    """
    return current_user


@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change password",
)
async def change_password(
    request: ChangePasswordRequest,
    current_user: CurrentUser,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Change the current user's password.

    Requires the current password for verification.
    """
    try:
        await auth_service.change_password(
            current_user.user_id,
            request.current_password,
            request.new_password,
        )
        return MessageResponse(message="Password changed successfully")
    except AuthError as e:
        if e.code == "INVALID_PASSWORD":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": e.code, "message": e.message},
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )
