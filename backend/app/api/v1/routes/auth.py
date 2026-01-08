"""Authentication endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies import CurrentUser, get_db
from app.models.user import Token, User, UserCreate
from app.services.auth.service import AuthError, AuthService

router = APIRouter()


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request body."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")


class RegisterRequest(UserCreate):
    """Registration request body (extends UserCreate)."""

    pass


class RefreshRequest(BaseModel):
    """Token refresh request body."""

    refresh_token: str = Field(..., description="Valid refresh token")


class AuthResponse(BaseModel):
    """Authentication response with user and tokens."""

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
async def get_auth_service(
    db: Annotated[AsyncSession, Depends(get_db)]
) -> AuthService:
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
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthResponse:
    """Register a new user account.

    Creates a new user with the provided credentials and returns
    authentication tokens for immediate login.
    """
    try:
        user, tokens = await auth_service.register(request)
        return AuthResponse(user=user, tokens=tokens)
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
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthResponse:
    """Authenticate user with email and password.

    Returns user information and authentication tokens on success.
    """
    try:
        user, tokens = await auth_service.login(request.email, request.password)
        return AuthResponse(user=user, tokens=tokens)
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
    response_model=Token,
    summary="Refresh access token",
)
async def refresh_token(
    request: RefreshRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> Token:
    """Refresh access token using a valid refresh token.

    Returns a new token pair (access + refresh tokens).
    """
    try:
        return await auth_service.refresh_tokens(request.refresh_token)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": e.code, "message": e.message},
            headers={"WWW-Authenticate": "Bearer"},
        )


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
