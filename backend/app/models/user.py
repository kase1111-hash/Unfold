"""User and authentication data models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, EmailStr, Field

from app.models.base import TimestampMixin


class UserRole(str, Enum):
    """User roles."""

    USER = "user"
    EDUCATOR = "educator"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class UserBase(BaseModel):
    """Base user model."""

    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    full_name: str | None = Field(None, max_length=100, description="Full name")
    orcid_id: str | None = Field(None, description="ORCID identifier")


class UserCreate(UserBase):
    """Model for user registration."""

    password: str = Field(..., min_length=8, max_length=100, description="Password")


class UserUpdate(BaseModel):
    """Model for updating user profile."""

    full_name: str | None = Field(None, max_length=100)
    orcid_id: str | None = None


class User(UserBase, TimestampMixin):
    """Full user model."""

    user_id: str = Field(..., description="Unique user identifier")
    role: UserRole = Field(UserRole.USER, description="User role")
    is_active: bool = Field(True, description="Account active status")
    is_verified: bool = Field(False, description="Email verified status")
    last_login: datetime | None = Field(None, description="Last login timestamp")

    class Config:
        """Pydantic config."""

        from_attributes = True


class UserInDB(User):
    """User model with hashed password (for internal use)."""

    hashed_password: str


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token expiration in seconds")


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str = Field(..., description="Subject (user ID)")
    exp: datetime = Field(..., description="Expiration time")
    iat: datetime = Field(..., description="Issued at time")
    type: str = Field(..., description="Token type (access/refresh)")


class LoginRequest(BaseModel):
    """Login request model."""

    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str
