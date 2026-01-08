"""Authentication service with login/register logic."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import Token, User, UserCreate
from app.repositories.user import UserRepository
from app.services.auth.jwt import (
    TokenError,
    create_token_pair,
    verify_token,
)
from app.utils.security import hash_password, verify_password


class AuthError(Exception):
    """Exception raised for authentication errors."""

    def __init__(self, message: str, code: str = "AUTH_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class AuthService:
    """Service for authentication operations."""

    def __init__(self, session: AsyncSession):
        """Initialize auth service with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        self.user_repo = UserRepository(session)

    async def register(self, user_data: UserCreate) -> tuple[User, Token]:
        """Register a new user.

        Args:
            user_data: User registration data including password

        Returns:
            Tuple of (created user, token pair)

        Raises:
            AuthError: If email or username already exists
        """
        # Check if email already exists
        if await self.user_repo.exists_by_email(user_data.email):
            raise AuthError("Email already registered", "EMAIL_EXISTS")

        # Check if username already exists
        if await self.user_repo.exists_by_username(user_data.username):
            raise AuthError("Username already taken", "USERNAME_EXISTS")

        # Hash password
        hashed_password = hash_password(user_data.password)

        # Create user
        user = await self.user_repo.create(user_data, hashed_password)

        # Generate tokens
        tokens = create_token_pair(user.user_id)

        return user, tokens

    async def login(self, email: str, password: str) -> tuple[User, Token]:
        """Authenticate user with email and password.

        Args:
            email: User email
            password: Plaintext password

        Returns:
            Tuple of (user, token pair)

        Raises:
            AuthError: If credentials are invalid or account is inactive
        """
        # Get user credentials
        credentials = await self.user_repo.get_password_hash_by_email(email)

        if credentials is None:
            raise AuthError("Invalid email or password", "INVALID_CREDENTIALS")

        user_id, hashed_password = credentials

        # Verify password
        if not verify_password(password, hashed_password):
            raise AuthError("Invalid email or password", "INVALID_CREDENTIALS")

        # Get full user object
        user = await self.user_repo.get_by_id(user_id)

        if user is None:
            raise AuthError("User not found", "USER_NOT_FOUND")

        # Check if account is active
        if not user.is_active:
            raise AuthError("Account is deactivated", "ACCOUNT_INACTIVE")

        # Update last login
        await self.user_repo.update_last_login(user_id)

        # Generate tokens
        tokens = create_token_pair(user_id)

        return user, tokens

    async def refresh_tokens(self, refresh_token: str) -> Token:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New token pair

        Raises:
            AuthError: If refresh token is invalid
        """
        try:
            payload = verify_token(refresh_token, token_type="refresh")
        except TokenError as e:
            raise AuthError(str(e), "INVALID_REFRESH_TOKEN")

        # Verify user still exists and is active
        user = await self.user_repo.get_by_id(payload.sub)

        if user is None:
            raise AuthError("User not found", "USER_NOT_FOUND")

        if not user.is_active:
            raise AuthError("Account is deactivated", "ACCOUNT_INACTIVE")

        # Generate new tokens
        return create_token_pair(payload.sub)

    async def get_current_user(self, access_token: str) -> User:
        """Get current user from access token.

        Args:
            access_token: Valid access token

        Returns:
            Current user

        Raises:
            AuthError: If token is invalid or user not found
        """
        try:
            payload = verify_token(access_token, token_type="access")
        except TokenError as e:
            raise AuthError(str(e), "INVALID_ACCESS_TOKEN")

        user = await self.user_repo.get_by_id(payload.sub)

        if user is None:
            raise AuthError("User not found", "USER_NOT_FOUND")

        if not user.is_active:
            raise AuthError("Account is deactivated", "ACCOUNT_INACTIVE")

        return user

    async def verify_user_email(self, user_id: str) -> bool:
        """Verify user's email address.

        Args:
            user_id: User identifier

        Returns:
            True if verification successful
        """
        return await self.user_repo.verify_email(user_id)

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> bool:
        """Change user's password.

        Args:
            user_id: User identifier
            current_password: Current plaintext password
            new_password: New plaintext password

        Returns:
            True if password changed successfully

        Raises:
            AuthError: If current password is incorrect
        """
        # Get current password hash
        hashed_password = await self.user_repo.get_password_hash(user_id)

        if hashed_password is None:
            raise AuthError("User not found", "USER_NOT_FOUND")

        # Verify current password
        if not verify_password(current_password, hashed_password):
            raise AuthError("Current password is incorrect", "INVALID_PASSWORD")

        # Hash new password and update
        new_hashed = hash_password(new_password)

        # For now, we'd need to add an update_password method to the repository
        # This is a placeholder - in production, add the proper repository method
        from sqlalchemy import update
        from app.db.models.user import UserORM

        await self.session.execute(
            update(UserORM)
            .where(UserORM.user_id == user_id)
            .values(hashed_password=new_hashed)
        )

        return True
