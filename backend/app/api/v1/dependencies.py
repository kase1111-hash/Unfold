"""API dependencies for dependency injection."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session, get_neo4j_session
from app.models.user import User, UserRole
from app.services.auth.jwt import TokenError, verify_token
from app.repositories.user import UserRepository

try:
    from neo4j import AsyncSession as Neo4jSession
except ImportError:
    Neo4jSession = None  # type: ignore

# Security scheme for JWT Bearer tokens
bearer_scheme = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL database session.

    Usage:
        @router.get("/items")
        async def get_items(db: Annotated[AsyncSession, Depends(get_db)]):
            ...
    """
    async for session in get_session():
        yield session


async def get_graph_db() -> AsyncGenerator[AsyncSession, None]:
    """Get Neo4j database session.

    Usage:
        @router.get("/nodes")
        async def get_nodes(graph: Annotated[Neo4jSession, Depends(get_graph_db)]):
            ...
    """
    async for session in get_neo4j_session():
        yield session


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Get currently authenticated user from JWT token.

    Usage:
        @router.get("/protected")
        async def protected_route(user: Annotated[User, Depends(get_current_user)]):
            ...

    Raises:
        HTTPException: 401 if token missing/invalid, 403 if account inactive
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "MISSING_TOKEN", "message": "Authorization token required"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = verify_token(credentials.credentials, token_type="access")
    except TokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_TOKEN", "message": str(e)},
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(payload.sub)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "USER_NOT_FOUND", "message": "User not found"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "ACCOUNT_INACTIVE", "message": "Account is deactivated"},
        )

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current user, ensuring they are active.

    This is an alias for get_current_user with the active check built-in.
    """
    return current_user


async def get_current_verified_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current user, ensuring they are verified.

    Raises:
        HTTPException: 403 if user's email is not verified
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "EMAIL_NOT_VERIFIED", "message": "Email verification required"},
        )
    return current_user


def require_role(required_role: UserRole):
    """Create a dependency that requires a specific user role.

    Usage:
        @router.get("/admin-only")
        async def admin_route(
            user: Annotated[User, Depends(require_role(UserRole.ADMIN))]
        ):
            ...
    """
    async def role_checker(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        # Define role hierarchy
        role_hierarchy = {
            UserRole.USER: 0,
            UserRole.EDUCATOR: 1,
            UserRole.RESEARCHER: 2,
            UserRole.ADMIN: 3,
        }

        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "INSUFFICIENT_PERMISSIONS",
                    "message": f"Role {required_role.value} or higher required",
                },
            )

        return current_user

    return role_checker


# Type aliases for cleaner dependency injection
DBSession = Annotated[AsyncSession, Depends(get_db)]
GraphSession = Annotated[AsyncSession, Depends(get_graph_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentActiveUser = Annotated[User, Depends(get_current_active_user)]
CurrentVerifiedUser = Annotated[User, Depends(get_current_verified_user)]
