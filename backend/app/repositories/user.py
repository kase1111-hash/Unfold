"""User repository for database operations."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import UserORM
from app.models.user import User, UserCreate, UserRole, UserUpdate


class UserRepository:
    """Repository for user database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create(
        self,
        user_data: UserCreate,
        hashed_password: str,
        role: UserRole = UserRole.USER,
    ) -> User:
        """Create a new user.

        Args:
            user_data: User creation data
            hashed_password: Pre-hashed password
            role: User role (default: USER)

        Returns:
            Created user
        """
        user_id = str(uuid4())

        user_orm = UserORM(
            user_id=user_id,
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            orcid_id=user_data.orcid_id,
            role=role,
            is_active=True,
            is_verified=False,
        )

        self.session.add(user_orm)
        await self.session.flush()
        await self.session.refresh(user_orm)

        return self._to_model(user_orm)

    async def get_by_id(self, user_id: str) -> User | None:
        """Get user by ID.

        Args:
            user_id: User identifier

        Returns:
            User if found, None otherwise
        """
        result = await self.session.execute(
            select(UserORM).where(UserORM.user_id == user_id)
        )
        user_orm = result.scalar_one_or_none()

        if user_orm is None:
            return None

        return self._to_model(user_orm)

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email.

        Args:
            email: User email address

        Returns:
            User if found, None otherwise
        """
        result = await self.session.execute(
            select(UserORM).where(UserORM.email == email)
        )
        user_orm = result.scalar_one_or_none()

        if user_orm is None:
            return None

        return self._to_model(user_orm)

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username.

        Args:
            username: Username

        Returns:
            User if found, None otherwise
        """
        result = await self.session.execute(
            select(UserORM).where(UserORM.username == username)
        )
        user_orm = result.scalar_one_or_none()

        if user_orm is None:
            return None

        return self._to_model(user_orm)

    async def get_password_hash(self, user_id: str) -> str | None:
        """Get user's hashed password.

        Args:
            user_id: User identifier

        Returns:
            Hashed password if user found, None otherwise
        """
        result = await self.session.execute(
            select(UserORM.hashed_password).where(UserORM.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_password_hash_by_email(self, email: str) -> tuple[str, str] | None:
        """Get user ID and hashed password by email.

        Args:
            email: User email address

        Returns:
            Tuple of (user_id, hashed_password) if found, None otherwise
        """
        result = await self.session.execute(
            select(UserORM.user_id, UserORM.hashed_password).where(
                UserORM.email == email
            )
        )
        row = result.one_or_none()

        if row is None:
            return None

        return row.user_id, row.hashed_password

    async def update(self, user_id: str, user_data: UserUpdate) -> User | None:
        """Update user data.

        Args:
            user_id: User identifier
            user_data: Update data

        Returns:
            Updated user if found, None otherwise
        """
        update_dict = user_data.model_dump(exclude_unset=True)

        if not update_dict:
            return await self.get_by_id(user_id)

        update_dict["updated_at"] = datetime.now(timezone.utc)

        await self.session.execute(
            update(UserORM).where(UserORM.user_id == user_id).values(**update_dict)
        )

        return await self.get_by_id(user_id)

    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp.

        Args:
            user_id: User identifier
        """
        await self.session.execute(
            update(UserORM)
            .where(UserORM.user_id == user_id)
            .values(last_login=datetime.now(timezone.utc))
        )

    async def verify_email(self, user_id: str) -> bool:
        """Mark user's email as verified.

        Args:
            user_id: User identifier

        Returns:
            True if user found and updated, False otherwise
        """
        result = await self.session.execute(
            update(UserORM)
            .where(UserORM.user_id == user_id)
            .values(is_verified=True, updated_at=datetime.now(timezone.utc))
        )
        return result.rowcount > 0

    async def deactivate(self, user_id: str) -> bool:
        """Deactivate a user account.

        Args:
            user_id: User identifier

        Returns:
            True if user found and deactivated, False otherwise
        """
        result = await self.session.execute(
            update(UserORM)
            .where(UserORM.user_id == user_id)
            .values(is_active=False, updated_at=datetime.now(timezone.utc))
        )
        return result.rowcount > 0

    async def exists_by_email(self, email: str) -> bool:
        """Check if user with email exists.

        Args:
            email: Email to check

        Returns:
            True if user exists, False otherwise
        """
        result = await self.session.execute(
            select(UserORM.user_id).where(UserORM.email == email).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def exists_by_username(self, username: str) -> bool:
        """Check if user with username exists.

        Args:
            username: Username to check

        Returns:
            True if user exists, False otherwise
        """
        result = await self.session.execute(
            select(UserORM.user_id).where(UserORM.username == username).limit(1)
        )
        return result.scalar_one_or_none() is not None

    def _to_model(self, user_orm: UserORM) -> User:
        """Convert ORM model to Pydantic model.

        Args:
            user_orm: SQLAlchemy ORM model

        Returns:
            Pydantic User model
        """
        return User(
            user_id=user_orm.user_id,
            email=user_orm.email,
            username=user_orm.username,
            full_name=user_orm.full_name,
            orcid_id=user_orm.orcid_id,
            role=user_orm.role,
            is_active=user_orm.is_active,
            is_verified=user_orm.is_verified,
            last_login=user_orm.last_login,
            created_at=user_orm.created_at,
            updated_at=user_orm.updated_at,
        )
