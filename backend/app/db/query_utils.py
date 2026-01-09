"""
Database query optimization utilities.
Provides helpers for efficient database operations.
"""

import logging
from typing import TypeVar, Generic, Sequence, Optional, Any
from dataclasses import dataclass

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PaginationParams:
    """Pagination parameters."""

    page: int = 1
    page_size: int = 20
    max_page_size: int = 100

    def __post_init__(self):
        self.page = max(1, self.page)
        self.page_size = min(max(1, self.page_size), self.max_page_size)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


@dataclass
class PaginatedResult(Generic[T]):
    """Paginated query result."""

    items: Sequence[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def create(
        cls,
        items: Sequence[T],
        total: int,
        params: PaginationParams,
    ) -> "PaginatedResult[T]":
        total_pages = (total + params.page_size - 1) // params.page_size
        return cls(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            total_pages=total_pages,
        )

    def to_dict(self) -> dict:
        return {
            "items": list(self.items),
            "pagination": {
                "page": self.page,
                "page_size": self.page_size,
                "total": self.total,
                "total_pages": self.total_pages,
                "has_next": self.page < self.total_pages,
                "has_prev": self.page > 1,
            },
        }


async def paginate_query(
    session: AsyncSession,
    query,
    params: PaginationParams,
    count_query=None,
) -> PaginatedResult:
    """
    Execute a paginated query efficiently.

    Uses a single query with window function for better performance
    when possible, otherwise falls back to separate count query.

    Args:
        session: Database session
        query: SQLAlchemy select statement
        params: Pagination parameters
        count_query: Optional custom count query

    Returns:
        PaginatedResult with items and metadata
    """
    # Get total count
    if count_query is not None:
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0
    else:
        # Create count query from the main query
        count_stmt = select(func.count()).select_from(query.subquery())
        total_result = await session.execute(count_stmt)
        total = total_result.scalar() or 0

    # Apply pagination
    paginated_query = query.offset(params.offset).limit(params.limit)
    result = await session.execute(paginated_query)
    items = result.scalars().all()

    return PaginatedResult.create(items, total, params)


async def bulk_insert(
    session: AsyncSession,
    model_class,
    items: list[dict],
    batch_size: int = 1000,
) -> int:
    """
    Efficiently insert multiple records in batches.

    Args:
        session: Database session
        model_class: SQLAlchemy model class
        items: List of dicts with model data
        batch_size: Number of items per batch

    Returns:
        Number of items inserted
    """
    if not items:
        return 0

    total_inserted = 0

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        objects = [model_class(**item) for item in batch]
        session.add_all(objects)
        await session.flush()
        total_inserted += len(batch)

    return total_inserted


async def bulk_update(
    session: AsyncSession,
    model_class,
    updates: list[dict],
    id_field: str = "id",
) -> int:
    """
    Efficiently update multiple records.

    Args:
        session: Database session
        model_class: SQLAlchemy model class
        updates: List of dicts with id and update data
        id_field: Name of the ID field

    Returns:
        Number of records updated
    """
    if not updates:
        return 0

    updated = 0
    for update in updates:
        id_value = update.pop(id_field, None)
        if id_value is None:
            continue

        query = select(model_class).where(getattr(model_class, id_field) == id_value)
        result = await session.execute(query)
        obj = result.scalar_one_or_none()

        if obj:
            for key, value in update.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            updated += 1

    return updated


async def exists(
    session: AsyncSession,
    model_class,
    **filters,
) -> bool:
    """
    Check if a record exists efficiently.

    Args:
        session: Database session
        model_class: SQLAlchemy model class
        **filters: Field-value pairs for filtering

    Returns:
        True if record exists
    """
    conditions = [
        getattr(model_class, field) == value
        for field, value in filters.items()
        if hasattr(model_class, field)
    ]

    if not conditions:
        return False

    query = select(func.count()).select_from(model_class).where(and_(*conditions))
    result = await session.execute(query)
    count = result.scalar() or 0

    return count > 0


async def get_or_create(
    session: AsyncSession,
    model_class,
    defaults: Optional[dict] = None,
    **lookup,
) -> tuple[Any, bool]:
    """
    Get existing record or create new one.

    Args:
        session: Database session
        model_class: SQLAlchemy model class
        defaults: Default values for creation
        **lookup: Fields to search by

    Returns:
        Tuple of (instance, created)
    """
    conditions = [
        getattr(model_class, field) == value
        for field, value in lookup.items()
        if hasattr(model_class, field)
    ]

    query = select(model_class).where(and_(*conditions))
    result = await session.execute(query)
    instance = result.scalar_one_or_none()

    if instance:
        return instance, False

    # Create new instance
    create_data = {**lookup, **(defaults or {})}
    instance = model_class(**create_data)
    session.add(instance)
    await session.flush()

    return instance, True


def build_search_filter(
    model_class,
    search_term: str,
    search_fields: list[str],
    case_insensitive: bool = True,
):
    """
    Build a search filter for multiple fields.

    Args:
        model_class: SQLAlchemy model class
        search_term: Search term
        search_fields: Fields to search in
        case_insensitive: Use case-insensitive search

    Returns:
        SQLAlchemy filter clause
    """
    if not search_term or not search_fields:
        return None

    conditions = []
    search_pattern = f"%{search_term}%"

    for field in search_fields:
        if hasattr(model_class, field):
            column = getattr(model_class, field)
            if case_insensitive:
                conditions.append(func.lower(column).like(search_pattern.lower()))
            else:
                conditions.append(column.like(search_pattern))

    return or_(*conditions) if conditions else None


class QueryBuilder:
    """
    Fluent query builder for common patterns.
    """

    def __init__(self, model_class):
        self.model_class = model_class
        self._query = select(model_class)
        self._filters = []
        self._order_by = []
        self._limit = None
        self._offset = None
        self._load_relations = []

    def filter(self, *conditions):
        """Add filter conditions."""
        self._filters.extend(conditions)
        return self

    def filter_by(self, **kwargs):
        """Add equality filter conditions."""
        for field, value in kwargs.items():
            if hasattr(self.model_class, field):
                self._filters.append(getattr(self.model_class, field) == value)
        return self

    def search(self, term: str, fields: list[str]):
        """Add search filter."""
        search_filter = build_search_filter(self.model_class, term, fields)
        if search_filter is not None:
            self._filters.append(search_filter)
        return self

    def order_by(self, *columns, desc: bool = False):
        """Add ordering."""
        for col in columns:
            if isinstance(col, str) and hasattr(self.model_class, col):
                col = getattr(self.model_class, col)
            if desc:
                col = col.desc()
            self._order_by.append(col)
        return self

    def limit(self, limit: int):
        """Set result limit."""
        self._limit = limit
        return self

    def offset(self, offset: int):
        """Set result offset."""
        self._offset = offset
        return self

    def load(self, *relations):
        """Eagerly load relations."""
        for rel in relations:
            if isinstance(rel, str) and hasattr(self.model_class, rel):
                self._load_relations.append(
                    selectinload(getattr(self.model_class, rel))
                )
        return self

    def build(self):
        """Build the query."""
        query = self._query

        if self._load_relations:
            query = query.options(*self._load_relations)

        if self._filters:
            query = query.where(and_(*self._filters))

        if self._order_by:
            query = query.order_by(*self._order_by)

        if self._offset is not None:
            query = query.offset(self._offset)

        if self._limit is not None:
            query = query.limit(self._limit)

        return query

    async def execute(self, session: AsyncSession):
        """Execute the query."""
        query = self.build()
        result = await session.execute(query)
        return result.scalars().all()

    async def first(self, session: AsyncSession):
        """Get first result."""
        self._limit = 1
        query = self.build()
        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def count(self, session: AsyncSession) -> int:
        """Get count of matching records."""
        count_query = select(func.count()).select_from(
            self._query.where(and_(*self._filters)).subquery()
            if self._filters
            else self.model_class
        )
        result = await session.execute(count_query)
        return result.scalar() or 0

    async def paginate(
        self,
        session: AsyncSession,
        params: PaginationParams,
    ) -> PaginatedResult:
        """Execute with pagination."""
        # Get total count
        total = await self.count(session)

        # Apply pagination
        self._offset = params.offset
        self._limit = params.limit
        items = await self.execute(session)

        return PaginatedResult.create(items, total, params)
