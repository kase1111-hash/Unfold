"""Base models and common schemas."""

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class APIResponse(BaseModel, Generic[DataT]):
    """Standard API response wrapper."""

    status: str = "success"
    data: DataT
    meta: dict[str, Any] | None = None


class APIErrorDetail(BaseModel):
    """Error detail model."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class APIErrorResponse(BaseModel):
    """Standard API error response."""

    status: str = "error"
    error: APIErrorDetail


class PaginatedResponse(BaseModel, Generic[DataT]):
    """Paginated response wrapper."""

    status: str = "success"
    data: list[DataT]
    pagination: "PaginationMeta"


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def from_query(cls, total: int, page: int, page_size: int) -> "PaginationMeta":
        """Create pagination meta from query parameters."""
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
