"""Document database models."""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.postgres import Base
from app.models.document import DocumentLicense, DocumentSource, DocumentStatus


class DocumentORM(Base):
    """SQLAlchemy model for documents table."""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    authors: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    doi: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)

    license: Mapped[DocumentLicense | None] = mapped_column(
        Enum(DocumentLicense, name="document_license"),
        nullable=True,
    )
    source: Mapped[DocumentSource] = mapped_column(
        Enum(DocumentSource, name="document_source"),
        default=DocumentSource.UPLOAD,
        nullable=False,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Storage and processing info
    vector_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    graph_nodes: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Content storage (for extracted text)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Owner relationship
    owner_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    owner: Mapped["UserORM | None"] = relationship("UserORM", back_populates="documents")
    validation: Mapped["DocumentValidationORM | None"] = relationship(
        "DocumentValidationORM",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Document(doc_id={self.doc_id}, title={self.title[:50]})>"


class DocumentValidationORM(Base):
    """SQLAlchemy model for document validation records."""

    __tablename__ = "document_validations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    is_valid: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    doi_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    license_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    author_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    provenance_hash: Mapped[str | None] = mapped_column(String(100), nullable=True)
    validation_errors: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)

    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    document: Mapped["DocumentORM"] = relationship("DocumentORM", back_populates="validation")

    def __repr__(self) -> str:
        return f"<DocumentValidation(doc_id={self.doc_id}, is_valid={self.is_valid})>"


# Import for relationship type hints
from app.db.models.user import UserORM  # noqa: E402
