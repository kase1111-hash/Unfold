"""Document-related data models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.models.base import TimestampMixin


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentSource(str, Enum):
    """Document source types."""

    UPLOAD = "upload"
    ARXIV = "arXiv"
    PUBMED = "PubMed"
    DOI = "DOI"
    URL = "URL"


class DocumentLicense(str, Enum):
    """Common document licenses."""

    CC_BY_4 = "CC-BY-4.0"
    CC_BY_SA_4 = "CC-BY-SA-4.0"
    CC_BY_NC_4 = "CC-BY-NC-4.0"
    CC0 = "CC0-1.0"
    MIT = "MIT"
    UNKNOWN = "unknown"
    PROPRIETARY = "proprietary"


class DocumentBase(BaseModel):
    """Base document model."""

    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    doi: str | None = Field(None, pattern=r"^10\.\d{4,}/.*$", description="DOI")
    abstract: str | None = Field(None, max_length=5000, description="Document abstract")
    license: DocumentLicense | None = Field(None, description="Document license")
    source: DocumentSource = Field(DocumentSource.UPLOAD, description="Document source")


class DocumentCreate(DocumentBase):
    """Model for creating a new document."""

    pass


class DocumentUpdate(BaseModel):
    """Model for updating document metadata."""

    title: str | None = Field(None, min_length=1, max_length=500)
    authors: list[str] | None = None
    doi: str | None = Field(None, pattern=r"^10\.\d{4,}/.*$")
    abstract: str | None = Field(None, max_length=5000)
    license: DocumentLicense | None = None


class Document(DocumentBase, TimestampMixin):
    """Full document model with all fields."""

    doc_id: str = Field(..., description="Unique document ID (SHA-256 hash)")
    status: DocumentStatus = Field(DocumentStatus.PENDING, description="Processing status")
    vector_id: str | None = Field(None, description="Vector store ID for embeddings")
    graph_nodes: list[str] = Field(default_factory=list, description="Associated graph node IDs")
    file_path: str | None = Field(None, description="Storage path for original file")
    file_size_bytes: int | None = Field(None, description="File size in bytes")
    page_count: int | None = Field(None, description="Number of pages")
    word_count: int | None = Field(None, description="Word count")

    class Config:
        """Pydantic config."""

        from_attributes = True


class DocumentValidation(BaseModel):
    """Document validation result."""

    doc_id: str
    is_valid: bool
    doi_verified: bool = False
    license_verified: bool = False
    author_verified: bool = False
    provenance_hash: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    validated_at: datetime | None = None
