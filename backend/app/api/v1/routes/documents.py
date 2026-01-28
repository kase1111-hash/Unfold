"""Document management endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies import CurrentUser, get_db
from app.models.document import Document, DocumentStatus
from app.services.ingestion.document_service import (
    DocumentProcessingError,
    DocumentService,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    status: str
    message: str
    document: Document


class DocumentListResponse(BaseModel):
    """Response model for document listing."""

    status: str
    data: list[Document]
    total: int
    page: int
    page_size: int


class ParaphraseResponse(BaseModel):
    """Response model for paraphrased content."""

    doc_id: str
    complexity: int
    content: str


# Dependency to get document service
async def get_document_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentService:
    """Get document service instance."""
    return DocumentService(db)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF or EPUB document to upload")],
    service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: CurrentUser,
) -> DocumentUploadResponse:
    """Upload a document for processing.

    Accepts PDF and EPUB files. The document will be:
    1. Validated for format and license compliance
    2. Parsed for metadata extraction
    3. Text content extracted
    4. Stored in the database

    Args:
        file: The document file to upload (PDF or EPUB)
        service: Document service
        current_user: Authenticated user

    Returns:
        Upload confirmation with document metadata
    """
    # Validate file type
    allowed_types = ["application/pdf", "application/epub+zip"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "UNSUPPORTED_TYPE",
                "message": f"Unsupported file type: {file.content_type}. Allowed: PDF, EPUB",
            },
        )

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "READ_ERROR",
                "message": "Failed to read uploaded file",
            },
        )

    # Validate file size (max 50MB)
    max_size = 50 * 1024 * 1024
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File too large. Maximum size: {max_size // (1024*1024)}MB",
            },
        )

    try:
        document = await service.upload_document(
            file_content=file_content,
            filename=file.filename or "document",
            content_type=file.content_type or "application/pdf",
            owner_id=current_user.user_id,
        )
    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )

    return DocumentUploadResponse(
        status="success",
        message="Document uploaded successfully",
        document=document,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: CurrentUser,
    page: int = 1,
    page_size: int = 20,
    status_filter: DocumentStatus | None = None,
) -> DocumentListResponse:
    """List all documents for the current user.

    Args:
        service: Document service
        current_user: Authenticated user
        page: Page number (1-indexed)
        page_size: Number of documents per page (max 100)
        status_filter: Optional status filter

    Returns:
        Paginated list of documents
    """
    documents, total = await service.list_documents(
        owner_id=current_user.user_id,
        status=status_filter,
        page=page,
        page_size=page_size,
    )

    return DocumentListResponse(
        status="success",
        data=documents,
        total=total,
        page=page,
        page_size=min(page_size, 100),
    )


@router.get("/{doc_id}", response_model=Document)
async def get_document(
    doc_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: CurrentUser,
) -> Document:
    """Get document metadata by ID.

    Args:
        doc_id: Document identifier (SHA-256 hash)
        service: Document service
        current_user: Authenticated user

    Returns:
        Document metadata
    """
    document = await service.get_document(doc_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Document not found: {doc_id}",
            },
        )

    return document


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: CurrentUser,
) -> None:
    """Delete a document and its associated data.

    This will remove:
    - Document record from database
    - Uploaded file from storage
    - Associated validation records

    Args:
        doc_id: Document identifier (SHA-256 hash)
        service: Document service
        current_user: Authenticated user
    """
    deleted = await service.delete_document(doc_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Document not found: {doc_id}",
            },
        )


@router.get("/{doc_id}/content")
async def get_document_content(
    doc_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: CurrentUser,
) -> dict:
    """Get the extracted text content of a document.

    Args:
        doc_id: Document identifier
        service: Document service
        current_user: Authenticated user

    Returns:
        Document text content
    """
    content = await service.get_document_content(doc_id)

    if content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Document not found or has no content: {doc_id}",
            },
        )

    return {"doc_id": doc_id, "content": content}


@router.get("/{doc_id}/paraphrase", response_model=ParaphraseResponse)
async def get_document_paraphrase(
    doc_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: CurrentUser,
    complexity: int = 50,
) -> ParaphraseResponse:
    """Get a paraphrased version of the document.

    The complexity parameter controls how much the text is simplified:
    - 0-30: Heavily simplified, basic vocabulary
    - 31-60: Moderately simplified
    - 61-80: Light simplification
    - 81-100: Near-original or original text

    Args:
        doc_id: Document identifier
        service: Document service
        current_user: Authenticated user
        complexity: Complexity level (0=simplest, 100=original)

    Returns:
        Paraphrased document content
    """
    # Validate complexity range
    if not 0 <= complexity <= 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_COMPLEXITY",
                "message": "Complexity must be between 0 and 100",
            },
        )

    content = await service.paraphrase_content(doc_id, complexity)

    if content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Document not found: {doc_id}",
            },
        )

    return ParaphraseResponse(
        doc_id=doc_id,
        complexity=complexity,
        content=content,
    )
