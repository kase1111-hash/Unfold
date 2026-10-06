"""Document management endpoints."""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.routing import APIRoute
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies import CurrentUser, get_db
from app.models.document import Document, DocumentStatus
from app.services.graph.document_graph import (
    build_document_graph_task,
    delete_document_graph,
)
from app.services.ingestion.document_service import (
    DocumentProcessingError,
    DocumentService,
)

logger = logging.getLogger(__name__)

# Maximum upload size (50MB)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# Allowance for the multipart framing around the file in Content-Length
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _file_too_large() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail={
            "code": "FILE_TOO_LARGE",
            "message": f"File too large. Maximum size: {MAX_UPLOAD_BYTES // (1024*1024)}MB",
        },
    )


class SizeLimitedRoute(APIRoute):
    """Route that rejects an oversized body from its Content-Length header.

    FastAPI reads and parses a multipart body before any dependency or
    endpoint code runs, so the check wraps the route handler itself in
    order to answer 413 without first receiving the whole upload.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def size_limited_handler(request: Request) -> Response:
            content_length = request.headers.get("content-length", "")
            if (
                content_length.isdigit()
                and int(content_length) > MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
            ):
                raise _file_too_large()
            return await handler(request)

        return size_limited_handler


router = APIRouter(route_class=SizeLimitedRoute)


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
    file: Annotated[UploadFile, File(description="PDF document to upload")],
    service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: CurrentUser,
    background_tasks: BackgroundTasks,
) -> DocumentUploadResponse:
    """Upload a document for processing.

    Accepts PDF files. The document will be:
    1. Validated for format
    2. Parsed for metadata extraction
    3. Text content extracted
    4. Stored in the database
    5. Built into the knowledge graph in the background (status becomes
       "indexed" when that succeeds)

    A file that can't be processed is rejected with 400 and its error code
    (EMPTY_FILE, CORRUPT_PDF, ENCRYPTED_PDF, NO_TEXT_EXTRACTED); nothing is
    stored for it.

    Args:
        file: The PDF file to upload
        service: Document service
        current_user: Authenticated user
        background_tasks: Runs the knowledge-graph build after the response

    Returns:
        Upload confirmation with the processed document
    """
    # Validate file type
    allowed_types = ["application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "UNSUPPORTED_TYPE",
                "message": f"Unsupported file type: {file.content_type}. Allowed: PDF",
            },
        )

    # Read file content, at most one byte past the size limit
    try:
        file_content = await file.read(MAX_UPLOAD_BYTES + 1)
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "READ_ERROR",
                "message": "Failed to read uploaded file",
            },
        )

    # Validate file size (also covers bodies sent without Content-Length)
    if len(file_content) > MAX_UPLOAD_BYTES:
        raise _file_too_large()

    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "EMPTY_FILE",
                "message": "The uploaded file is empty",
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

    # Build the knowledge graph after the response is sent. The task never
    # raises; it marks the document INDEXED when nodes were created.
    if document.status == DocumentStatus.VALIDATED:
        background_tasks.add_task(build_document_graph_task, document.doc_id)

    return DocumentUploadResponse(
        status="success",
        message="Document uploaded successfully",
        document=document,
    )


# Served with and without the trailing slash, so neither form gets a 307
@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse, include_in_schema=False)
async def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1),
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
        Document metadata (404 unless the current user owns it)
    """
    document = await service.get_document(doc_id, owner_id=current_user.user_id)

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
    - Its knowledge-graph nodes (best effort)

    Args:
        doc_id: Document identifier (SHA-256 hash)
        service: Document service
        current_user: Authenticated user
    """
    deleted = await service.delete_document(doc_id, owner_id=current_user.user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Document not found: {doc_id}",
            },
        )

    # A graph-database problem must never fail the delete
    try:
        await delete_document_graph(doc_id)
    except Exception:
        logger.exception(f"Failed to delete graph nodes for {doc_id}")


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
    content = await service.get_document_content(doc_id, owner_id=current_user.user_id)

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

    content = await service.paraphrase_content(
        doc_id, complexity, owner_id=current_user.user_id
    )

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
