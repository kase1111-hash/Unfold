"""Document processing service."""

import hashlib
import io
import logging
import os
import re
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.document import (
    Document,
    DocumentLicense,
    DocumentSource,
    DocumentStatus,
    DocumentUpdate,
)
from app.repositories.document import DocumentRepository

logger = logging.getLogger(__name__)
settings = get_settings()

# Try to import PDF parsing library
try:
    import pypdf

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("pypdf not installed - PDF text extraction disabled")


class DocumentProcessingError(Exception):
    """Error during document processing."""

    def __init__(self, message: str, code: str = "PROCESSING_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class DocumentService:
    """Service for document processing operations."""

    # Supported MIME types
    SUPPORTED_TYPES = {
        "application/pdf": ".pdf",
        "application/epub+zip": ".epub",
    }

    # Upload directory (relative to project root)
    UPLOAD_DIR = Path("uploads/documents")

    def __init__(self, session: AsyncSession):
        """Initialize document service.

        Args:
            session: Database session
        """
        self.session = session
        self.repo = DocumentRepository(session)

        # Ensure upload directory exists
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        owner_id: str | None = None,
    ) -> Document:
        """Upload and process a document.

        Args:
            file_content: Document file bytes
            filename: Original filename
            content_type: MIME type
            owner_id: Owner user ID

        Returns:
            Created document

        Raises:
            DocumentProcessingError: If processing fails
        """
        # Validate file type
        if content_type not in self.SUPPORTED_TYPES:
            raise DocumentProcessingError(
                f"Unsupported file type: {content_type}. Supported: PDF, EPUB",
                code="UNSUPPORTED_TYPE",
            )

        # Generate document ID from content hash
        doc_id = self.repo.generate_doc_id(file_content)

        # Check if document already exists
        existing = await self.repo.get_by_id(doc_id)
        if existing is not None:
            logger.info(f"Document already exists: {doc_id}")
            return existing

        # Save file to disk
        extension = self.SUPPORTED_TYPES[content_type]
        file_path = self.UPLOAD_DIR / f"{doc_id.replace(':', '_')}{extension}"

        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
        except OSError as e:
            raise DocumentProcessingError(
                f"Failed to save file: {e}",
                code="STORAGE_ERROR",
            )

        # Extract title from filename
        title = Path(filename).stem if filename else "Untitled"

        # Create document record
        document = await self.repo.create(
            doc_id=doc_id,
            title=title,
            owner_id=owner_id,
            source=DocumentSource.UPLOAD,
            file_path=str(file_path),
            file_size_bytes=len(file_content),
        )

        # Process document content asynchronously
        try:
            await self._process_document_content(document, file_content, content_type)
        except Exception as e:
            logger.error(f"Failed to process document content: {e}")
            await self.repo.update_status(doc_id, DocumentStatus.FAILED)

        return document

    async def _process_document_content(
        self,
        document: Document,
        file_content: bytes,
        content_type: str,
    ) -> None:
        """Process document to extract text and metadata.

        Args:
            document: Document record
            file_content: File bytes
            content_type: MIME type
        """
        await self.repo.update_status(document.doc_id, DocumentStatus.PROCESSING)

        text_content = ""
        page_count = None
        metadata: dict = {}

        if content_type == "application/pdf":
            text_content, page_count, metadata = self._extract_pdf_content(file_content)
        elif content_type == "application/epub+zip":
            text_content, metadata = self._extract_epub_content(file_content)

        # Count words
        word_count = len(text_content.split()) if text_content else 0

        # Update document with extracted content
        await self.repo.update_content(
            doc_id=document.doc_id,
            content=text_content,
            page_count=page_count,
            word_count=word_count,
        )

        # Update title if extracted from metadata
        if metadata.get("title"):
            await self.repo.update(
                document.doc_id,
                DocumentUpdate(title=metadata["title"]),
            )

        # Update authors if extracted
        if metadata.get("authors"):
            await self.repo.update(
                document.doc_id,
                DocumentUpdate(authors=metadata["authors"]),
            )

        # Mark as validated
        await self.repo.update_status(document.doc_id, DocumentStatus.VALIDATED)

        # Create validation record
        await self.repo.create_validation(
            doc_id=document.doc_id,
            is_valid=True,
            provenance_hash=hashlib.sha256(text_content.encode()).hexdigest()
            if text_content
            else None,
        )

        logger.info(
            f"Processed document {document.doc_id}: "
            f"{page_count or 'N/A'} pages, {word_count} words"
        )

    def _extract_pdf_content(
        self, file_content: bytes
    ) -> tuple[str, int | None, dict]:
        """Extract text content from PDF.

        Args:
            file_content: PDF file bytes

        Returns:
            Tuple of (text_content, page_count, metadata)
        """
        if not PYPDF_AVAILABLE:
            logger.warning("pypdf not available, skipping PDF extraction")
            return "", None, {}

        try:
            pdf_file = io.BytesIO(file_content)
            reader = pypdf.PdfReader(pdf_file)

            # Extract metadata
            metadata: dict = {}
            if reader.metadata:
                if reader.metadata.title:
                    metadata["title"] = reader.metadata.title
                if reader.metadata.author:
                    # Split multiple authors
                    authors = reader.metadata.author
                    if "," in authors:
                        metadata["authors"] = [a.strip() for a in authors.split(",")]
                    elif ";" in authors:
                        metadata["authors"] = [a.strip() for a in authors.split(";")]
                    else:
                        metadata["authors"] = [authors]

            # Extract text from each page
            text_parts = []
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page: {e}")

            text_content = "\n\n".join(text_parts)
            page_count = len(reader.pages)

            return text_content, page_count, metadata

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return "", None, {}

    def _extract_epub_content(self, file_content: bytes) -> tuple[str, dict]:
        """Extract text content from EPUB.

        Args:
            file_content: EPUB file bytes

        Returns:
            Tuple of (text_content, metadata)
        """
        # EPUB extraction would require ebooklib
        # For now, return empty content
        logger.warning("EPUB extraction not implemented")
        return "", {}

    async def get_document(self, doc_id: str) -> Document | None:
        """Get document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document if found, None otherwise
        """
        return await self.repo.get_by_id(doc_id)

    async def list_documents(
        self,
        owner_id: str | None = None,
        status: DocumentStatus | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Document], int]:
        """List documents with pagination.

        Args:
            owner_id: Filter by owner
            status: Filter by status
            page: Page number
            page_size: Items per page

        Returns:
            Tuple of (documents, total_count)
        """
        return await self.repo.list_documents(
            owner_id=owner_id,
            status=status,
            page=page,
            page_size=min(page_size, 100),
        )

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its file.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        # Get document to find file path
        document = await self.repo.get_by_id(doc_id)
        if document is None:
            return False

        # Delete file if it exists
        if document.file_path:
            try:
                file_path = Path(document.file_path)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to delete file: {e}")

        # Delete database record
        return await self.repo.delete(doc_id)

    async def get_document_content(self, doc_id: str) -> str | None:
        """Get extracted text content of a document.

        Args:
            doc_id: Document identifier

        Returns:
            Text content if found, None otherwise
        """
        return await self.repo.get_content(doc_id)

    async def paraphrase_content(
        self,
        doc_id: str,
        complexity: int = 50,
    ) -> str | None:
        """Get paraphrased version of document content.

        Args:
            doc_id: Document identifier
            complexity: Complexity level (0=simplest, 100=original)

        Returns:
            Paraphrased content, or None if document not found

        Note:
            This is a placeholder. Full implementation would use LLM.
        """
        content = await self.repo.get_content(doc_id)
        if content is None:
            return None

        # Without LLM, just return original or simplified version
        if complexity >= 80:
            return content

        # Basic simplification: shorter sentences, common words
        # This is a placeholder - real implementation would use LLM
        sentences = re.split(r"[.!?]+", content)
        simplified = []

        for sentence in sentences[:50]:  # Limit for demo
            sentence = sentence.strip()
            if len(sentence) > 20:
                simplified.append(sentence)

        return ". ".join(simplified) + "."


# Dependency injection helper
def get_document_service(session: AsyncSession) -> DocumentService:
    """Get document service instance.

    Args:
        session: Database session

    Returns:
        DocumentService instance
    """
    return DocumentService(session)
