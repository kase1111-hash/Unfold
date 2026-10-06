"""Document processing service."""

import hashlib
import io
import logging
import re
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.document import (
    Document,
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

        The file is parsed before anything is stored, so a rejected upload
        leaves no document row and no file behind (and uploading the same
        bytes again is rejected again, not deduplicated to a failed record).

        Args:
            file_content: Document file bytes
            filename: Original filename
            content_type: MIME type
            owner_id: Owner user ID

        Returns:
            The processed document, or the owner's existing copy of it

        Raises:
            DocumentProcessingError: If the type is unsupported or no text can
                be extracted (corrupt, password-protected or image-only PDF)
        """
        # Validate file type
        if content_type not in self.SUPPORTED_TYPES:
            raise DocumentProcessingError(
                f"Unsupported file type: {content_type}. Supported: PDF",
                code="UNSUPPORTED_TYPE",
            )

        # Generate document ID from owner + content hash
        doc_id = self.repo.generate_doc_id(file_content, owner_id)

        # Check if this owner already uploaded the document
        existing = await self.repo.get_by_id(doc_id, owner_id=owner_id)
        if existing is not None:
            logger.info(f"Document already exists: {doc_id}")
            return existing

        # Extract text before persisting anything (raises DocumentProcessingError)
        text_content, page_count, metadata = self._extract_pdf_content(file_content)
        if not text_content.split():
            raise DocumentProcessingError(
                "No text content could be extracted from the document. "
                "Scanned or image-only PDFs are not supported.",
                code="NO_TEXT_EXTRACTED",
            )

        extension = self.SUPPORTED_TYPES[content_type]
        file_path = self.UPLOAD_DIR / f"{doc_id.replace(':', '_')}{extension}"

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
        await self._process_document_content(document, text_content, page_count, metadata)

        # Return the processed document, not the 'pending' snapshot from create()
        processed = await self.repo.get_by_id(doc_id) or document

        # Save the file last: if any step above fails, the request's
        # transaction is rolled back and no orphan file is left on disk
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
        except OSError as e:
            file_path.unlink(missing_ok=True)
            raise DocumentProcessingError(
                f"Failed to save file: {e}",
                code="STORAGE_ERROR",
            )

        return processed

    async def _process_document_content(
        self,
        document: Document,
        text_content: str,
        page_count: int | None,
        metadata: dict,
    ) -> None:
        """Store extracted text and metadata and mark the document VALIDATED.

        Args:
            document: Document record
            text_content: Extracted text (non-empty)
            page_count: Number of pages
            metadata: Title/authors extracted from the file
        """
        # Count words
        word_count = len(text_content.split())

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

        await self.repo.update_status(document.doc_id, DocumentStatus.VALIDATED)
        await self.repo.create_validation(
            doc_id=document.doc_id,
            is_valid=True,
            provenance_hash=hashlib.sha256(text_content.encode()).hexdigest(),
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

        Raises:
            DocumentProcessingError: If the PDF is encrypted or corrupt.
        """
        if not PYPDF_AVAILABLE:
            raise DocumentProcessingError(
                "pypdf library is not installed — cannot extract PDF content",
                code="MISSING_DEPENDENCY",
            )

        try:
            pdf_file = io.BytesIO(file_content)
            reader = pypdf.PdfReader(pdf_file)
        except Exception as e:
            raise DocumentProcessingError(
                f"File appears to be corrupt or is not a valid PDF: {e}",
                code="CORRUPT_PDF",
            )

        # Reject password-protected PDFs. Many publisher PDFs are encrypted
        # with an empty user password (permission flags only); those open
        # without a password, so only reject when "" does not decrypt.
        if reader.is_encrypted and not self._decrypts_with_empty_password(reader):
            raise DocumentProcessingError(
                "Password-protected PDFs are not supported. "
                "Please upload an unprotected PDF.",
                code="ENCRYPTED_PDF",
            )

        # pypdf parses lazily, so a broken page tree or metadata object only
        # fails here; report it as a corrupt file rather than a server error.
        # NULs are stripped throughout: PostgreSQL text columns reject them.
        try:
            # Extract metadata
            metadata: dict = {}
            if reader.metadata:
                if reader.metadata.title:
                    metadata["title"] = reader.metadata.title.replace("\x00", "")[:500]
                if reader.metadata.author:
                    authors = reader.metadata.author.replace("\x00", "")
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

            page_count = len(reader.pages)
        except Exception as e:
            raise DocumentProcessingError(
                f"File appears to be corrupt or is not a valid PDF: {e}",
                code="CORRUPT_PDF",
            )

        text_content = "\n\n".join(text_parts).replace("\x00", "")

        return text_content, page_count, metadata

    @staticmethod
    def _decrypts_with_empty_password(reader) -> bool:
        """Return True if an encrypted PDF opens with an empty user password."""
        try:
            return bool(reader.decrypt(""))
        except Exception:
            return False

    async def get_document(
        self, doc_id: str, owner_id: str | None = None
    ) -> Document | None:
        """Get document by ID.

        Args:
            doc_id: Document identifier
            owner_id: If given, only return the document when this user owns it

        Returns:
            Document if found, None otherwise
        """
        return await self.repo.get_by_id(doc_id, owner_id=owner_id)

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

    async def delete_document(self, doc_id: str, owner_id: str | None = None) -> bool:
        """Delete a document and its file.

        Args:
            doc_id: Document identifier
            owner_id: If given, only delete the document when this user owns it

        Returns:
            True if deleted, False if not found
        """
        # Get document to find file path
        document = await self.repo.get_by_id(doc_id, owner_id=owner_id)
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

    async def get_document_content(
        self, doc_id: str, owner_id: str | None = None
    ) -> str | None:
        """Get extracted text content of a document.

        Args:
            doc_id: Document identifier
            owner_id: If given, only return content when this user owns it

        Returns:
            Text content if found, None otherwise
        """
        return await self.repo.get_content(doc_id, owner_id=owner_id)

    async def paraphrase_content(
        self,
        doc_id: str,
        complexity: int = 50,
        owner_id: str | None = None,
    ) -> str | None:
        """Get paraphrased version of document content.

        Args:
            doc_id: Document identifier
            complexity: Complexity level (0=simplest, 100=original)
            owner_id: If given, only paraphrase when this user owns the document

        Returns:
            Paraphrased content, or None if document not found

        Note:
            This is a placeholder. Full implementation would use LLM.
        """
        content = await self.repo.get_content(doc_id, owner_id=owner_id)
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
