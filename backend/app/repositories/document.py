"""Document repository for database operations."""

import hashlib
import logging
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import DocumentORM, DocumentValidationORM
from app.models.document import (
    Document,
    DocumentCreate,
    DocumentLicense,
    DocumentSource,
    DocumentStatus,
    DocumentUpdate,
    DocumentValidation,
)

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for document database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    @staticmethod
    def generate_doc_id(content: bytes) -> str:
        """Generate document ID from content hash.

        Args:
            content: Document file content

        Returns:
            SHA-256 hash prefixed with 'sha256:'
        """
        hash_value = hashlib.sha256(content).hexdigest()
        return f"sha256:{hash_value}"

    async def create(
        self,
        doc_id: str,
        title: str,
        owner_id: str | None = None,
        authors: list[str] | None = None,
        doi: str | None = None,
        abstract: str | None = None,
        license: DocumentLicense | None = None,
        source: DocumentSource = DocumentSource.UPLOAD,
        file_path: str | None = None,
        file_size_bytes: int | None = None,
        content: str | None = None,
    ) -> Document:
        """Create a new document.

        Args:
            doc_id: Document ID (SHA-256 hash)
            title: Document title
            owner_id: Owner user ID
            authors: List of author names
            doi: Digital Object Identifier
            abstract: Document abstract
            license: Document license
            source: Document source
            file_path: Path to stored file
            file_size_bytes: File size in bytes
            content: Extracted text content

        Returns:
            Created document
        """
        doc_orm = DocumentORM(
            doc_id=doc_id,
            title=title,
            owner_id=owner_id,
            authors=authors or [],
            doi=doi,
            abstract=abstract,
            license=license,
            source=source,
            status=DocumentStatus.PENDING,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            content=content,
            graph_nodes=[],
        )

        self.session.add(doc_orm)
        await self.session.flush()
        await self.session.refresh(doc_orm)

        logger.info(f"Created document: {doc_id}")
        return self._to_model(doc_orm)

    async def get_by_id(self, doc_id: str) -> Document | None:
        """Get document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document if found, None otherwise
        """
        result = await self.session.execute(
            select(DocumentORM).where(DocumentORM.doc_id == doc_id)
        )
        doc_orm = result.scalar_one_or_none()

        if doc_orm is None:
            return None

        return self._to_model(doc_orm)

    async def get_by_doi(self, doi: str) -> Document | None:
        """Get document by DOI.

        Args:
            doi: Digital Object Identifier

        Returns:
            Document if found, None otherwise
        """
        result = await self.session.execute(
            select(DocumentORM).where(DocumentORM.doi == doi)
        )
        doc_orm = result.scalar_one_or_none()

        if doc_orm is None:
            return None

        return self._to_model(doc_orm)

    async def list_documents(
        self,
        owner_id: str | None = None,
        status: DocumentStatus | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Document], int]:
        """List documents with pagination.

        Args:
            owner_id: Filter by owner (optional)
            status: Filter by status (optional)
            page: Page number (1-indexed)
            page_size: Items per page

        Returns:
            Tuple of (documents list, total count)
        """
        query = select(DocumentORM)
        count_query = select(func.count(DocumentORM.doc_id))

        if owner_id is not None:
            query = query.where(DocumentORM.owner_id == owner_id)
            count_query = count_query.where(DocumentORM.owner_id == owner_id)

        if status is not None:
            query = query.where(DocumentORM.status == status)
            count_query = count_query.where(DocumentORM.status == status)

        # Get total count
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        # Get paginated results
        offset = (page - 1) * page_size
        query = (
            query.order_by(DocumentORM.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(query)
        docs_orm = result.scalars().all()

        return [self._to_model(doc) for doc in docs_orm], total

    async def update(self, doc_id: str, update_data: DocumentUpdate) -> Document | None:
        """Update document metadata.

        Args:
            doc_id: Document identifier
            update_data: Fields to update

        Returns:
            Updated document if found, None otherwise
        """
        update_dict = update_data.model_dump(exclude_unset=True)

        if not update_dict:
            return await self.get_by_id(doc_id)

        update_dict["updated_at"] = datetime.now(timezone.utc)

        await self.session.execute(
            update(DocumentORM)
            .where(DocumentORM.doc_id == doc_id)
            .values(**update_dict)
        )

        return await self.get_by_id(doc_id)

    async def update_status(
        self, doc_id: str, status: DocumentStatus
    ) -> Document | None:
        """Update document processing status.

        Args:
            doc_id: Document identifier
            status: New status

        Returns:
            Updated document if found, None otherwise
        """
        await self.session.execute(
            update(DocumentORM)
            .where(DocumentORM.doc_id == doc_id)
            .values(status=status, updated_at=datetime.now(timezone.utc))
        )
        return await self.get_by_id(doc_id)

    async def update_content(
        self,
        doc_id: str,
        content: str,
        page_count: int | None = None,
        word_count: int | None = None,
    ) -> Document | None:
        """Update document extracted content.

        Args:
            doc_id: Document identifier
            content: Extracted text content
            page_count: Number of pages
            word_count: Word count

        Returns:
            Updated document if found, None otherwise
        """
        values: dict = {
            "content": content,
            "updated_at": datetime.now(timezone.utc),
        }
        if page_count is not None:
            values["page_count"] = page_count
        if word_count is not None:
            values["word_count"] = word_count

        await self.session.execute(
            update(DocumentORM).where(DocumentORM.doc_id == doc_id).values(**values)
        )
        return await self.get_by_id(doc_id)

    async def add_graph_nodes(self, doc_id: str, node_ids: list[str]) -> bool:
        """Add graph node IDs to document.

        Args:
            doc_id: Document identifier
            node_ids: List of graph node IDs to add

        Returns:
            True if updated, False if document not found
        """
        doc = await self.get_by_id(doc_id)
        if doc is None:
            return False

        existing_nodes = set(doc.graph_nodes)
        new_nodes = list(existing_nodes | set(node_ids))

        await self.session.execute(
            update(DocumentORM)
            .where(DocumentORM.doc_id == doc_id)
            .values(graph_nodes=new_nodes, updated_at=datetime.now(timezone.utc))
        )
        return True

    async def delete(self, doc_id: str) -> bool:
        """Delete a document.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            delete(DocumentORM).where(DocumentORM.doc_id == doc_id)
        )
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted document: {doc_id}")
        return deleted

    async def exists(self, doc_id: str) -> bool:
        """Check if document exists.

        Args:
            doc_id: Document identifier

        Returns:
            True if exists, False otherwise
        """
        result = await self.session.execute(
            select(DocumentORM.doc_id).where(DocumentORM.doc_id == doc_id).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def get_content(self, doc_id: str) -> str | None:
        """Get document text content.

        Args:
            doc_id: Document identifier

        Returns:
            Text content if found, None otherwise
        """
        result = await self.session.execute(
            select(DocumentORM.content).where(DocumentORM.doc_id == doc_id)
        )
        return result.scalar_one_or_none()

    # Validation methods

    async def create_validation(
        self,
        doc_id: str,
        is_valid: bool,
        doi_verified: bool = False,
        license_verified: bool = False,
        author_verified: bool = False,
        provenance_hash: str | None = None,
        validation_errors: list[str] | None = None,
    ) -> DocumentValidation:
        """Create validation record for a document.

        Args:
            doc_id: Document identifier
            is_valid: Whether document passed validation
            doi_verified: Whether DOI was verified
            license_verified: Whether license was verified
            author_verified: Whether authors were verified
            provenance_hash: Content provenance hash
            validation_errors: List of validation errors

        Returns:
            Created validation record
        """
        validation_orm = DocumentValidationORM(
            doc_id=doc_id,
            is_valid=is_valid,
            doi_verified=doi_verified,
            license_verified=license_verified,
            author_verified=author_verified,
            provenance_hash=provenance_hash,
            validation_errors=validation_errors or [],
            validated_at=datetime.now(timezone.utc),
        )

        self.session.add(validation_orm)
        await self.session.flush()
        await self.session.refresh(validation_orm)

        return DocumentValidation(
            doc_id=validation_orm.doc_id,
            is_valid=validation_orm.is_valid,
            doi_verified=validation_orm.doi_verified,
            license_verified=validation_orm.license_verified,
            author_verified=validation_orm.author_verified,
            provenance_hash=validation_orm.provenance_hash,
            validation_errors=validation_orm.validation_errors,
            validated_at=validation_orm.validated_at,
        )

    async def get_validation(self, doc_id: str) -> DocumentValidation | None:
        """Get validation record for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Validation record if found, None otherwise
        """
        result = await self.session.execute(
            select(DocumentValidationORM).where(
                DocumentValidationORM.doc_id == doc_id
            )
        )
        validation_orm = result.scalar_one_or_none()

        if validation_orm is None:
            return None

        return DocumentValidation(
            doc_id=validation_orm.doc_id,
            is_valid=validation_orm.is_valid,
            doi_verified=validation_orm.doi_verified,
            license_verified=validation_orm.license_verified,
            author_verified=validation_orm.author_verified,
            provenance_hash=validation_orm.provenance_hash,
            validation_errors=validation_orm.validation_errors,
            validated_at=validation_orm.validated_at,
        )

    def _to_model(self, doc_orm: DocumentORM) -> Document:
        """Convert ORM model to Pydantic model.

        Args:
            doc_orm: SQLAlchemy ORM model

        Returns:
            Pydantic Document model
        """
        return Document(
            doc_id=doc_orm.doc_id,
            title=doc_orm.title,
            authors=doc_orm.authors,
            doi=doc_orm.doi,
            abstract=doc_orm.abstract,
            license=doc_orm.license,
            source=doc_orm.source,
            status=doc_orm.status,
            vector_id=doc_orm.vector_id,
            graph_nodes=doc_orm.graph_nodes,
            file_path=doc_orm.file_path,
            file_size_bytes=doc_orm.file_size_bytes,
            page_count=doc_orm.page_count,
            word_count=doc_orm.word_count,
            created_at=doc_orm.created_at,
            updated_at=doc_orm.updated_at,
        )
