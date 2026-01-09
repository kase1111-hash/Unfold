"""
Provenance System for content authenticity and verification.
Implements C2PA-style manifests and SHA-256 content fingerprinting.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class ProvenanceAction(str, Enum):
    """Types of provenance actions."""

    CREATED = "created"
    MODIFIED = "modified"
    IMPORTED = "imported"
    VALIDATED = "validated"
    TRANSFORMED = "transformed"
    ANNOTATED = "annotated"


class ValidationStatus(str, Enum):
    """Document validation status."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    INVALID = "invalid"
    PENDING = "pending"


@dataclass
class ProvenanceAssertion:
    """A single provenance assertion/claim."""

    assertion_id: str
    action: ProvenanceAction
    actor: str  # User ID or system component
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "assertion_id": self.assertion_id,
            "action": self.action.value,
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class ContentCredential:
    """C2PA-style content credential for a document."""

    credential_id: str
    document_id: str
    content_hash: str  # SHA-256 hash of content
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Validation
    validation_status: ValidationStatus = ValidationStatus.PENDING
    doi_verified: bool = False
    license_verified: bool = False
    author_verified: bool = False

    # Provenance chain
    assertions: list[ProvenanceAssertion] = field(default_factory=list)

    # Signatures
    signature: Optional[str] = None  # Digital signature
    signer_id: Optional[str] = None

    # Parent credentials for derived content
    parent_credentials: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "credential_id": self.credential_id,
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "validation": {
                "status": self.validation_status.value,
                "doi_verified": self.doi_verified,
                "license_verified": self.license_verified,
                "author_verified": self.author_verified,
            },
            "assertions": [a.to_dict() for a in self.assertions],
            "signature": self.signature,
            "signer_id": self.signer_id,
            "parent_credentials": self.parent_credentials,
        }


@dataclass
class ProvenanceManifest:
    """Complete C2PA-style manifest for a document."""

    manifest_id: str
    document_id: str
    title: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Content credential
    credential: Optional[ContentCredential] = None

    # Metadata
    authors: list[str] = field(default_factory=list)
    doi: Optional[str] = None
    license: Optional[str] = None
    source: Optional[str] = None

    # Integrity
    manifest_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "manifest_id": self.manifest_id,
            "document_id": self.document_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "credential": self.credential.to_dict() if self.credential else None,
            "metadata": {
                "authors": self.authors,
                "doi": self.doi,
                "license": self.license,
                "source": self.source,
            },
            "manifest_hash": self.manifest_hash,
        }


class ProvenanceService:
    """
    Service for managing content provenance and C2PA-style credentials.
    """

    def __init__(self):
        self._manifests: dict[str, ProvenanceManifest] = {}
        self._credentials: dict[str, ContentCredential] = {}

    def compute_content_hash(self, content: bytes | str) -> str:
        """
        Compute SHA-256 hash of content.

        Args:
            content: Content to hash (bytes or string)

        Returns:
            Hex-encoded SHA-256 hash
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def create_credential(
        self,
        document_id: str,
        content: bytes | str,
        actor: str,
    ) -> ContentCredential:
        """
        Create a new content credential for a document.

        Args:
            document_id: Document identifier
            content: Document content for hashing
            actor: User or system creating the credential

        Returns:
            New ContentCredential
        """
        content_hash = self.compute_content_hash(content)
        credential_id = f"cred_{uuid.uuid4().hex[:12]}"

        initial_assertion = ProvenanceAssertion(
            assertion_id=f"assert_{uuid.uuid4().hex[:8]}",
            action=ProvenanceAction.CREATED,
            actor=actor,
            description="Initial content credential creation",
        )

        credential = ContentCredential(
            credential_id=credential_id,
            document_id=document_id,
            content_hash=content_hash,
            assertions=[initial_assertion],
        )

        self._credentials[credential_id] = credential
        return credential

    def verify_content_integrity(
        self,
        credential_id: str,
        content: bytes | str,
    ) -> tuple[bool, str]:
        """
        Verify content integrity against stored credential.

        Args:
            credential_id: Credential to verify against
            content: Current content to verify

        Returns:
            Tuple of (is_valid, message)
        """
        if credential_id not in self._credentials:
            return False, "Credential not found"

        credential = self._credentials[credential_id]
        current_hash = self.compute_content_hash(content)

        if current_hash == credential.content_hash:
            return True, "Content integrity verified"
        else:
            return (
                False,
                f"Content hash mismatch: expected {credential.content_hash[:16]}..., got {current_hash[:16]}...",
            )

    def add_assertion(
        self,
        credential_id: str,
        action: ProvenanceAction,
        actor: str,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[ProvenanceAssertion]:
        """
        Add a provenance assertion to a credential.

        Args:
            credential_id: Credential to add assertion to
            action: Type of action being recorded
            actor: User or system performing the action
            description: Human-readable description
            metadata: Additional metadata

        Returns:
            New assertion or None if credential not found
        """
        if credential_id not in self._credentials:
            return None

        assertion = ProvenanceAssertion(
            assertion_id=f"assert_{uuid.uuid4().hex[:8]}",
            action=action,
            actor=actor,
            description=description,
            metadata=metadata or {},
        )

        self._credentials[credential_id].assertions.append(assertion)
        return assertion

    def update_validation_status(
        self,
        credential_id: str,
        doi_verified: Optional[bool] = None,
        license_verified: Optional[bool] = None,
        author_verified: Optional[bool] = None,
    ) -> Optional[ContentCredential]:
        """
        Update validation status of a credential.

        Args:
            credential_id: Credential to update
            doi_verified: DOI verification status
            license_verified: License verification status
            author_verified: Author verification status

        Returns:
            Updated credential or None if not found
        """
        if credential_id not in self._credentials:
            return None

        credential = self._credentials[credential_id]

        if doi_verified is not None:
            credential.doi_verified = doi_verified
        if license_verified is not None:
            credential.license_verified = license_verified
        if author_verified is not None:
            credential.author_verified = author_verified

        # Update overall status
        if credential.doi_verified and credential.license_verified:
            credential.validation_status = ValidationStatus.VERIFIED
        elif not credential.doi_verified and not credential.license_verified:
            credential.validation_status = ValidationStatus.UNVERIFIED
        else:
            credential.validation_status = ValidationStatus.PENDING

        # Add validation assertion
        self.add_assertion(
            credential_id,
            ProvenanceAction.VALIDATED,
            "system",
            f"Validation updated: DOI={doi_verified}, License={license_verified}, Author={author_verified}",
        )

        return credential

    def create_manifest(
        self,
        document_id: str,
        title: str,
        content: bytes | str,
        actor: str,
        authors: Optional[list[str]] = None,
        doi: Optional[str] = None,
        license: Optional[str] = None,
        source: Optional[str] = None,
    ) -> ProvenanceManifest:
        """
        Create a complete provenance manifest for a document.

        Args:
            document_id: Document identifier
            title: Document title
            content: Document content for hashing
            actor: User creating the manifest
            authors: Document authors
            doi: Document DOI
            license: Document license
            source: Document source

        Returns:
            New ProvenanceManifest
        """
        credential = self.create_credential(document_id, content, actor)

        manifest_id = f"manifest_{uuid.uuid4().hex[:12]}"

        manifest = ProvenanceManifest(
            manifest_id=manifest_id,
            document_id=document_id,
            title=title,
            credential=credential,
            authors=authors or [],
            doi=doi,
            license=license,
            source=source,
        )

        # Compute manifest hash
        manifest_data = json.dumps(manifest.to_dict(), sort_keys=True)
        manifest.manifest_hash = self.compute_content_hash(manifest_data)

        self._manifests[manifest_id] = manifest
        return manifest

    def get_manifest(self, manifest_id: str) -> Optional[ProvenanceManifest]:
        """Get a manifest by ID."""
        return self._manifests.get(manifest_id)

    def get_manifest_by_document(
        self, document_id: str
    ) -> Optional[ProvenanceManifest]:
        """Get a manifest by document ID."""
        for manifest in self._manifests.values():
            if manifest.document_id == document_id:
                return manifest
        return None

    def get_credential(self, credential_id: str) -> Optional[ContentCredential]:
        """Get a credential by ID."""
        return self._credentials.get(credential_id)

    def get_provenance_chain(self, credential_id: str) -> list[ProvenanceAssertion]:
        """
        Get the complete provenance chain for a credential.

        Args:
            credential_id: Credential to get chain for

        Returns:
            List of assertions in chronological order
        """
        if credential_id not in self._credentials:
            return []

        return sorted(
            self._credentials[credential_id].assertions,
            key=lambda a: a.timestamp,
        )

    def verify_manifest_integrity(self, manifest_id: str) -> tuple[bool, str]:
        """
        Verify manifest integrity by recomputing hash.

        Args:
            manifest_id: Manifest to verify

        Returns:
            Tuple of (is_valid, message)
        """
        if manifest_id not in self._manifests:
            return False, "Manifest not found"

        manifest = self._manifests[manifest_id]
        stored_hash = manifest.manifest_hash

        # Temporarily clear hash for recomputation
        manifest.manifest_hash = None
        manifest_data = json.dumps(manifest.to_dict(), sort_keys=True)
        current_hash = self.compute_content_hash(manifest_data)
        manifest.manifest_hash = stored_hash

        if current_hash == stored_hash:
            return True, "Manifest integrity verified"
        else:
            return False, "Manifest hash mismatch - possible tampering detected"

    def export_manifest_json(self, manifest_id: str) -> Optional[str]:
        """
        Export manifest as JSON string.

        Args:
            manifest_id: Manifest to export

        Returns:
            JSON string or None if not found
        """
        manifest = self.get_manifest(manifest_id)
        if manifest is None:
            return None

        return json.dumps(manifest.to_dict(), indent=2)


# Singleton instance
_provenance_service: Optional[ProvenanceService] = None


def get_provenance_service() -> ProvenanceService:
    """Get or create singleton ProvenanceService instance."""
    global _provenance_service
    if _provenance_service is None:
        _provenance_service = ProvenanceService()
    return _provenance_service
