"""
Collaborative Annotation Service with CRDT-based synchronization.
Enables real-time collaborative annotations on documents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import uuid
import json


class AnnotationType(str, Enum):
    """Types of annotations."""
    HIGHLIGHT = "highlight"
    COMMENT = "comment"
    QUESTION = "question"
    ANSWER = "answer"
    LINK = "link"
    TAG = "tag"


class AnnotationVisibility(str, Enum):
    """Visibility levels for annotations."""
    PRIVATE = "private"  # Only creator can see
    GROUP = "group"  # Shared with specific group
    PUBLIC = "public"  # Visible to all document readers


@dataclass
class Annotation:
    """Represents a single annotation."""
    annotation_id: str
    document_id: str
    user_id: str
    user_name: str
    annotation_type: AnnotationType
    visibility: AnnotationVisibility = AnnotationVisibility.PRIVATE

    # Content
    content: str = ""  # Comment text, question, etc.
    selected_text: Optional[str] = None  # Highlighted text

    # Position
    start_offset: int = 0  # Character offset start
    end_offset: int = 0  # Character offset end
    section_id: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None  # For replies/threads
    tags: list[str] = field(default_factory=list)
    reactions: dict[str, list[str]] = field(default_factory=dict)  # emoji -> user_ids

    # CRDT fields
    vector_clock: dict[str, int] = field(default_factory=dict)
    is_deleted: bool = False

    def to_dict(self) -> dict:
        return {
            "annotation_id": self.annotation_id,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "type": self.annotation_type.value,
            "visibility": self.visibility.value,
            "content": self.content,
            "selected_text": self.selected_text,
            "position": {
                "start": self.start_offset,
                "end": self.end_offset,
                "section_id": self.section_id,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parent_id": self.parent_id,
            "tags": self.tags,
            "reactions": self.reactions,
            "is_deleted": self.is_deleted,
        }


@dataclass
class CRDTOperation:
    """Represents a CRDT operation for synchronization."""
    operation_id: str
    timestamp: datetime
    user_id: str
    operation_type: str  # "insert", "update", "delete"
    annotation_id: str
    data: dict = field(default_factory=dict)
    vector_clock: dict[str, int] = field(default_factory=dict)


class AnnotationCRDT:
    """
    CRDT-based conflict-free replicated data structure for annotations.
    Uses Last-Writer-Wins (LWW) with vector clocks for ordering.
    """

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self._annotations: dict[str, Annotation] = {}
        self._vector_clock: dict[str, int] = {replica_id: 0}
        self._pending_operations: list[CRDTOperation] = []

    def increment_clock(self) -> dict[str, int]:
        """Increment local vector clock."""
        self._vector_clock[self.replica_id] = (
            self._vector_clock.get(self.replica_id, 0) + 1
        )
        return self._vector_clock.copy()

    def merge_clock(self, remote_clock: dict[str, int]) -> None:
        """Merge remote vector clock with local."""
        for replica, count in remote_clock.items():
            self._vector_clock[replica] = max(
                self._vector_clock.get(replica, 0),
                count,
            )

    def _clock_compare(
        self,
        clock1: dict[str, int],
        clock2: dict[str, int],
    ) -> int:
        """
        Compare two vector clocks.
        Returns: -1 if clock1 < clock2, 1 if clock1 > clock2, 0 if concurrent
        """
        all_keys = set(clock1.keys()) | set(clock2.keys())
        less = False
        greater = False

        for key in all_keys:
            v1 = clock1.get(key, 0)
            v2 = clock2.get(key, 0)

            if v1 < v2:
                less = True
            elif v1 > v2:
                greater = True

        if less and not greater:
            return -1
        elif greater and not less:
            return 1
        else:
            return 0  # Concurrent

    def insert(self, annotation: Annotation) -> CRDTOperation:
        """Insert a new annotation."""
        clock = self.increment_clock()
        annotation.vector_clock = clock

        self._annotations[annotation.annotation_id] = annotation

        operation = CRDTOperation(
            operation_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=annotation.user_id,
            operation_type="insert",
            annotation_id=annotation.annotation_id,
            data=annotation.to_dict(),
            vector_clock=clock,
        )

        self._pending_operations.append(operation)
        return operation

    def update(
        self,
        annotation_id: str,
        user_id: str,
        updates: dict,
    ) -> Optional[CRDTOperation]:
        """Update an existing annotation."""
        if annotation_id not in self._annotations:
            return None

        annotation = self._annotations[annotation_id]
        clock = self.increment_clock()

        # Apply updates
        for key, value in updates.items():
            if hasattr(annotation, key):
                setattr(annotation, key, value)

        annotation.updated_at = datetime.utcnow()
        annotation.vector_clock = clock

        operation = CRDTOperation(
            operation_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            operation_type="update",
            annotation_id=annotation_id,
            data=updates,
            vector_clock=clock,
        )

        self._pending_operations.append(operation)
        return operation

    def delete(
        self,
        annotation_id: str,
        user_id: str,
    ) -> Optional[CRDTOperation]:
        """Soft delete an annotation (tombstone)."""
        if annotation_id not in self._annotations:
            return None

        annotation = self._annotations[annotation_id]
        clock = self.increment_clock()

        annotation.is_deleted = True
        annotation.updated_at = datetime.utcnow()
        annotation.vector_clock = clock

        operation = CRDTOperation(
            operation_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            operation_type="delete",
            annotation_id=annotation_id,
            data={},
            vector_clock=clock,
        )

        self._pending_operations.append(operation)
        return operation

    def apply_remote(self, operation: CRDTOperation) -> bool:
        """
        Apply a remote operation.

        Returns:
            True if applied, False if rejected
        """
        self.merge_clock(operation.vector_clock)

        if operation.operation_type == "insert":
            if operation.annotation_id not in self._annotations:
                # Reconstruct annotation from data
                data = operation.data
                annotation = Annotation(
                    annotation_id=data["annotation_id"],
                    document_id=data["document_id"],
                    user_id=data["user_id"],
                    user_name=data["user_name"],
                    annotation_type=AnnotationType(data["type"]),
                    visibility=AnnotationVisibility(data["visibility"]),
                    content=data.get("content", ""),
                    selected_text=data.get("selected_text"),
                    start_offset=data.get("position", {}).get("start", 0),
                    end_offset=data.get("position", {}).get("end", 0),
                    section_id=data.get("position", {}).get("section_id"),
                    tags=data.get("tags", []),
                    vector_clock=operation.vector_clock,
                )
                self._annotations[annotation.annotation_id] = annotation
                return True
            return False

        elif operation.operation_type == "update":
            if operation.annotation_id in self._annotations:
                existing = self._annotations[operation.annotation_id]

                # LWW: Apply if remote clock is greater or concurrent (use timestamp)
                cmp = self._clock_compare(
                    operation.vector_clock,
                    existing.vector_clock,
                )

                if cmp >= 0:  # Remote is newer or concurrent
                    for key, value in operation.data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.vector_clock = operation.vector_clock
                    return True
            return False

        elif operation.operation_type == "delete":
            if operation.annotation_id in self._annotations:
                existing = self._annotations[operation.annotation_id]
                existing.is_deleted = True
                existing.vector_clock = operation.vector_clock
                return True
            return False

        return False

    def get_pending_operations(self) -> list[CRDTOperation]:
        """Get and clear pending operations for sync."""
        ops = self._pending_operations
        self._pending_operations = []
        return ops

    def get_annotations(
        self,
        include_deleted: bool = False,
    ) -> list[Annotation]:
        """Get all annotations."""
        annotations = list(self._annotations.values())
        if not include_deleted:
            annotations = [a for a in annotations if not a.is_deleted]
        return annotations


class AnnotationService:
    """
    High-level service for managing collaborative annotations.
    """

    def __init__(self):
        self._documents: dict[str, AnnotationCRDT] = {}

    def _get_crdt(self, document_id: str, replica_id: str) -> AnnotationCRDT:
        """Get or create CRDT for a document."""
        if document_id not in self._documents:
            self._documents[document_id] = AnnotationCRDT(replica_id)
        return self._documents[document_id]

    def create_annotation(
        self,
        document_id: str,
        user_id: str,
        user_name: str,
        annotation_type: AnnotationType,
        content: str,
        selected_text: Optional[str] = None,
        start_offset: int = 0,
        end_offset: int = 0,
        section_id: Optional[str] = None,
        visibility: AnnotationVisibility = AnnotationVisibility.PRIVATE,
        parent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Annotation:
        """Create a new annotation."""
        crdt = self._get_crdt(document_id, user_id)

        annotation = Annotation(
            annotation_id=str(uuid.uuid4()),
            document_id=document_id,
            user_id=user_id,
            user_name=user_name,
            annotation_type=annotation_type,
            visibility=visibility,
            content=content,
            selected_text=selected_text,
            start_offset=start_offset,
            end_offset=end_offset,
            section_id=section_id,
            parent_id=parent_id,
            tags=tags or [],
        )

        crdt.insert(annotation)
        return annotation

    def update_annotation(
        self,
        document_id: str,
        annotation_id: str,
        user_id: str,
        updates: dict,
    ) -> Optional[Annotation]:
        """Update an existing annotation."""
        crdt = self._get_crdt(document_id, user_id)
        crdt.update(annotation_id, user_id, updates)

        annotations = crdt.get_annotations()
        for a in annotations:
            if a.annotation_id == annotation_id:
                return a
        return None

    def delete_annotation(
        self,
        document_id: str,
        annotation_id: str,
        user_id: str,
    ) -> bool:
        """Delete an annotation."""
        crdt = self._get_crdt(document_id, user_id)
        op = crdt.delete(annotation_id, user_id)
        return op is not None

    def add_reaction(
        self,
        document_id: str,
        annotation_id: str,
        user_id: str,
        emoji: str,
    ) -> Optional[Annotation]:
        """Add a reaction to an annotation."""
        crdt = self._get_crdt(document_id, user_id)

        for annotation in crdt.get_annotations():
            if annotation.annotation_id == annotation_id:
                if emoji not in annotation.reactions:
                    annotation.reactions[emoji] = []
                if user_id not in annotation.reactions[emoji]:
                    annotation.reactions[emoji].append(user_id)
                    crdt.update(
                        annotation_id,
                        user_id,
                        {"reactions": annotation.reactions},
                    )
                return annotation
        return None

    def get_annotations(
        self,
        document_id: str,
        user_id: str,
        visibility_filter: Optional[list[AnnotationVisibility]] = None,
        type_filter: Optional[list[AnnotationType]] = None,
        section_id: Optional[str] = None,
    ) -> list[Annotation]:
        """
        Get annotations for a document with filters.

        Args:
            document_id: Document to get annotations for
            user_id: Current user (for visibility filtering)
            visibility_filter: Filter by visibility levels
            type_filter: Filter by annotation types
            section_id: Filter by section

        Returns:
            List of matching annotations
        """
        crdt = self._get_crdt(document_id, user_id)
        annotations = crdt.get_annotations()

        # Apply filters
        result = []
        for a in annotations:
            # Visibility filter
            if a.visibility == AnnotationVisibility.PRIVATE and a.user_id != user_id:
                continue

            if visibility_filter and a.visibility not in visibility_filter:
                continue

            if type_filter and a.annotation_type not in type_filter:
                continue

            if section_id and a.section_id != section_id:
                continue

            result.append(a)

        # Sort by position
        result.sort(key=lambda x: x.start_offset)
        return result

    def get_thread(
        self,
        document_id: str,
        parent_id: str,
        user_id: str,
    ) -> list[Annotation]:
        """Get all replies in a thread."""
        annotations = self.get_annotations(document_id, user_id)
        return [a for a in annotations if a.parent_id == parent_id]

    def get_annotation_stats(
        self,
        document_id: str,
        user_id: str,
    ) -> dict:
        """Get annotation statistics for a document."""
        annotations = self.get_annotations(document_id, user_id)

        stats = {
            "total": len(annotations),
            "by_type": {},
            "by_user": {},
            "by_section": {},
        }

        for a in annotations:
            # By type
            type_key = a.annotation_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1

            # By user
            stats["by_user"][a.user_name] = stats["by_user"].get(a.user_name, 0) + 1

            # By section
            if a.section_id:
                stats["by_section"][a.section_id] = (
                    stats["by_section"].get(a.section_id, 0) + 1
                )

        return stats


# Singleton instance
_annotation_service: Optional[AnnotationService] = None


def get_annotation_service() -> AnnotationService:
    """Get or create singleton AnnotationService instance."""
    global _annotation_service
    if _annotation_service is None:
        _annotation_service = AnnotationService()
    return _annotation_service
