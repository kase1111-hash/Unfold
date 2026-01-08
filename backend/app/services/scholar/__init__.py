# Scholar services for Phase 5
from .citations import (
    CitationService,
    CitationTree,
    CitationNode,
    get_citation_service,
)
from .credibility import (
    CredibilityScorer,
    CredibilityScore,
    CredibilityLevel,
    get_credibility_scorer,
)
from .zotero import (
    ZoteroExporter,
    ZoteroCollection,
    ZoteroItem,
    get_zotero_exporter,
)
from .reflection import (
    ReflectionEngine,
    ReadingSnapshot,
    ReflectionDiff,
    ReflectionType,
    get_reflection_engine,
)
from .collaboration import (
    AnnotationService,
    Annotation,
    AnnotationCRDT,
    AnnotationType,
    AnnotationVisibility,
    get_annotation_service,
)

__all__ = [
    # Citations
    "CitationService",
    "CitationTree",
    "CitationNode",
    "get_citation_service",
    # Credibility
    "CredibilityScorer",
    "CredibilityScore",
    "CredibilityLevel",
    "get_credibility_scorer",
    # Zotero
    "ZoteroExporter",
    "ZoteroCollection",
    "ZoteroItem",
    "get_zotero_exporter",
    # Reflection
    "ReflectionEngine",
    "ReadingSnapshot",
    "ReflectionDiff",
    "ReflectionType",
    "get_reflection_engine",
    # Collaboration
    "AnnotationService",
    "Annotation",
    "AnnotationCRDT",
    "AnnotationType",
    "AnnotationVisibility",
    "get_annotation_service",
]
