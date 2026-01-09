"""
Ethics Services Module for Unfold.
Implements provenance tracking, bias detection, privacy compliance, and analytics.
"""

from .provenance import (
    ProvenanceService,
    ProvenanceManifest,
    ContentCredential,
    ProvenanceAssertion,
    ProvenanceAction,
    ValidationStatus,
    get_provenance_service,
)

from .audit import (
    BiasAuditor,
    BiasAuditReport,
    BiasFinding,
    InclusivityMetrics,
    SentimentResult,
    BiasCategory,
    SeverityLevel,
    get_bias_auditor,
)

from .privacy import (
    PrivacyCompliance,
    DifferentialPrivacy,
    ConsentRecord,
    DataRetentionPolicy,
    PrivacyReport,
    ConsentType,
    ConsentStatus,
    DataCategory,
    get_privacy_compliance,
)

from .analytics import (
    EthicsAnalytics,
    EthicsDashboard,
    AIOperation,
    EthicsMetric,
    UserEthicsProfile,
    AggregatedReport,
    MetricType,
    TransparencyLevel,
    get_ethics_analytics,
)

__all__ = [
    # Provenance
    "ProvenanceService",
    "ProvenanceManifest",
    "ContentCredential",
    "ProvenanceAssertion",
    "ProvenanceAction",
    "ValidationStatus",
    "get_provenance_service",
    # Audit
    "BiasAuditor",
    "BiasAuditReport",
    "BiasFinding",
    "InclusivityMetrics",
    "SentimentResult",
    "BiasCategory",
    "SeverityLevel",
    "get_bias_auditor",
    # Privacy
    "PrivacyCompliance",
    "DifferentialPrivacy",
    "ConsentRecord",
    "DataRetentionPolicy",
    "PrivacyReport",
    "ConsentType",
    "ConsentStatus",
    "DataCategory",
    "get_privacy_compliance",
    # Analytics
    "EthicsAnalytics",
    "EthicsDashboard",
    "AIOperation",
    "EthicsMetric",
    "UserEthicsProfile",
    "AggregatedReport",
    "MetricType",
    "TransparencyLevel",
    "get_ethics_analytics",
]
