"""
Privacy Compliance Module for GDPR and differential privacy.
Implements data anonymization and consent management.
"""

import hashlib
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from enum import Enum
import uuid


class ConsentType(str, Enum):
    """Types of user consent."""

    ANALYTICS = "analytics"
    PERSONALIZATION = "personalization"
    THIRD_PARTY = "third_party"
    MARKETING = "marketing"
    RESEARCH = "research"


class ConsentStatus(str, Enum):
    """Consent status."""

    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"


class DataCategory(str, Enum):
    """Categories of personal data."""

    IDENTIFIER = "identifier"
    CONTACT = "contact"
    BEHAVIORAL = "behavioral"
    PREFERENCES = "preferences"
    LOCATION = "location"
    BIOMETRIC = "biometric"
    FINANCIAL = "financial"
    SENSITIVE = "sensitive"


@dataclass
class ConsentRecord:
    """Record of user consent."""

    consent_id: str
    user_id: str
    consent_type: ConsentType
    status: ConsentStatus
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    ip_address: Optional[str] = None  # Anonymized
    user_agent: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "consent_id": self.consent_id,
            "user_id": self.user_id,
            "consent_type": self.consent_type.value,
            "status": self.status.value,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "withdrawn_at": (
                self.withdrawn_at.isoformat() if self.withdrawn_at else None
            ),
        }


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""

    category: DataCategory
    retention_days: int
    anonymize_after_days: int
    delete_after_days: int
    legal_basis: str

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "retention_days": self.retention_days,
            "anonymize_after_days": self.anonymize_after_days,
            "delete_after_days": self.delete_after_days,
            "legal_basis": self.legal_basis,
        }


@dataclass
class PrivacyReport:
    """User privacy report (GDPR data subject access request)."""

    report_id: str
    user_id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Data categories held
    data_categories: list[DataCategory] = field(default_factory=list)

    # Consent status
    consents: list[ConsentRecord] = field(default_factory=list)

    # Data summary
    data_summary: dict = field(default_factory=dict)

    # Processing activities
    processing_activities: list[str] = field(default_factory=list)

    # Third-party sharing
    third_party_recipients: list[str] = field(default_factory=list)

    # Rights information
    user_rights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "user_id": self.user_id,
            "generated_at": self.generated_at.isoformat(),
            "data_categories": [c.value for c in self.data_categories],
            "consents": [c.to_dict() for c in self.consents],
            "data_summary": self.data_summary,
            "processing_activities": self.processing_activities,
            "third_party_recipients": self.third_party_recipients,
            "user_rights": self.user_rights,
        }


class DifferentialPrivacy:
    """
    Implements differential privacy mechanisms for data protection.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy with privacy budget.

        Args:
            epsilon: Privacy parameter (lower = more private)
            delta: Probability bound for privacy guarantee
        """
        self.epsilon = epsilon
        self.delta = delta

    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """
        Add Laplace noise for epsilon-differential privacy.

        Args:
            value: Original value
            sensitivity: Query sensitivity

        Returns:
            Noised value
        """
        scale = sensitivity / self.epsilon
        noise = random.uniform(-0.5, 0.5)
        laplace_noise = -scale * math.copysign(1, noise) * math.log(1 - 2 * abs(noise))
        return value + laplace_noise

    def add_gaussian_noise(self, value: float, sensitivity: float) -> float:
        """
        Add Gaussian noise for (epsilon, delta)-differential privacy.

        Args:
            value: Original value
            sensitivity: Query sensitivity

        Returns:
            Noised value
        """
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        noise = random.gauss(0, sigma)
        return value + noise

    def randomized_response(self, value: bool, probability: float = 0.75) -> bool:
        """
        Randomized response for boolean values.

        Args:
            value: True value
            probability: Probability of reporting true value

        Returns:
            Possibly flipped value
        """
        if random.random() < probability:
            return value
        else:
            return random.choice([True, False])

    def private_count(self, count: int, sensitivity: int = 1) -> int:
        """
        Return differentially private count.

        Args:
            count: True count
            sensitivity: Maximum change from one record

        Returns:
            Private count (rounded)
        """
        return max(0, round(self.add_laplace_noise(count, sensitivity)))

    def private_mean(self, values: list[float], bounds: tuple[float, float]) -> float:
        """
        Return differentially private mean.

        Args:
            values: List of values
            bounds: (min, max) bounds for values

        Returns:
            Private mean
        """
        if not values:
            return 0.0

        # Clip values to bounds
        clipped = [max(bounds[0], min(bounds[1], v)) for v in values]
        true_mean = sum(clipped) / len(clipped)

        # Sensitivity is (max - min) / n
        sensitivity = (bounds[1] - bounds[0]) / len(values)

        return self.add_laplace_noise(true_mean, sensitivity)

    def private_histogram(
        self,
        values: list[Any],
        num_bins: int,
    ) -> dict[Any, int]:
        """
        Return differentially private histogram.

        Args:
            values: List of categorical values
            num_bins: Number of bins/categories

        Returns:
            Private histogram
        """
        # Count actual values
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1

        # Add noise to each count
        private_counts = {}
        for key, count in counts.items():
            private_counts[key] = self.private_count(count)

        return private_counts


class PrivacyCompliance:
    """
    Service for managing privacy compliance and GDPR requirements.
    """

    # Default retention policies
    DEFAULT_POLICIES = {
        DataCategory.IDENTIFIER: DataRetentionPolicy(
            category=DataCategory.IDENTIFIER,
            retention_days=365,
            anonymize_after_days=180,
            delete_after_days=730,
            legal_basis="Contract performance",
        ),
        DataCategory.BEHAVIORAL: DataRetentionPolicy(
            category=DataCategory.BEHAVIORAL,
            retention_days=90,
            anonymize_after_days=30,
            delete_after_days=180,
            legal_basis="Legitimate interest",
        ),
        DataCategory.PREFERENCES: DataRetentionPolicy(
            category=DataCategory.PREFERENCES,
            retention_days=365,
            anonymize_after_days=365,
            delete_after_days=730,
            legal_basis="Consent",
        ),
    }

    # User rights under GDPR
    USER_RIGHTS = [
        "Right to access your personal data",
        "Right to rectification of inaccurate data",
        "Right to erasure ('right to be forgotten')",
        "Right to restrict processing",
        "Right to data portability",
        "Right to object to processing",
        "Right not to be subject to automated decision-making",
    ]

    def __init__(self, epsilon: float = 1.0):
        self._consents: dict[str, dict[ConsentType, ConsentRecord]] = {}
        self._data_records: dict[str, dict] = {}
        self.differential_privacy = DifferentialPrivacy(epsilon=epsilon)

    def anonymize_identifier(self, identifier: str, salt: str = "") -> str:
        """
        Anonymize an identifier using one-way hashing.

        Args:
            identifier: Original identifier
            salt: Optional salt for additional security

        Returns:
            Anonymized identifier
        """
        combined = f"{identifier}{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def anonymize_ip_address(self, ip: str) -> str:
        """
        Anonymize IP address by zeroing last octet(s).

        Args:
            ip: Original IP address

        Returns:
            Anonymized IP
        """
        parts = ip.split(".")
        if len(parts) == 4:
            # IPv4: zero last octet
            return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
        # IPv6 or invalid: hash it
        return self.anonymize_identifier(ip)

    def mask_email(self, email: str) -> str:
        """
        Mask email address for display.

        Args:
            email: Original email

        Returns:
            Masked email
        """
        if "@" not in email:
            return "***"

        local, domain = email.rsplit("@", 1)
        if len(local) <= 2:
            masked_local = "*" * len(local)
        else:
            masked_local = local[0] + "*" * (len(local) - 2) + local[-1]

        return f"{masked_local}@{domain}"

    def record_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expires_days: int = 365,
    ) -> ConsentRecord:
        """
        Record user consent decision.

        Args:
            user_id: User identifier
            consent_type: Type of consent
            granted: Whether consent was granted
            ip_address: User's IP address
            user_agent: User's browser
            expires_days: Consent expiration in days

        Returns:
            ConsentRecord
        """
        consent = ConsentRecord(
            consent_id=f"consent_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            consent_type=consent_type,
            status=ConsentStatus.GRANTED if granted else ConsentStatus.DENIED,
            granted_at=datetime.utcnow() if granted else None,
            expires_at=(
                datetime.utcnow() + timedelta(days=expires_days) if granted else None
            ),
            ip_address=self.anonymize_ip_address(ip_address) if ip_address else None,
            user_agent=user_agent[:100] if user_agent else None,
        )

        if user_id not in self._consents:
            self._consents[user_id] = {}
        self._consents[user_id][consent_type] = consent

        return consent

    def withdraw_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> Optional[ConsentRecord]:
        """
        Withdraw previously granted consent.

        Args:
            user_id: User identifier
            consent_type: Type of consent to withdraw

        Returns:
            Updated ConsentRecord or None
        """
        if user_id not in self._consents:
            return None
        if consent_type not in self._consents[user_id]:
            return None

        consent = self._consents[user_id][consent_type]
        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = datetime.utcnow()

        return consent

    def check_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> bool:
        """
        Check if user has granted consent for a specific type.

        Args:
            user_id: User identifier
            consent_type: Type of consent to check

        Returns:
            True if consent is active
        """
        if user_id not in self._consents:
            return False
        if consent_type not in self._consents[user_id]:
            return False

        consent = self._consents[user_id][consent_type]

        # Check status
        if consent.status != ConsentStatus.GRANTED:
            return False

        # Check expiration
        if consent.expires_at and consent.expires_at < datetime.utcnow():
            return False

        return True

    def get_user_consents(self, user_id: str) -> list[ConsentRecord]:
        """
        Get all consent records for a user.

        Args:
            user_id: User identifier

        Returns:
            List of consent records
        """
        if user_id not in self._consents:
            return []
        return list(self._consents[user_id].values())

    def generate_privacy_report(
        self,
        user_id: str,
        user_data: Optional[dict] = None,
    ) -> PrivacyReport:
        """
        Generate a GDPR-compliant privacy report for a user.

        Args:
            user_id: User identifier
            user_data: Optional user data summary

        Returns:
            PrivacyReport
        """
        consents = self.get_user_consents(user_id)

        # Determine data categories held
        data_categories = [
            DataCategory.IDENTIFIER,
            DataCategory.PREFERENCES,
        ]
        if self.check_consent(user_id, ConsentType.ANALYTICS):
            data_categories.append(DataCategory.BEHAVIORAL)

        # Processing activities
        processing_activities = [
            "Account management and authentication",
            "Document storage and retrieval",
            "Knowledge graph construction",
        ]
        if self.check_consent(user_id, ConsentType.PERSONALIZATION):
            processing_activities.append("Learning recommendations and personalization")
        if self.check_consent(user_id, ConsentType.ANALYTICS):
            processing_activities.append("Usage analytics and service improvement")

        # Third-party sharing
        third_party_recipients = []
        if self.check_consent(user_id, ConsentType.THIRD_PARTY):
            third_party_recipients = [
                "Cloud storage providers (encrypted data only)",
                "AI model providers (anonymized queries only)",
            ]

        # Data summary
        data_summary = user_data or {
            "documents_uploaded": "See document list",
            "annotations_created": "See annotations",
            "flashcards_generated": "See flashcard deck",
            "reading_sessions": "See learning history",
        }

        return PrivacyReport(
            report_id=f"privacy_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            data_categories=data_categories,
            consents=consents,
            data_summary=data_summary,
            processing_activities=processing_activities,
            third_party_recipients=third_party_recipients,
            user_rights=self.USER_RIGHTS,
        )

    def request_data_deletion(
        self,
        user_id: str,
    ) -> dict:
        """
        Process a data deletion request (right to be forgotten).

        Args:
            user_id: User identifier

        Returns:
            Deletion status
        """
        # In production, this would trigger actual data deletion
        # Here we simulate the process

        deletion_id = f"del_{uuid.uuid4().hex[:12]}"

        return {
            "deletion_id": deletion_id,
            "user_id": user_id,
            "requested_at": datetime.utcnow().isoformat(),
            "status": "processing",
            "estimated_completion": (
                datetime.utcnow() + timedelta(days=30)
            ).isoformat(),
            "categories_to_delete": [
                DataCategory.IDENTIFIER.value,
                DataCategory.BEHAVIORAL.value,
                DataCategory.PREFERENCES.value,
            ],
            "message": "Your data deletion request has been received. "
            "Per GDPR requirements, we will process this within 30 days.",
        }

    def export_user_data(
        self,
        user_id: str,
        user_data: dict,
    ) -> dict:
        """
        Export user data for portability (right to data portability).

        Args:
            user_id: User identifier
            user_data: User's data to export

        Returns:
            Exportable data package
        """
        return {
            "export_id": f"export_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "exported_at": datetime.utcnow().isoformat(),
            "format": "application/json",
            "data": user_data,
            "consents": [c.to_dict() for c in self.get_user_consents(user_id)],
        }

    def get_retention_policy(self, category: DataCategory) -> DataRetentionPolicy:
        """
        Get retention policy for a data category.

        Args:
            category: Data category

        Returns:
            DataRetentionPolicy
        """
        return self.DEFAULT_POLICIES.get(
            category,
            DataRetentionPolicy(
                category=category,
                retention_days=365,
                anonymize_after_days=180,
                delete_after_days=730,
                legal_basis="Legitimate interest",
            ),
        )


# Singleton instance
_privacy_compliance: Optional[PrivacyCompliance] = None


def get_privacy_compliance() -> PrivacyCompliance:
    """Get or create singleton PrivacyCompliance instance."""
    global _privacy_compliance
    if _privacy_compliance is None:
        _privacy_compliance = PrivacyCompliance()
    return _privacy_compliance
