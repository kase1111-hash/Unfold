"""
Ethics Analytics Module for user transparency dashboard.
Provides metrics and insights about AI operations and ethical practices.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from enum import Enum
import uuid
from collections import defaultdict


class MetricType(str, Enum):
    """Types of ethics metrics."""
    AI_USAGE = "ai_usage"
    CONTENT_PROCESSING = "content_processing"
    DATA_ACCESS = "data_access"
    RECOMMENDATION = "recommendation"
    BIAS_DETECTION = "bias_detection"
    PRIVACY_ACTION = "privacy_action"


class TransparencyLevel(str, Enum):
    """Transparency levels for operations."""
    FULL = "full"          # Complete details available
    SUMMARY = "summary"    # Aggregated information
    MINIMAL = "minimal"    # Basic acknowledgment only
    REDACTED = "redacted"  # Sensitive details hidden


@dataclass
class AIOperation:
    """Record of an AI operation for transparency."""
    operation_id: str
    operation_type: str
    timestamp: datetime
    user_id: str

    # Operation details
    model_used: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0

    # Transparency
    purpose: str = ""
    data_accessed: list[str] = field(default_factory=list)

    # Results
    confidence_score: Optional[float] = None
    human_review_required: bool = False

    def to_dict(self) -> dict:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "timestamp": self.timestamp.isoformat(),
            "model_used": self.model_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "purpose": self.purpose,
            "data_accessed": self.data_accessed,
            "confidence_score": self.confidence_score,
            "human_review_required": self.human_review_required,
        }


@dataclass
class EthicsMetric:
    """A single ethics metric."""
    metric_id: str
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metric_id": self.metric_id,
            "metric_type": self.metric_type.value,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class UserEthicsProfile:
    """User's ethics and transparency profile."""
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Preferences
    transparency_level: TransparencyLevel = TransparencyLevel.FULL
    receive_ethics_reports: bool = True
    allow_aggregated_analytics: bool = True

    # Statistics
    total_ai_operations: int = 0
    total_documents_processed: int = 0
    bias_alerts_received: int = 0
    privacy_actions_taken: int = 0

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "transparency_level": self.transparency_level.value,
            "receive_ethics_reports": self.receive_ethics_reports,
            "allow_aggregated_analytics": self.allow_aggregated_analytics,
            "total_ai_operations": self.total_ai_operations,
            "total_documents_processed": self.total_documents_processed,
            "bias_alerts_received": self.bias_alerts_received,
            "privacy_actions_taken": self.privacy_actions_taken,
        }


@dataclass
class EthicsDashboard:
    """Complete ethics dashboard for a user."""
    user_id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    period_start: datetime = field(default_factory=lambda: datetime.utcnow() - timedelta(days=30))
    period_end: datetime = field(default_factory=datetime.utcnow)

    # Summary stats
    ai_operations_count: int = 0
    documents_analyzed: int = 0
    bias_findings_count: int = 0
    privacy_score: float = 100.0

    # Breakdowns
    operations_by_type: dict[str, int] = field(default_factory=dict)
    operations_by_day: dict[str, int] = field(default_factory=dict)

    # Recent operations
    recent_operations: list[AIOperation] = field(default_factory=list)

    # Metrics
    metrics: list[EthicsMetric] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "summary": {
                "ai_operations_count": self.ai_operations_count,
                "documents_analyzed": self.documents_analyzed,
                "bias_findings_count": self.bias_findings_count,
                "privacy_score": self.privacy_score,
            },
            "operations_by_type": self.operations_by_type,
            "operations_by_day": self.operations_by_day,
            "recent_operations": [op.to_dict() for op in self.recent_operations],
            "metrics": [m.to_dict() for m in self.metrics],
            "recommendations": self.recommendations,
        }


@dataclass
class AggregatedReport:
    """Aggregated ethics report (anonymized across users)."""
    report_id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    period_start: datetime = field(default_factory=lambda: datetime.utcnow() - timedelta(days=30))
    period_end: datetime = field(default_factory=datetime.utcnow)

    # Aggregated metrics
    total_users: int = 0
    total_operations: int = 0
    avg_operations_per_user: float = 0.0

    # Distribution data
    operation_type_distribution: dict[str, float] = field(default_factory=dict)
    bias_category_distribution: dict[str, float] = field(default_factory=dict)

    # Platform health
    avg_bias_score: float = 0.0
    avg_privacy_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "aggregated_metrics": {
                "total_users": self.total_users,
                "total_operations": self.total_operations,
                "avg_operations_per_user": self.avg_operations_per_user,
            },
            "distributions": {
                "operation_types": self.operation_type_distribution,
                "bias_categories": self.bias_category_distribution,
            },
            "platform_health": {
                "avg_bias_score": self.avg_bias_score,
                "avg_privacy_score": self.avg_privacy_score,
            },
        }


class EthicsAnalytics:
    """
    Service for tracking and reporting ethics metrics.
    Provides transparency into AI operations and data handling.
    """

    # Standard operation types
    OPERATION_TYPES = [
        "document_upload",
        "ai_summary",
        "ai_extraction",
        "ai_flashcard",
        "ai_recommendation",
        "bias_check",
        "citation_lookup",
        "knowledge_graph",
    ]

    def __init__(self):
        self._operations: dict[str, list[AIOperation]] = defaultdict(list)
        self._metrics: dict[str, list[EthicsMetric]] = defaultdict(list)
        self._profiles: dict[str, UserEthicsProfile] = {}

    def get_or_create_profile(self, user_id: str) -> UserEthicsProfile:
        """Get or create user ethics profile."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserEthicsProfile(user_id=user_id)
        return self._profiles[user_id]

    def update_profile_preferences(
        self,
        user_id: str,
        transparency_level: Optional[TransparencyLevel] = None,
        receive_reports: Optional[bool] = None,
        allow_aggregated: Optional[bool] = None,
    ) -> UserEthicsProfile:
        """Update user ethics preferences."""
        profile = self.get_or_create_profile(user_id)

        if transparency_level is not None:
            profile.transparency_level = transparency_level
        if receive_reports is not None:
            profile.receive_ethics_reports = receive_reports
        if allow_aggregated is not None:
            profile.allow_aggregated_analytics = allow_aggregated

        return profile

    def record_operation(
        self,
        user_id: str,
        operation_type: str,
        purpose: str,
        model_used: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        data_accessed: Optional[list[str]] = None,
        confidence_score: Optional[float] = None,
        human_review_required: bool = False,
    ) -> AIOperation:
        """
        Record an AI operation for transparency.

        Args:
            user_id: User who initiated the operation
            operation_type: Type of operation
            purpose: Human-readable purpose
            model_used: AI model identifier
            input_tokens: Tokens in input
            output_tokens: Tokens in output
            data_accessed: List of data sources accessed
            confidence_score: Model confidence (0-1)
            human_review_required: Whether human review is needed

        Returns:
            AIOperation record
        """
        operation = AIOperation(
            operation_id=f"op_{uuid.uuid4().hex[:12]}",
            operation_type=operation_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            purpose=purpose,
            data_accessed=data_accessed or [],
            confidence_score=confidence_score,
            human_review_required=human_review_required,
        )

        self._operations[user_id].append(operation)

        # Update profile stats
        profile = self.get_or_create_profile(user_id)
        profile.total_ai_operations += 1
        if operation_type == "document_upload":
            profile.total_documents_processed += 1

        return operation

    def record_metric(
        self,
        user_id: str,
        metric_type: MetricType,
        name: str,
        value: float,
        unit: str,
        context: Optional[dict] = None,
    ) -> EthicsMetric:
        """
        Record an ethics metric.

        Args:
            user_id: Associated user
            metric_type: Type of metric
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            context: Additional context

        Returns:
            EthicsMetric record
        """
        metric = EthicsMetric(
            metric_id=f"metric_{uuid.uuid4().hex[:12]}",
            metric_type=metric_type,
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            context=context or {},
        )

        self._metrics[user_id].append(metric)
        return metric

    def record_bias_alert(self, user_id: str):
        """Record that a bias alert was shown to user."""
        profile = self.get_or_create_profile(user_id)
        profile.bias_alerts_received += 1

    def record_privacy_action(self, user_id: str):
        """Record that user took a privacy action."""
        profile = self.get_or_create_profile(user_id)
        profile.privacy_actions_taken += 1

    def get_user_operations(
        self,
        user_id: str,
        operation_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AIOperation]:
        """
        Get user's AI operations.

        Args:
            user_id: User identifier
            operation_type: Filter by type
            since: Filter by timestamp
            limit: Maximum results

        Returns:
            List of operations
        """
        operations = self._operations.get(user_id, [])

        if operation_type:
            operations = [op for op in operations if op.operation_type == operation_type]

        if since:
            operations = [op for op in operations if op.timestamp >= since]

        # Sort by timestamp descending
        operations = sorted(operations, key=lambda x: x.timestamp, reverse=True)

        return operations[:limit]

    def get_user_metrics(
        self,
        user_id: str,
        metric_type: Optional[MetricType] = None,
        since: Optional[datetime] = None,
    ) -> list[EthicsMetric]:
        """Get user's ethics metrics."""
        metrics = self._metrics.get(user_id, [])

        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]

        if since:
            metrics = [m for m in metrics if m.timestamp >= since]

        return sorted(metrics, key=lambda x: x.timestamp, reverse=True)

    def generate_dashboard(
        self,
        user_id: str,
        period_days: int = 30,
    ) -> EthicsDashboard:
        """
        Generate ethics dashboard for a user.

        Args:
            user_id: User identifier
            period_days: Number of days to include

        Returns:
            EthicsDashboard
        """
        period_start = datetime.utcnow() - timedelta(days=period_days)
        period_end = datetime.utcnow()

        # Get operations in period
        operations = self.get_user_operations(user_id, since=period_start, limit=1000)

        # Count by type
        operations_by_type: dict[str, int] = defaultdict(int)
        for op in operations:
            operations_by_type[op.operation_type] += 1

        # Count by day
        operations_by_day: dict[str, int] = defaultdict(int)
        for op in operations:
            day_key = op.timestamp.strftime("%Y-%m-%d")
            operations_by_day[day_key] += 1

        # Count bias findings
        metrics = self.get_user_metrics(user_id, since=period_start)
        bias_findings = sum(
            1 for m in metrics
            if m.metric_type == MetricType.BIAS_DETECTION
        )

        # Calculate privacy score based on actions and settings
        profile = self.get_or_create_profile(user_id)
        privacy_score = self._calculate_privacy_score(profile)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            profile, operations, metrics
        )

        return EthicsDashboard(
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            ai_operations_count=len(operations),
            documents_analyzed=operations_by_type.get("document_upload", 0),
            bias_findings_count=bias_findings,
            privacy_score=privacy_score,
            operations_by_type=dict(operations_by_type),
            operations_by_day=dict(operations_by_day),
            recent_operations=operations[:10],
            metrics=metrics[:20],
            recommendations=recommendations,
        )

    def _calculate_privacy_score(self, profile: UserEthicsProfile) -> float:
        """Calculate privacy health score (0-100)."""
        score = 100.0

        # Deduct for lower transparency preference
        if profile.transparency_level == TransparencyLevel.MINIMAL:
            score -= 10
        elif profile.transparency_level == TransparencyLevel.REDACTED:
            score -= 20

        # Bonus for opting out of aggregated analytics
        if not profile.allow_aggregated_analytics:
            score += 5

        # Bonus for taking privacy actions
        if profile.privacy_actions_taken > 0:
            score += min(10, profile.privacy_actions_taken * 2)

        return min(100.0, max(0.0, score))

    def _generate_recommendations(
        self,
        profile: UserEthicsProfile,
        operations: list[AIOperation],
        metrics: list[EthicsMetric],
    ) -> list[str]:
        """Generate personalized ethics recommendations."""
        recommendations = []

        # Check transparency level
        if profile.transparency_level in [TransparencyLevel.MINIMAL, TransparencyLevel.REDACTED]:
            recommendations.append(
                "Consider increasing your transparency level to get more detailed insights "
                "into how AI processes your documents."
            )

        # Check for low confidence operations
        low_confidence_ops = [
            op for op in operations
            if op.confidence_score is not None and op.confidence_score < 0.7
        ]
        if low_confidence_ops:
            recommendations.append(
                f"Review {len(low_confidence_ops)} AI operations that had lower confidence scores. "
                "Human verification is recommended for these results."
            )

        # Check for bias alerts
        if profile.bias_alerts_received > 5:
            recommendations.append(
                "You've received multiple bias alerts. Consider reviewing the bias audit "
                "reports for your documents to understand potential issues."
            )

        # Check for no ethics reports
        if not profile.receive_ethics_reports:
            recommendations.append(
                "Enable ethics reports to receive regular updates about how your data is being processed."
            )

        # Check for human review needed
        needs_review = [op for op in operations if op.human_review_required]
        if needs_review:
            recommendations.append(
                f"{len(needs_review)} AI operations flagged for human review. "
                "Please verify these results for accuracy."
            )

        # General recommendation if no specific issues
        if not recommendations:
            recommendations.append(
                "Your ethics profile looks good! Keep monitoring the dashboard "
                "for transparency into AI operations."
            )

        return recommendations

    def generate_aggregated_report(
        self,
        period_days: int = 30,
    ) -> AggregatedReport:
        """
        Generate aggregated ethics report across all users.
        Only includes users who opted in to aggregated analytics.

        Args:
            period_days: Number of days to include

        Returns:
            AggregatedReport with anonymized data
        """
        period_start = datetime.utcnow() - timedelta(days=period_days)

        # Filter to consenting users
        consenting_users = [
            uid for uid, profile in self._profiles.items()
            if profile.allow_aggregated_analytics
        ]

        if not consenting_users:
            return AggregatedReport(
                report_id=f"agg_{uuid.uuid4().hex[:12]}",
                period_start=period_start,
            )

        # Aggregate operations
        all_operations: list[AIOperation] = []
        for user_id in consenting_users:
            ops = self.get_user_operations(user_id, since=period_start, limit=10000)
            all_operations.extend(ops)

        # Calculate distributions
        operation_counts: dict[str, int] = defaultdict(int)
        for op in all_operations:
            operation_counts[op.operation_type] += 1

        total_ops = len(all_operations)
        operation_distribution = {
            k: (v / total_ops * 100) if total_ops > 0 else 0
            for k, v in operation_counts.items()
        }

        # Aggregate metrics for bias
        bias_counts: dict[str, int] = defaultdict(int)
        for user_id in consenting_users:
            metrics = self.get_user_metrics(
                user_id, MetricType.BIAS_DETECTION, since=period_start
            )
            for m in metrics:
                category = m.context.get("category", "unknown")
                bias_counts[category] += 1

        total_bias = sum(bias_counts.values())
        bias_distribution = {
            k: (v / total_bias * 100) if total_bias > 0 else 0
            for k, v in bias_counts.items()
        }

        # Calculate platform averages
        privacy_scores = [
            self._calculate_privacy_score(self._profiles[uid])
            for uid in consenting_users
        ]
        avg_privacy = sum(privacy_scores) / len(privacy_scores) if privacy_scores else 0

        return AggregatedReport(
            report_id=f"agg_{uuid.uuid4().hex[:12]}",
            period_start=period_start,
            total_users=len(consenting_users),
            total_operations=total_ops,
            avg_operations_per_user=total_ops / len(consenting_users) if consenting_users else 0,
            operation_type_distribution=operation_distribution,
            bias_category_distribution=bias_distribution,
            avg_privacy_score=avg_privacy,
        )

    def export_user_ethics_data(self, user_id: str) -> dict:
        """
        Export all ethics data for a user (GDPR compliance).

        Args:
            user_id: User identifier

        Returns:
            Complete ethics data package
        """
        profile = self.get_or_create_profile(user_id)
        operations = self.get_user_operations(user_id, limit=10000)
        metrics = self.get_user_metrics(user_id)

        return {
            "export_id": f"ethics_export_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "exported_at": datetime.utcnow().isoformat(),
            "profile": profile.to_dict(),
            "operations": [op.to_dict() for op in operations],
            "metrics": [m.to_dict() for m in metrics],
        }

    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all ethics data for a user (right to erasure).

        Args:
            user_id: User identifier

        Returns:
            True if data was deleted
        """
        deleted = False

        if user_id in self._operations:
            del self._operations[user_id]
            deleted = True

        if user_id in self._metrics:
            del self._metrics[user_id]
            deleted = True

        if user_id in self._profiles:
            del self._profiles[user_id]
            deleted = True

        return deleted


# Singleton instance
_ethics_analytics: Optional[EthicsAnalytics] = None


def get_ethics_analytics() -> EthicsAnalytics:
    """Get or create singleton EthicsAnalytics instance."""
    global _ethics_analytics
    if _ethics_analytics is None:
        _ethics_analytics = EthicsAnalytics()
    return _ethics_analytics
