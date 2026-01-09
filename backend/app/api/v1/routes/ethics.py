"""
Ethics API routes for Phase 6 features.
Includes provenance tracking, bias auditing, privacy compliance, and analytics.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.v1.dependencies import CurrentUser
from app.services.ethics import (
    get_provenance_service,
    get_bias_auditor,
    get_privacy_compliance,
    get_ethics_analytics,
    ConsentType,
    TransparencyLevel,
    MetricType,
    ProvenanceAction,
)

router = APIRouter(prefix="/ethics", tags=["ethics"])


# ============== Pydantic Models ==============


class ProvenanceCreateRequest(BaseModel):
    """Request to create content provenance."""

    document_id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document content")


class ProvenanceVerifyRequest(BaseModel):
    """Request to verify content integrity."""

    credential_id: str = Field(..., description="Credential identifier")
    content: str = Field(..., description="Content to verify")


class ProvenanceAssertionRequest(BaseModel):
    """Request to add assertion to credential."""

    credential_id: str
    assertion_type: str = Field(default="ai_processed")
    actor: str = Field(default="system")
    description: str = ""
    metadata: Optional[dict] = None


class BiasAuditRequest(BaseModel):
    """Request to audit document for bias."""

    document_id: str
    content: str
    sections: Optional[list[str]] = None


class ConsentRequest(BaseModel):
    """Request to record consent."""

    consent_type: str = Field(..., description="Type of consent")
    granted: bool = Field(..., description="Whether consent is granted")
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class ConsentWithdrawRequest(BaseModel):
    """Request to withdraw consent."""

    consent_type: str


class PrivacyPreferencesRequest(BaseModel):
    """Request to update privacy preferences."""

    transparency_level: Optional[str] = None
    receive_reports: Optional[bool] = None
    allow_aggregated: Optional[bool] = None


class AIOperationRequest(BaseModel):
    """Request to record AI operation."""

    operation_type: str
    purpose: str
    model_used: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    data_accessed: Optional[list[str]] = None
    confidence_score: Optional[float] = None
    human_review_required: bool = False


class MetricRequest(BaseModel):
    """Request to record ethics metric."""

    metric_type: str
    name: str
    value: float
    unit: str
    context: Optional[dict] = None


# ============== Provenance Tracking ==============


@router.post("/provenance/create")
async def create_provenance(
    request: ProvenanceCreateRequest,
    current_user: CurrentUser,
):
    """
    Create content provenance credential for a document.
    Establishes SHA-256 content hash for integrity verification.
    """
    service = get_provenance_service()

    credential = service.create_credential(
        document_id=request.document_id,
        content=request.content,
        actor=current_user.username,
    )

    return credential.to_dict()


@router.post("/provenance/verify")
async def verify_provenance(
    request: ProvenanceVerifyRequest,
    current_user: CurrentUser,
):
    """
    Verify content integrity against stored credential.
    """
    service = get_provenance_service()

    valid, message = service.verify_content_integrity(
        credential_id=request.credential_id,
        content=request.content,
    )

    return {
        "valid": valid,
        "message": message,
        "credential_id": request.credential_id,
    }


@router.post("/provenance/assertion")
async def add_assertion(
    request: ProvenanceAssertionRequest,
    current_user: CurrentUser,
):
    """
    Add an assertion to a content credential.
    Records processing history for transparency.
    """
    service = get_provenance_service()

    try:
        action = ProvenanceAction(request.assertion_type)
    except ValueError:
        action = ProvenanceAction.MODIFIED

    assertion = service.add_assertion(
        credential_id=request.credential_id,
        action=action,
        actor=request.actor or current_user.username,
        description=request.description,
        metadata=request.metadata,
    )

    if assertion is None:
        raise HTTPException(status_code=404, detail="Credential not found")

    return assertion.to_dict()


@router.get("/provenance/{credential_id}")
async def get_provenance(
    credential_id: str,
    current_user: CurrentUser,
):
    """
    Get full provenance information for a credential.
    """
    service = get_provenance_service()
    credential = service.get_credential(credential_id)

    if credential is None:
        raise HTTPException(status_code=404, detail="Credential not found")

    return credential.to_dict()


@router.get("/provenance/document/{document_id}")
async def get_document_provenance(
    document_id: str,
    current_user: CurrentUser,
):
    """
    Get provenance manifest for a document.
    """
    service = get_provenance_service()
    manifest = service.get_manifest_by_document(document_id)

    if manifest is None:
        return {
            "document_id": document_id,
            "manifest": None,
            "message": "No provenance manifest found for this document",
        }

    return {
        "document_id": document_id,
        "manifest": manifest.to_dict(),
    }


class ManifestCreateRequest(BaseModel):
    """Request to create provenance manifest."""

    document_id: str
    title: str
    content: str
    authors: Optional[list[str]] = None
    doi: Optional[str] = None
    license: Optional[str] = None
    source: Optional[str] = None


@router.post("/provenance/manifest")
async def create_manifest(
    request: ManifestCreateRequest,
    current_user: CurrentUser,
):
    """
    Create a complete provenance manifest for a document.
    """
    service = get_provenance_service()
    manifest = service.create_manifest(
        document_id=request.document_id,
        title=request.title,
        content=request.content,
        actor=current_user.username,
        authors=request.authors,
        doi=request.doi,
        license=request.license,
        source=request.source,
    )

    return manifest.to_dict()


# ============== Bias Auditing ==============


@router.post("/bias/audit")
async def audit_document(
    request: BiasAuditRequest,
    current_user: CurrentUser,
):
    """
    Perform comprehensive bias audit on document content.
    Analyzes sentiment, detects biased language, and calculates inclusivity metrics.
    """
    auditor = get_bias_auditor()
    analytics = get_ethics_analytics()

    report = auditor.audit_document(
        document_id=request.document_id,
        content=request.content,
        sections=request.sections,
    )

    # Record bias alert if findings exist
    if report.findings:
        analytics.record_bias_alert(current_user.user_id)

    return report.to_dict()


@router.post("/bias/sentiment")
async def analyze_sentiment(
    text: str = Query(..., min_length=1),
    current_user: CurrentUser = None,
):
    """
    Analyze sentiment of text content.
    """
    auditor = get_bias_auditor()
    result = auditor.analyze_sentiment(text)

    return result.to_dict()


@router.get("/bias/categories")
async def get_bias_categories(
    current_user: CurrentUser,
):
    """
    Get available bias categories and their descriptions.
    """
    from app.services.ethics.audit import BiasCategory

    return {
        "categories": [
            {
                "id": cat.value,
                "name": cat.name.replace("_", " ").title(),
            }
            for cat in BiasCategory
        ],
    }


# ============== Privacy Compliance ==============


@router.post("/privacy/consent")
async def record_consent(
    request: ConsentRequest,
    current_user: CurrentUser,
):
    """
    Record user consent decision (GDPR compliant).
    """
    compliance = get_privacy_compliance()
    analytics = get_ethics_analytics()

    try:
        consent_type = ConsentType(request.consent_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid consent type. Valid types: {[t.value for t in ConsentType]}",
        )

    consent = compliance.record_consent(
        user_id=current_user.user_id,
        consent_type=consent_type,
        granted=request.granted,
        ip_address=request.ip_address,
        user_agent=request.user_agent,
    )

    # Record privacy action
    analytics.record_privacy_action(current_user.user_id)

    return consent.to_dict()


@router.post("/privacy/consent/withdraw")
async def withdraw_consent(
    request: ConsentWithdrawRequest,
    current_user: CurrentUser,
):
    """
    Withdraw previously granted consent.
    """
    compliance = get_privacy_compliance()
    analytics = get_ethics_analytics()

    try:
        consent_type = ConsentType(request.consent_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid consent type")

    consent = compliance.withdraw_consent(
        user_id=current_user.user_id,
        consent_type=consent_type,
    )

    if consent is None:
        raise HTTPException(status_code=404, detail="Consent record not found")

    analytics.record_privacy_action(current_user.user_id)

    return consent.to_dict()


@router.get("/privacy/consents")
async def get_consents(
    current_user: CurrentUser,
):
    """
    Get all consent records for current user.
    """
    compliance = get_privacy_compliance()
    consents = compliance.get_user_consents(current_user.user_id)

    return {
        "consents": [c.to_dict() for c in consents],
        "count": len(consents),
    }


@router.get("/privacy/report")
async def get_privacy_report(
    current_user: CurrentUser,
):
    """
    Generate GDPR-compliant privacy report (data subject access request).
    """
    compliance = get_privacy_compliance()

    report = compliance.generate_privacy_report(
        user_id=current_user.user_id,
    )

    return report.to_dict()


@router.post("/privacy/delete")
async def request_deletion(
    current_user: CurrentUser,
):
    """
    Request data deletion (right to be forgotten).
    """
    compliance = get_privacy_compliance()
    analytics = get_ethics_analytics()

    result = compliance.request_data_deletion(current_user.user_id)
    analytics.record_privacy_action(current_user.user_id)

    return result


@router.get("/privacy/export")
async def export_data(
    current_user: CurrentUser,
):
    """
    Export all user data (right to data portability).
    """
    compliance = get_privacy_compliance()
    analytics = get_ethics_analytics()

    # Get user data from various sources
    ethics_data = analytics.export_user_ethics_data(current_user.user_id)

    user_data = {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": compliance.mask_email(current_user.email),
        "created_at": (
            current_user.created_at.isoformat() if current_user.created_at else None
        ),
    }

    result = compliance.export_user_data(
        user_id=current_user.user_id,
        user_data={
            "profile": user_data,
            "ethics_data": ethics_data,
        },
    )

    analytics.record_privacy_action(current_user.user_id)

    return result


@router.get("/privacy/retention/{category}")
async def get_retention_policy(
    category: str,
    current_user: CurrentUser,
):
    """
    Get data retention policy for a category.
    """
    from app.services.ethics.privacy import DataCategory

    compliance = get_privacy_compliance()

    try:
        data_category = DataCategory(category)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid categories: {[c.value for c in DataCategory]}",
        )

    policy = compliance.get_retention_policy(data_category)
    return policy.to_dict()


# ============== Ethics Analytics ==============


@router.post("/analytics/operation")
async def record_operation(
    request: AIOperationRequest,
    current_user: CurrentUser,
):
    """
    Record an AI operation for transparency tracking.
    """
    analytics = get_ethics_analytics()

    operation = analytics.record_operation(
        user_id=current_user.user_id,
        operation_type=request.operation_type,
        purpose=request.purpose,
        model_used=request.model_used,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        data_accessed=request.data_accessed,
        confidence_score=request.confidence_score,
        human_review_required=request.human_review_required,
    )

    return operation.to_dict()


@router.post("/analytics/metric")
async def record_metric(
    request: MetricRequest,
    current_user: CurrentUser,
):
    """
    Record an ethics metric.
    """
    analytics = get_ethics_analytics()

    try:
        metric_type = MetricType(request.metric_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric type. Valid types: {[t.value for t in MetricType]}",
        )

    metric = analytics.record_metric(
        user_id=current_user.user_id,
        metric_type=metric_type,
        name=request.name,
        value=request.value,
        unit=request.unit,
        context=request.context,
    )

    return metric.to_dict()


@router.get("/analytics/operations")
async def get_operations(
    operation_type: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    current_user: CurrentUser = None,
):
    """
    Get user's AI operations history.
    """
    analytics = get_ethics_analytics()

    operations = analytics.get_user_operations(
        user_id=current_user.user_id,
        operation_type=operation_type,
        limit=limit,
    )

    return {
        "operations": [op.to_dict() for op in operations],
        "count": len(operations),
    }


@router.get("/analytics/dashboard")
async def get_dashboard(
    period_days: int = Query(default=30, ge=1, le=365),
    current_user: CurrentUser = None,
):
    """
    Get complete ethics dashboard for user.
    Includes AI operations, bias findings, and privacy score.
    """
    analytics = get_ethics_analytics()

    dashboard = analytics.generate_dashboard(
        user_id=current_user.user_id,
        period_days=period_days,
    )

    return dashboard.to_dict()


@router.get("/analytics/profile")
async def get_ethics_profile(
    current_user: CurrentUser,
):
    """
    Get user's ethics profile and preferences.
    """
    analytics = get_ethics_analytics()
    profile = analytics.get_or_create_profile(current_user.user_id)
    return profile.to_dict()


@router.put("/analytics/preferences")
async def update_preferences(
    request: PrivacyPreferencesRequest,
    current_user: CurrentUser,
):
    """
    Update user's ethics and privacy preferences.
    """
    analytics = get_ethics_analytics()

    transparency = None
    if request.transparency_level:
        try:
            transparency = TransparencyLevel(request.transparency_level)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transparency level. Valid levels: {[level.value for level in TransparencyLevel]}",
            )

    profile = analytics.update_profile_preferences(
        user_id=current_user.user_id,
        transparency_level=transparency,
        receive_reports=request.receive_reports,
        allow_aggregated=request.allow_aggregated,
    )

    return profile.to_dict()


@router.get("/analytics/aggregated")
async def get_aggregated_report(
    period_days: int = Query(default=30, ge=1, le=365),
    current_user: CurrentUser = None,
):
    """
    Get aggregated ethics report across platform.
    Only includes data from users who opted in.
    """
    analytics = get_ethics_analytics()
    report = analytics.generate_aggregated_report(period_days=period_days)
    return report.to_dict()
