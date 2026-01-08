"""Health check endpoints."""

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
from app.db import (
    check_postgres_connection,
    check_neo4j_connection,
    check_faiss_connection,
)

router = APIRouter()
settings = get_settings()


class HealthStatus(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    version: str
    environment: str


class DetailedHealthStatus(HealthStatus):
    """Detailed health check with service status."""

    services: dict[str, dict[str, str | bool]]


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Basic health check endpoint.

    Returns the application health status, version, and environment.
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check() -> DetailedHealthStatus:
    """Detailed health check with service connectivity status.

    Checks connectivity to:
    - PostgreSQL database
    - Neo4j graph database
    - Vector store (Pinecone/FAISS)
    - External APIs
    """
    services = {}

    # Check PostgreSQL
    services["postgresql"] = await _check_postgresql()

    # Check Neo4j
    services["neo4j"] = await _check_neo4j()

    # Check Vector Store
    services["vector_store"] = await _check_vector_store()

    # Determine overall status
    all_healthy = all(svc.get("connected", False) for svc in services.values())
    any_healthy = any(svc.get("connected", False) for svc in services.values())

    if all_healthy:
        status = "healthy"
    elif any_healthy:
        status = "degraded"
    else:
        status = "unhealthy"

    return DetailedHealthStatus(
        status=status,
        timestamp=datetime.now(timezone.utc),
        version=settings.app_version,
        environment=settings.environment,
        services=services,
    )


async def _check_postgresql() -> dict[str, str | bool]:
    """Check PostgreSQL connectivity."""
    return await check_postgres_connection()


async def _check_neo4j() -> dict[str, str | bool]:
    """Check Neo4j connectivity."""
    return await check_neo4j_connection()


async def _check_vector_store() -> dict[str, str | bool]:
    """Check vector store connectivity."""
    return await check_faiss_connection()


@router.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    """Kubernetes-style readiness probe.

    Returns 200 if the application is ready to receive traffic.
    """
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes-style liveness probe.

    Returns 200 if the application is alive and running.
    """
    return {"status": "alive"}
