"""API v1 router configuration."""

from fastapi import APIRouter

from app.api.v1.routes import health, documents

router = APIRouter()

# Include route modules
router.include_router(health.router, tags=["Health"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
