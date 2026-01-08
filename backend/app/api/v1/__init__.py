"""API v1 router configuration."""

from fastapi import APIRouter

from app.api.v1.routes import auth, documents, graph, health

router = APIRouter()

# Include route modules
router.include_router(health.router, tags=["Health"])
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(graph.router, prefix="/graph", tags=["Knowledge Graph"])
