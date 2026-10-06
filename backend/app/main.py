"""Unfold API - Main application entry point."""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import router as api_v1_router
from app.config import get_settings
from app.db import (
    GRAPH_UNAVAILABLE_ERRORS,
    check_neo4j_connection,
    close_all_databases,
    init_postgres,
    init_neo4j,
    init_faiss,
    create_neo4j_indexes,
    create_tables,
)
from app.middleware import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# The multipart parser runs before authentication and logs a WARNING for
# malformed input; python-multipart 0.0.9 logged one per byte after the
# closing boundary, so one unauthenticated request could write gigabytes of
# logs. Keep only its errors. 0.0.20 logs as "python_multipart"; "multipart"
# is the old import name.
for _multipart_logger in ("multipart", "python_multipart"):
    logging.getLogger(_multipart_logger).setLevel(logging.ERROR)

settings = get_settings()

# Set log level based on environment
if settings.debug:
    logging.getLogger("app").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Initialize PostgreSQL
    try:
        await init_postgres()
        if settings.environment in ("development", "test"):
            await create_tables()
        logger.info("PostgreSQL connected successfully")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")

    # Initialize Neo4j
    try:
        await init_neo4j()
        # The driver connects lazily, so verify before claiming success.
        neo4j_status = await check_neo4j_connection()
        if neo4j_status.get("connected"):
            await create_neo4j_indexes()
            logger.info("Neo4j connected successfully")
        else:
            logger.warning(f"Neo4j unavailable: {neo4j_status.get('message')}")
    except Exception as e:
        logger.warning(f"Neo4j connection failed: {e}")

    # Initialize FAISS vector store
    try:
        await init_faiss()
        logger.info("FAISS vector store initialized")
    except ImportError:
        logger.info("FAISS not installed (optional dependency)")
    except Exception as e:
        logger.warning(f"FAISS initialization failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_all_databases()
    logger.info("All database connections closed")


app = FastAPI(
    title=settings.app_name,
    description="AI-assisted reading and comprehension platform API",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Rate limiting middleware (must be added before CORS)
app.add_middleware(RateLimitMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_v1_router, prefix="/api/v1")


async def graph_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map Neo4j connectivity failures to 503 instead of an unhandled 500."""
    logger.warning(f"Graph database unavailable on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "detail": {
                "code": "GRAPH_UNAVAILABLE",
                "message": "The knowledge graph database is unavailable. Please try again later.",
            }
        },
    )


for _exc in GRAPH_UNAVAILABLE_ERRORS:
    app.add_exception_handler(_exc, graph_unavailable_handler)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
