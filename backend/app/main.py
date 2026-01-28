"""Unfold API - Main application entry point."""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router as api_v1_router
from app.config import get_settings
from app.db import (
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
        if settings.environment == "development":
            await create_tables()
        logger.info("PostgreSQL connected successfully")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")

    # Initialize Neo4j
    try:
        await init_neo4j()
        await create_neo4j_indexes()
        logger.info("Neo4j connected successfully")
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


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
