"""Unfold API - Main application entry point."""

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

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")

    # Initialize PostgreSQL
    try:
        await init_postgres()
        if settings.environment == "development":
            await create_tables()
        print("✓ PostgreSQL connected")
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")

    # Initialize Neo4j
    try:
        await init_neo4j()
        await create_neo4j_indexes()
        print("✓ Neo4j connected")
    except Exception as e:
        print(f"✗ Neo4j connection failed: {e}")

    # Initialize FAISS vector store
    try:
        await init_faiss()
        print("✓ FAISS vector store initialized")
    except ImportError:
        print("✗ FAISS not installed (optional)")
    except Exception as e:
        print(f"✗ FAISS initialization failed: {e}")

    yield

    # Shutdown
    print("Shutting down...")
    await close_all_databases()
    print("All database connections closed")


app = FastAPI(
    title=settings.app_name,
    description="AI-assisted reading and comprehension platform API",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

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
