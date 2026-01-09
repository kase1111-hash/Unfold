# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Unfold Portable Edition - Full Featured Build.
Builds a standalone Windows executable with all dependencies.

Usage:
    pyinstaller unfold.spec

Or use the provided build.bat script.
"""

import os
import sys
from pathlib import Path

# Get the base directory
BASE_DIR = Path(SPECPATH)
BACKEND_DIR = BASE_DIR / 'backend'

# Analysis configuration
a = Analysis(
    ['unfold_portable.py'],
    pathex=[str(BACKEND_DIR)],
    binaries=[],
    datas=[
        # Include backend source files
        (str(BACKEND_DIR / 'app'), 'backend/app'),
    ],
    hiddenimports=[
        # FastAPI and web server
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'starlette',
        'starlette.routing',
        'starlette.middleware',
        'starlette.middleware.cors',
        'starlette.responses',
        'starlette.requests',

        # Pydantic
        'pydantic',
        'pydantic_settings',
        'pydantic.deprecated.decorator',
        'pydantic_core',

        # Database
        'sqlite3',
        'asyncpg',
        'sqlalchemy',
        'sqlalchemy.ext.asyncio',
        'sqlalchemy.dialects.postgresql',
        'sqlalchemy.dialects.sqlite',

        # Neo4j
        'neo4j',

        # HTTP clients
        'httpx',
        'httpx._transports',
        'httpx._transports.default',
        'httpcore',
        'aiohttp',
        'anyio',
        'anyio._backends',
        'anyio._backends._asyncio',

        # AI/ML
        'openai',
        'anthropic',
        'langchain',
        'langchain.llms',
        'langchain.chat_models',
        'langchain_openai',
        'langchain_community',

        # NLP
        'spacy',
        'spacy.lang.en',
        'spacy.pipeline',
        'spacy.tokens',

        # Document Processing
        'pypdf2',
        'docx',
        'ebooklib',

        # JSON and serialization
        'json',
        'orjson',

        # Graph services
        'app.services.graph.coreference',
        'app.services.graph.dependency_parsing',
        'app.services.graph.llm_relations',
        'app.services.graph.integrated_pipeline',
        'app.services.graph.extractor',
        'app.services.graph.builder',
        'app.services.graph.relations',
        'app.services.graph.spacy_loader',
        'app.services.graph.enhanced_relations',

        # Models
        'app.models.graph',
        'app.models.document',
        'app.models.user',

        # Config
        'app.config',

        # Database modules
        'app.db',
        'app.db.neo4j',
        'app.db.postgres',
        'app.db.vector',

        # Authentication
        'jose',
        'passlib',
        'passlib.hash',
        'bcrypt',

        # Utilities
        're',
        'threading',
        'webbrowser',
        'argparse',
        'pathlib',
        'dotenv',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude test frameworks
        'pytest',
        'pytest_asyncio',
        'pytest_cov',

        # Exclude development tools
        'black',
        'ruff',
        'mypy',

        # Exclude unused heavy dependencies
        'torch',
        'tensorflow',
        'transformers',
        'matplotlib',
        'pandas',
        'scipy',
        'sklearn',

        # Exclude cloud-only services
        'pinecone',
        'boto3',
        'botocore',
        'google',
        'azure',
        'redis',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Create the PYZ archive
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# Create the executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Unfold',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Enable UPX compression if available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Show console window for output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if available: 'assets/icon.ico'
    version='version_info.txt',  # Add version info if available
)
