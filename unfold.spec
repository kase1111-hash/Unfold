# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Unfold Portable Edition.
Builds a standalone Windows executable for USB drive deployment.

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

# Collect all backend Python files
backend_datas = []
for root, dirs, files in os.walk(BACKEND_DIR):
    # Skip __pycache__, tests, and other unnecessary directories
    dirs[:] = [d for d in dirs if d not in ['__pycache__', 'tests', '.pytest_cache', 'alembic']]

    for file in files:
        if file.endswith('.py'):
            src = os.path.join(root, file)
            # Compute relative destination path
            rel_path = os.path.relpath(root, BASE_DIR)
            backend_datas.append((src, rel_path))

# Analysis configuration
a = Analysis(
    ['unfold_portable.py'],
    pathex=[str(BACKEND_DIR)],
    binaries=[],
    datas=[
        # Include backend source files
        (str(BACKEND_DIR / 'app'), 'backend/app'),
        # Include any static files if they exist
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

        # Pydantic
        'pydantic',
        'pydantic_settings',
        'pydantic.deprecated.decorator',

        # Database
        'sqlite3',
        'asyncpg',
        'sqlalchemy',
        'sqlalchemy.ext.asyncio',

        # HTTP clients
        'httpx',
        'httpx._transports',
        'httpx._transports.default',
        'aiohttp',

        # NLP and ML (if available)
        'spacy',
        'spacy.lang.en',

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

        # Models
        'app.models.graph',
        'app.models.document',

        # Utilities
        're',
        'threading',
        'webbrowser',
        'argparse',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy optional dependencies
        'torch',
        'tensorflow',
        'transformers',
        'matplotlib',
        'pandas',
        'numpy',  # Re-add if needed for spacy
        'scipy',
        'sklearn',

        # Exclude test frameworks
        'pytest',
        'pytest_asyncio',
        'pytest_cov',

        # Exclude development tools
        'black',
        'ruff',
        'mypy',

        # Exclude unused database drivers
        'neo4j',
        'pinecone',

        # Exclude cloud providers
        'boto3',
        'botocore',
        'google',
        'azure',
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

# Optional: Create a directory bundle instead of single file
# Useful for debugging or if single file mode has issues
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='Unfold',
# )
