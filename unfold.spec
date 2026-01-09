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
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# Get the base directory
BASE_DIR = Path(SPECPATH)
BACKEND_DIR = BASE_DIR / 'backend'

# Collect all files from key packages
datas = []
binaries = []
hiddenimports = []

# Packages that need full collection (all submodules and data files)
packages_to_collect = [
    'pydantic',
    'pydantic_settings',
    'pydantic_core',
    'sqlalchemy',
    'starlette',
    'fastapi',
    'uvicorn',
    'httpx',
    'httpcore',
    'anyio',
    'sniffio',
    'h11',
    'certifi',
    'idna',
    'email_validator',
    'dnspython',
]

for package in packages_to_collect:
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
        datas.extend(pkg_datas)
        binaries.extend(pkg_binaries)
        hiddenimports.extend(pkg_hiddenimports)
        print(f"Collected: {package}")
    except Exception as e:
        print(f"Warning: Could not collect {package}: {e}")

# Add backend app files
datas.append((str(BACKEND_DIR / 'app'), 'backend/app'))

# Analysis configuration
a = Analysis(
    ['unfold_portable.py'],
    pathex=[str(BACKEND_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        # Additional hidden imports
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

        # SQLAlchemy dialects
        'sqlalchemy.ext.asyncio',
        'sqlalchemy.dialects.postgresql',
        'sqlalchemy.dialects.sqlite',
        'sqlalchemy.pool',
        'sqlalchemy.sql.default_comparator',

        # Async support
        'asyncio',
        'concurrent.futures',

        # Database
        'sqlite3',

        # HTTP
        'httpx._transports',
        'httpx._transports.default',

        # JSON
        'json',
        'orjson',

        # Standard lib
        're',
        'threading',
        'webbrowser',
        'argparse',
        'pathlib',
        'typing_extensions',
        'annotated_types',
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

        # Exclude heavy ML dependencies not needed
        'torch',
        'tensorflow',
        'transformers',
        'matplotlib',
        'pandas',
        'scipy',
        'sklearn',
        'numpy',

        # Exclude NLP (too heavy, use Ollama instead)
        'spacy',
        'thinc',

        # Exclude cloud services
        'pinecone',
        'boto3',
        'botocore',
        'google',
        'azure',
        'redis',

        # Exclude optional backends
        'neo4j',
        'asyncpg',
        'aiohttp',

        # Exclude LangChain (too heavy)
        'langchain',
        'langchain_openai',
        'langchain_community',
        'openai',
        'anthropic',
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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    version='version_info.txt',
)
