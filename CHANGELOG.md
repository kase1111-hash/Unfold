# Changelog

All notable changes to Unfold will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- v0.2: Semantic graph + embeddings enhancements
- v0.3: Reading interface MVP
- v0.4: Adaptive focus mode
- v0.5: Scholar Mode + reflection engine
- v1.0: Ethics + provenance + public beta

## [0.1.0] - 2025-01-23

### Added

#### Core Platform
- Initial project structure with FastAPI backend and Next.js frontend
- Docker Compose configuration for development and production environments
- GitHub Actions CI/CD pipeline with testing, security scanning, and deployment
- Makefile with common development commands

#### Document Management
- PDF and EPUB document ingestion and processing
- DOI validation via CrossRef API
- C2PA-compliant content provenance tracking
- Creative Commons license compliance validation
- Document metadata extraction and storage

#### Knowledge Graph System
- Neo4j integration for semantic graph storage
- Integrated relation extraction pipeline:
  - Coreference resolution for pronoun and reference linking
  - spaCy dependency parsing for syntactic structure
  - LLM-based extraction with multi-provider support (Ollama, llama.cpp, OpenAI, Anthropic)
  - Pattern matching with multi-word entity support
- D3.js interactive graph visualization
- Wikipedia and Semantic Scholar external linking

#### Reading Interface
- Dual-view mode (technical and conceptual views)
- Complexity slider for dynamic content adjustment
- Inline annotations and highlighting
- Semantic overlays with term tooltips

#### Adaptive Learning
- AI-powered flashcard generation using T5/FLAN models
- SM2 spaced repetition algorithm implementation
- User engagement and comprehension tracking
- Export to Anki, Obsidian, and Markdown formats

#### Scholar Mode
- Citation tree exploration (up to 3 hops)
- Paper credibility scoring via CrossRef and Altmetrics
- Zotero export in RIS, BibTeX, and CSL-JSON formats
- Reading reflection snapshots

#### Ethics & Privacy
- Bias auditing with sentiment analysis
- GDPR compliance with consent management
- Differential privacy for analytics
- AI operation transparency dashboard
- Data portability and export functionality

#### Authentication & Security
- JWT-based authentication with OAuth2 support
- User registration and login endpoints
- Token refresh mechanism
- CORS configuration

#### API Endpoints
- `/api/v1/auth/*` - Authentication endpoints
- `/api/v1/documents/*` - Document management
- `/api/v1/graph/*` - Knowledge graph operations
- `/api/v1/learning/*` - Flashcards and spaced repetition
- `/api/v1/scholar/*` - Citation trees and credibility scoring
- `/api/v1/ethics/*` - Provenance, bias auditing, and privacy

#### Testing
- Backend pytest suite with integration tests
- Frontend Playwright E2E tests
- Test coverage reporting

#### Documentation
- Comprehensive README with architecture overview
- API documentation via FastAPI auto-generated docs
- AI implementation instructions
- Environment variable templates

### Technical Stack
- **Backend**: Python 3.11+, FastAPI 0.109.2, SQLAlchemy 2.0, asyncpg
- **Frontend**: Next.js 14.1, React 18.2, TypeScript 5.3, Tailwind CSS 3.4
- **Databases**: PostgreSQL 14+, Neo4j 5+
- **Vector Store**: FAISS, Pinecone
- **AI/ML**: LangChain, spaCy 3.7, OpenAI, Anthropic
- **Infrastructure**: Docker, Nginx, GitHub Actions

---

## Version History

| Version | Status | Description |
|---------|--------|-------------|
| 0.1.0 | Current | Document ingestion + validation |
| 0.2.0 | Planned | Semantic graph + embeddings |
| 0.3.0 | Planned | Reading interface MVP |
| 0.4.0 | Planned | Adaptive focus mode |
| 0.5.0 | Planned | Scholar Mode + reflection engine |
| 1.0.0 | Planned | Ethics + provenance + public beta |

[Unreleased]: https://github.com/kase1111-hash/Unfold/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kase1111-hash/Unfold/releases/tag/v0.1.0
