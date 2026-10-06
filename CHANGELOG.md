# Changelog

All notable changes to Unfold will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Focus on the core loop: PDF upload → knowledge graph → flashcards → spaced
repetition (see REFOCUS_PLAN.md).

### Added
- `POST /api/v1/graph/documents/{doc_id}/build` rebuilds a document's graph
  from its stored text (idempotent) and marks the document `indexed`.
- The knowledge graph is built automatically in a background task after a
  successful upload; the document status becomes `indexed` when it succeeds.
- Flashcard endpoints: `GET/POST /api/v1/learning/flashcards`,
  `DELETE /api/v1/learning/flashcards/{card_id}`; `flashcards/generate` takes a
  `document_id` and reads the document text server-side.
- `backend/requirements-dev.txt` for test and lint tools.

### Changed
- Flashcards and their SM2 review state are stored per user in PostgreSQL
  (previously one in-memory scheduler shared by all users and lost on restart).
- Documents, graph data and flashcards are scoped to their owner. Other users'
  resources return 404, and `doc_id` is per owner (two users uploading the same
  file get separate documents). All graph routes now require authentication.
- Upload accepts PDF only. Failures return 400 with a code (`EMPTY_FILE`,
  `CORRUPT_PDF`, `ENCRYPTED_PDF`, `NO_TEXT_EXTRACTED`) and leave no document or
  file behind; 413 `FILE_TOO_LARGE`, 415 `UNSUPPORTED_TYPE`. The response holds
  the processed document.
- Any graph route returns 503 `GRAPH_UNAVAILABLE` when Neo4j is unreachable.
- `GET /api/v1/health` and `/health/ready` return 503 when PostgreSQL is
  unreachable, so container health checks fail.
- Production/staging reject an empty `NEO4J_PASSWORD` and placeholder secrets
  (any `JWT_SECRET`/`NEO4J_PASSWORD` containing "change").
- The rate limiter keys on the peer address only; client-supplied
  `X-Forwarded-For`/`X-Real-IP` are ignored. Behind nginx, uvicorn takes the
  client IP from `X-Forwarded-For` (`FORWARDED_ALLOW_IPS` in
  `docker-compose.prod.yml`), and nginx overwrites that header.
- `NEXT_PUBLIC_API_URL` includes `/api/v1` (the frontend also appends it when
  missing); production defaults to same-origin `/api/v1` behind nginx.
- `docker-compose.prod.yml` requires `POSTGRES_PASSWORD`, `NEO4J_PASSWORD` and
  `JWT_SECRET` (run it with `--env-file .env.production`) and runs
  `alembic upgrade head` before starting the API.
- The Neo4j admin user is always `neo4j`; the dev default password is
  `changeme` everywhere. Neo4j applies `NEO4J_AUTH` only when its data volume
  is created, so for an existing `neo4j-data` volume either reset the password
  or recreate the volume (`docker compose down -v` deletes all local data).

### Fixed
- Backend CI: installable requirements (pytest-asyncio, bcrypt pins), the
  `test` environment, a lifespan-managed test client, rate limiting disabled
  in tests, a Neo4j service for the graph tests and a `pip check` step.
- `GET /api/v1/graph/documents/{doc_id}/relations` no longer returns 500 for
  graphs built by the extraction pipeline.
- Graph node lookups accept the `node_id` values the list endpoints return.
- Frontend type errors that broke the build, and the wiring of upload, graph
  and flashcard pages to the API.
- `/health/detailed` no longer fails once FAISS holds 2 or more vectors;
  deleted FAISS vectors are no longer returned by search.
- Zotero export accepts a numeric `year`.

### Removed
- Ethics module and `/api/v1/ethics/*` endpoints, EPUB ingestion, Pinecone and
  LangChain (none were used by the core loop).
- Unused backend code and dependencies: the in-memory cache module,
  `aiohttp`, `python-docx`, and Tesseract in the backend image.

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
| 1.0.0 | Planned | Public beta (ethics suite deferred to the backlog, see REFOCUS_PLAN.md) |

[Unreleased]: https://github.com/kase1111-hash/Unfold/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kase1111-hash/Unfold/releases/tag/v0.1.0
