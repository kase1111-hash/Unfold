# Unfold

**LLM-Powered AI Reading Assistant & Semantic Comprehension Platform**

Unfold is a natural language processing platform that bridges the gap between dense academic/technical texts and genuine understanding. This AI-assisted reading tool uses semantic understanding, knowledge graph construction, and human-AI collaboration to transform how students and researchers engage with complex material. Built for those asking "how to understand academic papers faster" and "AI tools for research comprehension," Unfold emphasizes ethics, explainability, and educational collaboration while preserving human cognitive work and authorship in the learning process.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Frontend Development](#frontend-development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Connected Repositories](#connected-repositories)
- [License](#license)

## Features

### Document Management
- **PDF Ingestion** - Upload academic PDFs; the knowledge graph is built automatically afterwards
- **DOI Validation** - Verify document authenticity via CrossRef
- **Provenance Hash** - SHA-256 fingerprint of the extracted text, recorded at upload
- **License Compliance** - Creative Commons validation

### Knowledge Graph
Build a semantic knowledge graph from any document using LLM-powered entity extraction and natural language understanding.
- **Entity Extraction** - Automatic concept identification using spaCy + LLMs for prose-based semantic analysis
- **Relation Mapping** - Build semantic connections between concepts with intent-native relationship detection
- **Graph Visualization** - Interactive D3.js exploration for human-readable knowledge representation
- **External Linking** - Wikipedia and Semantic Scholar integration for knowledge augmentation

### Reading Interface
- **Dual-View Mode** - Toggle between technical and conceptual views
- **Complexity Slider** - Adjust content difficulty dynamically
- **Inline Annotations** - Highlight and comment on passages
- **Semantic Overlays** - Term tooltips with concept previews

### Adaptive Learning
Cognitive version control for your learning journey - track what you've learned and why it matters.
- **Flashcard Generation** - AI-powered question synthesis (T5/FLAN) for automated study material creation
- **Spaced Repetition** - SM2 algorithm for optimal review scheduling and reasoning audit trails
- **Engagement Tracking** - Monitor reading patterns, comprehension metrics, and cognitive work attribution
- **Export Options** - Anki, Obsidian, and Markdown formats for sovereign data ownership

### Scholar Mode
- **Citation Trees** - Explore reference chains (up to 3 hops)
- **Credibility Scoring** - CrossRef + Altmetrics integration
- **Zotero Export** - RIS, BibTeX, and CSL-JSON formats
- **Reflection Engine** - Track understanding evolution over time

## Architecture

```
Frontend (Next.js + React + Tailwind)
│
├── Reading Interface (dual-view, semantic overlays)
├── Knowledge Graph Visualization (D3.js)
├── Flashcard System
│
Backend (FastAPI + Python 3.11+)
│
├── Document Ingestion Service
├── Knowledge Graph Engine (Neo4j)
├── Learning Services (SM2, Flashcards)
├── Scholar Services (Citations, Credibility)
│
Storage Layer
├── PostgreSQL (users, documents, flashcards)
├── Neo4j (semantic graph)
└── FAISS (vector embeddings)
```

### B.1 Integrated Relation Extraction Pipeline

The knowledge graph construction uses an integrated pipeline combining multiple extraction methods:

**Extraction Methods (in priority order):**

1. **Coreference Resolution** (`coreference.py`)
   - Resolves pronouns and anaphoric references (he, she, it, they)
   - Handles definite descriptions ("the model", "this approach")
   - Links references to their antecedent entities for better relation coverage

2. **Dependency Parsing** (`dependency_parsing.py`)
   - Uses spaCy's dependency parser for syntactic structure
   - Extracts SVO (subject-verb-object) patterns
   - Falls back to pattern-based parsing when spaCy unavailable

3. **LLM-Based Extraction** (`llm_relations.py`)
   - Semantic understanding for complex relations
   - Supports multiple providers with automatic fallback:
     - **Ollama** (default) - Local LLM server for offline use
     - **llama.cpp** - Direct model loading for fully offline inference
     - **OpenAI** - Cloud API (requires OPENAI_API_KEY)
     - **Anthropic** - Cloud API (requires ANTHROPIC_API_KEY)

4. **Pattern Matching** (`integrated_pipeline.py`)
   - Rule-based extraction for common patterns
   - Multi-word entity matching with prefer-longer strategy
   - Co-occurrence fallback for uncovered entity pairs

**Setup for Offline LLM (Ollama):**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama3.2

# Start server
ollama serve
```

**Setup for spaCy (Optional but recommended):**
```bash
python setup_spacy.py
# Or manually:
python -m spacy download en_core_web_sm
```

**Usage:**
```python
from app.services.graph.builder import get_graph_builder

# Default: Uses integrated pipeline with Ollama
builder = get_graph_builder()

# With specific LLM provider
builder = get_graph_builder(llm_provider="openai")

# Without LLM (pattern-based only)
builder = get_graph_builder(use_llm=False)

# Build graph from text
result = await builder.build_from_text(text, doc_id)
```

**Extraction Pipeline Files:**
- `backend/app/services/graph/integrated_pipeline.py` - Main orchestrator
- `backend/app/services/graph/coreference.py` - Pronoun resolution
- `backend/app/services/graph/dependency_parsing.py` - Syntactic parsing
- `backend/app/services/graph/llm_relations.py` - LLM providers
- `backend/app/services/graph/spacy_loader.py` - Cached spaCy loader

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/unfold.git
cd unfold

# Start the backend, PostgreSQL, Neo4j and Redis with Docker Compose
cp .env.example .env
docker compose up -d

# Start the frontend (not part of the dev compose file)
cd frontend && cp .env.example .env.local && npm install && npm run dev

# Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Neo4j 5+ (optional, for knowledge graph)
- Docker & Docker Compose (recommended)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the server. With ENVIRONMENT=development it creates the database
# tables at startup; production/staging use migrations (alembic upgrade head).
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Upgrading an existing dev database

Flashcards used to live in memory; they are now stored in PostgreSQL, and the
schema is managed by Alembic. A development database created by an earlier
version still has the old `flashcards` table, because the dev server's
`create_tables` never alters existing tables. Every flashcard route then
returns 500, and `alembic upgrade head` fails because the tables already
exist. That table never held data, so drop it, let the backend recreate it,
and record the schema as current:

```bash
docker compose exec postgres psql -U postgres -d unfold -c 'DROP TABLE flashcards'
docker compose restart backend          # the dev server recreates the table
docker compose exec backend alembic stamp head
```

A database whose tables were created by the dev server (`ENVIRONMENT=development`)
must be stamped (`alembic stamp head`), not upgraded; later migrations then
apply with `alembic upgrade head` (`make db-migrate`).

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
cp .env.example .env.local
# Edit .env.local with your API URL

# Run development server
npm run dev
```

## Configuration

### Environment Variables

**Backend (.env)**

See `backend/.env.example` for the full list.

```bash
# Application: development | test | staging | production
ENVIRONMENT=development

# Database (required in production/staging)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/unfold

# Neo4j (optional; graph routes return 503 GRAPH_UNAVAILABLE without it)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme

# Security (production/staging: 32+ characters, no placeholder values)
JWT_SECRET=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30

# AI Services (optional)
OPENAI_API_KEY=your-openai-key

# CORS (comma-separated list, a single URL, or a JSON list)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Rate limiting per client IP (auth endpoints: 10 requests/min)
RATE_LIMIT_REQUESTS_PER_MINUTE=300
```

In production and staging the backend refuses to start with a missing or
short `JWT_SECRET`, a missing or empty `NEO4J_PASSWORD`, or placeholder values
(anything containing "change", such as `changeme`).

**Frontend (.env.local)**

```bash
# API base URL including the /api/v1 prefix (inlined at build time)
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

## API Documentation

### Authentication

All protected endpoints require a Bearer token in the Authorization header.

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecurePassword123!"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

Response (the refresh token is set as an httpOnly cookie):
```json
{
  "user": {
    "user_id": "uuid",
    "email": "user@example.com",
    "username": "johndoe"
  },
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Documents

#### Upload Document
```http
POST /api/v1/documents/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <PDF file>
```

Returns 201 with the processed document (status `validated`). The knowledge
graph is then built in the background and the status becomes `indexed`.
Errors: 400 with `detail.code` `EMPTY_FILE`, `CORRUPT_PDF`, `ENCRYPTED_PDF` or
`NO_TEXT_EXTRACTED`; 413 `FILE_TOO_LARGE`; 415 `UNSUPPORTED_TYPE`.

Documents are private to their owner: other users get 404.

#### List Documents
```http
GET /api/v1/documents?page=1&page_size=20
Authorization: Bearer <token>
```

#### Get Document with Paraphrase
```http
GET /api/v1/documents/{doc_id}/paraphrase?complexity=50
Authorization: Bearer <token>
```

### Knowledge Graph

All graph routes require a bearer token and only expose graphs of the
caller's documents. When Neo4j is unreachable they return 503 with
`detail.code` `GRAPH_UNAVAILABLE`.

#### Rebuild a Document's Graph
```http
POST /api/v1/graph/documents/{doc_id}/build
Authorization: Bearer <token>
```

Rebuilds the graph from the stored document text (idempotent). Response:
`{doc_id, nodes_created, relations_created, errors}`.

#### Get a Document's Relations
```http
GET /api/v1/graph/documents/{doc_id}/relations
Authorization: Bearer <token>
```

#### Search Nodes
```http
GET /api/v1/graph/nodes?source_doc_id={doc_id}&query=quantum&node_type=Concept&limit=50
Authorization: Bearer <token>
```

#### Get Related Nodes
```http
GET /api/v1/graph/nodes/{node_id}/related?max_depth=2&limit=50
Authorization: Bearer <token>
```

#### Link to Wikipedia
```http
GET /api/v1/graph/link/wikipedia/{entity}
Authorization: Bearer <token>
```

#### Search Academic Papers
```http
GET /api/v1/graph/link/papers?query=machine+learning&limit=10
Authorization: Bearer <token>
```

### Learning System

Flashcards are stored per user in PostgreSQL and scheduled with SM2.

#### Generate Flashcards
```http
POST /api/v1/learning/flashcards/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "document_id": "<doc_id>",
  "num_cards": 10
}
```

Reads the text of one of your documents server-side; the new cards are due
immediately. `GET /api/v1/learning/flashcards?document_id=` lists your cards,
`POST /api/v1/learning/flashcards` adds one manually and
`DELETE /api/v1/learning/flashcards/{card_id}` removes one.

#### Review Flashcard (SM2)
```http
POST /api/v1/learning/flashcards/review
Authorization: Bearer <token>
Content-Type: application/json

{
  "card_id": "card-123",
  "quality": 4
}
```

Quality ratings:
- 0: Complete blackout
- 1: Incorrect, recognized after
- 2: Incorrect, seemed easy after
- 3: Correct with difficulty
- 4: Correct with hesitation
- 5: Perfect recall

#### Get Due Flashcards
```http
GET /api/v1/learning/flashcards/due?limit=20
Authorization: Bearer <token>
```

#### Export Flashcards
```http
POST /api/v1/learning/export/flashcards
Authorization: Bearer <token>
Content-Type: application/json

{
  "flashcards": [
    {"question": "What is...", "answer": "It is..."}
  ],
  "format": "anki_csv",
  "title": "My Study Deck"
}
```

Supported formats: `json`, `anki_csv`, `anki_txt`, `anki_json`, `obsidian_sr`, `obsidian_callout`, `markdown_table`

### Scholar Mode

#### Build Citation Tree
```http
POST /api/v1/scholar/citations/tree
Authorization: Bearer <token>
Content-Type: application/json

{
  "doi": "10.1038/nature12373",
  "max_depth": 2,
  "refs_per_level": 10,
  "cites_per_level": 10
}
```

#### Score Paper Credibility
```http
POST /api/v1/scholar/credibility/score
Authorization: Bearer <token>
Content-Type: application/json

{
  "doi": "10.1038/nature12373"
}
```

🧠 5. AI Models
Purpose	Suggested Model	Notes
Entity/Relation Extraction	Ollama (llama3.2) / GPT-4o / Claude 3.5	Local-first with cloud fallback
Dependency Parsing	spaCy (en_core_web_sm)	Syntactic structure analysis
Coreference Resolution	Rule-based + LLM hybrid	Pronoun and reference linking
Summarization/Paraphrasing	GPT-4o-mini / Mistral 8x7B	Multi-level simplification
Question Generation	T5 / FLAN-UL2	SRS integration
Image Captioning	BLIP-2 / Pix2Struct	Diagram understanding

**Local/Offline LLM Options:**
- Ollama (recommended): Easy setup, runs llama3.2, mistral, qwen2.5 locally
- llama.cpp: Direct GGUF model loading, fully offline
- Both options enable knowledge graph construction without internet/API keys
🧭 6. APIs and Integrations
Function	API
Document Validation	CrossRef, Unpaywall, CORE
Metadata & Author ID	ORCID, ROR
Citation Management	Zotero
Knowledge Links	Wikipedia, arXiv, Semantic Scholar
LMS Integration	LTI 1.3 (Canvas, Moodle)
Analytics	Mixpanel / PostHog
Storage	AWS S3 / GCS / IPFS (optional decentralized mode)
💡 7. Security & Privacy

All data encrypted (AES-256 at rest, TLS 1.3 in transit)

OpenAI usage governed under academic license agreements

🧰 8. Development Environment
Stack	Tool
Backend	Python 3.11+, FastAPI, spaCy
Frontend	Next.js (React 18+), Tailwind, D3.js
Database	PostgreSQL + Neo4j
Embeddings	FAISS
Auth	OAuth2 + JWT (optional ORCID login)
Testing	PyTest + Playwright
DevOps	Docker Compose + GitHub Actions CI/CD
🚀 9. Roadmap Summary
Milestone	Deliverables
v0.1	Document ingestion + validation
v0.2	Semantic graph + embeddings
v0.3	Reading interface MVP
v0.4	Adaptive focus mode
v0.5	Scholar Mode + reflection engine
v1.0	Public beta (ethics suite deferred to the backlog, see REFOCUS_PLAN.md)
🧩 10. Licensing & Open Science

License: AGPL v3 (to ensure community benefit)

Open access to non-proprietary models and datasets

Opt-in Transparency Portal:

Model prompts

Audit logs

Citation sources

#### Export to Zotero
```http
POST /api/v1/scholar/zotero/export
Authorization: Bearer <token>
Content-Type: application/json

{
  "items": [
    {
      "title": "Paper Title",
      "authors": ["Author Name"],
      "year": 2024,
      "doi": "10.1234/example"
    }
  ],
  "format": "bibtex"
}
```

#### Create Reading Snapshot
```http
POST /api/v1/scholar/reflection/snapshot
Authorization: Bearer <token>
Content-Type: application/json

{
  "document_id": "doc-123",
  "reflection_type": "deep_analysis",
  "complexity_level": 70,
  "summary": "Key understanding...",
  "key_takeaways": ["Point 1", "Point 2"]
}
```

### Health Check

```http
GET /api/v1/health
```

Returns 503 when PostgreSQL is unreachable (`/health/ready` likewise);
`/health/live` only reports that the process is up.

## Frontend Development

### Project Structure

```
frontend/
├── src/
│   ├── app/              # Next.js App Router pages
│   ├── components/       # React components
│   │   ├── ui/          # Shared UI components
│   │   ├── graph/       # Knowledge graph components
│   │   ├── learning/    # Flashcard/learning components
│   │   └── reader/      # Reading interface components
│   ├── services/        # API client
│   ├── store/           # Zustand state management
│   ├── types/           # TypeScript types
│   └── utils/           # Utilities and helpers
├── e2e/                 # Playwright E2E tests
└── public/              # Static assets
```

### Available Scripts

```bash
# Development
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server

# Linting & Type Checking
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript compiler

# Testing
npm run test:e2e           # Run Playwright tests
npm run test:e2e:ui        # Run with Playwright UI
npm run test:e2e:headed    # Run in headed browser
npm run test:e2e:report    # Show test report
```

## Testing

### Backend Tests

The suite needs PostgreSQL and drops/recreates the schema of the database it
runs against, so `DATABASE_URL` must name a database ending in `_test`
(default: `postgresql://test:test@localhost:5432/unfold_test`). Neo4j is
optional: tests marked `requires_neo4j` are skipped when `NEO4J_URI` is not
reachable, and tests marked `requires_no_neo4j` (the 503 paths) run only then.
CI runs the suite with both services, then runs the `requires_no_neo4j` tests
again with `NEO4J_URI` pointing at an unused port.

```bash
cd backend
pip install -r requirements-dev.txt   # runtime + test/lint tools

createdb -h localhost -U postgres unfold_test
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/unfold_test
# Optional: export NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=...

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/                    # Unit tests
pytest tests/integration/             # Integration tests

# Run specific test file
pytest tests/integration/test_document_flow.py
```

### Frontend E2E Tests

```bash
cd frontend

# Install the Playwright browser (the suite runs on Chromium)
npx playwright install chromium

# Run all E2E tests (as CI does)
npm run test:e2e -- --project=chromium

# Run specific test file
npx playwright test e2e/auth.spec.ts

# Run in headed mode for debugging
npm run test:e2e:headed

# Open Playwright UI
npm run test:e2e:ui
```

### Test Coverage

The test suite includes:

**Backend Tests:**
- Authentication, health checks, configuration and rate limiting
- Document upload, error codes and per-user ownership
- Knowledge graph building, queries and the Neo4j-down (503) path
- Flashcard generation, persistence and SM2 scheduling

**Frontend E2E Tests** (Chromium, against a mocked API; `frontend/e2e/`):
- Authentication (`auth.spec.ts`)
- Documents and upload (`documents.spec.ts`)
- Reader (`reader.spec.ts`)
- Knowledge graph (`graph.spec.ts`)
- Flashcards and review (`flashcards.spec.ts`)
- Navigation (`navigation.spec.ts`)

## Deployment

### Docker Compose (Recommended)

- **Development:** `docker compose up -d` starts the backend (port 8000),
  PostgreSQL, Neo4j and Redis using `docker-compose.yml` and `.env`
  (copy `.env.example`). Their ports are published on 127.0.0.1 only. Run
  the frontend with `npm run dev`.
- **Production:** `docker-compose.prod.yml` adds nginx (TLS, `/api/` routed to
  the backend) and the frontend, and runs `alembic upgrade head` before the API
  starts. Secrets are required:

```bash
cp .env.production.example .env.production   # fill in the secrets
make ssl-generate                            # or put real certificates in nginx/ssl/
docker compose --env-file .env.production -f docker-compose.prod.yml up -d
```

The frontend is built with `NEXT_PUBLIC_API_URL=/api/v1` (same origin through
nginx) unless you set another value.

### Production Considerations

1. **Security**
   - Use strong, unique passwords for all services
   - Enable HTTPS with valid SSL certificates
   - Configure CORS appropriately
   - Use environment variables for secrets

2. **Performance**
   - Enable Redis caching for API responses
   - Configure connection pooling for databases
   - Use CDN for static frontend assets
   - Enable gzip compression

3. **Monitoring**
   - Set up health checks
   - Configure logging (structured JSON)
   - Use APM tools (Datadog, New Relic)
   - Set up alerts for errors

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Connected Repositories

Unfold is part of a broader ecosystem of tools focused on human-AI collaboration, cognitive work preservation, and natural language-first computing. Explore these related projects:

### NatLangChain Ecosystem
- **[NatLangChain](https://github.com/kase1111-hash/NatLangChain)** - Prose-first, intent-native blockchain protocol for recording human intent in natural language
- **[IntentLog](https://github.com/kase1111-hash/IntentLog)** - Git for human reasoning; tracks "why" changes happen via prose commits and semantic version control
- **[RRA-Module](https://github.com/kase1111-hash/RRA-Module)** - Revenant Repo Agent for autonomous licensing and abandoned repo monetization
- **[mediator-node](https://github.com/kase1111-hash/mediator-node)** - LLM mediation layer for semantic matching and natural language negotiation
- **[ILR-module](https://github.com/kase1111-hash/ILR-module)** - IP & Licensing Reconciliation for automated dispute resolution
- **[Finite-Intent-Executor](https://github.com/kase1111-hash/Finite-Intent-Executor)** - Posthumous smart contract execution for digital estate automation

### Agent-OS Ecosystem
- **[Agent-OS](https://github.com/kase1111-hash/Agent-OS)** - Natural language operating system for AI agents with constitutional AI governance
- **[synth-mind](https://github.com/kase1111-hash/synth-mind)** - NLOS-based agent with psychological AI architecture for emergent continuity and empathy
- **[boundary-daemon-](https://github.com/kase1111-hash/boundary-daemon-)** - Trust enforcement layer defining AI cognition boundaries and security policies
- **[memory-vault](https://github.com/kase1111-hash/memory-vault)** - Sovereign, offline-capable storage for cognitive artifacts and AI memory ownership
- **[value-ledger](https://github.com/kase1111-hash/value-ledger)** - Economic accounting layer for cognitive work, idea attribution, and thought valuation
- **[learning-contracts](https://github.com/kase1111-hash/learning-contracts)** - Safety protocols and governance framework for AI learning boundaries

### Security & Infrastructure
- **[Boundary-SIEM](https://github.com/kase1111-hash/Boundary-SIEM)** - Security Information and Event Management for AI agent monitoring

### Games
- **[Shredsquatch](https://github.com/kase1111-hash/Shredsquatch)** - 3D first-person snowboarding infinite runner (SkiFree spiritual successor)
- **[Midnight-pulse](https://github.com/kase1111-hash/Midnight-pulse)** - Procedurally generated synthwave night driving experience
- **[Long-Home](https://github.com/kase1111-hash/Long-Home)** - Atmospheric narrative indie game built with Godot

## License

This project is licensed under the AGPL v3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Frontend powered by [Next.js](https://nextjs.org/)
- Knowledge graphs with [Neo4j](https://neo4j.com/)
- Vector search with [FAISS](https://github.com/facebookresearch/faiss)
- Spaced repetition based on [SM2 algorithm](https://www.supermemo.com/en/archives1990-2015/english/ol/sm2)

## Part of the Authenticity Economy Ecosystem

Unfold is part of a broader ecosystem of tools focused on **human-AI collaboration**, **natural language programming**, and **owned AI infrastructure**. These connected projects share common principles of preserving human intent and cognitive value.

### NatLangChain Ecosystem

| Repository | Description |
|------------|-------------|
| [NatLangChain](https://github.com/kase1111-hash/NatLangChain) | Prose-first, intent-native blockchain protocol for recording human intent in natural language |
| [IntentLog](https://github.com/kase1111-hash/IntentLog) | Git for human reasoning - tracks "why" changes happen via prose commits |
| [RRA-Module](https://github.com/kase1111-hash/RRA-Module) | Revenant Repo Agent - converts abandoned GitHub repositories into autonomous AI agents |
| [mediator-node](https://github.com/kase1111-hash/mediator-node) | LLM mediation layer for matching, negotiation, and closure proposals |
| [ILR-module](https://github.com/kase1111-hash/ILR-module) | IP & Licensing Reconciliation - dispute resolution for intellectual property conflicts |
| [Finite-Intent-Executor](https://github.com/kase1111-hash/Finite-Intent-Executor) | Posthumous execution of predefined intent via Solidity smart contracts |

### Agent-OS Ecosystem

| Repository | Description |
|------------|-------------|
| [Agent-OS](https://github.com/kase1111-hash/Agent-OS) | Natural-language native operating system (NLOS) for AI agents |
| [synth-mind](https://github.com/kase1111-hash/synth-mind) | NLOS-based agent with six psychological modules for emergent continuity and empathy |
| [boundary-daemon](https://github.com/kase1111-hash/boundary-daemon-) | Mandatory trust enforcement layer for Agent OS defining cognition boundaries |
| [memory-vault](https://github.com/kase1111-hash/memory-vault) | Secure, offline-capable, owner-sovereign storage for cognitive artifacts |
| [value-ledger](https://github.com/kase1111-hash/value-ledger) | Economic accounting layer for cognitive work (ideas, effort, novelty) |
| [learning-contracts](https://github.com/kase1111-hash/learning-contracts) | Safety protocols for AI learning and data management |

### Security & Games

| Repository | Description |
|------------|-------------|
| [Boundary-SIEM](https://github.com/kase1111-hash/Boundary-SIEM) | Security Information and Event Management system for AI |
| [Shredsquatch](https://github.com/kase1111-hash/Shredsquatch) | 3D first-person snowboarding infinite runner (SkiFree homage) |
| [Midnight-pulse](https://github.com/kase1111-hash/Midnight-pulse) | Procedurally generated night drive game |
| [Long-Home](https://github.com/kase1111-hash/Long-Home) | Narrative indie game built with Godot |
