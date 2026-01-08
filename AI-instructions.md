# AI Instructions for Unfold Development

This document provides step-by-step guidance for implementing the Unfold platform — an AI-assisted reading and comprehension system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack Reference](#2-technology-stack-reference)
3. [Project Structure](#3-project-structure)
4. [Development Setup](#4-development-setup)
5. [Implementation Order](#5-implementation-order)
6. [Module Implementation Guide](#6-module-implementation-guide)
7. [Data Schemas](#7-data-schemas)
8. [API Design Conventions](#8-api-design-conventions)
9. [Testing Strategy](#9-testing-strategy)
10. [Security & Privacy Checklist](#10-security--privacy-checklist)
11. [Code Style & Conventions](#11-code-style--conventions)

---

## 1. Project Overview

**Unfold** is a modular AI-assisted reading platform with five primary layers:

| Layer | Purpose |
|-------|---------|
| Document Ingestion & Validation | Securely parse, verify, and index documents |
| Semantic Parsing & Knowledge Graph | Convert text into structured, queryable nodes |
| Reading Interface | Interactive dual-view document explorer |
| Adaptive Learning & Focus Engine | Personalized summarization and spaced-repetition |
| Ethical & Academic Framework | Provenance, compliance, and bias audit systems |

---

## 2. Technology Stack Reference

### Backend
- **Language:** Python 3.11+
- **Framework:** FastAPI
- **AI Orchestration:** LangChain / LangGraph
- **Authentication:** OAuth2 + JWT (optional ORCID login)

### Frontend
- **Framework:** Next.js (React 18+)
- **Styling:** Tailwind CSS
- **Visualization:** D3.js
- **State Management:** Zustand or Redux Toolkit

### Databases
- **User Data:** PostgreSQL
- **Knowledge Graph:** Neo4j or Weaviate
- **Vector Embeddings:** Pinecone or FAISS

### DevOps
- **Containerization:** Docker Compose
- **CI/CD:** GitHub Actions
- **Testing:** PyTest (backend), Cypress (frontend)

---

## 3. Project Structure

```
unfold/
├── frontend/                    # Next.js application
│   ├── src/
│   │   ├── app/                 # App router pages
│   │   ├── components/          # React components
│   │   │   ├── reading/         # Reading interface components
│   │   │   ├── graph/           # Knowledge graph visualizations
│   │   │   ├── learning/        # Adaptive learning UI
│   │   │   └── common/          # Shared UI components
│   │   ├── hooks/               # Custom React hooks
│   │   ├── store/               # Zustand/Redux state
│   │   ├── services/            # API client services
│   │   ├── types/               # TypeScript type definitions
│   │   └── utils/               # Utility functions
│   ├── public/                  # Static assets
│   ├── tailwind.config.js
│   ├── next.config.js
│   └── package.json
│
├── backend/                     # FastAPI application
│   ├── app/
│   │   ├── main.py              # Application entry point
│   │   ├── config.py            # Configuration settings
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── routes/
│   │   │   │   │   ├── documents.py
│   │   │   │   │   ├── graph.py
│   │   │   │   │   ├── learning.py
│   │   │   │   │   ├── scholar.py
│   │   │   │   │   └── auth.py
│   │   │   │   └── dependencies.py
│   │   │   └── __init__.py
│   │   ├── services/
│   │   │   ├── ingestion/       # Document ingestion service
│   │   │   │   ├── parser.py    # PDF/EPUB parsing
│   │   │   │   ├── ocr.py       # OCR with Tesseract
│   │   │   │   └── validator.py # DOI/metadata validation
│   │   │   ├── graph/           # Knowledge graph service
│   │   │   │   ├── extractor.py # Entity extraction
│   │   │   │   ├── builder.py   # Graph construction
│   │   │   │   └── linker.py    # External API linking
│   │   │   ├── llm/             # LLM orchestration
│   │   │   │   ├── chains.py    # LangChain chains
│   │   │   │   ├── paraphrase.py
│   │   │   │   └── summarize.py
│   │   │   ├── learning/        # Adaptive learning service
│   │   │   │   ├── tracker.py   # Engagement tracking
│   │   │   │   ├── flashcards.py
│   │   │   │   └── scheduler.py # SM2 algorithm
│   │   │   ├── ethics/          # Bias audit module
│   │   │   │   ├── provenance.py
│   │   │   │   └── audit.py
│   │   │   └── external/        # External API integrations
│   │   │       ├── crossref.py
│   │   │       ├── orcid.py
│   │   │       ├── semantic_scholar.py
│   │   │       └── zotero.py
│   │   ├── models/              # Pydantic models
│   │   │   ├── document.py
│   │   │   ├── graph.py
│   │   │   ├── user.py
│   │   │   └── learning.py
│   │   ├── db/                  # Database connections
│   │   │   ├── postgres.py
│   │   │   ├── neo4j.py
│   │   │   └── vector.py        # Pinecone/FAISS
│   │   └── utils/
│   │       ├── hashing.py       # SHA-256, C2PA
│   │       └── embeddings.py    # OpenAI embeddings
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── conftest.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── backend-ci.yml
│       └── frontend-ci.yml
├── README.md
├── LICENSE.md
└── AI-instructions.md
```

---

## 4. Development Setup

### Step 1: Prerequisites
```bash
# Required installations
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+
- Neo4j 5+ (or Weaviate)
```

### Step 2: Clone and Configure
```bash
git clone <repository-url>
cd unfold

# Copy environment templates
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

### Step 3: Environment Variables

**Backend `.env`:**
```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/unfold
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Vector Store
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=your-env

# AI Models
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# External APIs
CROSSREF_EMAIL=your-email
ORCID_CLIENT_ID=your-id
ORCID_CLIENT_SECRET=your-secret

# Security
JWT_SECRET=your-secret-key
```

**Frontend `.env.local`:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

### Step 4: Install Dependencies
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### Step 5: Start Services
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or manually:
# Terminal 1 - Backend
cd backend && uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend && npm run dev
```

---

## 5. Implementation Order

Follow this sequence for a logical build progression:

### Phase 1: Foundation (v0.1)
1. **Backend skeleton** - FastAPI setup with health endpoints
2. **Database connections** - PostgreSQL and Neo4j clients
3. **Authentication** - JWT-based auth system
4. **Document ingestion** - PDF/EPUB parsing with Apache Tika
5. **Validation service** - DOI/CrossRef integration

### Phase 2: Knowledge Graph (v0.2)
6. **Entity extraction** - spaCy + LLM extraction pipeline
7. **Graph builder** - Neo4j node/relationship creation
8. **Embedding service** - OpenAI text-embedding-3-large integration
9. **Vector storage** - Pinecone/FAISS setup
10. **External linkers** - Wikipedia, Semantic Scholar APIs

### Phase 3: Reading Interface (v0.3)
11. **Frontend skeleton** - Next.js with Tailwind setup
12. **Document viewer** - Technical view component
13. **Paraphrase service** - LLM-powered simplification
14. **Dual-view toggle** - Complexity slider component
15. **Graph visualization** - D3.js knowledge map
16. **Highlight Q&A** - Chat-on-selection feature

### Phase 4: Focus & Learning (v0.4)
17. **Relevance scoring** - TF-IDF + BERT relevance engine
18. **Focus mode UI** - Progressive reveal accordion
19. **Engagement tracker** - Dwell time, scroll tracking
20. **Flashcard generator** - T5/FLAN question synthesis
21. **SM2 scheduler** - Spaced repetition algorithm
22. **Export service** - Anki/Obsidian markdown export

### Phase 5: Scholar Mode (v0.5)
23. **Citation tree** - Reference chain visualization
24. **Zotero integration** - Citation export
25. **Credibility scoring** - CrossRef/Altmetrics integration
26. **Reflection engine** - Time-based snapshot diffs
27. **Collaborative features** - CRDT-based annotations

### Phase 6: Ethics & Launch (v1.0)
28. **Provenance system** - C2PA manifest implementation
29. **Bias audit module** - RoBERTa sentiment analysis
30. **Privacy compliance** - Differential privacy, GDPR
31. **Analytics dashboard** - User ethics transparency
32. **Performance optimization** - Caching, CDN setup

---

## 6. Module Implementation Guide

### A. Document Ingestion Service

**Location:** `backend/app/services/ingestion/`

**Step 1: Create parser.py**
```python
# Implement PDF/EPUB parsing
# - Use PyPDF2 or Apache Tika for PDF extraction
# - Handle multi-column layouts
# - Extract metadata (title, authors, abstract)
```

**Step 2: Create ocr.py**
```python
# OCR fallback for scanned documents
# - Integrate Tesseract or Donut
# - Handle tables and formulas
```

**Step 3: Create validator.py**
```python
# Document validation pipeline
# - DOI validation via CrossRef API
# - License check via Creative Commons API
# - Author verification via ORCID
# - Generate SHA-256 hash for provenance
```

**Data Flow:**
```
Upload → Validate hash & license → Extract metadata → Store record → Forward to Graph Builder
```

---

### B. Knowledge Graph Service

**Location:** `backend/app/services/graph/`

**Step 1: Create extractor.py**
```python
# Entity and relation extraction
# - Use spaCy for NER
# - Use GPT-4o for relationship extraction
# - Output structured JSON

# Entity types: Concept, Author, Paper, Method, Dataset
# Relation types: EXPLAINS, CITES, CONTRASTS_WITH, DERIVES_FROM
```

**Step 2: Create builder.py**
```python
# Graph construction in Neo4j
# - Create nodes with embeddings
# - Establish relationships
# - Handle graph updates and merging
```

**Step 3: Create linker.py**
```python
# External knowledge linking
# - Wikipedia API integration
# - Semantic Scholar lookups
# - arXiv metadata fetching
# - DOI resolution
```

---

### C. Reading Interface

**Location:** `frontend/src/components/reading/`

**Key Components:**
```
TechnicalView.tsx    - Raw formatted text with reference popovers
ConceptualView.tsx   - LLM-generated paraphrases
HybridSlider.tsx     - Complexity gradient toggle
TermTooltip.tsx      - Hover tooltips with graph preview
HighlightChat.tsx    - Q&A popup on selected text
```

**State Management:**
```typescript
// store/readingStore.ts
interface ReadingState {
  documentId: string;
  complexityLevel: number;  // 0-100 slider
  viewMode: 'technical' | 'conceptual' | 'hybrid';
  selectedText: string | null;
  activeNodes: GraphNode[];
}
```

---

### D. Adaptive Learning Service

**Location:** `backend/app/services/learning/`

**SM2 Algorithm Implementation:**
```python
# scheduler.py
def calculate_next_review(
    quality: int,      # 0-5 user rating
    repetitions: int,
    easiness: float,
    interval: int
) -> tuple[int, float, int]:
    # Implement SM2 spaced repetition
    pass
```

**Flashcard Generation:**
```python
# flashcards.py
# Use T5 or FLAN-UL2 for question synthesis
# Input: Text passage + context
# Output: Question-answer pairs
```

---

### E. Ethical Framework

**Location:** `backend/app/services/ethics/`

**Provenance Implementation:**
```python
# provenance.py
# - C2PA manifest creation
# - SHA-256 content fingerprinting
# - DOI revalidation checks
```

**Bias Audit:**
```python
# audit.py
# - Sentiment analysis with RoBERTa
# - Perspective API integration
# - Language inclusivity metrics
```

---

## 7. Data Schemas

### Document Record
```json
{
  "doc_id": "sha256:xxxx",
  "title": "string",
  "authors": ["John Doe", "Jane Smith"],
  "doi": "10.xxxx/abc123",
  "license": "CC-BY-4.0",
  "source": "arXiv",
  "vector_id": "uuid",
  "graph_nodes": ["concept_001", "author_002"],
  "created_at": "2026-01-08T00:00:00Z",
  "updated_at": "2026-01-08T00:00:00Z"
}
```

### Knowledge Graph Node
```json
{
  "node_id": "concept_001",
  "type": "Concept",
  "label": "Quantum Entanglement",
  "embedding": [0.34, 0.56, ...],
  "relations": [
    {"type": "CITES", "target": "concept_004"}
  ],
  "metadata": {
    "source_doc": "sha256:xxxx",
    "confidence": 0.95
  }
}
```

### User Learning Record
```json
{
  "user_id": "uuid",
  "document_id": "sha256:xxxx",
  "engagement": {
    "dwell_time_seconds": 1200,
    "scroll_depth": 0.85,
    "annotations_count": 5
  },
  "flashcards": [
    {
      "card_id": "uuid",
      "next_review": "2026-01-15T00:00:00Z",
      "easiness": 2.5,
      "interval": 7,
      "repetitions": 3
    }
  ]
}
```

### Reflection Snapshot
```json
{
  "user_id": "uuid",
  "timestamp": "2026-01-05T13:00:00Z",
  "nodes_added": 12,
  "nodes_deleted": 1,
  "notes": "Now understand Bell's theorem implications"
}
```

---

## 8. API Design Conventions

### Endpoint Structure
```
/api/v1/
├── /auth
│   ├── POST /login
│   ├── POST /register
│   └── POST /refresh
├── /documents
│   ├── POST /upload
│   ├── GET /{doc_id}
│   ├── GET /{doc_id}/paraphrase
│   └── DELETE /{doc_id}
├── /graph
│   ├── GET /nodes/{node_id}
│   ├── GET /search
│   └── GET /traverse
├── /learning
│   ├── GET /flashcards
│   ├── POST /flashcards/{card_id}/review
│   ├── GET /progress
│   └── POST /export
├── /scholar
│   ├── GET /citations/{doc_id}
│   ├── GET /credibility/{doc_id}
│   └── POST /zotero/export
└── /ethics
    ├── GET /provenance/{doc_id}
    └── GET /bias-report/{doc_id}
```

### Response Format
```json
{
  "status": "success",
  "data": { ... },
  "meta": {
    "timestamp": "2026-01-08T00:00:00Z",
    "request_id": "uuid"
  }
}
```

### Error Format
```json
{
  "status": "error",
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "Document with ID xyz not found",
    "details": {}
  }
}
```

---

## 9. Testing Strategy

### Backend Testing (PyTest)

**Unit Tests:**
```bash
backend/tests/unit/
├── test_parser.py           # Document parsing
├── test_validator.py        # Validation logic
├── test_extractor.py        # Entity extraction
├── test_scheduler.py        # SM2 algorithm
└── test_flashcards.py       # Question generation
```

**Integration Tests:**
```bash
backend/tests/integration/
├── test_document_flow.py    # Full ingestion pipeline
├── test_graph_queries.py    # Neo4j operations
└── test_external_apis.py    # CrossRef, ORCID mocks
```

**Run Tests:**
```bash
cd backend
pytest --cov=app tests/
```

### Frontend Testing (Cypress)

**E2E Tests:**
```bash
frontend/cypress/e2e/
├── document-upload.cy.ts
├── reading-interface.cy.ts
├── complexity-slider.cy.ts
└── flashcard-review.cy.ts
```

**Run Tests:**
```bash
cd frontend
npm run cypress:run
```

---

## 10. Security & Privacy Checklist

### Encryption
- [ ] AES-256 encryption at rest for all stored data
- [ ] TLS 1.3 for all data in transit
- [ ] Secure key management (AWS KMS or similar)

### Authentication
- [ ] JWT with short expiration + refresh tokens
- [ ] Rate limiting on auth endpoints
- [ ] ORCID OAuth2 integration (optional)

### Privacy
- [ ] Differential privacy for analytics
- [ ] User consent management system
- [ ] Data retention policies (GDPR compliance)
- [ ] Anonymization of tracking data

### API Security
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] CORS configuration

### Content Security
- [ ] C2PA provenance verification
- [ ] Copyright/license validation before ingestion
- [ ] Rate limiting on external API calls

---

## 11. Code Style & Conventions

### Python (Backend)
- Follow PEP 8
- Use type hints everywhere
- Docstrings for all public functions
- Use Pydantic for data validation
- Async/await for I/O operations

```python
from pydantic import BaseModel

class DocumentCreate(BaseModel):
    """Schema for document creation."""
    title: str
    authors: list[str]
    doi: str | None = None

async def create_document(data: DocumentCreate) -> Document:
    """Create a new document record.

    Args:
        data: Document creation payload.

    Returns:
        Created document with generated ID.
    """
    pass
```

### TypeScript (Frontend)
- Strict TypeScript configuration
- Functional components with hooks
- Named exports for components
- CSS modules or Tailwind classes

```typescript
interface ReadingViewProps {
  documentId: string;
  initialMode?: 'technical' | 'conceptual';
}

export function ReadingView({
  documentId,
  initialMode = 'technical'
}: ReadingViewProps) {
  // Component implementation
}
```

### Git Conventions
- Branch naming: `feature/`, `fix/`, `docs/`
- Commit messages: Conventional Commits format
- PR reviews required before merge
- Squash merge to main branch

```
feat(graph): add entity extraction pipeline
fix(auth): resolve JWT refresh race condition
docs(api): update endpoint documentation
```

---

## Quick Reference Commands

```bash
# Start development environment
docker-compose up -d

# Run backend
cd backend && uvicorn app.main:app --reload

# Run frontend
cd frontend && npm run dev

# Run all tests
cd backend && pytest
cd frontend && npm test

# Database migrations
cd backend && alembic upgrade head

# Generate API docs
# Visit http://localhost:8000/docs (Swagger UI)
```

---

## External API Quick Reference

| Service | Purpose | Documentation |
|---------|---------|---------------|
| CrossRef | DOI validation | https://api.crossref.org |
| ORCID | Author verification | https://orcid.org/developer |
| Semantic Scholar | Paper metadata | https://api.semanticscholar.org |
| Unpaywall | Open access lookup | https://unpaywall.org/api |
| Zotero | Citation export | https://www.zotero.org/support/dev |
| OpenAI | Embeddings & LLM | https://platform.openai.com/docs |

---

*Last updated: 2026-01-08*
*Version: 1.0*
*License: AGPL v3*
