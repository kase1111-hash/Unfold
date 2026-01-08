# Unfold

**AI-Assisted Reading and Comprehension Platform**

Unfold bridges the gap between dense academic/technical texts and genuine understanding through modular AI assistance, emphasizing ethics, explainability, and educational collaboration.

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
- [License](#license)

## Features

### Document Management
- **PDF/EPUB Ingestion** - Upload and process academic documents
- **DOI Validation** - Verify document authenticity via CrossRef
- **Provenance Tracking** - C2PA-compliant content fingerprinting
- **License Compliance** - Creative Commons validation

### Knowledge Graph
- **Entity Extraction** - Automatic concept identification using spaCy + LLMs
- **Relation Mapping** - Build semantic connections between concepts
- **Graph Visualization** - Interactive D3.js exploration
- **External Linking** - Wikipedia and Semantic Scholar integration

### Reading Interface
- **Dual-View Mode** - Toggle between technical and conceptual views
- **Complexity Slider** - Adjust content difficulty dynamically
- **Inline Annotations** - Highlight and comment on passages
- **Semantic Overlays** - Term tooltips with concept previews

### Adaptive Learning
- **Flashcard Generation** - AI-powered question synthesis (T5/FLAN)
- **Spaced Repetition** - SM2 algorithm for optimal review scheduling
- **Engagement Tracking** - Monitor reading patterns and comprehension
- **Export Options** - Anki, Obsidian, and Markdown formats

### Scholar Mode
- **Citation Trees** - Explore reference chains (up to 3 hops)
- **Credibility Scoring** - CrossRef + Altmetrics integration
- **Zotero Export** - RIS, BibTeX, and CSL-JSON formats
- **Reflection Engine** - Track understanding evolution over time

### Ethics & Privacy
- **Bias Auditing** - Sentiment analysis and inclusivity checks
- **GDPR Compliance** - Consent management and data portability
- **Differential Privacy** - Anonymized analytics
- **Transparency Dashboard** - AI operation tracking

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
├── Ethics Services (Provenance, Privacy)
│
Storage Layer
├── PostgreSQL (user data, sessions)
├── Neo4j (semantic graph)
└── FAISS (vector embeddings)
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/unfold.git
cd unfold

# Start with Docker Compose
docker-compose up -d

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

# Initialize database
python -c "from app.db import create_tables; import asyncio; asyncio.run(create_tables())"

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

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

```bash
# Application
APP_NAME=Unfold
APP_VERSION=1.0.0
ENVIRONMENT=development

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=unfold
POSTGRES_USER=unfold
POSTGRES_PASSWORD=your-secure-password

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Security
JWT_SECRET=your-jwt-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Services (optional)
OPENAI_API_KEY=your-openai-key

# CORS
CORS_ORIGINS=["http://localhost:3000"]
```

**Frontend (.env.local)**

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
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

Response:
```json
{
  "user": {
    "user_id": "uuid",
    "email": "user@example.com",
    "username": "johndoe"
  },
  "tokens": {
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "bearer"
  }
}
```

### Documents

#### Upload Document
```http
POST /api/v1/documents/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <PDF or EPUB file>
```

#### List Documents
```http
GET /api/v1/documents/?page=1&page_size=20
Authorization: Bearer <token>
```

#### Get Document with Paraphrase
```http
GET /api/v1/documents/{doc_id}/paraphrase?complexity=50
Authorization: Bearer <token>
```

### Knowledge Graph

#### Build Graph from Text
```http
POST /api/v1/graph/build
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "Your document text here...",
  "source_doc_id": "doc-123",
  "extract_relations": true,
  "generate_embeddings": true
}
```

#### Search Nodes
```http
GET /api/v1/graph/nodes?query=quantum&node_type=Concept&limit=50
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
```

#### Search Academic Papers
```http
GET /api/v1/graph/link/papers?query=machine+learning&limit=10
```

### Learning System

#### Generate Flashcards
```http
POST /api/v1/learning/flashcards/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "Your study material here...",
  "num_cards": 5,
  "difficulty": "intermediate"
}
```

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

### Ethics & Privacy

#### Create Provenance Credential
```http
POST /api/v1/ethics/provenance/create
Authorization: Bearer <token>
Content-Type: application/json

{
  "document_id": "doc-123",
  "content": "Document content..."
}
```

#### Audit Document for Bias
```http
POST /api/v1/ethics/bias/audit
Authorization: Bearer <token>
Content-Type: application/json

{
  "document_id": "doc-123",
  "content": "Content to analyze..."
}
```

#### Record Consent (GDPR)
```http
POST /api/v1/ethics/privacy/consent
Authorization: Bearer <token>
Content-Type: application/json

{
  "consent_type": "analytics",
  "granted": true
}
```

Consent types: `essential`, `analytics`, `personalization`, `marketing`, `research`

#### Export User Data
```http
GET /api/v1/ethics/privacy/export
Authorization: Bearer <token>
```

#### Get Ethics Dashboard
```http
GET /api/v1/ethics/analytics/dashboard?period_days=30
Authorization: Bearer <token>
```

### Health Check

```http
GET /api/v1/health
```

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
│   │   └── ethics/      # Ethics dashboard components
│   ├── hooks/           # Custom React hooks
│   ├── lib/             # Utilities and helpers
│   └── stores/          # Zustand state management
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

```bash
cd backend

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

# Install Playwright browsers
npx playwright install

# Run all E2E tests
npm run test:e2e

# Run specific test file
npx playwright test e2e/auth.spec.ts

# Run in headed mode for debugging
npm run test:e2e:headed

# Open Playwright UI
npm run test:e2e:ui
```

### Test Coverage

The test suite includes:

**Backend Integration Tests:**
- Document upload and processing flow
- Knowledge graph operations
- Flashcard generation and SM2 scheduling
- Scholar mode citation trees
- Ethics provenance and bias auditing
- Privacy compliance (GDPR)

**Frontend E2E Tests:**
- Navigation and routing
- Authentication flow
- Document management
- Accessibility checks
- Responsive design
- Performance benchmarks

## Deployment

### Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_HOST=db
      - NEO4J_URI=bolt://neo4j:7687
    depends_on:
      - db
      - neo4j

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000

  db:
    image: postgres:14
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=unfold
      - POSTGRES_USER=unfold
      - POSTGRES_PASSWORD=changeme

  neo4j:
    image: neo4j:5
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/changeme

volumes:
  postgres_data:
  neo4j_data:
```

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

## License

This project is licensed under the AGPL v3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Frontend powered by [Next.js](https://nextjs.org/)
- Knowledge graphs with [Neo4j](https://neo4j.com/)
- Vector search with [FAISS](https://github.com/facebookresearch/faiss)
- Spaced repetition based on [SM2 algorithm](https://www.supermemo.com/en/archives1990-2015/english/ol/sm2)
