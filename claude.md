# Unfold - Claude Code Guidelines

## Project Overview

Unfold is an LLM-powered AI reading assistant and semantic comprehension platform designed to help students and researchers engage with complex academic and technical texts. Key features include document management, knowledge graph construction, adaptive learning with spaced repetition, and multi-LLM support.

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI (async)
- **AI/ML**: LangChain, spaCy, multiple LLM providers (Ollama, OpenAI, Anthropic)
- **Databases**: PostgreSQL (user data), Neo4j 5+ (knowledge graphs), FAISS/Pinecone (vectors), Redis (cache)
- **Testing**: pytest, pytest-asyncio, pytest-cov

### Frontend
- **Framework**: Next.js 14 with React 18+
- **Language**: TypeScript (strict mode)
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Visualization**: D3.js
- **Testing**: Playwright (E2E)

## Project Structure

```
Unfold/
├── backend/                    # FastAPI Python application
│   ├── app/
│   │   ├── main.py            # FastAPI entry point
│   │   ├── config.py          # Settings management
│   │   ├── api/v1/routes/     # API endpoints
│   │   ├── db/                # Database connections and ORM models
│   │   ├── services/          # Business logic layer
│   │   ├── models/            # Pydantic schemas
│   │   ├── middleware/        # Rate limiting, CORS
│   │   └── repositories/      # Data access layer
│   ├── tests/                 # Unit and integration tests
│   ├── alembic/               # Database migrations
│   └── requirements.txt
├── frontend/                   # Next.js React application
│   ├── src/
│   │   ├── app/               # App Router pages
│   │   ├── components/        # React components
│   │   ├── store/             # Zustand state management
│   │   ├── services/          # API client services
│   │   └── types/             # TypeScript interfaces
│   ├── e2e/                   # Playwright E2E tests
│   └── package.json
├── docker-compose.yml         # Development environment
├── docker-compose.prod.yml    # Production environment
└── Makefile                   # Development shortcuts
```

## Development Commands

```bash
# Start development environment
make dev

# Run all tests
make test

# Backend tests with coverage
make test-backend-cov

# Frontend E2E tests
make test-e2e

# Database migrations
make db-migrate

# Clean Docker resources
make clean
```

## Key Entry Points

- **Backend API**: `backend/app/main.py` - FastAPI initialization
- **Frontend**: `frontend/src/app/` - Next.js App Router pages
- **API Docs**: http://localhost:8000/docs (Swagger UI)

## Architecture

The project follows a 5-layer modular architecture:
1. **Frontend** (Next.js + React)
2. **API Layer** (FastAPI routes)
3. **Service Layer** (Business logic in `backend/app/services/`)
4. **Repository Layer** (Data access abstraction)
5. **Storage Layer** (PostgreSQL, Neo4j, FAISS, Redis)

## Coding Conventions

### Backend (Python)
- Use `async/await` for all I/O operations
- Format with `black`, lint with `ruff`, type check with `mypy`
- Follow repository pattern for data access
- Place business logic in services, not routes
- Use Pydantic models for request/response validation

### Frontend (TypeScript)
- Use TypeScript strict mode
- Prefer functional components with hooks
- Use Zustand for global state
- Follow Next.js App Router conventions
- Lint with ESLint

## Key Services

### Knowledge Graph (`backend/app/services/graph/`)
- `integrated_pipeline.py` - Main orchestrator for graph construction
- `extractor.py` - Entity extraction
- `builder.py` - Graph construction
- `llm_relations.py` - LLM-based relation extraction

### Learning System (`backend/app/services/learning/`)
- `sm2.py` - SM2 spaced repetition algorithm
- `flashcards.py` - Flashcard generation
- `export.py` - Anki/Obsidian export

### Authentication (`backend/app/services/auth/`)
- JWT-based authentication with httpOnly cookies
- bcrypt password hashing (12 rounds)

## Database Notes

- **PostgreSQL**: User data, sessions, documents
- **Neo4j**: Knowledge graphs with semantic relationships
- **FAISS**: Vector embeddings for semantic search
- **Redis**: Caching layer

Run migrations with: `make db-migrate`

## Testing

### Backend
```bash
cd backend && pytest                    # All tests
cd backend && pytest tests/unit/        # Unit tests only
cd backend && pytest --cov=app          # With coverage
```

### Frontend
```bash
cd frontend && npm run test:e2e         # Headless E2E
cd frontend && npm run test:e2e:ui      # Interactive UI
```

## Environment Variables

Copy `.env.example` files in backend and frontend directories. Key variables:
- `JWT_SECRET` - Required in production (32+ chars)
- `DATABASE_URL` - PostgreSQL connection
- `NEO4J_URI`, `NEO4J_PASSWORD` - Neo4j connection
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` - LLM providers (optional with Ollama)

## Security Considerations

- Never commit secrets or `.env` files
- Use parameterized queries for Neo4j (allowlist validation in place)
- Rate limiting: 60 req/min general, 10 req/min for auth endpoints
- Production config enforces explicit secrets (no weak defaults)

## Multi-LLM Support

The platform supports multiple LLM providers with automatic fallback:
1. **Ollama** - Local/offline inference
2. **OpenAI** - GPT models
3. **Anthropic** - Claude models

Configuration in `backend/app/services/llm/`. Falls back to pattern-based extraction if no LLM available.
