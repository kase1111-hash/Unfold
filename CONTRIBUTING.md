# Contributing to Unfold

Thank you for your interest in contributing to Unfold! This document provides guidelines and instructions for contributing to this AI-powered reading comprehension platform.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.11 or higher
- Node.js 18 or higher
- Docker and Docker Compose
- Git
- A GitHub account

### Understanding the Project

Unfold consists of two main components:

1. **Backend** (`/backend`) - Python/FastAPI REST API with PostgreSQL, Neo4j, and vector stores
2. **Frontend** (`/frontend`) - Next.js 14 React application with TypeScript

We recommend reading the [README.md](README.md) and [AI-instructions.md](AI-instructions.md) to understand the architecture and implementation details.

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Unfold.git
cd Unfold

# Add upstream remote
git remote add upstream https://github.com/kase1111-hash/Unfold.git
```

### 2. Start Development Environment

```bash
# Start all services with Docker Compose
make dev

# Or manually:
docker-compose up -d
```

### 3. Backend Setup (for local development)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Frontend Setup (for local development)

```bash
cd frontend

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Run development server
npm run dev
```

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/kase1111-hash/Unfold/issues) to avoid duplicates
2. Use the bug report template when creating a new issue
3. Include:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, browser, versions)
   - Screenshots or logs if applicable

### Suggesting Features

1. Check existing issues and discussions for similar ideas
2. Open a new issue with the feature request template
3. Describe:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches considered
   - How it aligns with Unfold's mission of ethical AI-assisted learning

### Submitting Code

#### Types of Contributions Welcome

- Bug fixes
- New features (aligned with roadmap)
- Performance improvements
- Documentation improvements
- Test coverage improvements
- Accessibility enhancements
- Internationalization support

#### Areas of Focus

- **Document Processing** - Improving PDF/EPUB parsing
- **Knowledge Graph** - Entity extraction and relation mapping
- **Learning System** - Flashcard generation and spaced repetition
- **Ethics & Privacy** - Bias detection and GDPR compliance
- **UI/UX** - Accessibility and responsive design

## Pull Request Process

### 1. Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the [coding standards](#coding-standards)
- Write tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Commit Your Changes

We follow conventional commit messages:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(graph): add temporal edge support for concept evolution"
git commit -m "fix(auth): resolve token refresh race condition"
git commit -m "docs(api): add examples for flashcard export endpoints"
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- Clear title describing the change
- Description of what was changed and why
- Link to related issue(s)
- Screenshots for UI changes
- Checklist confirmation

### 5. PR Review Process

- At least one maintainer review is required
- CI checks must pass (tests, linting, type checking)
- Address review feedback promptly
- Squash commits if requested before merge

## Coding Standards

### Python (Backend)

- **Style**: Follow PEP 8, enforced by `black` and `ruff`
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google style for modules, classes, and functions
- **Async**: Use async/await for I/O operations

```bash
# Format code
black backend/
ruff check backend/ --fix

# Type checking
mypy backend/
```

### TypeScript (Frontend)

- **Style**: ESLint with Next.js configuration
- **Types**: Strict TypeScript, no `any` without justification
- **Components**: Functional components with hooks
- **State**: Zustand for global state

```bash
# Lint code
npm run lint

# Type check
npm run type-check
```

### General Guidelines

- Keep functions small and focused
- Prefer composition over inheritance
- Write self-documenting code with clear naming
- Comment complex logic, not obvious code
- Handle errors gracefully with informative messages

## Testing Guidelines

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/integration/test_document_flow.py
```

**Test Requirements:**
- Unit tests for utility functions and services
- Integration tests for API endpoints
- Mock external services (OpenAI, CrossRef, etc.)
- Aim for 80%+ coverage on new code

### Frontend Tests

```bash
cd frontend

# Install Playwright browsers (first time)
npx playwright install

# Run E2E tests
npm run test:e2e

# Run with UI
npm run test:e2e:ui
```

**Test Requirements:**
- E2E tests for critical user flows
- Accessibility tests for UI components
- Test responsive behavior

## Documentation

### When to Update Docs

- Adding new API endpoints
- Changing configuration options
- Adding new features
- Modifying installation steps
- Deprecating functionality

### Documentation Locations

- `README.md` - Main project documentation
- `AI-instructions.md` - Implementation guidance
- Inline code comments - Complex logic
- API docstrings - Endpoint documentation (auto-generated via FastAPI)

## Community Guidelines

### Communication

- Be respectful and constructive
- Assume good intentions
- Welcome newcomers
- Help others learn

### Decision Making

- Major changes require discussion in issues first
- Maintainers have final say on architectural decisions
- Community feedback is valued and considered

### Recognition

Contributors are recognized in:
- GitHub contributors list
- Release notes for significant contributions
- README acknowledgments for major features

## Questions?

- Open a [Discussion](https://github.com/kase1111-hash/Unfold/discussions) for general questions
- Check existing issues and documentation first
- Be specific and provide context in questions

Thank you for contributing to Unfold!
