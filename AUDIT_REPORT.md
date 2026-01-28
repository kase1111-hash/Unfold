# Software Audit Report: Unfold

**Audit Date:** 2026-01-28
**Auditor:** Automated Code Review
**Repository:** Unfold - LLM-Powered AI Reading Assistant
**Version:** 0.1.0

---

## Executive Summary

Unfold is a well-architected LLM-powered reading assistant designed to help users comprehend complex academic and technical texts. The codebase demonstrates solid software engineering practices with a modular 5-layer architecture. However, several areas require attention before production deployment.

### Overall Assessment: **GOOD with Reservations**

| Category | Score | Status |
|----------|-------|--------|
| Architecture & Design | 8/10 | Good |
| Security | 7/10 | Needs Improvement |
| Code Quality | 8/10 | Good |
| Test Coverage | 6/10 | Needs Improvement |
| Documentation | 8/10 | Good |
| Production Readiness | 6/10 | Needs Improvement |

---

## 1. Architecture & Design Analysis

### Strengths

1. **Clean Layered Architecture**: The 5-layer design (Frontend, API, Service, Data Access, Storage) provides excellent separation of concerns.

2. **Technology Stack Choices**:
   - FastAPI for async Python backend - modern and performant
   - Next.js 14 with App Router - current best practices
   - Neo4j for knowledge graphs - appropriate for the domain
   - PostgreSQL for relational data - reliable choice

3. **Modular Services**: Each service (auth, graph, learning, ethics, scholar) is independently organized with clear responsibilities.

4. **Multi-LLM Provider Support**: The system supports Ollama (local), OpenAI, and Anthropic with graceful fallbacks - excellent for privacy-first deployment.

5. **Ethics-First Design**: Built-in GDPR compliance, differential privacy, bias auditing, and provenance tracking.

### Areas for Improvement

1. **Singleton Pattern Overuse**: Multiple services use global singleton instances (`_builder`, `_sm2_scheduler`, `_privacy_compliance`). This can cause issues with:
   - Testing (mocking becomes difficult)
   - Concurrent request isolation
   - Memory leaks if not properly managed

   **Recommendation**: Consider dependency injection containers.

2. **Missing Service Layer Transactions**: Services don't implement distributed transaction patterns for operations spanning PostgreSQL and Neo4j.

---

## 2. Security Analysis

### Strengths

1. **Proper Password Hashing**: Uses bcrypt with 12 rounds via passlib - industry standard.
   ```python
   # backend/app/utils/security.py:6-10
   pwd_context = CryptContext(
       schemes=["bcrypt"],
       deprecated="auto",
       bcrypt__rounds=12,
   )
   ```

2. **JWT Implementation**: Proper token separation (access/refresh), type checking, and expiration validation.

3. **CORS Configuration**: Configurable origins via settings.

4. **Database Statement Timeouts**: 30-second statement timeout prevents runaway queries.

### Critical Issues

1. **CRITICAL: Insecure Default JWT Secret**
   ```python
   # backend/app/config.py:60
   jwt_secret: str = "change-me-in-production"
   ```
   **Impact**: If deployed with default, all tokens can be forged.
   **Recommendation**: Remove default, require explicit configuration, fail startup if not set in production.

2. **CRITICAL: Default Database Credentials**
   ```python
   # backend/app/config.py:32-38
   database_url: PostgresDsn = "postgresql://postgres:postgres@localhost:5432/unfold"
   neo4j_password: str = "password"
   ```
   **Recommendation**: No defaults for sensitive credentials in production mode.

3. **HIGH: Potential Cypher Injection in Neo4j**
   ```python
   # backend/app/db/neo4j.py:181-183
   query = f"""
   CREATE (n:{node_type} $props)
   ```
   The `node_type` is interpolated directly into the query. While properties are parameterized, node labels could be vulnerable if user-controlled.
   **Recommendation**: Validate node_type against an allowlist.

4. **MEDIUM: Token Storage in localStorage**
   ```typescript
   // frontend/src/services/api.ts:92-94
   localStorage.setItem("access_token", tokens.access_token);
   localStorage.setItem("refresh_token", tokens.refresh_token);
   ```
   **Impact**: Vulnerable to XSS attacks.
   **Recommendation**: Use httpOnly cookies for refresh tokens.

5. **MEDIUM: Missing Rate Limiting**
   No rate limiting on authentication endpoints.
   **Recommendation**: Implement rate limiting on `/auth/login`, `/auth/register`, `/auth/refresh`.

6. **LOW: API Key Exposure Risk**
   ```python
   # backend/app/config.py:46-52
   openai_api_key: str | None = None
   anthropic_api_key: str | None = None
   ```
   Keys in environment variables is standard, but ensure they're never logged.

---

## 3. Code Quality Analysis

### Strengths

1. **Type Hints**: Extensive use of Python type hints and TypeScript types.

2. **Pydantic Validation**: All API inputs validated with Pydantic models.

3. **Error Handling**: Custom exception classes with error codes.

4. **Async/Await**: Proper async patterns throughout backend.

5. **Code Organization**: Clear file naming and directory structure.

### Issues

1. **Incomplete Document Routes**
   ```python
   # backend/app/api/v1/routes/documents.py:74-80
   # TODO: Implement actual document processing
   ```
   Multiple endpoints return placeholder responses or always 404.

2. **Exception Swallowing**
   ```python
   # backend/app/db/neo4j.py:419-421
   except Exception:
       # Index might already exist
       pass
   ```
   Silent exception catching can hide real problems.

3. **Print Statements in Production Code**
   ```python
   # backend/app/main.py:26-27
   print(f"Starting {settings.app_name} v{settings.app_version}")
   ```
   **Recommendation**: Use structured logging instead.

4. **HTTP Client Not Properly Closed**
   ```python
   # backend/app/services/learning/flashcards.py:344-348
   async def close(self):
       """Close the HTTP client."""
       if self._client:
           await self._client.aclose()
   ```
   The `close()` method exists but is never called. Use context managers or ensure cleanup.

---

## 4. Database Operations

### Strengths

1. **Connection Pooling**: Proper pool configuration with size limits and recycling.
   ```python
   # backend/app/db/postgres.py:56-64
   pool_size=settings.db_pool_size,
   max_overflow=settings.db_max_overflow,
   pool_recycle=settings.db_pool_recycle,
   ```

2. **Health Checks**: Both databases have connectivity checks for monitoring.

3. **Proper Session Management**: Context managers ensure cleanup.

4. **Indexes**: Neo4j indexes created for common query patterns.

### Issues

1. **No Database Migrations in Production Path**
   ```python
   # backend/app/main.py:32-33
   if settings.environment == "development":
       await create_tables()
   ```
   Production relies on Alembic but no migration files found.

2. **Neo4j Optional Handling**: When Neo4j is unavailable, graph operations silently return empty results rather than failing. This could mask configuration problems.

---

## 5. Frontend Analysis

### Strengths

1. **Modern React Patterns**: Uses hooks, Zustand for state, proper component organization.

2. **Protected Routes**: Dashboard layout checks authentication.

3. **Token Refresh**: Automatic token refresh on 401 responses.

4. **Type Safety**: Full TypeScript with proper interfaces.

### Issues

1. **No Client-Side Input Sanitization**: While server validates, client should also sanitize for XSS.

2. **Auth State Persistence Issue**
   ```typescript
   // frontend/src/store/auth.ts:103-108
   partialize: (state) => ({
       user: state.user,
       isAuthenticated: state.isAuthenticated,
   }),
   ```
   Persisting `isAuthenticated` can cause stale auth state on page reload.

3. **Missing Error Boundaries**: No React error boundaries for graceful failure handling.

---

## 6. Test Coverage Analysis

### Current State

- **Unit Tests**: Present for auth, documents, graph, health
- **Integration Tests**: Present for document flow, knowledge graph, ethics, learning, scholar mode
- **E2E Tests**: Playwright tests present in frontend

### Issues

1. **Test Coverage Unknown**: No coverage reports configured.

2. **Tests May Rely on External Services**: Some tests need database connections to fully run.

3. **Missing Tests**:
   - Security/penetration tests
   - Performance/load tests
   - Negative path tests for edge cases

4. **Test Fixture Issue**
   ```python
   # backend/tests/unit/test_auth.py:184
   assert response.status_code in [201, 500]  # May fail without DB
   ```
   Tests accept 500 as valid - masks real failures.

---

## 7. Fitness for Purpose Assessment

### Intended Purpose
"LLM-powered AI reading assistant for comprehending complex academic/technical texts"

### Assessment

| Feature | Implementation Status | Production Ready |
|---------|----------------------|------------------|
| Document Upload | Placeholder | No |
| Knowledge Graph | Implemented | Yes (with caveats) |
| Flashcard Generation | Implemented | Yes |
| Spaced Repetition (SM2) | Fully Implemented | Yes |
| Citation Analysis | Implemented | Yes |
| Privacy/GDPR | Implemented | Yes |
| Bias Detection | Not Found | No |
| User Authentication | Implemented | Yes (with security fixes) |
| Multi-LLM Support | Implemented | Yes |

### Verdict

The software is **fit for purpose as a beta/development platform** but requires the following before production:

1. Complete document processing pipeline
2. Address all critical security issues
3. Implement rate limiting
4. Add production monitoring/logging
5. Create database migration strategy
6. Improve test coverage to >80%

---

## 8. Recommendations

### Immediate (Before Any Deployment)

1. **Remove all default secrets/passwords**
2. **Implement rate limiting on auth endpoints**
3. **Add input validation for Neo4j node types**
4. **Move refresh tokens to httpOnly cookies**

### Short-Term (Before Beta)

1. **Complete document upload/processing pipeline**
2. **Add structured logging (replace print statements)**
3. **Implement proper HTTP client lifecycle management**
4. **Add React error boundaries**
5. **Configure test coverage reporting**

### Medium-Term (Before Production)

1. **Add database migrations with Alembic**
2. **Implement distributed tracing**
3. **Add performance monitoring**
4. **Security penetration testing**
5. **Load testing for concurrent users**
6. **Implement proper dependency injection**

---

## 9. Positive Highlights

Despite the issues identified, this codebase demonstrates several excellent practices:

1. **Privacy-First Architecture**: The ethics module with GDPR compliance, differential privacy, and consent management is exemplary.

2. **SM2 Algorithm Implementation**: The spaced repetition system is well-implemented and mathematically correct.

3. **Multi-Provider LLM Design**: Supporting local (Ollama) and cloud providers with fallback shows forward-thinking architecture.

4. **Comprehensive API Design**: OpenAPI documentation, proper HTTP status codes, structured error responses.

5. **Type Safety**: Consistent use of type hints in Python and TypeScript throughout.

---

## Appendix: Files Reviewed

### Backend
- `backend/app/main.py`
- `backend/app/config.py`
- `backend/app/services/auth/service.py`
- `backend/app/services/auth/jwt.py`
- `backend/app/services/graph/builder.py`
- `backend/app/services/learning/sm2.py`
- `backend/app/services/learning/flashcards.py`
- `backend/app/services/ethics/privacy.py`
- `backend/app/api/v1/routes/*.py`
- `backend/app/db/postgres.py`
- `backend/app/db/neo4j.py`
- `backend/app/utils/security.py`
- `backend/requirements.txt`
- `backend/tests/unit/test_auth.py`

### Frontend
- `frontend/src/app/(dashboard)/layout.tsx`
- `frontend/src/services/api.ts`
- `frontend/src/store/auth.ts`

---

**End of Audit Report**
