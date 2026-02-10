# UNFOLD REFOCUS PLAN

This plan strips Unfold to its core value proposition — **PDF upload → Knowledge Graph → Flashcards → Spaced Repetition** — and fixes the broken wiring between backend and frontend before adding anything new.

---

## PHASE 0: CUT DEAD WEIGHT (Day 1)

**Goal:** Remove unused code and dependencies to reduce maintenance surface.

### 0.1 Remove Unused Dependencies

**File:** `backend/requirements.txt`

Remove these 6 packages (zero imports across `backend/app/`):

| Package | Why it's dead |
|---------|---------------|
| `langchain==0.1.7` | Never imported anywhere in codebase |
| `langchain-openai==0.0.6` | Never imported |
| `langchain-community==0.0.20` | Never imported |
| `ebooklib==0.4.1` | Imported in `document_service.py` but handler returns empty string |
| `pytesseract==0.3.10` | Never imported |
| `pinecone-client==3.0.2` | 140 lines of wrapper code in `db/vector.py`, zero callers |

Keep `aiohttp` (used transitively by LLM providers) and `python-docx` (low cost, future use plausible).

### 0.2 Delete Dead Code

| File | Lines | Why |
|------|-------|-----|
| `backend/app/db/vector.py` lines 240-397 | 157 | Pinecone wrapper functions — zero callers |
| `backend/app/services/cache/cache.py` | 80+ | Abstract cache class, never imported |
| `backend/app/services/scholar/collaboration.py` | ~50 | Empty stub, no route calls it |
| `Keywords.md` | 350+ | SEO strategy doc, not application code |

### 0.3 Remove Ethics Module

**Delete entirely:**
- `backend/app/services/ethics/` (4 files, ~1,100 lines total)
  - `provenance.py` — C2PA tracking, in-memory only, never persisted
  - `audit.py` — bias auditing, no real data source
  - `privacy.py` — GDPR consent, 629 lines, in-memory only
  - `analytics.py` — differential privacy noise on empty data
- `backend/app/api/v1/routes/ethics.py` (~650 lines)
- `frontend/src/components/ethics/` (5 components, all mock data)
  - `BiasAuditPanel.tsx` — uses `mockReport`
  - `EthicsDashboard.tsx` — uses `mockDashboard`
  - `ProvenanceBadge.tsx` — uses `mockManifest`
  - `PrivacyCenter.tsx` — uses `mockConsents`, `mockProfile`
  - `TransparencyDashboard.tsx`

**Update:**
- Remove ethics router registration in `backend/app/api/v1/router.py`
- Remove ethics imports from `backend/app/main.py`
- Remove ethics-related pages/links from frontend navigation

**Rationale:** The entire ethics module is in-memory-only on the backend and mock-data-only on the frontend. It serves a different user need (enterprise compliance) than the core product (academic reading). It can be extracted into a standalone package later if demand materializes.

---

## PHASE 1: WIRE THE CORE PIPELINE (Days 2-5)

**Goal:** Connect the working backend to the disconnected frontend. The backend is ~90% complete; the frontend consumes ~40% of the available APIs.

### 1.1 Fix Graph Visualization Relations (Critical)

**Problem:** `frontend/src/store/graph.ts` line 59-61 fetches nodes but creates an **empty links array**. The graph renders isolated nodes without edges.

**Fix:**
- Add a backend endpoint or use existing search to return relations for a document
- In `graph.ts:loadGraphForDocument()`, after fetching nodes, fetch relations from the backend
- Map Neo4j relationships to D3 link objects (`{ source, target, type, confidence }`)
- The D3 force simulation in `KnowledgeGraph.tsx` already handles link rendering — it just needs data

**Files to modify:**
- `frontend/src/store/graph.ts` — add relation fetching
- `frontend/src/services/api.ts` — add `getRelations(docId)` method if not present
- `backend/app/api/v1/routes/graph.py` — verify relation query endpoint exists and returns source/target pairs

### 1.2 Wire Flashcards Page to Backend (Critical)

**Problem:** `frontend/src/app/(dashboard)/flashcards/page.tsx` line 8-57 uses hardcoded `DEMO_FLASHCARDS` array.

**Fix:**
- Replace `useState(DEMO_FLASHCARDS)` with a `useEffect` that calls `api.getFlashcards()` or `api.getDueCards()`
- Add loading/empty states
- Backend endpoint `/learning/flashcards/due` already exists and returns real data

**Files to modify:**
- `frontend/src/app/(dashboard)/flashcards/page.tsx` — replace mock data with API call
- `frontend/src/services/api.ts` — add flashcard API methods

### 1.3 Wire SM2 Review to Backend (Critical)

**Problem:** `FlashcardReview.tsx` captures user quality ratings (0-5) but the `onReview` callback never POSTs to the backend. Review data is lost on page refresh.

**Fix:**
- In `FlashcardReview.tsx`, call `api.reviewFlashcard(cardId, quality)` inside the `handleRate` function
- Backend endpoint `POST /learning/flashcards/review` exists and returns next review date, interval, easiness factor
- Update the card's local state with the response

**Files to modify:**
- `frontend/src/components/learning/FlashcardReview.tsx` — add API call on review
- `frontend/src/services/api.ts` — add `reviewFlashcard()` method
- `frontend/src/app/(dashboard)/flashcards/page.tsx` — pass real `onReview` handler

### 1.4 Wire StudyStats to Backend

**Problem:** `StudyStats.tsx` line 50 says "For now, use mock data." Backend has `get_study_stats()` in sm2.py.

**Fix:**
- Replace mock stats with API call to learning stats endpoint
- Display: cards due, total reviewed, average easiness, retention rate

**Files to modify:**
- `frontend/src/components/learning/StudyStats.tsx` — replace mock with API call
- `frontend/src/services/api.ts` — add `getStudyStats()` method

### 1.5 Wire DocumentViewer to Real Content

**Problem:** `DocumentViewer.tsx` line 67 says "Mock content for demo." The document text is stored in the database after upload but the viewer doesn't fetch it.

**Fix:**
- Fetch document content from `GET /documents/{doc_id}`
- Render the actual extracted text in the viewer component
- The dual-view / complexity slider UI already exists — it just needs real text input

**Files to modify:**
- `frontend/src/components/reader/DocumentViewer.tsx` — fetch real content
- `frontend/src/services/api.ts` — verify `getDocument(docId)` returns content field

---

## PHASE 2: FIX THE TEST SUITE (Days 6-8)

**Goal:** Replace the 88 permissive assertions with deterministic tests that verify actual behavior.

### 2.1 Rewrite Integration Tests

**Problem:** Tests like `assert response.status_code in [200, 401, 404, 422, 500]` pass whether the feature works or crashes. They provide zero signal.

**Pattern to replace:**
```python
# BEFORE (passes if endpoint crashes)
assert response.status_code in [200, 401, 404]

# AFTER (verifies specific behavior)
# Test: authenticated user gets document
response = client.get("/documents/1", headers=auth_headers)
assert response.status_code == 200
data = response.json()
assert "title" in data

# Test: unauthenticated user gets 401
response = client.get("/documents/1")
assert response.status_code == 401
```

**Test files to rewrite (6 files, ~88 weak assertions):**

| File | Weak Assertions | Priority |
|------|----------------|----------|
| `tests/integration/test_knowledge_graph.py` | 15 | High — core feature |
| `tests/integration/test_document_flow.py` | 9 | High — core feature |
| `tests/integration/test_learning_system.py` | 17 | High — core feature |
| `tests/integration/test_scholar_mode.py` | 30 | Low — defer with module |
| `tests/integration/test_ethics_system.py` | 27 | Delete — ethics module removed |
| `tests/unit/test_graph.py` | 3 | High — core feature |

**Approach:**
- Set up proper test fixtures in `conftest.py`: create test user, get auth token, upload test document
- Each test should set up its preconditions, make ONE request, and assert ONE specific outcome
- Add response body assertions, not just status codes

### 2.2 Add Missing Unit Tests

**Priority unit tests to add:**

| Component | What to Test | Why Missing |
|-----------|-------------|-------------|
| Entity Extractor | Input text → entity list with types/confidence | Only integration-tested via graph route |
| Integrated Pipeline | Text → relations with source/target/type | Only tested via E2E scripts in root dir |
| Flashcard Generator | Document text → flashcard JSON structure | Only tested via route, not isolated |
| Document Service | PDF bytes → extracted text + metadata | Only tested via upload route |

### 2.3 Delete Obsolete Test Files

Remove after ethics module deletion:
- `tests/integration/test_ethics_system.py`
- `tests/integration/test_scholar_mode.py` (defer with scholar module)

---

## PHASE 3: HARDEN THE CORE (Days 9-14)

**Goal:** Make the core pipeline production-reliable.

### 3.1 Graph Builder Error Handling

**Problem:** `builder.py` lines 148-162 has an async/sync inconsistency — the same method name is called synchronously or asynchronously depending on a config flag.

**Fix:**
- Standardize on async for all relation extraction
- Add retry logic for LLM provider failures (Ollama timeout, OpenAI rate limits)
- Add fallback chain: Ollama → OpenAI → pattern-based extraction (already partially implemented in integrated pipeline, needs to be robust)

### 3.2 Document Processing Robustness

**Fix:**
- Handle corrupt/encrypted PDFs gracefully (currently may crash)
- Add file size limits enforcement at upload
- Add progress tracking for large documents (the status field exists in DB but isn't updated during processing)
- Remove EPUB extraction stub — return a clear "PDF only" error instead of silently returning empty text

### 3.3 Frontend Error States

**Fix:**
- Add error boundaries around D3.js graph visualization (Canvas crashes can freeze the page)
- Add retry buttons on failed API calls
- Add empty states for: no documents uploaded, no flashcards generated, no graph data
- Handle auth token expiration with redirect to login

### 3.4 Database Hardening

**Fix:**
- Replace hardcoded passwords in `docker-compose.yml` with environment variable references
- Add proper connection pool limits for Neo4j (currently using defaults)
- Add database health checks that actually verify query capability, not just TCP connection

---

## PHASE 4: DEFER TO BACKLOG (Post-v1.0)

These features are removed from active development but preserved for future consideration:

| Feature | Current State | Backlog Condition |
|---------|--------------|-------------------|
| **Scholar Mode** (citations, credibility, Zotero, reflection) | Backend services exist, frontend partially wired | Re-add when core pipeline has 1,000+ active users |
| **Ethics Suite** (provenance, bias, GDPR, analytics) | All mock/in-memory | Extract as standalone npm/pip package |
| **EPUB Support** | Stub returning empty | Add when PDF pipeline is battle-tested |
| **Pinecone Vector Store** | Dead code | Add when FAISS proves insufficient for scale |
| **Multi-Provider LLM** (Anthropic, llama.cpp) | Stubs/partial | Stabilize Ollama + OpenAI first |
| **Collaboration Features** | Empty stub | Wrong phase for a v0.x product |
| **Differential Privacy** | Implemented on empty data | Solve when there's data worth anonymizing |

---

## SUCCESS METRICS

After completing Phases 0-3, the product should:

1. **Core Loop Works End-to-End:** User uploads PDF → sees knowledge graph with nodes AND edges → generates flashcards → reviews with SM2 → comes back tomorrow and sees due cards
2. **Zero Mock Data in Core Path:** No hardcoded demo data in the upload → graph → flashcard → review flow
3. **Tests Verify Behavior:** All core pipeline tests assert specific outcomes, not "server didn't crash"
4. **Dependency Count:** Reduced from 32 to ~24 backend packages
5. **Lines of Code:** Reduced by ~3,000 lines (ethics module, dead code, Pinecone wrappers, stubs)
6. **Cold Start:** `docker-compose up` → working product in < 60 seconds

---

## FILE IMPACT SUMMARY

| Action | Files | Lines Removed | Lines Added |
|--------|-------|---------------|-------------|
| Phase 0: Cut dead weight | ~20 files deleted | ~3,200 | 0 |
| Phase 1: Wire frontend | ~8 files modified | ~200 (mock data) | ~400 (API calls) |
| Phase 2: Fix tests | ~6 files rewritten | ~500 (weak tests) | ~600 (real tests) |
| Phase 3: Harden core | ~10 files modified | ~50 | ~300 |
| **Total** | **~44 files** | **~3,950** | **~1,300** |

Net result: **~2,650 fewer lines of code** with more working features.
