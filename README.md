🧬 Unfold — Technical Specification (v1.0, 2026)

Mission:
Build a modular AI-assisted reading and comprehension platform that bridges the gap between dense academic/technical texts and genuine understanding — emphasizing ethics, explainability, and educational collaboration.

🧱 1. System Overview

Unfold is composed of five primary layers, each designed to interact modularly:

Layer	Purpose
1️⃣ Document Ingestion & Validation	Securely parse, verify, and index academic/technical documents.
2️⃣ Semantic Parsing & Knowledge Graph	Convert text into structured, queryable nodes and relationships.
3️⃣ Reading Interface	Interactive dual-view (technical ↔ conceptual) document explorer.
4️⃣ Adaptive Learning & Focus Engine	Personalized summarization, focus, and spaced-repetition systems.
5️⃣ Ethical & Academic Framework	Provenance, compliance, and bias audit systems for transparency.
⚙️ 2. System Architecture
Core Components:
Frontend (Next.js + React)
│
├── Reading Interface (dual-view, semantic overlays)
│
Backend (FastAPI + LangGraph + Neo4j/Weaviate)
│
├── Ingestion Service
├── Validation Service (CrossRef, ORCID, DOI)
├── Knowledge Graph Engine
├── Paraphrasing/LLM Engine
├── Adaptive Learning Service
├── Provenance/Bias Audit Module
│
Storage Layer
├── PostgreSQL (user data)
├── Neo4j or Weaviate (semantic graph)
├── Pinecone/FAISS (vector embeddings)
└── C2PA Ledger (content fingerprints)

🧩 3. Functional Modules
A. Document Ingestion & Validation

Goal: Verify authenticity and legality of documents before parsing.

Key Features:

PDF/EPUB ingestion via Apache Tika or PyPDF2

OCR fallback for scanned docs via Tesseract

DOI/metadata validation via CrossRef, Unpaywall, CORE

Copyright compliance with Creative Commons API

Provenance hashing with W3C C2PA and SHA-256

Author validation via ORCID or ROR APIs

Data Flow:

Upload → Validate hash & license

Extract metadata (title, authors, DOI, abstract)

Store record → forward to Semantic Parser

B. Semantic Parsing & Knowledge Graph

Goal: Transform documents into structured knowledge entities.

Implementation:

Use LangGraph / LangChain for orchestration.

Entity extraction via spaCy + GPT-4o for relations.

Graph database: Neo4j or Weaviate (vector-native preferred).

Multimodal support:

Text embeddings: OpenAI text-embedding-3-large

Image embeddings: CLIP or OpenCLIP

Diagram parsing: Pix2Struct or BLIP-2

Graph schema:

Node types: Concept, Author, Paper, Method, Dataset
Relationships: EXPLAINS, CITES, CONTRASTS_WITH, DERIVES_FROM


External linkers:

Wikipedia, Semantic Scholar, arXiv API

DOI resolver for persistent references

Storage:

Semantic relationships in graph DB

Text vectors in Pinecone/FAISS

Cached summaries in PostgreSQL

C. Reading Interface (Frontend)

Goal: Human-centered reading experience with dual complexity modes.

Stack:

Frontend: Next.js + React + Tailwind + D3.js

API Layer: Axios → FastAPI

State Management: Zustand or Redux Toolkit

Views:

Technical View: Raw formatted text + inline reference popovers

Conceptual View: LLM-generated paraphrases & summaries

Hybrid Slider: Dynamic complexity gradient toggle

Features:

Term hover tooltips → concept graph preview

“Explain Like I’m 5” / “Expert Mode” toggle

Semantic scroll-sync (text ↔ node map)

Highlight-based Q&A popup (ChatGPT-style chat on selected content)

D. Intelligent Focus Mode

Goal: Prioritize sections relevant to user goals.

Techniques:

TF-IDF + BERT embeddings → initial relevance scoring

Graph traversal (shortest path from target node)

Attention heatmaps via Captum/BertViz (explainability)

Relevance feedback loop:

User feedback → stored as reinforcement signals (RLHF-lite)

Progressive reveal UI (accordion expansion for deeper dives)

E. Adaptive Learning Layer

Goal: Track comprehension and enable active recall.

Subsystems:

Engagement tracking: dwell time, scroll velocity, annotation density

Flashcard generation via T5 or FLAN-UL2 question synthesis

Spaced repetition scheduling (SM2 algorithm)

Export to Anki / Obsidian via Markdown

Gamification (optional):

Progress streaks, mastery badges, reflection streaks

F. Scholar Mode

Goal: Recursive academic exploration and citation hygiene.

Capabilities:

Depth-controlled exploration (cap recursion at 3 hops)

Tree view of references with citation chains

Citation generation via Zotero API

Source credibility scoring:

CrossRef impact factor

Altmetrics data integration

Peer-review flags (Scopus/OpenAlex APIs)

G. Bookmarking & Reflection Engine

Goal: Capture metacognitive evolution.

Implementation:

Time-based snapshot diffs (graph-diff algorithm)

User reflection storage → graph node annotation

Visual timelines / mind maps (D3.js)

Collaborative reflection clusters (CRDTs)

H. Ethical & Academic Framework

Goal: Ensure academic integrity, privacy, and inclusivity.

Components:

Provenance via C2PA manifest + DOI revalidation

Anonymized analytics (Mixpanel or PostHog)

Bias audit module (text sentiment & fairness analysis)

Partner integrations:

Creative Commons

OpenAI/Anthropic safety endpoints

UNESCO / FAIR AI compliance tracking

🔐 4. Data Schemas
Document Record
{
  "doc_id": "sha256:xxxx",
  "title": "string",
  "authors": ["John Doe", "Jane Smith"],
  "doi": "10.xxxx/abc123",
  "license": "CC-BY-4.0",
  "source": "arXiv",
  "vector_id": "uuid",
  "graph_nodes": ["concept_001", "author_002"]
}

Knowledge Graph Node
{
  "node_id": "concept_001",
  "type": "Concept",
  "label": "Quantum Entanglement",
  "embedding": [0.34, 0.56, ...],
  "relations": [{"type": "CITES", "target": "concept_004"}]
}

Reflection Snapshot
{
  "user_id": "uuid",
  "timestamp": "2026-01-05T13:00:00Z",
  "nodes_added": 12,
  "nodes_deleted": 1,
  "notes": "Now understand Bell's theorem implications"
}

🧠 5. AI Models
Purpose	Suggested Model	Notes
Entity/Relation Extraction	GPT-4o / Claude 3.5	Structured JSON output
Summarization/Paraphrasing	GPT-4o-mini / Mistral 8x7B	Multi-level simplification
Question Generation	T5 / FLAN-UL2	SRS integration
Image Captioning	BLIP-2 / Pix2Struct	Diagram understanding
Bias Audit	RoBERTa Sentiment / Perspective API	Language inclusivity checks
🧭 6. APIs and Integrations
Function	API
Document Validation	CrossRef, Unpaywall, CORE
Metadata & Author ID	ORCID, ROR
Citation Management	Zotero
Provenance	C2PA
Knowledge Links	Wikipedia, arXiv, Semantic Scholar
LMS Integration	LTI 1.3 (Canvas, Moodle)
Analytics	Mixpanel / PostHog
Storage	AWS S3 / GCS / IPFS (optional decentralized mode)
💡 7. Security & Privacy

All data encrypted (AES-256 at rest, TLS 1.3 in transit)

Differential privacy for analytics

Strict user consent for data collection

OpenAI usage governed under academic license agreements

🧰 8. Development Environment
Stack	Tool
Backend	Python 3.11+, FastAPI, LangChain/LangGraph
Frontend	Next.js (React 18+), Tailwind, D3.js
Database	PostgreSQL + Neo4j (or Weaviate)
Embeddings	Pinecone or FAISS
Auth	OAuth2 + JWT (optional ORCID login)
Testing	PyTest + Cypress
DevOps	Docker Compose + GitHub Actions CI/CD
🚀 9. Roadmap Summary
Milestone	Deliverables
v0.1	Document ingestion + validation
v0.2	Semantic graph + embeddings
v0.3	Reading interface MVP
v0.4	Adaptive focus mode
v0.5	Scholar Mode + reflection engine
v1.0	Ethics + provenance + public beta
🧩 10. Licensing & Open Science

License: AGPL v3 (to ensure community benefit)

Open access to non-proprietary models and datasets

Opt-in Transparency Portal:

Model prompts

Bias metrics

Audit logs

Citation sources
