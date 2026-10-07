"""Build or remove the knowledge graph for a stored document.

This is the single entry point the rest of the app uses to turn an
uploaded document's extracted text into graph nodes and relations: the
upload route schedules ``build_document_graph_task`` after a successful
upload, and ``POST /graph/documents/{doc_id}/build`` rebuilds on demand.

While a build runs the document's status is PROCESSING; it becomes INDEXED
when the build created nodes, and goes back to VALIDATED otherwise (no
nodes, a failure, or the graph database being unreachable). No database
session is held while the graph is built: builds can take minutes, and
PostgreSQL closes connections that sit idle in a transaction.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.db import GRAPH_UNAVAILABLE_ERRORS, get_session_context
from app.models.document import DocumentStatus
from app.repositories.document import DocumentRepository
from app.services.graph.builder import GraphBuildResult, get_graph_builder

logger = logging.getLogger(__name__)

# BuildGraphRequest caps text at 50k chars; stay well inside spaCy's limits too.
MAX_CHUNK_CHARS = 50_000

# At most one build per document at a time: two interleaved builds each
# delete the document's nodes once and then append their own, leaving
# duplicates. Callers never wait for a build to finish; they skip (or answer
# 409) instead. The app runs a single process, so in-memory locks suffice.
_build_locks: dict[str, asyncio.Lock] = {}

# Background builds are CPU-heavy (spaCy); bound how many run at once.
MAX_CONCURRENT_BACKGROUND_BUILDS = 2
_background_build_slots = asyncio.Semaphore(MAX_CONCURRENT_BACKGROUND_BUILDS)


class BuildInProgressError(Exception):
    """A graph build for this document is already running."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        super().__init__(f"A graph build for {doc_id} is already running")


@asynccontextmanager
async def exclusive_build(doc_id: str) -> AsyncIterator[None]:
    """Hold the document's build lock for the duration of the block.

    Raises:
        BuildInProgressError: If a build of ``doc_id`` is already running
    """
    lock = _build_locks.setdefault(doc_id, asyncio.Lock())
    if lock.locked():
        raise BuildInProgressError(doc_id)
    async with lock:
        try:
            yield
        finally:
            # Nobody waits on these locks, so the entry can go now
            if _build_locks.get(doc_id) is lock:
                del _build_locks[doc_id]


def normalize_text(text: str) -> str:
    """Undo PDF hard line wraps so entities aren't split across lines.

    Single newlines become spaces; blank lines (paragraph breaks) are kept.
    """
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks of at most ``max_chars``, on paragraph or
    sentence boundaries where possible."""
    chunks: list[str] = []
    while len(text) > max_chars:
        window = text[:max_chars]
        cut = window.rfind("\n\n")
        if cut < max_chars // 2:
            cut = window.rfind(". ")
            cut = cut + 1 if cut >= max_chars // 2 else max_chars
        chunks.append(text[:cut].strip())
        text = text[cut:]
    if text.strip():
        chunks.append(text.strip())
    return chunks


async def build_document_graph(doc_id: str, content: str) -> GraphBuildResult:
    """(Re)build the graph for ``doc_id`` from its text.

    Existing nodes for the document are deleted first, so rebuilding is
    idempotent. An entity that appears in several chunks becomes one node.
    Graph-database connectivity errors propagate to the caller.
    """
    builder = get_graph_builder()
    await builder.delete_document_nodes(doc_id)

    total = GraphBuildResult(
        nodes_created=0, relations_created=0, node_ids=[], relation_ids=[], errors=[]
    )
    # Lowercased label -> node_id for the whole document, shared by all chunks
    document_nodes: dict[str, str] = {}
    for chunk in chunk_text(normalize_text(content)):
        result = await builder.build_from_text(
            chunk, source_doc_id=doc_id, known_nodes=document_nodes
        )
        total.nodes_created += result.nodes_created
        total.relations_created += result.relations_created
        total.node_ids.extend(result.node_ids)
        total.relation_ids.extend(result.relation_ids)
        total.errors.extend(result.errors)
    return total


async def save_build_result(
    repo: DocumentRepository, doc_id: str, result: GraphBuildResult
) -> bool:
    """Store a finished build's node IDs and final status.

    Returns False if the document was deleted while it was being built; the
    graph the build just wrote is then deleted again.
    """
    if not await repo.set_graph_nodes(doc_id, result.node_ids):
        logger.info(f"{doc_id} was deleted during its graph build; removing the graph")
        await delete_document_graph(doc_id)
        return False
    status = DocumentStatus.INDEXED if result.nodes_created else DocumentStatus.VALIDATED
    await repo.update_status(doc_id, status)
    return True


async def mark_build_failed(doc_id: str) -> None:
    """Put a document whose build failed back to VALIDATED (own session)."""
    try:
        async with get_session_context() as session:
            await DocumentRepository(session).update_status(
                doc_id, DocumentStatus.VALIDATED
            )
    except Exception:
        logger.exception(f"Could not reset the status of {doc_id}")


async def _mark_processing(doc_id: str) -> bool:
    """Set PROCESSING if the document has text to build from."""
    async with get_session_context() as session:
        repo = DocumentRepository(session)
        if not await repo.get_content(doc_id):
            return False
        await repo.update_status(doc_id, DocumentStatus.PROCESSING)
        return True


async def _build_and_save(doc_id: str) -> None:
    """Read the text, build the graph, save the result: one short session
    before the build and one after it, none during it."""
    async with get_session_context() as session:
        content = await DocumentRepository(session).get_content(doc_id)
    if not content:
        return  # deleted while queued

    result = await build_document_graph(doc_id, content)
    if result.errors:
        logger.warning(
            f"Graph build for {doc_id} had {len(result.errors)} errors; "
            f"first: {result.errors[0]}"
        )
    async with get_session_context() as session:
        saved = await save_build_result(DocumentRepository(session), doc_id, result)

    if saved:
        logger.info(
            f"Graph built for {doc_id}: {result.nodes_created} nodes, "
            f"{result.relations_created} relations"
        )


async def build_document_graph_task(doc_id: str) -> None:
    """Background task: build the graph for a stored document.

    Opens its own database sessions (the request's session is closed by the
    time background tasks run) and never raises: failures are logged and
    the document goes back to VALIDATED, so the user can retry from the
    reader. Skipped when a build of the same document is already running.
    """
    try:
        async with exclusive_build(doc_id):
            # PROCESSING already while waiting for a build slot: otherwise the
            # reader gives up on a "validated" document and offers a Build
            # button that can only get 409 BUILD_IN_PROGRESS.
            if not await _mark_processing(doc_id):
                logger.warning(f"Graph build skipped for {doc_id}: no content")
                return
            try:
                async with _background_build_slots:
                    await _build_and_save(doc_id)
            except BaseException:
                # Also on cancellation (shutdown), so it isn't left PROCESSING
                await mark_build_failed(doc_id)
                raise
    except BuildInProgressError:
        logger.info(f"Graph build for {doc_id} skipped: a build is already running")
    except GRAPH_UNAVAILABLE_ERRORS as e:
        logger.warning(f"Graph build for {doc_id} skipped, graph DB unavailable: {e}")
    except Exception:
        logger.exception(f"Graph build for {doc_id} failed")


async def reset_interrupted_builds() -> int:
    """Return documents left in PROCESSING by a killed process to VALIDATED.

    Call at startup: build locks live in memory, so no build can be running
    yet, and without this a crash mid-build leaves the document showing
    "building" forever. Returns how many documents were reset.
    """
    async with get_session_context() as session:
        return await DocumentRepository(session).reset_status(
            DocumentStatus.PROCESSING, DocumentStatus.VALIDATED
        )


async def delete_document_graph(doc_id: str) -> int:
    """Delete a document's graph nodes; returns how many were removed.

    Best effort: if the graph database is unreachable, logs and returns 0.
    """
    try:
        return await get_graph_builder().delete_document_nodes(doc_id)
    except GRAPH_UNAVAILABLE_ERRORS as e:
        logger.warning(f"Could not delete graph for {doc_id}, graph DB unavailable: {e}")
        return 0
