"""Build or remove the knowledge graph for a stored document.

This is the single entry point the rest of the app uses to turn an
uploaded document's extracted text into graph nodes and relations: the
upload route schedules ``build_document_graph_task`` after a successful
upload, and ``POST /graph/documents/{doc_id}/build`` rebuilds on demand.
"""

import logging
import re

from app.db import GRAPH_UNAVAILABLE_ERRORS, get_session_context
from app.models.document import DocumentStatus
from app.repositories.document import DocumentRepository
from app.services.graph.builder import GraphBuildResult, get_graph_builder

logger = logging.getLogger(__name__)

# BuildGraphRequest caps text at 50k chars; stay well inside spaCy's limits too.
MAX_CHUNK_CHARS = 50_000


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
    idempotent. Graph-database connectivity errors propagate to the caller.
    """
    builder = get_graph_builder()
    await builder.delete_document_nodes(doc_id)

    total = GraphBuildResult(
        nodes_created=0, relations_created=0, node_ids=[], relation_ids=[], errors=[]
    )
    for chunk in chunk_text(normalize_text(content)):
        result = await builder.build_from_text(chunk, source_doc_id=doc_id)
        total.nodes_created += result.nodes_created
        total.relations_created += result.relations_created
        total.node_ids.extend(result.node_ids)
        total.relation_ids.extend(result.relation_ids)
        total.errors.extend(result.errors)
    return total


async def build_document_graph_task(doc_id: str) -> None:
    """Background task: build the graph for a stored document.

    Opens its own database session (the request's session is closed by the
    time background tasks run) and never raises: failures are logged and
    the document stays VALIDATED, so the user can retry from the reader.
    """
    try:
        async with get_session_context() as session:
            repo = DocumentRepository(session)
            content = await repo.get_content(doc_id)
            if not content:
                logger.warning(f"Graph build skipped for {doc_id}: no content")
                return

            result = await build_document_graph(doc_id, content)
            if result.errors:
                logger.warning(
                    f"Graph build for {doc_id} had {len(result.errors)} errors; "
                    f"first: {result.errors[0]}"
                )
            if result.nodes_created:
                await repo.add_graph_nodes(doc_id, result.node_ids)
                await repo.update_status(doc_id, DocumentStatus.INDEXED)
            logger.info(
                f"Graph built for {doc_id}: {result.nodes_created} nodes, "
                f"{result.relations_created} relations"
            )
    except GRAPH_UNAVAILABLE_ERRORS as e:
        logger.warning(f"Graph build for {doc_id} skipped, graph DB unavailable: {e}")
    except Exception:
        logger.exception(f"Graph build for {doc_id} failed")


async def delete_document_graph(doc_id: str) -> int:
    """Delete a document's graph nodes; returns how many were removed.

    Best effort: if the graph database is unreachable, logs and returns 0.
    """
    try:
        return await get_graph_builder().delete_document_nodes(doc_id)
    except GRAPH_UNAVAILABLE_ERRORS as e:
        logger.warning(f"Could not delete graph for {doc_id}, graph DB unavailable: {e}")
        return 0
