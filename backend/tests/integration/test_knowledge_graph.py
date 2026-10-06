"""
Integration tests for the Knowledge Graph system.

Ownership checks run against PostgreSQL before Neo4j is touched, so those
tests run everywhere. Tests marked requires_neo4j need a reachable Neo4j
(CI provides one); tests marked requires_no_neo4j check the 503 contract
when it is down (the usual developer setup).
"""

import asyncio
import threading
import uuid
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

import app.api.v1.routes.graph as graph_routes
import app.services.graph.builder as builder_module
import app.services.graph.document_graph as document_graph
from app.api.v1.routes import documents as documents_routes
from app.db import get_neo4j_session_context, get_session_context
from app.models.document import DocumentStatus
from app.repositories.document import DocumentRepository
from app.services.graph.builder import GraphBuildResult
from tests.integration.test_document_flow import (
    document_status,
    fake_build_result,
    idle_in_transaction_connections,
)
from tests.pdf_utils import DEFAULT_LINES, make_text_pdf

BUILD_TEXT = (
    "Marie Curie was a physicist who worked at the University of Paris. "
    "Marie Curie discovered polonium and radium with Pierre Curie."
)

# Driver error text that must never reach a response
DRIVER_ERROR = "Couldn't connect to graph-internal.example:7687 (resolved to 10.20.30.40)"


@pytest.fixture
def graph_document(client: TestClient, api_prefix: str, auth_headers: dict) -> dict:
    """Upload a real text PDF as the ``auth_headers`` user; return the document.

    A unique trailing line keeps the doc_id distinct from every other
    test's upload, whatever the server's dedup rules are.
    """
    pdf = make_text_pdf(DEFAULT_LINES + [f"Reference code {uuid.uuid4().hex[:8]}."])
    response = client.post(
        f"{api_prefix}/documents/upload",
        files={"file": ("curie.pdf", pdf, "application/pdf")},
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["document"]


@pytest.fixture
def no_background_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uploads in this test schedule no graph build."""

    async def skip(doc_id: str) -> None:
        return None

    monkeypatch.setattr(documents_routes, "build_document_graph_task", skip)


def _upload_unique(client: TestClient, api_prefix: str, headers: dict) -> dict:
    pdf = make_text_pdf(DEFAULT_LINES + [f"Reference code {uuid.uuid4().hex[:8]}."])
    response = client.post(
        f"{api_prefix}/documents/upload",
        files={"file": ("curie.pdf", pdf, "application/pdf")},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["document"]


@pytest.fixture
def empty_graph_document(
    client: TestClient, api_prefix: str, auth_headers: dict, no_background_build: None
) -> dict:
    """Like graph_document, but no graph is built for it."""
    return _upload_unique(client, api_prefix, auth_headers)


def _create_node(client: TestClient, api_prefix: str, headers: dict, doc_id: str, label: str) -> str:
    response = client.post(
        f"{api_prefix}/graph/nodes",
        json={"label": label, "type": "Concept", "source_doc_id": doc_id},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["node_id"]


def _set_document_state(client: TestClient, doc_id: str, status: DocumentStatus, node_ids: list[str]):
    async def update():
        async with get_session_context() as session:
            repo = DocumentRepository(session)
            assert await repo.set_graph_nodes(doc_id, node_ids)
            await repo.update_status(doc_id, status)

    client.portal.call(update)


def _assert_graph_unavailable(response) -> None:
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "GRAPH_UNAVAILABLE"


def _assert_not_found(response, code: str = "NOT_FOUND") -> None:
    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == code


def _node_ids(response) -> set[str]:
    assert response.status_code == 200, response.text
    return {node["node_id"] for node in response.json()["nodes"]}


class TestGraphOwnership:
    """Graph routes only act on the caller's documents (no Neo4j needed)."""

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", "/graph/documents/{doc}/build", None),
            ("post", "/graph/build", {"text": BUILD_TEXT, "source_doc_id": "{doc}"}),
            ("post", "/graph/nodes", {"label": "Radium", "type": "Concept", "source_doc_id": "{doc}"}),
            ("get", "/graph/nodes?source_doc_id={doc}", None),
            ("get", "/graph/documents/{doc}/relations", None),
            ("delete", "/graph/documents/{doc}/nodes", None),
        ],
    )
    def test_other_users_document_is_not_found(
        self,
        client: TestClient,
        api_prefix: str,
        graph_document: dict,
        other_auth_headers: dict,
        method: str,
        path: str,
        body: dict | None,
    ):
        doc_id = graph_document["doc_id"]
        kwargs = {}
        if body is not None:
            kwargs["json"] = {
                k: (v.format(doc=doc_id) if isinstance(v, str) else v) for k, v in body.items()
            }

        response = getattr(client, method)(
            f"{api_prefix}{path.format(doc=doc_id)}", headers=other_auth_headers, **kwargs
        )

        _assert_not_found(response)

    def test_build_unknown_document_is_not_found(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        response = client.post(
            f"{api_prefix}/graph/documents/sha256:does_not_exist/build", headers=auth_headers
        )

        _assert_not_found(response)

    def test_build_document_without_text_is_rejected(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """A document with no extracted text can't be built: 422 NO_CONTENT."""
        me = client.get(f"{api_prefix}/auth/me", headers=auth_headers)
        assert me.status_code == 200
        doc_id = f"sha256:empty_{uuid.uuid4().hex}"

        async def create_empty_document():
            async with get_session_context() as session:
                await DocumentRepository(session).create(
                    doc_id=doc_id, title="Scanned images", owner_id=me.json()["user_id"]
                )

        client.portal.call(create_empty_document)

        response = client.post(
            f"{api_prefix}/graph/documents/{doc_id}/build", headers=auth_headers
        )

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "NO_CONTENT"


@pytest.mark.requires_no_neo4j
class TestGraphUnavailable:
    """With Neo4j down, every graph route reports 503 GRAPH_UNAVAILABLE."""

    def test_search(self, client: TestClient, api_prefix: str, auth_headers: dict):
        _assert_graph_unavailable(client.get(f"{api_prefix}/graph/nodes", headers=auth_headers))

    def test_search_by_document(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        response = client.get(
            f"{api_prefix}/graph/nodes",
            params={"source_doc_id": graph_document["doc_id"]},
            headers=auth_headers,
        )
        _assert_graph_unavailable(response)

    def test_get_node(self, client: TestClient, api_prefix: str, auth_headers: dict):
        _assert_graph_unavailable(
            client.get(f"{api_prefix}/graph/nodes/node_123", headers=auth_headers)
        )

    def test_related_nodes(self, client: TestClient, api_prefix: str, auth_headers: dict):
        _assert_graph_unavailable(
            client.get(f"{api_prefix}/graph/nodes/node_123/related", headers=auth_headers)
        )

    def test_document_relations(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        response = client.get(
            f"{api_prefix}/graph/documents/{graph_document['doc_id']}/relations",
            headers=auth_headers,
        )
        _assert_graph_unavailable(response)

    def test_build_from_text(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        response = client.post(
            f"{api_prefix}/graph/build",
            json={
                "text": BUILD_TEXT,
                "source_doc_id": graph_document["doc_id"],
                "generate_embeddings": False,
            },
            headers=auth_headers,
        )
        _assert_graph_unavailable(response)

    def test_build_document(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        response = client.post(
            f"{api_prefix}/graph/documents/{graph_document['doc_id']}/build",
            headers=auth_headers,
        )
        _assert_graph_unavailable(response)

    def test_create_node(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        response = client.post(
            f"{api_prefix}/graph/nodes",
            json={"label": "Radium", "type": "Concept", "source_doc_id": graph_document["doc_id"]},
            headers=auth_headers,
        )
        _assert_graph_unavailable(response)

    def test_create_relation(self, client: TestClient, api_prefix: str, auth_headers: dict):
        response = client.post(
            f"{api_prefix}/graph/relations",
            json={
                "source_node_id": "node_1",
                "target_node_id": "node_2",
                "relation_type": "EXPLAINS",
            },
            headers=auth_headers,
        )
        _assert_graph_unavailable(response)

    def test_delete_document_nodes(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        response = client.delete(
            f"{api_prefix}/graph/documents/{graph_document['doc_id']}/nodes",
            headers=auth_headers,
        )
        _assert_graph_unavailable(response)


@pytest.fixture
def graph_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every graph-database session fails as if Neo4j were down."""

    @asynccontextmanager
    async def unreachable():
        raise ServiceUnavailable(DRIVER_ERROR)
        yield  # pragma: no cover

    monkeypatch.setattr(builder_module, "get_neo4j_session_context", unreachable)


class TestGraphOutageContract:
    """503 GRAPH_UNAVAILABLE without the driver's text, whether or not a real
    Neo4j is running (the requires_no_neo4j tests above never run in CI)."""

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("get", "/graph/nodes", None),
            ("get", "/graph/nodes?source_doc_id={doc}", None),
            ("get", "/graph/nodes/node_123", None),
            ("get", "/graph/nodes/node_123/related", None),
            ("get", "/graph/documents/{doc}/relations", None),
            ("post", "/graph/documents/{doc}/build", None),
            ("post", "/graph/nodes", {"label": "Radium", "type": "Concept", "source_doc_id": "{doc}"}),
            (
                "post",
                "/graph/relations",
                {"source_node_id": "node_1", "target_node_id": "node_2", "relation_type": "EXPLAINS"},
            ),
            ("delete", "/graph/documents/{doc}/nodes", None),
        ],
    )
    def test_route_reports_graph_unavailable(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
        graph_outage: None,
        method: str,
        path: str,
        body: dict | None,
    ):
        doc_id = empty_graph_document["doc_id"]
        kwargs = {}
        if body is not None:
            kwargs["json"] = {
                k: (v.format(doc=doc_id) if isinstance(v, str) else v) for k, v in body.items()
            }

        response = getattr(client, method)(
            f"{api_prefix}{path.format(doc=doc_id)}", headers=auth_headers, **kwargs
        )

        _assert_graph_unavailable(response)
        assert "graph-internal" not in response.text
        assert "10.20.30.40" not in response.text

    def test_rebuild_outage_returns_document_to_validated(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
        graph_outage: None,
    ):
        doc_id = empty_graph_document["doc_id"]
        _set_document_state(client, doc_id, DocumentStatus.INDEXED, ["node_old"])

        response = client.post(
            f"{api_prefix}/graph/documents/{doc_id}/build", headers=auth_headers
        )

        _assert_graph_unavailable(response)
        assert client.portal.call(document_status, doc_id) == "validated"


class PausedBuild:
    """Stands in for build_document_graph and waits until released."""

    def __init__(self, result: GraphBuildResult):
        self.result = result
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []
        self.idle_in_transaction: list[int] = []

    async def __call__(self, doc_id: str, content: str) -> GraphBuildResult:
        self.calls.append(doc_id)
        self.idle_in_transaction.append(await idle_in_transaction_connections())
        self.started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        return self.result


class TestDocumentRebuild:
    """POST /graph/documents/{id}/build with the graph build replaced."""

    def test_startup_resets_builds_interrupted_by_a_crash(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
    ):
        """A process killed mid-build leaves the document in PROCESSING;
        the startup reset puts it back to VALIDATED so it can be rebuilt."""
        doc_id = empty_graph_document["doc_id"]
        _set_document_state(client, doc_id, DocumentStatus.PROCESSING, [])

        reset = client.portal.call(document_graph.reset_interrupted_builds)

        assert reset >= 1
        response = client.get(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "validated"

    def test_conflict_while_building_and_status_transitions(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        doc_id = empty_graph_document["doc_id"]
        url = f"{api_prefix}/graph/documents/{doc_id}/build"
        paused = PausedBuild(fake_build_result("node_a", "node_b"))
        monkeypatch.setattr(graph_routes, "build_document_graph", paused)
        monkeypatch.setattr(document_graph, "build_document_graph", paused)
        safety = threading.Timer(20, paused.release.set)  # never hang the suite
        safety.start()
        results: list = []
        first = threading.Thread(target=lambda: results.append(client.post(url, headers=auth_headers)))
        first.start()
        try:
            assert paused.started.wait(10)
            status_during = client.portal.call(document_status, doc_id)
            second = client.post(url, headers=auth_headers)
            # A background build of the same document is skipped, not run
            client.portal.call(document_graph.build_document_graph_task, doc_id)
        finally:
            paused.release.set()
            first.join(30)
            safety.cancel()

        assert status_during == "processing"
        assert second.status_code == 409, second.text
        assert second.json()["detail"]["code"] == "BUILD_IN_PROGRESS"
        assert paused.calls == [doc_id]
        # The request's transaction was over before the build started
        assert paused.idle_in_transaction == [0]
        (response,) = results
        assert response.status_code == 200, response.text
        assert response.json()["nodes_created"] == 2
        document = client.get(f"{api_prefix}/documents/{doc_id}", headers=auth_headers).json()
        assert document["status"] == "indexed"
        assert document["graph_nodes"] == ["node_a", "node_b"]
        # The lock is released afterwards
        assert client.post(url, headers=auth_headers).status_code == 200

    def test_rebuild_without_nodes_clears_graph_nodes(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        doc_id = empty_graph_document["doc_id"]
        _set_document_state(client, doc_id, DocumentStatus.INDEXED, ["node_stale_1", "node_stale_2"])

        async def build(doc_id: str, content: str) -> GraphBuildResult:
            return fake_build_result()

        monkeypatch.setattr(graph_routes, "build_document_graph", build)

        response = client.post(
            f"{api_prefix}/graph/documents/{doc_id}/build", headers=auth_headers
        )

        assert response.status_code == 200, response.text
        assert response.json()["nodes_created"] == 0
        document = client.get(f"{api_prefix}/documents/{doc_id}", headers=auth_headers).json()
        assert document["graph_nodes"] == []
        assert document["status"] == "validated"

    def test_rebuild_of_document_deleted_meanwhile(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        doc_id = empty_graph_document["doc_id"]
        graph_deletes: list[str] = []

        async def record_graph_delete(doc_id: str) -> int:
            graph_deletes.append(doc_id)
            return 0

        async def delete_then_build(doc_id: str, content: str) -> GraphBuildResult:
            async with get_session_context() as session:
                await DocumentRepository(session).delete(doc_id)
            return fake_build_result("node_a")

        monkeypatch.setattr(document_graph, "delete_document_graph", record_graph_delete)
        monkeypatch.setattr(graph_routes, "build_document_graph", delete_then_build)

        response = client.post(
            f"{api_prefix}/graph/documents/{doc_id}/build", headers=auth_headers
        )

        _assert_not_found(response)
        # The graph the build wrote for the deleted document is removed again
        assert graph_deletes == [doc_id]


@pytest.mark.requires_neo4j
class TestGraphNodeOperations:
    """Single-node and single-relation operations against a live Neo4j."""

    def test_create_and_get_node(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        doc_id = graph_document["doc_id"]
        response = client.post(
            f"{api_prefix}/graph/nodes",
            json={
                "label": "Polonium",
                "type": "Concept",
                "description": "A chemical element",
                "source_doc_id": doc_id,
                # Metadata may not move the node into someone else's document
                "metadata": {"source_doc_id": "sha256:someone_else", "note": "kept"},
            },
            headers=auth_headers,
        )

        assert response.status_code == 201, response.text
        created = response.json()
        assert created["label"] == "Polonium"
        assert created["type"] == "Concept"
        assert created["source_doc_id"] == doc_id
        assert created["node_id"].startswith("node_")

        fetched = client.get(
            f"{api_prefix}/graph/nodes/{created['node_id']}", headers=auth_headers
        )
        assert fetched.status_code == 200
        body = fetched.json()
        assert body["node_id"] == created["node_id"]
        assert body["label"] == "Polonium"
        assert body["description"] == "A chemical element"
        assert body["source_doc_id"] == doc_id
        assert body["metadata"]["note"] == "kept"

    def test_get_nonexistent_node(self, client: TestClient, api_prefix: str, auth_headers: dict):
        response = client.get(
            f"{api_prefix}/graph/nodes/node_does_not_exist", headers=auth_headers
        )
        _assert_not_found(response, "NODE_NOT_FOUND")

    def test_traverse_from_nonexistent_node(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        response = client.get(
            f"{api_prefix}/graph/nodes/node_does_not_exist/related?max_depth=2",
            headers=auth_headers,
        )
        _assert_not_found(response, "NODE_NOT_FOUND")

    def test_create_relation_nonexistent_nodes(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        response = client.post(
            f"{api_prefix}/graph/relations",
            json={
                "source_node_id": "node_does_not_exist_1",
                "target_node_id": "node_does_not_exist_2",
                "relation_type": "EXPLAINS",
                "weight": 0.8,
            },
            headers=auth_headers,
        )
        _assert_not_found(response, "NODE_NOT_FOUND")

    def test_create_relation_between_nodes(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        doc_id = graph_document["doc_id"]
        node_ids = []
        for label in ("Radioactivity", "Radium"):
            response = client.post(
                f"{api_prefix}/graph/nodes",
                json={"label": label, "type": "Concept", "source_doc_id": doc_id},
                headers=auth_headers,
            )
            assert response.status_code == 201, response.text
            node_ids.append(response.json()["node_id"])

        response = client.post(
            f"{api_prefix}/graph/relations",
            json={
                "source_node_id": node_ids[1],
                "target_node_id": node_ids[0],
                "relation_type": "EXPLAINS",
                "weight": 0.8,
            },
            headers=auth_headers,
        )

        assert response.status_code == 201, response.text
        relation = response.json()
        assert relation["relation_id"].startswith("rel_")
        assert relation["type"] == "EXPLAINS"

        relations = client.get(
            f"{api_prefix}/graph/documents/{doc_id}/relations", headers=auth_headers
        )
        assert relations.status_code == 200
        # The upload may also have built a graph; pick out our relation
        mine = [
            r for r in relations.json()["relations"]
            if r["relation_id"] == relation["relation_id"]
        ]
        assert mine == [
            {
                "relation_id": relation["relation_id"],
                "source_node_id": node_ids[1],
                "target_node_id": node_ids[0],
                "type": "EXPLAINS",
                "weight": 0.8,
            }
        ]

    def test_relations_without_stored_id_or_weight(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        """Relationships written before relation_id/weight existed still load."""
        doc_id = graph_document["doc_id"]
        node_ids = []
        for label in ("Henri Becquerel", "Uranium"):
            response = client.post(
                f"{api_prefix}/graph/nodes",
                json={"label": label, "type": "Concept", "source_doc_id": doc_id},
                headers=auth_headers,
            )
            assert response.status_code == 201, response.text
            node_ids.append(response.json()["node_id"])

        async def create_legacy_relationship():
            async with get_neo4j_session_context() as session:
                result = await session.run(
                    "MATCH (a {node_id: $a}), (b {node_id: $b}) "
                    "CREATE (a)-[r:RELATED_TO {confidence: 0.6}]->(b) "
                    "RETURN elementId(r) AS id",
                    a=node_ids[0],
                    b=node_ids[1],
                )
                return (await result.single())["id"]

        element_id = client.portal.call(create_legacy_relationship)

        response = client.get(
            f"{api_prefix}/graph/documents/{doc_id}/relations", headers=auth_headers
        )

        assert response.status_code == 200, response.text
        legacy = [r for r in response.json()["relations"] if r["relation_id"] == element_id]
        assert legacy == [
            {
                "relation_id": element_id,
                "source_node_id": node_ids[0],
                "target_node_id": node_ids[1],
                "type": "RELATED_TO",
                "weight": 0.6,
            }
        ]

    def test_build_from_text_returns_public_node_ids(
        self, client: TestClient, api_prefix: str, auth_headers: dict, graph_document: dict
    ):
        response = client.post(
            f"{api_prefix}/graph/build",
            json={
                "text": BUILD_TEXT,
                "source_doc_id": graph_document["doc_id"],
                "generate_embeddings": False,
            },
            headers=auth_headers,
        )

        assert response.status_code == 201, response.text
        body = response.json()
        assert body["errors"] == []
        assert body["nodes_created"] == len(body["node_ids"]) > 0
        for node_id in body["node_ids"]:
            fetched = client.get(f"{api_prefix}/graph/nodes/{node_id}", headers=auth_headers)
            assert fetched.status_code == 200
            assert fetched.json()["node_id"] == node_id

    def test_other_user_cannot_link_owners_nodes(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        other_auth_headers: dict,
        empty_graph_document: dict,
    ):
        doc_id = empty_graph_document["doc_id"]
        a = _create_node(client, api_prefix, auth_headers, doc_id, "Polonium")
        b = _create_node(client, api_prefix, auth_headers, doc_id, "Radium")

        response = client.post(
            f"{api_prefix}/graph/relations",
            json={"source_node_id": a, "target_node_id": b, "relation_type": "EXPLAINS"},
            headers=other_auth_headers,
        )

        _assert_not_found(response, "NODE_NOT_FOUND")
        relations = client.get(
            f"{api_prefix}/graph/documents/{doc_id}/relations", headers=auth_headers
        )
        assert relations.json() == {"relations": [], "total": 0}

    def test_relation_needs_both_nodes_owned_by_caller(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        other_auth_headers: dict,
        empty_graph_document: dict,
    ):
        my_doc = empty_graph_document["doc_id"]
        their_doc = _upload_unique(client, api_prefix, other_auth_headers)["doc_id"]
        mine = _create_node(client, api_prefix, auth_headers, my_doc, "Polonium")
        theirs = _create_node(client, api_prefix, other_auth_headers, their_doc, "Radium")

        for source, target in ((mine, theirs), (theirs, mine)):
            response = client.post(
                f"{api_prefix}/graph/relations",
                json={"source_node_id": source, "target_node_id": target, "relation_type": "EXPLAINS"},
                headers=auth_headers,
            )
            _assert_not_found(response, "NODE_NOT_FOUND")

        for doc, headers in ((my_doc, auth_headers), (their_doc, other_auth_headers)):
            relations = client.get(f"{api_prefix}/graph/documents/{doc}/relations", headers=headers)
            assert relations.json() == {"relations": [], "total": 0}

    def test_related_nodes_never_include_other_users_nodes(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        other_auth_headers: dict,
        empty_graph_document: dict,
    ):
        my_doc = empty_graph_document["doc_id"]
        their_doc = _upload_unique(client, api_prefix, other_auth_headers)["doc_id"]
        start = _create_node(client, api_prefix, auth_headers, my_doc, "Marie Curie")
        mine = _create_node(client, api_prefix, auth_headers, my_doc, "Radium")
        theirs = _create_node(client, api_prefix, other_auth_headers, their_doc, "Polonium")

        async def link(source: str, target: str) -> None:
            # The API refuses cross-user edges, so seed one directly
            async with get_neo4j_session_context() as session:
                result = await session.run(
                    "MATCH (a {node_id: $a}), (b {node_id: $b}) "
                    "CREATE (a)-[:RELATED_TO {relation_id: $rid}]->(b)",
                    a=source,
                    b=target,
                    rid=f"rel_{uuid.uuid4().hex}",
                )
                await result.consume()

        client.portal.call(link, start, mine)
        client.portal.call(link, start, theirs)

        related = client.get(
            f"{api_prefix}/graph/nodes/{start}/related",
            params={"max_depth": 1},
            headers=auth_headers,
        )

        assert _node_ids(related) == {mine}
        assert related.json()["total"] == 1

    def test_overlong_stored_label_is_truncated_when_listed(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
    ):
        """Builds used to store run-on 'entities' over GraphNode's 500-char
        limit; one such node made the document's whole listing a 500."""
        doc_id = empty_graph_document["doc_id"]
        node_id = _create_node(client, api_prefix, auth_headers, doc_id, "placeholder")
        long_label = " ".join(f"value{i} measurement" for i in range(60))
        assert len(long_label) > 800

        async def store_long_label():
            async with get_neo4j_session_context() as session:
                result = await session.run(
                    "MATCH (n {node_id: $id}) SET n.label = $label", id=node_id, label=long_label
                )
                await result.consume()

        client.portal.call(store_long_label)

        by_doc = client.get(
            f"{api_prefix}/graph/nodes",
            params={"source_doc_id": doc_id, "limit": 100},
            headers=auth_headers,
        )
        everything = client.get(f"{api_prefix}/graph/nodes", params={"limit": 200}, headers=auth_headers)
        single = client.get(f"{api_prefix}/graph/nodes/{node_id}", headers=auth_headers)

        assert by_doc.status_code == 200, by_doc.text
        assert [n["label"] for n in by_doc.json()["nodes"]] == [long_label[:500]]
        assert everything.status_code == 200, everything.text
        assert single.status_code == 200, single.text
        assert single.json()["label"] == long_label[:500]

    def test_flat_metadata_with_arrays_is_stored(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
    ):
        doc_id = empty_graph_document["doc_id"]
        metadata = {
            "aliases": ["Po", "element 84"],
            "year": 1898,
            "half_life_days": 138.4,
            "radioactive": True,
            "pages": [3, 4.5],
            "empty": [],
        }
        created = client.post(
            f"{api_prefix}/graph/nodes",
            json={"label": "Polonium", "type": "Concept", "source_doc_id": doc_id, "metadata": metadata},
            headers=auth_headers,
        )
        assert created.status_code == 201, created.text
        other = _create_node(client, api_prefix, auth_headers, doc_id, "Radium")

        relation = client.post(
            f"{api_prefix}/graph/relations",
            json={
                "source_node_id": created.json()["node_id"],
                "target_node_id": other,
                "relation_type": "RELATED_TO",
                "metadata": {"pages": [1, 2], "note": "same paper"},
            },
            headers=auth_headers,
        )

        assert relation.status_code == 201, relation.text
        fetched = client.get(
            f"{api_prefix}/graph/nodes/{created.json()['node_id']}", headers=auth_headers
        ).json()
        stored = {k: fetched["metadata"][k] for k in metadata}
        assert stored == {**metadata, "pages": [3.0, 4.5]}

    def test_search_is_ordered_by_label_then_node_id(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        empty_graph_document: dict,
    ):
        """A limited result is always the same subset (it used to be
        whatever Neo4j returned first), and up to 1000 nodes come back."""
        doc_id = empty_graph_document["doc_id"]
        for label in ("Zeta", "Alpha", "Mu", "Mu", "Beta"):
            _create_node(client, api_prefix, auth_headers, doc_id, label)

        listed = client.get(
            f"{api_prefix}/graph/nodes",
            params={"source_doc_id": doc_id, "limit": 1000},
            headers=auth_headers,
        )
        first_two = client.get(
            f"{api_prefix}/graph/nodes",
            params={"source_doc_id": doc_id, "limit": 2},
            headers=auth_headers,
        )

        assert listed.status_code == 200, listed.text
        pairs = [(n["label"], n["node_id"]) for n in listed.json()["nodes"]]
        assert [label for label, _ in pairs] == ["Alpha", "Beta", "Mu", "Mu", "Zeta"]
        assert pairs == sorted(pairs)
        assert [(n["label"], n["node_id"]) for n in first_two.json()["nodes"]] == pairs[:2]


@pytest.mark.requires_neo4j
class TestDocumentGraphRoundTrip:
    """Upload -> build -> browse -> rebuild -> isolation -> delete."""

    def test_round_trip(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        other_auth_headers: dict,
        graph_document: dict,
    ):
        doc_id = graph_document["doc_id"]
        graph = f"{api_prefix}/graph"

        # Build from the stored text
        build = client.post(f"{graph}/documents/{doc_id}/build", headers=auth_headers)
        assert build.status_code == 200, build.text
        built = build.json()
        assert built["doc_id"] == doc_id
        assert built["nodes_created"] > 0
        assert built["relations_created"] > 0
        assert built["errors"] == []

        document = client.get(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)
        assert document.status_code == 200
        assert document.json()["status"] == "indexed"

        # Nodes of this document
        nodes = client.get(
            f"{graph}/nodes", params={"source_doc_id": doc_id, "limit": 200}, headers=auth_headers
        )
        node_ids = _node_ids(nodes)
        assert nodes.json()["total"] == len(node_ids) == built["nodes_created"]
        labels = [n["label"] for n in nodes.json()["nodes"]]
        assert "Marie Curie" in labels
        assert all("\n" not in label for label in labels)
        assert all(n["source_doc_id"] == doc_id for n in nodes.json()["nodes"])

        # Relations: every endpoint is one of the listed nodes, no self-loops
        relations = client.get(f"{graph}/documents/{doc_id}/relations", headers=auth_headers)
        assert relations.status_code == 200, relations.text
        rels = relations.json()["relations"]
        assert relations.json()["total"] == len(rels) == built["relations_created"]
        for rel in rels:
            assert rel["relation_id"].startswith("rel_")
            assert 0.0 <= rel["weight"] <= 1.0
            assert rel["source_node_id"] in node_ids
            assert rel["target_node_id"] in node_ids
            assert rel["source_node_id"] != rel["target_node_id"]

        # A listed node_id works for the single-node and related endpoints
        source, target = rels[0]["source_node_id"], rels[0]["target_node_id"]
        node = client.get(f"{graph}/nodes/{source}", headers=auth_headers)
        assert node.status_code == 200
        assert node.json()["node_id"] == source
        assert node.json()["source_doc_id"] == doc_id

        related = client.get(
            f"{graph}/nodes/{source}/related",
            params={"max_depth": 1, "limit": 200},
            headers=auth_headers,
        )
        related_ids = _node_ids(related)
        assert related.json()["total"] > 0
        assert target in related_ids
        assert related_ids <= node_ids

        # Rebuilding replaces the graph instead of duplicating it
        rebuild = client.post(f"{graph}/documents/{doc_id}/build", headers=auth_headers)
        assert rebuild.status_code == 200
        assert rebuild.json()["nodes_created"] == built["nodes_created"]
        rebuilt = client.get(
            f"{graph}/nodes", params={"source_doc_id": doc_id, "limit": 200}, headers=auth_headers
        )
        node_ids = _node_ids(rebuilt)
        assert len(node_ids) == built["nodes_created"]
        relations = client.get(f"{graph}/documents/{doc_id}/relations", headers=auth_headers)
        assert relations.json()["total"] == rebuild.json()["relations_created"]
        # The stored node IDs are replaced too, not merged with the old ones
        document = client.get(f"{api_prefix}/documents/{doc_id}", headers=auth_headers).json()
        assert document["status"] == "indexed"
        assert sorted(document["graph_nodes"]) == sorted(node_ids)

        # An unfiltered search is scoped to the caller's documents
        mine = client.get(f"{graph}/nodes", params={"limit": 200}, headers=auth_headers)
        assert _node_ids(mine) == node_ids

        # Another user can neither see nor touch this graph
        some_node = sorted(node_ids)[0]
        other = other_auth_headers
        _assert_not_found(client.post(f"{graph}/documents/{doc_id}/build", headers=other))
        _assert_not_found(client.get(f"{graph}/documents/{doc_id}/relations", headers=other))
        _assert_not_found(
            client.get(f"{graph}/nodes", params={"source_doc_id": doc_id}, headers=other)
        )
        _assert_not_found(client.delete(f"{graph}/documents/{doc_id}/nodes", headers=other))
        _assert_not_found(client.get(f"{graph}/nodes/{some_node}", headers=other), "NODE_NOT_FOUND")
        _assert_not_found(
            client.get(f"{graph}/nodes/{some_node}/related", headers=other), "NODE_NOT_FOUND"
        )
        theirs = client.get(f"{graph}/nodes", params={"limit": 200}, headers=other)
        assert _node_ids(theirs).isdisjoint(node_ids)

        # The owner deletes the document's graph
        deleted = client.delete(f"{graph}/documents/{doc_id}/nodes", headers=auth_headers)
        assert deleted.status_code == 200
        assert deleted.json()["deleted_count"] == len(node_ids)
        after = client.get(f"{graph}/nodes", params={"source_doc_id": doc_id}, headers=auth_headers)
        assert after.json() == {"nodes": [], "total": 0}
        relations = client.get(f"{graph}/documents/{doc_id}/relations", headers=auth_headers)
        assert relations.json() == {"relations": [], "total": 0}


class TestEntityExtraction:
    """Test entity extraction from text."""

    @pytest.fixture
    def extractor(self):
        spacy = pytest.importorskip("spacy")
        if not spacy.util.is_package("en_core_web_sm"):
            pytest.skip("spaCy model en_core_web_sm is not installed")
        from app.services.graph.extractor import EntityExtractor

        return EntityExtractor()

    def test_extract_entities_basic(self, extractor, sample_document_content: str):
        """Named organisations in the sample are extracted."""
        entities = extractor.extract_entities(sample_document_content)

        texts = {e.text for e in entities}
        assert len(entities) > 0
        assert {"MIT", "Google"} <= texts

    def test_extract_entities_empty_text(self, extractor):
        """Test entity extraction from empty text."""
        assert extractor.extract_entities("") == []
