"""
Integration tests for the Knowledge Graph system.

Ownership checks run against PostgreSQL before Neo4j is touched, so those
tests run everywhere. Tests marked requires_neo4j need a reachable Neo4j
(CI provides one); tests marked requires_no_neo4j check the 503 contract
when it is down (the usual developer setup).
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.db import get_neo4j_session_context, get_session_context
from app.repositories.document import DocumentRepository
from tests.pdf_utils import DEFAULT_LINES, make_text_pdf

BUILD_TEXT = (
    "Marie Curie was a physicist who worked at the University of Paris. "
    "Marie Curie discovered polonium and radium with Pierre Curie."
)


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
