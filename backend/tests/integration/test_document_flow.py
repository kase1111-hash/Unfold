"""
Integration tests for the document ingestion and management flow.
Tests the complete lifecycle: upload -> validate -> process -> retrieve -> delete
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from app.api.v1.routes import documents as documents_routes
from app.db import get_neo4j_session_context
from app.services.ingestion.document_service import DocumentService
from tests.pdf_utils import DEFAULT_LINES, make_text_pdf

EXPECTED_WORD_COUNT = sum(len(line.split()) for line in DEFAULT_LINES)


def _upload(client: TestClient, api_prefix: str, headers: dict, name: str, data: bytes):
    return client.post(
        f"{api_prefix}/documents/upload",
        files={"file": (name, data, "application/pdf")},
        headers=headers,
    )


def _list(client: TestClient, api_prefix: str, headers: dict, query: str = "") -> dict:
    response = client.get(f"{api_prefix}/documents?{query}", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _stored_files() -> set[str]:
    upload_dir = DocumentService.UPLOAD_DIR
    return set(p.name for p in upload_dir.iterdir()) if upload_dir.exists() else set()


def _encrypted_pdf(user_password: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_text_pdf())))
    writer.encrypt(user_password=user_password, owner_password="owner-secret")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


async def _count_graph_nodes(doc_id: str) -> int:
    async with get_neo4j_session_context() as session:
        result = await session.run(
            "MATCH (n {source_doc_id: $doc_id}) RETURN count(n) AS count",
            doc_id=doc_id,
        )
        record = await result.single()
        return record["count"]


@pytest.fixture
def graph_build_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the background graph build with a recorder."""
    calls: list[str] = []

    async def record(doc_id: str) -> None:
        calls.append(doc_id)

    monkeypatch.setattr(documents_routes, "build_document_graph_task", record)
    return calls


class TestUploadSuccess:
    """A real PDF is processed synchronously and returned processed."""

    def test_upload_returns_processed_document(
        self, client: TestClient, api_prefix: str, auth_headers: dict, text_pdf: bytes
    ):
        response = _upload(client, api_prefix, auth_headers, "curie.pdf", text_pdf)

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "success"
        assert body["message"] == "Document uploaded successfully"
        document = body["document"]
        assert document["status"] == "validated"
        assert document["word_count"] == EXPECTED_WORD_COUNT
        assert document["page_count"] == 1
        assert document["title"] == "curie"
        assert document["file_size_bytes"] == len(text_pdf)
        assert document["doc_id"].startswith("sha256:")
        assert Path(document["file_path"]).read_bytes() == text_pdf

    def test_content_returns_extracted_text(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        doc_id = uploaded_document["doc_id"]

        response = client.get(
            f"{api_prefix}/documents/{doc_id}/content", headers=auth_headers
        )

        assert response.status_code == 200
        body = response.json()
        assert body["doc_id"] == doc_id
        for line in DEFAULT_LINES:
            assert line in body["content"]

    def test_get_document_metadata(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        doc_id = uploaded_document["doc_id"]

        response = client.get(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)

        assert response.status_code == 200
        document = response.json()
        assert document["doc_id"] == doc_id
        assert document["title"] == "curie"
        assert document["word_count"] == EXPECTED_WORD_COUNT
        assert document["page_count"] == 1

    def test_upload_schedules_graph_build(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        text_pdf: bytes,
        graph_build_calls: list[str],
    ):
        response = _upload(client, api_prefix, auth_headers, "curie.pdf", text_pdf)

        assert response.status_code == 201
        assert graph_build_calls == [response.json()["document"]["doc_id"]]

    def test_reupload_returns_same_document(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        text_pdf: bytes,
        graph_build_calls: list[str],
    ):
        first = _upload(client, api_prefix, auth_headers, "curie.pdf", text_pdf)
        second = _upload(client, api_prefix, auth_headers, "copy.pdf", text_pdf)

        assert first.status_code == 201
        assert second.status_code == 201
        doc_id = first.json()["document"]["doc_id"]
        assert second.json()["document"]["doc_id"] == doc_id
        assert second.json()["document"]["status"] == "validated"
        listing = _list(client, api_prefix, auth_headers)
        assert listing["total"] == 1
        # Not indexed yet (the build was stubbed), so the re-upload retries it
        assert graph_build_calls == [doc_id, doc_id]

    def test_pdf_encrypted_with_empty_user_password_accepted(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """Publisher PDFs often carry permission-only encryption; they open
        without a password and must not be rejected as password-protected."""
        response = _upload(
            client, api_prefix, auth_headers, "publisher.pdf", _encrypted_pdf("")
        )

        assert response.status_code == 201
        document = response.json()["document"]
        assert document["status"] == "validated"
        assert document["word_count"] == EXPECTED_WORD_COUNT

    def test_nul_bytes_stripped_from_text(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """PostgreSQL text rejects NUL; it used to cause a 500 and an orphan file."""
        pdf = make_text_pdf(["Hello\x00World neural networks"])

        response = _upload(client, api_prefix, auth_headers, "nul.pdf", pdf)

        assert response.status_code == 201
        assert response.json()["document"]["word_count"] == 3
        doc_id = response.json()["document"]["doc_id"]
        content = client.get(
            f"{api_prefix}/documents/{doc_id}/content", headers=auth_headers
        ).json()["content"]
        assert content == "HelloWorld neural networks"

    @pytest.mark.requires_no_neo4j
    def test_document_stays_validated_when_graph_unavailable(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        """The background build swallows the outage; the upload still succeeds."""
        doc_id = uploaded_document["doc_id"]

        document = client.get(
            f"{api_prefix}/documents/{doc_id}", headers=auth_headers
        ).json()

        assert document["status"] == "validated"
        assert document["graph_nodes"] == []

    @pytest.mark.requires_neo4j
    def test_document_indexed_after_background_graph_build(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        doc_id = uploaded_document["doc_id"]

        document = client.get(
            f"{api_prefix}/documents/{doc_id}", headers=auth_headers
        ).json()

        assert document["status"] == "indexed"
        node_count = client.portal.call(_count_graph_nodes, doc_id)
        assert node_count > 0
        assert len(document["graph_nodes"]) == node_count


class TestUploadRejections:
    """Unprocessable PDFs get 400 with a specific code and leave nothing behind."""

    def _assert_rejected(
        self,
        client: TestClient,
        api_prefix: str,
        headers: dict,
        data: bytes,
        code: str,
    ) -> None:
        files_before = _stored_files()

        response = _upload(client, api_prefix, headers, "upload.pdf", data)

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == code
        assert _stored_files() == files_before
        assert _list(client, api_prefix, headers)["total"] == 0

    def test_corrupt_pdf_rejected_every_time(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """No failed row survives, so a retry is rejected again rather than
        deduplicated into a 201 'success'."""
        corrupt = b"%PDF-1.4 garbage that is not a real PDF"
        self._assert_rejected(client, api_prefix, auth_headers, corrupt, "CORRUPT_PDF")
        self._assert_rejected(client, api_prefix, auth_headers, corrupt, "CORRUPT_PDF")

    def test_broken_page_tree_rejected_as_corrupt(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """pypdf parses lazily: a catalog /Pages pointing at a missing object
        only fails during extraction, and must not escape as a 500."""
        pdf = make_text_pdf().replace(b"/Pages 2 0 R", b"/Pages 9 0 R", 1)
        self._assert_rejected(client, api_prefix, auth_headers, pdf, "CORRUPT_PDF")

    def test_password_protected_pdf_rejected(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        self._assert_rejected(
            client, api_prefix, auth_headers, _encrypted_pdf("secret"), "ENCRYPTED_PDF"
        )

    def test_pdf_without_text_rejected(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        self._assert_rejected(
            client, api_prefix, auth_headers, _blank_pdf(), "NO_TEXT_EXTRACTED"
        )

    def test_rejected_upload_does_not_schedule_graph_build(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        graph_build_calls: list[str],
    ):
        response = _upload(client, api_prefix, auth_headers, "blank.pdf", _blank_pdf())

        assert response.status_code == 400
        assert graph_build_calls == []

    def test_upload_requires_authentication(
        self, client: TestClient, api_prefix: str, text_pdf: bytes
    ):
        response = _upload(client, api_prefix, {}, "curie.pdf", text_pdf)
        assert response.status_code == 401


class TestDocumentOwnership:
    """Documents are visible only to their owner."""

    @pytest.mark.parametrize(
        "method,suffix",
        [
            ("GET", ""),
            ("GET", "/content"),
            ("GET", "/paraphrase"),
            ("DELETE", ""),
        ],
    )
    def test_other_user_gets_404(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        other_auth_headers: dict,
        uploaded_document: dict,
        method: str,
        suffix: str,
    ):
        doc_url = f"{api_prefix}/documents/{uploaded_document['doc_id']}"
        url = f"{doc_url}{suffix}"

        response = client.request(method, url, headers=other_auth_headers)

        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "NOT_FOUND"
        # The owner still has access (and a foreign DELETE removed nothing)
        owner_url = url if method == "GET" else doc_url
        assert client.get(owner_url, headers=auth_headers).status_code == 200

    def test_same_file_uploaded_by_two_users(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        other_auth_headers: dict,
        text_pdf: bytes,
    ):
        """doc_id is per owner, so each user gets (and sees) their own copy."""
        mine = _upload(client, api_prefix, auth_headers, "curie.pdf", text_pdf)
        theirs = _upload(client, api_prefix, other_auth_headers, "curie.pdf", text_pdf)

        assert mine.status_code == 201
        assert theirs.status_code == 201
        my_id = mine.json()["document"]["doc_id"]
        their_id = theirs.json()["document"]["doc_id"]
        assert my_id != their_id

        my_list = _list(client, api_prefix, auth_headers)
        their_list = _list(client, api_prefix, other_auth_headers)
        assert my_list["total"] == 1
        assert [d["doc_id"] for d in my_list["data"]] == [my_id]
        assert their_list["total"] == 1
        assert [d["doc_id"] for d in their_list["data"]] == [their_id]

        # Each user can read their own copy but not the other's
        assert client.get(
            f"{api_prefix}/documents/{their_id}", headers=other_auth_headers
        ).status_code == 200
        assert client.get(
            f"{api_prefix}/documents/{their_id}", headers=auth_headers
        ).status_code == 404


class TestDocumentList:
    """Listing, pagination and status filtering over real documents."""

    def test_list_shows_uploaded_document(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        listing = _list(client, api_prefix, auth_headers)

        assert listing["total"] == 1
        assert [d["doc_id"] for d in listing["data"]] == [uploaded_document["doc_id"]]

    def test_pagination_over_two_documents(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        first = _upload(
            client, api_prefix, auth_headers, "one.pdf", make_text_pdf("First paper text")
        )
        second = _upload(
            client, api_prefix, auth_headers, "two.pdf", make_text_pdf("Second paper text")
        )
        assert first.status_code == 201
        assert second.status_code == 201

        page1 = _list(client, api_prefix, auth_headers, "page=1&page_size=1")
        page2 = _list(client, api_prefix, auth_headers, "page=2&page_size=1")
        page3 = _list(client, api_prefix, auth_headers, "page=3&page_size=1")

        assert page1["total"] == page2["total"] == page3["total"] == 2
        # Newest first
        assert [d["title"] for d in page1["data"]] == ["two"]
        assert [d["title"] for d in page2["data"]] == ["one"]
        assert page3["data"] == []

    def test_status_filter(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        doc_id = uploaded_document["doc_id"]
        # "validated", or "indexed" once the background graph build succeeded
        current = client.get(
            f"{api_prefix}/documents/{doc_id}", headers=auth_headers
        ).json()["status"]

        matching = _list(client, api_prefix, auth_headers, f"status_filter={current}")
        failed = _list(client, api_prefix, auth_headers, "status_filter=failed")

        assert [d["doc_id"] for d in matching["data"]] == [doc_id]
        assert matching["total"] == 1
        assert failed["data"] == []
        assert failed["total"] == 0


class TestParaphrase:
    """GET /documents/{id}/paraphrase for the owner."""

    def test_paraphrase_original_complexity(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        doc_id = uploaded_document["doc_id"]

        response = client.get(
            f"{api_prefix}/documents/{doc_id}/paraphrase?complexity=100",
            headers=auth_headers,
        )
        content = client.get(
            f"{api_prefix}/documents/{doc_id}/content", headers=auth_headers
        ).json()["content"]

        assert response.status_code == 200
        assert response.json() == {
            "doc_id": doc_id,
            "complexity": 100,
            "content": content,
        }

    def test_paraphrase_invalid_complexity(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        response = client.get(
            f"{api_prefix}/documents/{uploaded_document['doc_id']}/paraphrase?complexity=150",
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "INVALID_COMPLEXITY"

    def test_paraphrase_requires_auth(
        self, client: TestClient, api_prefix: str, mock_document_id: str
    ):
        response = client.get(f"{api_prefix}/documents/{mock_document_id}/paraphrase")
        assert response.status_code == 401


class TestDocumentDeletion:
    """DELETE /documents/{id} removes the row, the file and the graph nodes."""

    def test_delete_then_get_returns_404(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        doc_id = uploaded_document["doc_id"]
        file_path = Path(uploaded_document["file_path"])
        assert file_path.exists()

        response = client.delete(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)

        assert response.status_code == 204
        assert response.content == b""
        for suffix in ("", "/content"):
            after = client.get(
                f"{api_prefix}/documents/{doc_id}{suffix}", headers=auth_headers
            )
            assert after.status_code == 404
            assert after.json()["detail"]["code"] == "NOT_FOUND"
        assert _list(client, api_prefix, auth_headers)["total"] == 0
        assert not file_path.exists()

    def test_delete_removes_graph_nodes(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        uploaded_document: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: list[str] = []

        async def record(doc_id: str) -> int:
            calls.append(doc_id)
            return 0

        monkeypatch.setattr(documents_routes, "delete_document_graph", record)
        doc_id = uploaded_document["doc_id"]

        response = client.delete(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)

        assert response.status_code == 204
        assert calls == [doc_id]

    def test_graph_failure_does_not_fail_delete(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        uploaded_document: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        async def broken(doc_id: str) -> int:
            raise RuntimeError("graph exploded")

        monkeypatch.setattr(documents_routes, "delete_document_graph", broken)
        doc_id = uploaded_document["doc_id"]

        response = client.delete(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)

        assert response.status_code == 204
        after = client.get(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)
        assert after.status_code == 404

    def test_foreign_delete_does_not_touch_graph(
        self,
        client: TestClient,
        api_prefix: str,
        other_auth_headers: dict,
        uploaded_document: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: list[str] = []

        async def record(doc_id: str) -> int:
            calls.append(doc_id)
            return 0

        monkeypatch.setattr(documents_routes, "delete_document_graph", record)

        response = client.delete(
            f"{api_prefix}/documents/{uploaded_document['doc_id']}",
            headers=other_auth_headers,
        )

        assert response.status_code == 404
        assert calls == []

    @pytest.mark.requires_neo4j
    def test_delete_removes_graph_nodes_from_neo4j(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        doc_id = uploaded_document["doc_id"]
        assert client.portal.call(_count_graph_nodes, doc_id) > 0

        response = client.delete(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)

        assert response.status_code == 204
        assert client.portal.call(_count_graph_nodes, doc_id) == 0

    @pytest.mark.requires_no_neo4j
    def test_delete_succeeds_when_graph_unavailable(
        self, client: TestClient, api_prefix: str, auth_headers: dict, uploaded_document: dict
    ):
        doc_id = uploaded_document["doc_id"]

        response = client.delete(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)

        assert response.status_code == 204
        after = client.get(f"{api_prefix}/documents/{doc_id}", headers=auth_headers)
        assert after.status_code == 404

    def test_delete_nonexistent_document(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        response = client.delete(
            f"{api_prefix}/documents/nonexistent_doc_id", headers=auth_headers
        )
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "NOT_FOUND"
