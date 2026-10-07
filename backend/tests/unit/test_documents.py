"""Tests for document endpoints: auth, listing, validation and IDs."""

import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes import documents as documents_routes
from app.repositories.document import DocumentRepository
from app.services.ingestion.document_service import DocumentService

UNKNOWN_DOC_ID = "sha256:" + "0" * 64


def _upload(client: TestClient, api_prefix: str, headers: dict, name: str,
            data: bytes, content_type: str = "application/pdf", **kwargs):
    return client.post(
        f"{api_prefix}/documents/upload",
        files={"file": (name, data, content_type)},
        headers=headers,
        **kwargs,
    )


def _stored_files() -> set[str]:
    upload_dir = DocumentService.UPLOAD_DIR
    return set(p.name for p in upload_dir.iterdir()) if upload_dir.exists() else set()


class TestDocumentAuth:
    """Every document route requires a bearer token."""

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/documents"),
            ("GET", "/documents/"),
            ("GET", f"/documents/{UNKNOWN_DOC_ID}"),
            ("GET", f"/documents/{UNKNOWN_DOC_ID}/content"),
            ("GET", f"/documents/{UNKNOWN_DOC_ID}/paraphrase"),
            ("DELETE", f"/documents/{UNKNOWN_DOC_ID}"),
        ],
    )
    def test_requires_token(
        self, client: TestClient, api_prefix: str, method: str, path: str
    ):
        response = client.request(method, f"{api_prefix}{path}")
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "MISSING_TOKEN"

    def test_upload_requires_token(
        self, client: TestClient, api_prefix: str, text_pdf: bytes
    ):
        response = _upload(client, api_prefix, {}, "curie.pdf", text_pdf)
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "MISSING_TOKEN"

    def test_invalid_token_rejected(self, client: TestClient, api_prefix: str):
        response = client.get(
            f"{api_prefix}/documents",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "INVALID_TOKEN"


class TestDocumentListing:
    """GET /documents: shape, pagination bounds and filters."""

    def test_list_documents_empty(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        response = client.get(f"{api_prefix}/documents/", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "data": [],
            "total": 0,
            "page": 1,
            "page_size": 20,
        }

    def test_list_served_without_trailing_slash(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        """The frontend calls /documents; it must not get a 307 redirect."""
        without = client.get(
            f"{api_prefix}/documents?page=1&page_size=10",
            headers=auth_headers,
            follow_redirects=False,
        )
        with_slash = client.get(
            f"{api_prefix}/documents/?page=1&page_size=10",
            headers=auth_headers,
            follow_redirects=False,
        )
        assert without.status_code == 200
        assert with_slash.status_code == 200
        assert without.json() == with_slash.json()
        assert without.json()["page_size"] == 10

    def test_list_documents_pagination(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        response = client.get(
            f"{api_prefix}/documents/?page=2&page_size=10", headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10
        assert data["data"] == []
        assert data["total"] == 0

    @pytest.mark.parametrize("page_size", [101, 200, 1000])
    def test_list_documents_caps_page_size(
        self, client: TestClient, api_prefix: str, auth_headers: dict, page_size: int
    ):
        response = client.get(
            f"{api_prefix}/documents/?page=1&page_size={page_size}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["page_size"] == 100

    @pytest.mark.parametrize(
        "query,param",
        [
            ("page=0", "page"),
            ("page=-1", "page"),
            ("page_size=0", "page_size"),
            ("page_size=-5", "page_size"),
        ],
    )
    def test_list_documents_rejects_non_positive_paging(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        query: str,
        param: str,
    ):
        """page/page_size below 1 is a 422, not a 500 from a negative OFFSET."""
        response = client.get(f"{api_prefix}/documents/?{query}", headers=auth_headers)
        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["query", param]
        assert error["type"] == "greater_than_equal"

    def test_list_documents_rejects_unknown_status_filter(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        response = client.get(
            f"{api_prefix}/documents/?status_filter=bogus", headers=auth_headers
        )
        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["query", "status_filter"]
        assert error["type"] == "enum"


class TestDocumentNotFound:
    """Unknown document IDs give 404 NOT_FOUND on every per-document route."""

    @pytest.mark.parametrize(
        "method,suffix",
        [
            ("GET", ""),
            ("GET", "/content"),
            ("GET", "/paraphrase"),
            ("DELETE", ""),
        ],
    )
    def test_unknown_document(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        method: str,
        suffix: str,
    ):
        response = client.request(
            method,
            f"{api_prefix}/documents/{UNKNOWN_DOC_ID}{suffix}",
            headers=auth_headers,
        )
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "NOT_FOUND"


class TestUploadValidation:
    """Upload requests rejected before any processing."""

    @pytest.mark.parametrize(
        "name,content_type",
        [
            ("test.txt", "text/plain"),
            ("test.exe", "application/octet-stream"),
            ("book.epub", "application/epub+zip"),
        ],
    )
    def test_upload_unsupported_type(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        name: str,
        content_type: str,
    ):
        files_before = _stored_files()

        response = _upload(
            client, api_prefix, auth_headers, name, b"PK\x03\x04 not a pdf", content_type
        )

        assert response.status_code == 415
        detail = response.json()["detail"]
        assert detail["code"] == "UNSUPPORTED_TYPE"
        assert detail["message"] == (
            f"Unsupported file type: {content_type}. Allowed: PDF"
        )
        assert _stored_files() == files_before
        listing = client.get(f"{api_prefix}/documents", headers=auth_headers).json()
        assert listing["total"] == 0

    def test_upload_empty_file(
        self, client: TestClient, api_prefix: str, auth_headers: dict
    ):
        response = _upload(client, api_prefix, auth_headers, "empty.pdf", b"")

        assert response.status_code == 400
        assert response.json()["detail"] == {
            "code": "EMPTY_FILE",
            "message": "The uploaded file is empty",
        }
        listing = client.get(f"{api_prefix}/documents", headers=auth_headers).json()
        assert listing["total"] == 0

    def test_upload_rejected_from_content_length_header(
        self, client: TestClient, api_prefix: str, auth_headers: dict, text_pdf: bytes
    ):
        """An oversized Content-Length is refused before the body is read.

        The body itself is a small valid PDF, so only the header check can
        produce this 413.
        """
        declared = documents_routes.MAX_UPLOAD_BYTES + 10 * 1024 * 1024
        response = _upload(
            client,
            api_prefix,
            {**auth_headers, "Content-Length": str(declared)},
            "curie.pdf",
            text_pdf,
        )

        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"
        listing = client.get(f"{api_prefix}/documents", headers=auth_headers).json()
        assert listing["total"] == 0

    def test_upload_rejected_after_bounded_read(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        text_pdf: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A file over the limit is refused even when Content-Length passes."""
        monkeypatch.setattr(documents_routes, "MAX_UPLOAD_BYTES", len(text_pdf) - 1)

        response = _upload(client, api_prefix, auth_headers, "curie.pdf", text_pdf)

        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"
        listing = client.get(f"{api_prefix}/documents", headers=auth_headers).json()
        assert listing["total"] == 0

    @staticmethod
    def _chunked_multipart(chunks: int, chunk_size: int, consumed: list[int]):
        """A multipart file body as a generator: httpx sends it chunked,
        without Content-Length. ``consumed`` counts the chunks pulled."""
        boundary = "chunkedboundary"

        def body():
            yield (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"big.pdf\"\r\nContent-Type: application/pdf\r\n\r\n"
            ).encode()
            for _ in range(chunks):
                consumed.append(1)
                yield b"A" * chunk_size
            yield f"\r\n--{boundary}--\r\n".encode()

        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        return body(), headers

    def test_chunked_upload_over_limit_rejected(
        self,
        client: TestClient,
        api_prefix: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """No Content-Length (chunked): the 413 comes from counting the body
        as it is received, before authentication, not after parsing it all
        (which used to answer 401 here, and only after spooling the body)."""
        monkeypatch.setattr(documents_routes, "MAX_UPLOAD_BYTES", 4096)
        monkeypatch.setattr(documents_routes, "MULTIPART_OVERHEAD_BYTES", 1024)
        body, headers = self._chunked_multipart(64, 1024, [])

        response = client.post(
            f"{api_prefix}/documents/upload", content=body, headers=headers
        )

        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"

    def test_chunked_upload_stops_reading_at_limit(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        """Streamed over ASGI, the body is not read past the limit."""
        import httpx

        from app.main import app

        monkeypatch.setattr(documents_routes, "MAX_UPLOAD_BYTES", 4096)
        monkeypatch.setattr(documents_routes, "MULTIPART_OVERHEAD_BYTES", 1024)
        consumed: list[int] = []
        body, headers = self._chunked_multipart(1000, 1024, consumed)

        async def stream_body():
            for part in body:
                # Like a network read, give the server a chance to respond
                await asyncio.sleep(0)
                yield part

        async def post():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as async_client:
                return await async_client.post(
                    "/api/v1/documents/upload", content=stream_body(), headers=headers
                )

        response = client.portal.call(post)

        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"
        # 5 KiB allowed in 1 KiB chunks: the 413 went out right after the
        # limit was crossed, not after the whole 1000 KiB were parsed
        assert len(consumed) < 20, len(consumed)

    def test_upload_at_size_limit_accepted(
        self,
        client: TestClient,
        api_prefix: str,
        auth_headers: dict,
        text_pdf: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(documents_routes, "MAX_UPLOAD_BYTES", len(text_pdf))

        response = _upload(client, api_prefix, auth_headers, "curie.pdf", text_pdf)

        assert response.status_code == 201
        assert response.json()["document"]["file_size_bytes"] == len(text_pdf)


class TestDocIdGeneration:
    """doc_id = "sha256:" + sha256(owner_id + NUL + file bytes)."""

    def test_doc_id_format(self):
        doc_id = DocumentRepository.generate_doc_id(b"%PDF-1.4 bytes", "user-1")
        expected = hashlib.sha256(b"user-1\x00%PDF-1.4 bytes").hexdigest()
        assert doc_id == f"sha256:{expected}"
        assert len(doc_id) == 71

    def test_doc_id_is_deterministic(self):
        first = DocumentRepository.generate_doc_id(b"same bytes", "user-1")
        second = DocumentRepository.generate_doc_id(b"same bytes", "user-1")
        assert first == second

    def test_doc_id_differs_per_owner(self):
        alice = DocumentRepository.generate_doc_id(b"same bytes", "alice")
        bob = DocumentRepository.generate_doc_id(b"same bytes", "bob")
        assert alice != bob

    def test_doc_id_differs_per_content(self):
        original = DocumentRepository.generate_doc_id(b"content", "user-1")
        modified = DocumentRepository.generate_doc_id(b"content modified", "user-1")
        assert original != modified

    def test_owner_and_content_boundary_is_unambiguous(self):
        """The NUL separator keeps ("ab", "c") and ("a", "bc") apart."""
        assert DocumentRepository.generate_doc_id(
            b"c", "ab"
        ) != DocumentRepository.generate_doc_id(b"bc", "a")

    def test_uploaded_document_id_matches_owner_and_bytes(
        self, client: TestClient, api_prefix: str, auth_headers: dict, text_pdf: bytes
    ):
        user_id = client.get(f"{api_prefix}/auth/me", headers=auth_headers).json()[
            "user_id"
        ]

        response = _upload(client, api_prefix, auth_headers, "curie.pdf", text_pdf)

        assert response.status_code == 201
        expected = hashlib.sha256(user_id.encode() + b"\x00" + text_pdf).hexdigest()
        assert response.json()["document"]["doc_id"] == f"sha256:{expected}"
