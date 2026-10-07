"""
Integration tests for the scholar annotation routes.

Annotations are shared per document_id: anyone with the id sees the PUBLIC
ones. Only the author may edit or delete an annotation, and another user's
PRIVATE annotation behaves exactly like a missing one.
"""

import uuid

import pytest


@pytest.fixture
def annotations(api_prefix: str) -> str:
    return f"{api_prefix}/scholar/annotations"


@pytest.fixture
def document_id() -> str:
    """A fresh document id per test (the annotation store is in memory)."""
    return f"sha256:{uuid.uuid4().hex}"


def _create(client, annotations, headers, document_id, content, visibility):
    response = client.post(
        annotations,
        json={"document_id": document_id, "content": content, "visibility": visibility},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _listed(client, annotations, headers, document_id) -> list[tuple[str, str, str]]:
    response = client.get(f"{annotations}/{document_id}", headers=headers)
    assert response.status_code == 200
    return [
        (a["annotation_id"], a["content"], a["visibility"])
        for a in response.json()["annotations"]
    ]


@pytest.fixture
def public_note(client, annotations, auth_headers, document_id) -> dict:
    return _create(
        client, annotations, auth_headers, document_id, "Alice public note", "public"
    )


@pytest.fixture
def private_note(client, annotations, auth_headers, document_id) -> dict:
    return _create(
        client, annotations, auth_headers, document_id, "Alice PRIVATE note", "private"
    )


def _assert_not_found(response, secret: str = "Alice PRIVATE note") -> None:
    assert response.status_code == 404
    assert response.json() == {"detail": "Annotation not found"}
    assert secret not in response.text


class TestAuthorOnlyChanges:
    def test_other_user_cannot_edit_public_annotation(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        public_note,
    ):
        annotation_id = public_note["annotation_id"]
        response = client.put(
            f"{annotations}/{document_id}/{annotation_id}",
            json={"content": "EDITED BY BOB", "visibility": "private"},
            headers=other_auth_headers,
        )
        _assert_not_found(response, secret="EDITED BY BOB")

        expected = [(annotation_id, "Alice public note", "public")]
        assert _listed(client, annotations, auth_headers, document_id) == expected
        assert _listed(client, annotations, other_auth_headers, document_id) == expected

    def test_other_user_cannot_delete_annotation(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        public_note,
    ):
        annotation_id = public_note["annotation_id"]
        response = client.delete(
            f"{annotations}/{document_id}/{annotation_id}", headers=other_auth_headers
        )
        _assert_not_found(response)
        assert _listed(client, annotations, auth_headers, document_id) == [
            (annotation_id, "Alice public note", "public")
        ]

    def test_other_user_cannot_make_private_annotation_public(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        private_note,
    ):
        annotation_id = private_note["annotation_id"]
        response = client.put(
            f"{annotations}/{document_id}/{annotation_id}",
            json={"visibility": "public"},
            headers=other_auth_headers,
        )
        _assert_not_found(response)

        assert _listed(client, annotations, other_auth_headers, document_id) == []
        assert _listed(client, annotations, auth_headers, document_id) == [
            (annotation_id, "Alice PRIVATE note", "private")
        ]

    def test_other_user_cannot_delete_private_annotation(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        private_note,
    ):
        annotation_id = private_note["annotation_id"]
        response = client.delete(
            f"{annotations}/{document_id}/{annotation_id}", headers=other_auth_headers
        )
        _assert_not_found(response)
        assert _listed(client, annotations, auth_headers, document_id) == [
            (annotation_id, "Alice PRIVATE note", "private")
        ]

    def test_author_can_edit_and_delete(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        private_note,
    ):
        annotation_id = private_note["annotation_id"]
        response = client.put(
            f"{annotations}/{document_id}/{annotation_id}",
            json={"content": "Shared now", "visibility": "public", "tags": ["x"]},
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert (body["content"], body["visibility"], body["tags"]) == (
            "Shared now",
            "public",
            ["x"],
        )
        assert _listed(client, annotations, other_auth_headers, document_id) == [
            (annotation_id, "Shared now", "public")
        ]

        response = client.delete(
            f"{annotations}/{document_id}/{annotation_id}", headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json() == {"status": "deleted", "annotation_id": annotation_id}
        assert _listed(client, annotations, auth_headers, document_id) == []

        # A deleted annotation can no longer be changed or deleted again.
        again = client.put(
            f"{annotations}/{document_id}/{annotation_id}",
            json={"content": "back"},
            headers=auth_headers,
        )
        _assert_not_found(again)
        _assert_not_found(
            client.delete(
                f"{annotations}/{document_id}/{annotation_id}", headers=auth_headers
            )
        )

    def test_unknown_annotation_returns_404(
        self, client, annotations, auth_headers, document_id
    ):
        missing = str(uuid.uuid4())
        _assert_not_found(
            client.put(
                f"{annotations}/{document_id}/{missing}",
                json={"content": "x"},
                headers=auth_headers,
            )
        )
        _assert_not_found(
            client.delete(f"{annotations}/{document_id}/{missing}", headers=auth_headers)
        )


class TestReactions:
    def test_reaction_on_other_users_private_annotation_returns_404(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        private_note,
    ):
        annotation_id = private_note["annotation_id"]
        response = client.post(
            f"{annotations}/{document_id}/{annotation_id}/reaction",
            json={"emoji": "x"},
            headers=other_auth_headers,
        )
        _assert_not_found(response)

        # The owner's annotation got no reaction.
        own = client.get(f"{annotations}/{document_id}", headers=auth_headers).json()
        [annotation] = own["annotations"]
        assert annotation["reactions"] == {}

    def test_reaction_on_public_annotation_allowed(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        public_note,
    ):
        annotation_id = public_note["annotation_id"]
        response = client.post(
            f"{annotations}/{document_id}/{annotation_id}/reaction",
            json={"emoji": "+1"},
            headers=other_auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["content"] == "Alice public note"
        assert list(body["reactions"]) == ["+1"]
        assert len(body["reactions"]["+1"]) == 1

    def test_author_can_react_to_own_private_annotation(
        self, client, annotations, auth_headers, document_id, private_note
    ):
        response = client.post(
            f"{annotations}/{document_id}/{private_note['annotation_id']}/reaction",
            json={"emoji": "star"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert list(response.json()["reactions"]) == ["star"]

    def test_reaction_on_deleted_annotation_returns_404(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        public_note,
    ):
        annotation_id = public_note["annotation_id"]
        deleted = client.delete(
            f"{annotations}/{document_id}/{annotation_id}", headers=auth_headers
        )
        assert deleted.status_code == 200
        response = client.post(
            f"{annotations}/{document_id}/{annotation_id}/reaction",
            json={"emoji": "x"},
            headers=other_auth_headers,
        )
        _assert_not_found(response, secret="Alice public note")


class TestVisibility:
    def test_private_annotations_hidden_from_lists_threads_and_stats(
        self, client, annotations, auth_headers, other_auth_headers, document_id,
        private_note, public_note,
    ):
        reply = client.post(
            annotations,
            json={
                "document_id": document_id,
                "content": "Alice private reply",
                "visibility": "private",
                "parent_id": public_note["annotation_id"],
            },
            headers=auth_headers,
        )
        assert reply.status_code == 200

        other_list = client.get(
            f"{annotations}/{document_id}", headers=other_auth_headers
        ).json()
        assert [a["content"] for a in other_list["annotations"]] == ["Alice public note"]

        thread = client.get(
            f"{annotations}/{document_id}/thread/{public_note['annotation_id']}",
            headers=other_auth_headers,
        ).json()
        assert thread["replies"] == []

        stats = client.get(
            f"{annotations}/{document_id}/stats", headers=other_auth_headers
        ).json()
        assert stats["total"] == 1

        own_thread = client.get(
            f"{annotations}/{document_id}/thread/{public_note['annotation_id']}",
            headers=auth_headers,
        ).json()
        assert [r["content"] for r in own_thread["replies"]] == ["Alice private reply"]
