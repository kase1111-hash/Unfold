"""Tests for the FAISS vector store helpers (app.db.vector)."""

import pytest

from app.db import vector

DIM = 4


def vec(*values: float) -> list[float]:
    return [float(v) for v in values]


@pytest.fixture
async def index():
    """A fresh 4-dim index; the app's own index is restored afterwards."""
    saved = {
        name: getattr(vector, name)
        for name in (
            "_faiss_index",
            "_faiss_id_map",
            "_faiss_metadata",
            "_faiss_deleted",
            "_faiss_dimension",
        )
    }
    await vector.close_faiss()
    await vector.init_faiss(dimension=DIM)
    # Nearest to the query (1, 0, 0, 0) first: a, b, c.
    await vector.faiss_add_vectors(
        [vec(1, 0, 0, 0), vec(0.9, 0.1, 0, 0), vec(0, 1, 0, 0)],
        ["a", "b", "c"],
        [{"doc": "A"}, {"doc": "A"}, {"doc": "B"}],
    )
    yield
    for name, value in saved.items():
        setattr(vector, name, value)


QUERY = vec(1, 0, 0, 0)


async def search_ids(k: int = 10, filter_metadata=None) -> list[str]:
    results = await vector.faiss_search(QUERY, k=k, filter_metadata=filter_metadata)
    return [r["id"] for r in results]


class TestFaissDelete:
    async def test_deleted_vector_not_returned(self, index):
        assert await vector.faiss_delete(["b"]) == 1
        assert await search_ids() == ["a", "c"]

    async def test_overfetches_past_deleted_vectors(self, index):
        """With k=1 and the nearest vector deleted, the next one is returned."""
        await vector.faiss_delete(["a"])
        assert await search_ids(k=1) == ["b"]

    async def test_overfetches_with_metadata_filter(self, index):
        await vector.faiss_delete(["a", "b"])
        assert await search_ids(k=1, filter_metadata={"doc": "B"}) == ["c"]
        assert await search_ids(k=1, filter_metadata={"doc": "A"}) == []

    async def test_vector_without_metadata_can_be_deleted(self, index):
        await vector.faiss_add_vectors([vec(1, 0, 0, 0)], ["plain"])
        assert await vector.faiss_delete(["plain"]) == 1
        assert "plain" not in await search_ids()

    async def test_shared_metadata_dict_only_hides_deleted_id(self, index):
        shared = {"doc": "S"}
        await vector.faiss_add_vectors(
            [vec(0, 0, 1, 0), vec(0, 0, 0.9, 0.1)], ["s1", "s2"], [shared, shared]
        )
        await vector.faiss_delete(["s1"])
        assert await search_ids(filter_metadata={"doc": "S"}) == ["s2"]
        assert shared == {"doc": "S"}

    async def test_unknown_or_repeated_ids_not_counted(self, index):
        assert await vector.faiss_delete(["missing"]) == 0
        assert await vector.faiss_delete(["a", "a"]) == 1
        assert await vector.faiss_delete(["a"]) == 0

    async def test_re_adding_deleted_id_makes_it_searchable(self, index):
        await vector.faiss_delete(["a"])
        await vector.faiss_add_vectors([vec(1, 0, 0, 0)], ["a"], [{"doc": "A"}])
        assert await search_ids(k=1) == ["a"]

    async def test_deletions_survive_save_and_load(self, index, tmp_path):
        await vector.faiss_delete(["a"])
        path = str(tmp_path / "index.faiss")
        await vector.save_faiss_index(path)

        await vector.close_faiss()
        await vector.init_faiss(dimension=DIM, index_path=path)
        assert await search_ids() == ["b", "c"]
