from __future__ import annotations

import asyncio
import hashlib

import pytest

from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

# A deadlocked call never returns, so every await here is bounded.
_TIMEOUT_SECONDS = 20


class _Note(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class _HashEmbeddingEngine:
    def get_vector_size(self):
        return 16

    def get_batch_size(self):
        return 32

    async def embed_text(self, texts):
        return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


def _make_adapter(tmp_path) -> LanceDBAdapter:
    return LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_HashEmbeddingEngine(),
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_create_data_points_creates_a_missing_collection(tmp_path):
    adapter = _make_adapter(tmp_path)

    await asyncio.wait_for(
        adapter.create_data_points("notes", [_Note(text="x")]), timeout=_TIMEOUT_SECONDS
    )

    collection = await adapter.get_collection("notes")
    assert await collection.count_rows() == 1


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_concurrent_create_data_points_share_one_new_collection(tmp_path):
    adapter = _make_adapter(tmp_path)

    await asyncio.wait_for(
        asyncio.gather(
            adapter.create_data_points("notes", [_Note(text="a")]),
            adapter.create_data_points("notes", [_Note(text="b")]),
        ),
        timeout=_TIMEOUT_SECONDS,
    )

    connection = await adapter.get_connection()
    assert (await connection.list_tables()).tables == ["notes"]
    collection = await adapter.get_collection("notes")
    assert await collection.count_rows() == 2
