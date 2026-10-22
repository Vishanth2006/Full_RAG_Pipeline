"""
Tests for the retrieval service.
"""

import pytest
from unittest.mock import patch, MagicMock

from services.retrieval_service import retrieve_chunks


@pytest.fixture
def populated_store(isolated_settings):
    """Insert some chunks into an isolated ChromaDB collection."""
    import services.vector_store_service as vs
    vs._client_cache.clear()

    from models.schemas import ChunkRecord
    from services.embedding_service import embed_texts
    from services.vector_store_service import upsert_chunks

    chunks = [
        ChunkRecord(
            chunk_id=f"r_chunk_{i}",
            document_id=f"doc{i%2}",
            document_name=f"doc{i%2}.pdf",
            chunk_index=i,
            page_start=1, page_end=1,
            source_page_numbers=[1],
            char_count=60,
            chunk_text=f"Retrieval test content number {i} with extra words to make unique.",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        for i in range(5)
    ]
    embeddings = embed_texts([c.chunk_text for c in chunks])
    upsert_chunks(chunks, embeddings, collection_name="test_collection")

    yield chunks
    vs._client_cache.clear()


def test_retrieve_returns_results(populated_store):
    results, query_vec, duration = retrieve_chunks(
        "test retrieval content",
        top_k=3,
    )
    assert len(results) <= 3
    assert isinstance(query_vec, list)
    assert len(query_vec) == 384
    assert duration >= 0.0


def test_retrieve_ranks_are_sequential(populated_store):
    results, _, _ = retrieve_chunks("test retrieval", top_k=3)
    for expected_rank, chunk in enumerate(results, start=1):
        assert chunk.rank == expected_rank


def test_retrieve_similarity_is_derived(populated_store):
    results, _, _ = retrieve_chunks("test retrieval", top_k=3)
    for chunk in results:
        assert abs(chunk.cosine_similarity - (1.0 - chunk.cosine_distance)) < 1e-6
        assert 0.0 <= chunk.cosine_similarity <= 1.0


def test_retrieve_with_document_filter(populated_store):
    results, _, _ = retrieve_chunks(
        "test content",
        top_k=5,
        document_ids=["doc0"],
    )
    for chunk in results:
        assert chunk.document_id == "doc0"


def test_retrieve_empty_query_raises():
    with pytest.raises(ValueError, match="empty"):
        retrieve_chunks("", top_k=3)


def test_retrieve_returns_empty_when_db_empty(isolated_settings):
    import services.vector_store_service as vs
    vs._client_cache.clear()

    results, query_vec, _ = retrieve_chunks("anything", top_k=3)
    assert results == []
    assert len(query_vec) == 384

    vs._client_cache.clear()


def test_retrieve_fewer_than_k_chunks_available(populated_store):
    """If fewer than k chunks exist, return what's available."""
    results, _, _ = retrieve_chunks("content", top_k=100)
    assert len(results) <= 5  # only 5 chunks were inserted


def test_retrieved_chunk_metadata(populated_store):
    results, _, _ = retrieve_chunks("test", top_k=1)
    if results:
        r = results[0]
        assert r.chunk_id.startswith("r_chunk_")
        assert r.document_name.endswith(".pdf")
        assert isinstance(r.source_page_numbers, list)
