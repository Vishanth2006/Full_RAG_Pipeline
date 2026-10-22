"""
Tests for ChromaDB vector store service.
Uses isolated temporary directories to avoid polluting production data.
"""

import pytest
from unittest.mock import patch

from models.schemas import ChunkRecord
from services.embedding_service import embed_texts


def _make_chunk(chunk_id: str, doc_id: str = "doc1", idx: int = 0) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_name="test.pdf",
        chunk_index=idx,
        page_start=1,
        page_end=1,
        source_page_numbers=[1],
        char_count=50,
        chunk_text=f"Sample text for chunk {chunk_id}",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


@pytest.fixture
def chroma_env(isolated_settings, tmp_path):
    """Provides an isolated ChromaDB environment."""
    import config.settings as settings
    # Patch the module-level client cache so we get a fresh client
    import services.vector_store_service as vs
    vs._client_cache.clear()
    yield settings
    vs._client_cache.clear()


def test_upsert_and_count(chroma_env):
    from services.vector_store_service import upsert_chunks, get_collection_stats

    chunks = [_make_chunk("chunk_001"), _make_chunk("chunk_002")]
    embeddings = embed_texts([c.chunk_text for c in chunks])

    total = upsert_chunks(chunks, embeddings, collection_name="test_collection")
    stats = get_collection_stats("test_collection")
    assert stats["total_records"] >= 2


def test_upsert_is_idempotent(chroma_env):
    from services.vector_store_service import upsert_chunks, get_collection_stats

    chunk = _make_chunk("chunk_idem")
    emb = embed_texts([chunk.chunk_text])

    upsert_chunks([chunk], emb, collection_name="test_collection")
    upsert_chunks([chunk], emb, collection_name="test_collection")  # repeat

    stats = get_collection_stats("test_collection")
    assert stats["total_records"] == 1  # not duplicated


def test_query_returns_results(chroma_env):
    from services.vector_store_service import upsert_chunks, query_collection

    chunks = [_make_chunk(f"chunk_{i}", idx=i) for i in range(3)]
    embeddings = embed_texts([c.chunk_text for c in chunks])
    upsert_chunks(chunks, embeddings, collection_name="test_collection")

    query_vec = embed_texts(["sample text"])[0]
    result = query_collection(
        query_embedding=query_vec,
        top_k=2,
        collection_name="test_collection",
    )
    assert len(result["ids"][0]) <= 2


def test_delete_document_chunks(chroma_env):
    from services.vector_store_service import upsert_chunks, delete_document_chunks, get_collection_stats

    chunks_a = [_make_chunk("a1", doc_id="docA", idx=0), _make_chunk("a2", doc_id="docA", idx=1)]
    chunks_b = [_make_chunk("b1", doc_id="docB", idx=0)]
    all_chunks = chunks_a + chunks_b
    embeddings = embed_texts([c.chunk_text for c in all_chunks])

    upsert_chunks(all_chunks, embeddings, collection_name="test_collection")
    remaining = delete_document_chunks("docA", collection_name="test_collection")
    assert remaining == 1  # only docB's chunk remains


def test_document_filter_in_query(chroma_env):
    from services.vector_store_service import upsert_chunks, query_collection

    chunk_a = _make_chunk("fa1", doc_id="docX", idx=0)
    chunk_b = _make_chunk("fb1", doc_id="docY", idx=0)
    embeddings = embed_texts([chunk_a.chunk_text, chunk_b.chunk_text])
    upsert_chunks([chunk_a, chunk_b], embeddings, collection_name="test_collection")

    query_vec = embed_texts(["sample text"])[0]
    result = query_collection(
        query_embedding=query_vec,
        top_k=5,
        document_ids=["docX"],
        collection_name="test_collection",
    )
    for meta in result["metadatas"][0]:
        assert meta.get("document_id") == "docX"


def test_empty_database_returns_empty(chroma_env):
    from services.vector_store_service import query_collection

    query_vec = embed_texts(["anything"])[0]
    result = query_collection(
        query_embedding=query_vec,
        top_k=4,
        collection_name="test_collection",
    )
    assert result["ids"] == [[]] or result["ids"][0] == []


def test_cosine_distance_range(chroma_env):
    from services.vector_store_service import upsert_chunks, query_collection

    chunk = _make_chunk("cos_chunk")
    emb = embed_texts([chunk.chunk_text])
    upsert_chunks([chunk], emb, collection_name="test_collection")

    query_vec = embed_texts([chunk.chunk_text])[0]  # identical text
    result = query_collection(query_vec, top_k=1, collection_name="test_collection")
    distances = result.get("distances", [[]])[0]
    if distances:
        d = distances[0]
        # Cosine distance of identical normalised vectors should be very small
        assert 0.0 <= d <= 0.1
        similarity = 1.0 - d
        assert 0.9 <= similarity <= 1.0
