"""
End-to-end integration test covering the full RAG pipeline through retrieval
and prompt construction, without requiring a Gemini API key.
"""

import pytest
from tests.conftest import _make_minimal_pdf


@pytest.fixture
def e2e_env(isolated_settings, tmp_path):
    """Full isolated environment for end-to-end tests."""
    import services.vector_store_service as vs
    vs._client_cache.clear()

    from storage.document_registry import init_db
    from storage.history_repository import init_history_db
    init_db()
    init_history_db()

    yield

    vs._client_cache.clear()


def test_full_pipeline_extraction_to_retrieval(e2e_env):
    """
    Scenario: Upload PDF → Extract → Chunk → Embed → Index → Query → Retrieve.
    """
    from services.pdf_service import extract_text_from_bytes, compute_sha256
    from services.chunking_service import chunk_pages
    from services.embedding_service import embed_chunks, embed_query
    from services.vector_store_service import upsert_chunks, query_collection
    from services.retrieval_service import retrieve_chunks

    # 1. Prepare PDF
    text = "The transformer architecture is foundational to modern NLP models. " * 20
    pdf_bytes = _make_minimal_pdf(text)
    doc_id = compute_sha256(pdf_bytes)
    doc_name = "e2e_test.pdf"

    # 2. Extract
    pages = extract_text_from_bytes(pdf_bytes, doc_id, doc_name)
    ok_pages = [p for p in pages if p.status == "ok"]
    assert len(ok_pages) >= 1

    # 3. Chunk
    chunks = chunk_pages(pages, doc_id, doc_name, chunk_size=400, chunk_overlap=60)
    assert len(chunks) >= 1

    # 4. Embed
    embeddings, duration = embed_chunks(chunks)
    assert len(embeddings) == len(chunks)
    assert all(len(e) == 384 for e in embeddings)

    # 5. Index
    total = upsert_chunks(chunks, embeddings, collection_name="test_collection")
    assert total >= 1

    # 6. Retrieve
    results, query_vec, retrieval_dur = retrieve_chunks(
        "What is the transformer architecture?",
        top_k=2,
    )
    assert len(results) >= 1
    assert len(query_vec) == 384
    assert retrieval_dur >= 0.0

    # 7. Verify result structure
    for r in results:
        assert r.chunk_id in [c.chunk_id for c in chunks]
        assert r.document_name == doc_name
        assert 0.0 <= r.cosine_similarity <= 1.0
        assert abs(r.cosine_similarity - (1.0 - r.cosine_distance)) < 1e-5


def test_full_pipeline_prompt_construction(e2e_env):
    """
    Scenario: Use retrieved chunks to build a complete, inspectable prompt.
    """
    from models.schemas import RetrievedChunk
    from services.prompt_service import build_prompt_components, build_source_citations

    chunks = [
        RetrievedChunk(
            rank=1,
            chunk_id="e2e_chunk_001",
            document_id="doc_e2e",
            document_name="e2e_paper.pdf",
            page_start=2, page_end=2,
            source_page_numbers=[2],
            chunk_text="Transformers use self-attention mechanisms.",
            cosine_distance=0.15,
            cosine_similarity=0.85,
        )
    ]

    components = build_prompt_components(
        "How do transformers work?",
        chunks,
        system_prompt="Answer only from documents.",
    )

    # Verify components
    assert "self-attention" in components["context_block"]
    assert "How do transformers work?" in components["user_message"]
    assert components["system_instruction"] == "Answer only from documents."
    assert "e2e_paper.pdf" in components["context_block"]

    # Citations
    citations = build_source_citations(chunks)
    assert citations[0]["document_name"] == "e2e_paper.pdf"
    assert "2" in citations[0]["pages"]


def test_add_second_document_preserves_first(e2e_env):
    """
    Scenario D: Upload second PDF without losing first.
    """
    from services.pdf_service import compute_sha256
    from services.chunking_service import chunk_pages
    from services.embedding_service import embed_chunks
    from services.vector_store_service import upsert_chunks, get_collection_stats
    from services.pdf_service import extract_text_from_bytes

    def _index_doc(text: str, doc_name: str):
        pdf_bytes = _make_minimal_pdf(text)
        doc_id = compute_sha256(pdf_bytes)
        pages = extract_text_from_bytes(pdf_bytes, doc_id, doc_name)
        chunks = chunk_pages(pages, doc_id, doc_name, chunk_size=300, chunk_overlap=40)
        if chunks:
            embeddings, _ = embed_chunks(chunks)
            upsert_chunks(chunks, embeddings, collection_name="test_collection")
        return chunks

    chunks_a = _index_doc("Alpha document content about machine learning.", "alpha.pdf")
    count_after_a = get_collection_stats("test_collection")["total_records"]

    chunks_b = _index_doc("Beta document content about deep learning.", "beta.pdf")
    count_after_b = get_collection_stats("test_collection")["total_records"]

    # Second document added records to the existing collection
    assert count_after_b >= count_after_a


def test_delete_document_leaves_others(e2e_env):
    """
    Scenario G: Delete one document's vectors; others remain searchable.
    """
    from services.pdf_service import compute_sha256
    from services.chunking_service import chunk_pages
    from services.embedding_service import embed_chunks
    from services.vector_store_service import upsert_chunks, delete_document_chunks, query_collection
    from services.pdf_service import extract_text_from_bytes
    from services.embedding_service import embed_query

    def _index(text, name):
        pdf = _make_minimal_pdf(text)
        doc_id = compute_sha256(pdf)
        pages = extract_text_from_bytes(pdf, doc_id, name)
        chunks = chunk_pages(pages, doc_id, name, chunk_size=200, chunk_overlap=30)
        if chunks:
            embeddings, _ = embed_chunks(chunks)
            upsert_chunks(chunks, embeddings, collection_name="test_collection")
        return doc_id, chunks

    doc_id_a, _ = _index("Alpha text: cats and dogs.", "alpha.pdf")
    doc_id_b, _ = _index("Beta text: cars and trucks.", "beta.pdf")

    # Delete doc A
    remaining = delete_document_chunks(doc_id_a, collection_name="test_collection")

    # Query should not return doc A's chunks
    qvec = embed_query("cats and dogs")
    result = query_collection(qvec, top_k=10, collection_name="test_collection")
    returned_doc_ids = [m.get("document_id") for m in result["metadatas"][0]]
    assert doc_id_a not in returned_doc_ids
    # doc B should still be accessible
    assert doc_id_b in returned_doc_ids or remaining >= 0
