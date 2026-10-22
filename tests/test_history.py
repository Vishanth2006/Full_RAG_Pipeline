"""
Tests for query history repository.
"""

import pytest
from models.schemas import QueryHistoryRecord, RetrievedChunk
from services.history_service import record_query, reconstruct_retrieved_chunks


@pytest.fixture
def history_env(isolated_settings):
    from storage.history_repository import init_history_db
    init_history_db()
    yield


def _make_retrieved_chunk(rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=f"chunk_{rank}",
        document_id="doc1",
        document_name="doc1.pdf",
        page_start=1, page_end=1,
        source_page_numbers=[1],
        chunk_text=f"Text for chunk {rank}",
        cosine_distance=0.2,
        cosine_similarity=0.8,
    )


def test_save_and_retrieve_query(history_env):
    from storage.history_repository import list_queries

    chunks = [_make_retrieved_chunk(1), _make_retrieved_chunk(2)]
    rec = record_query(
        user_question="What is RAG?",
        system_prompt="You are a helpful assistant.",
        retrieved_chunks=chunks,
        gemini_model="gemini-2.0-flash",
        generated_answer="RAG stands for Retrieval-Augmented Generation.",
        temperature=0.2,
        max_output_tokens=2048,
        top_k=4,
        document_filter=None,
        workspace_doc_ids=["doc1"],
    )

    queries = list_queries()
    assert len(queries) >= 1
    found = next((q for q in queries if q.query_id == rec.query_id), None)
    assert found is not None
    assert found.user_question == "What is RAG?"
    assert found.generated_answer == "RAG stands for Retrieval-Augmented Generation."


def test_context_snapshot_stored(history_env):
    chunks = [_make_retrieved_chunk(1)]
    rec = record_query(
        user_question="Test?",
        system_prompt="sys",
        retrieved_chunks=chunks,
        gemini_model="gemini-2.0-flash",
        generated_answer="Answer.",
        temperature=0.2,
        max_output_tokens=1024,
        top_k=1,
        document_filter=None,
        workspace_doc_ids=["doc1"],
    )
    assert len(rec.retrieved_metadata) == 1
    assert rec.retrieved_metadata[0]["chunk_text"] == "Text for chunk 1"


def test_reconstruct_chunks_from_history(history_env):
    chunks_in = [_make_retrieved_chunk(1), _make_retrieved_chunk(2)]
    rec = record_query(
        user_question="Reconstruct test?",
        system_prompt="sys",
        retrieved_chunks=chunks_in,
        gemini_model="gemini-2.0-flash",
        generated_answer="ok",
        temperature=0.2,
        max_output_tokens=1024,
        top_k=2,
        document_filter=None,
        workspace_doc_ids=["doc1"],
    )

    from storage.history_repository import get_query
    stored = get_query(rec.query_id)
    reconstructed = reconstruct_retrieved_chunks(stored)

    assert len(reconstructed) == 2
    assert reconstructed[0].chunk_id == "chunk_1"
    assert reconstructed[0].chunk_text == "Text for chunk 1"


def test_query_count(history_env):
    from storage.history_repository import count_queries

    before = count_queries()
    for i in range(3):
        record_query(
            user_question=f"Question {i}?",
            system_prompt="sys",
            retrieved_chunks=[_make_retrieved_chunk(1)],
            gemini_model="gemini-2.0-flash",
            generated_answer="ans",
            temperature=0.2,
            max_output_tokens=1024,
            top_k=1,
            document_filter=None,
            workspace_doc_ids=[],
        )
    after = count_queries()
    assert after == before + 3


def test_clear_history(history_env):
    from storage.history_repository import clear_history, count_queries

    record_query(
        user_question="Q?",
        system_prompt="s",
        retrieved_chunks=[_make_retrieved_chunk(1)],
        gemini_model="gemini-2.0-flash",
        generated_answer="A",
        temperature=0.0,
        max_output_tokens=512,
        top_k=1,
        document_filter=None,
        workspace_doc_ids=[],
    )
    clear_history()
    assert count_queries() == 0


def test_document_filter_stored(history_env):
    from storage.history_repository import get_query

    rec = record_query(
        user_question="Filtered?",
        system_prompt="s",
        retrieved_chunks=[_make_retrieved_chunk(1)],
        gemini_model="gemini-2.0-flash",
        generated_answer="A",
        temperature=0.0,
        max_output_tokens=512,
        top_k=1,
        document_filter=["doc1", "doc2"],
        workspace_doc_ids=["doc1", "doc2"],
    )
    stored = get_query(rec.query_id)
    assert stored.document_filter == ["doc1", "doc2"]
