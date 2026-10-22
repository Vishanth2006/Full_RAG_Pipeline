"""
Tests for prompt construction service.
"""

import pytest
from models.schemas import RetrievedChunk
from services.prompt_service import (
    assemble_user_message,
    build_context_block,
    build_prompt_components,
    build_source_citations,
    CONTEXT_DELIMITER_START,
    CONTEXT_DELIMITER_END,
)


def _make_chunk(rank: int, doc_name: str = "doc.pdf", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=f"chunk_{rank:03d}",
        document_id=f"doc{rank}",
        document_name=doc_name,
        page_start=page,
        page_end=page,
        source_page_numbers=[page],
        chunk_text=f"This is the text content of chunk {rank}.",
        cosine_distance=0.1 * rank,
        cosine_similarity=1.0 - 0.1 * rank,
    )


def test_context_block_contains_delimiters():
    chunks = [_make_chunk(1), _make_chunk(2)]
    block = build_context_block(chunks)
    assert CONTEXT_DELIMITER_START in block
    assert CONTEXT_DELIMITER_END in block


def test_context_block_contains_chunk_text():
    chunks = [_make_chunk(1), _make_chunk(2)]
    block = build_context_block(chunks)
    assert "chunk 1" in block
    assert "chunk 2" in block


def test_context_block_contains_source_info():
    chunk = _make_chunk(1, doc_name="report.pdf", page=5)
    block = build_context_block([chunk])
    assert "report.pdf" in block
    assert "5" in block


def test_empty_chunks_context_block():
    block = build_context_block([])
    assert "No relevant document context" in block


def test_user_message_contains_question():
    chunks = [_make_chunk(1)]
    msg = assemble_user_message("What is the answer?", chunks)
    assert "What is the answer?" in msg


def test_user_message_contains_context():
    chunks = [_make_chunk(1)]
    msg = assemble_user_message("Query?", chunks)
    assert CONTEXT_DELIMITER_START in msg
    assert "chunk 1" in msg


def test_prompt_components_keys():
    chunks = [_make_chunk(1)]
    components = build_prompt_components("Question?", chunks)
    assert "system_instruction" in components
    assert "context_block" in components
    assert "user_message" in components
    assert "assembled_readable" in components
    assert "query_text" in components


def test_prompt_components_system_prompt():
    chunks = [_make_chunk(1)]
    custom_system = "Custom system instruction."
    components = build_prompt_components("Q?", chunks, system_prompt=custom_system)
    assert components["system_instruction"] == custom_system


def test_prompt_ordering_system_then_context_then_question():
    chunks = [_make_chunk(1)]
    components = build_prompt_components("Final question?", chunks)
    assembled = components["assembled_readable"]
    sys_pos = assembled.find("SYSTEM INSTRUCTION")
    user_pos = assembled.find("USER MESSAGE")
    assert sys_pos < user_pos


def test_source_citations_structure():
    chunks = [_make_chunk(1, "doc.pdf", 3), _make_chunk(2, "doc.pdf", 7)]
    citations = build_source_citations(chunks)
    assert len(citations) == 2
    for cite in citations:
        assert "rank" in cite
        assert "document_name" in cite
        assert "pages" in cite
        assert "chunk_id" in cite
        assert "similarity" in cite
        assert "chunk_text" in cite


def test_source_citations_page_numbers():
    chunk = _make_chunk(1, page=5)
    citations = build_source_citations([chunk])
    assert "5" in citations[0]["pages"]


def test_no_api_key_in_prompt():
    """Ensure no API key can appear in the prompt even if set."""
    import os
    os.environ["GEMINI_API_KEY"] = "TEST_SECRET_KEY_12345"
    chunks = [_make_chunk(1)]
    components = build_prompt_components("Question?", chunks)
    assert "TEST_SECRET_KEY_12345" not in components["assembled_readable"]
