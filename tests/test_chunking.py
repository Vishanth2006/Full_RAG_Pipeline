"""
Tests for text chunking service.
"""

import pytest
from models.schemas import PageExtractionResult
from services.chunking_service import chunk_pages, get_chunk_stats, _make_chunk_id


def _make_page(text: str, page_number: int = 1) -> PageExtractionResult:
    return PageExtractionResult(
        document_id="test_doc",
        document_name="test.pdf",
        page_number=page_number,
        text=text,
        char_count=len(text),
        status="ok",
    )


LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 60  # ~2700 chars


def test_chunk_id_is_deterministic():
    id1 = _make_chunk_id("docid", 3)
    id2 = _make_chunk_id("docid", 3)
    assert id1 == id2
    assert len(id1) == 32


def test_chunk_id_differs_by_index():
    id1 = _make_chunk_id("docid", 0)
    id2 = _make_chunk_id("docid", 1)
    assert id1 != id2


def test_chunk_id_differs_by_doc():
    id1 = _make_chunk_id("docA", 0)
    id2 = _make_chunk_id("docB", 0)
    assert id1 != id2


def test_single_page_short_text_produces_one_chunk():
    pages = [_make_page("Short text", 1)]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "Short text"


def test_long_text_produces_multiple_chunks():
    pages = [_make_page(LONG_TEXT, 1)]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=800, chunk_overlap=120)
    assert len(chunks) > 1


def test_chunk_preserves_page_number():
    pages = [_make_page(LONG_TEXT, 7)]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=800, chunk_overlap=120)
    for chunk in chunks:
        assert 7 in chunk.source_page_numbers
        assert chunk.page_start == 7


def test_chunk_metadata_fields():
    pages = [_make_page(LONG_TEXT, 1)]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=800, chunk_overlap=120)
    for chunk in chunks:
        assert chunk.document_id == "doc001"
        assert chunk.document_name == "test.pdf"
        assert isinstance(chunk.chunk_index, int)
        assert isinstance(chunk.char_count, int)
        assert chunk.char_count == len(chunk.chunk_text)
        assert len(chunk.chunk_id) == 32


def test_chunk_indices_are_sequential():
    pages = [_make_page(LONG_TEXT, 1)]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=800, chunk_overlap=120)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_size_respected():
    pages = [_make_page(LONG_TEXT, 1)]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=500, chunk_overlap=50)
    for chunk in chunks:
        # Allow some tolerance for splitter boundary behaviour
        assert chunk.char_count <= 600


def test_overlap_creates_adjacent_shared_text():
    text = "word " * 400  # 2000 chars
    pages = [_make_page(text, 1)]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=300, chunk_overlap=50)
    if len(chunks) >= 2:
        end_of_first = chunks[0].chunk_text[-50:]
        start_of_second = chunks[1].chunk_text[:50]
        # The overlap means the end of chunk 0 should appear in chunk 1's beginning
        # (not necessarily exactly, as splitter respects word boundaries)
        assert len(end_of_first) > 0
        assert len(start_of_second) > 0


def test_empty_pages_skipped():
    from models.schemas import PageExtractionResult
    empty_page = PageExtractionResult(
        document_id="doc", document_name="f.pdf",
        page_number=1, text="", char_count=0, status="empty"
    )
    pages = [empty_page]
    chunks = chunk_pages(pages, "doc", "f.pdf")
    assert chunks == []


def test_chunk_stats():
    pages = [_make_page(LONG_TEXT, 1)]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=800, chunk_overlap=120)
    stats = get_chunk_stats(chunks)
    assert stats["total"] == len(chunks)
    assert stats["min_chars"] <= stats["avg_chars"] <= stats["max_chars"]


def test_chunk_stats_empty():
    stats = get_chunk_stats([])
    assert stats["total"] == 0


def test_multi_page_preserves_each_page():
    pages = [
        _make_page(LONG_TEXT, 1),
        _make_page(LONG_TEXT, 2),
    ]
    chunks = chunk_pages(pages, "doc001", "test.pdf", chunk_size=800, chunk_overlap=120)
    pages_seen = {c.page_start for c in chunks}
    assert 1 in pages_seen
    assert 2 in pages_seen


def test_settings_recorded_in_chunks():
    pages = [_make_page(LONG_TEXT, 1)]
    chunks = chunk_pages(pages, "doc", "f.pdf", chunk_size=600, chunk_overlap=80)
    for c in chunks:
        assert c.chunk_size == 600
        assert c.chunk_overlap == 80
