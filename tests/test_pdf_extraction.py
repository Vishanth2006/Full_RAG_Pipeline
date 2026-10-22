"""
Tests for PDF text extraction service.
"""

import pytest
from services.pdf_service import (
    compute_sha256,
    extract_text_from_bytes,
    get_page_count,
    has_any_text,
    summarise_extraction,
)


def test_sha256_is_deterministic(sample_pdf_bytes):
    h1 = compute_sha256(sample_pdf_bytes)
    h2 = compute_sha256(sample_pdf_bytes)
    assert h1 == h2
    assert len(h1) == 64  # hex SHA-256


def test_sha256_differs_for_different_content():
    a = compute_sha256(b"document A content")
    b = compute_sha256(b"document B content")
    assert a != b


def test_get_page_count(sample_pdf_bytes):
    count = get_page_count(sample_pdf_bytes)
    assert count >= 1


def test_extract_text_returns_page_results(sample_pdf_bytes):
    results = extract_text_from_bytes(sample_pdf_bytes, "doc_id_001", "test.pdf")
    assert isinstance(results, list)
    assert len(results) >= 1


def test_page_numbers_are_one_indexed(sample_pdf_bytes):
    results = extract_text_from_bytes(sample_pdf_bytes, "doc_id_001", "test.pdf")
    page_numbers = [r.page_number for r in results]
    assert page_numbers[0] == 1  # 1-indexed


def test_extraction_result_has_required_fields(sample_pdf_bytes):
    results = extract_text_from_bytes(sample_pdf_bytes, "doc_id_001", "test.pdf")
    for r in results:
        assert r.document_id == "doc_id_001"
        assert r.document_name == "test.pdf"
        assert isinstance(r.page_number, int)
        assert isinstance(r.char_count, int)
        assert r.status in ("ok", "empty", "error")


def test_has_any_text_true(sample_pdf_bytes):
    results = extract_text_from_bytes(sample_pdf_bytes, "doc_id_001", "test.pdf")
    assert has_any_text(results)


def test_empty_pdf_bytes_raises():
    with pytest.raises(Exception):
        extract_text_from_bytes(b"not a pdf", "bad_id", "bad.pdf")


def test_summarise_extraction(sample_pdf_bytes):
    results = extract_text_from_bytes(sample_pdf_bytes, "doc_id_001", "test.pdf")
    summary = summarise_extraction(results)
    assert "total_pages" in summary
    assert "ok_pages" in summary
    assert "total_chars" in summary
    assert summary["total_pages"] == len(results)


def test_char_count_matches_text_length(sample_pdf_bytes):
    results = extract_text_from_bytes(sample_pdf_bytes, "doc_id_001", "test.pdf")
    for r in results:
        assert r.char_count == len(r.text)
