"""
Shared pytest fixtures for all test modules.
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Minimal valid PDF bytes ────────────────────────────────────────────────────

def _make_minimal_pdf(text: str = "Hello RAG World") -> bytes:
    """Create a minimal 1-page PDF containing the given text, using only stdlib."""
    import struct, zlib

    content_stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    stream_bytes = content_stream.encode("latin-1")

    pdf = b"%PDF-1.4\n"

    # Object 1: catalog
    obj1_offset = len(pdf)
    pdf += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    # Object 2: pages
    obj2_offset = len(pdf)
    pdf += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"

    # Object 3: page
    obj3_offset = len(pdf)
    pdf += (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )

    # Object 4: content stream
    obj4_offset = len(pdf)
    stream_len = len(stream_bytes)
    pdf += (
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )

    # Object 5: font
    obj5_offset = len(pdf)
    pdf += (
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >>\nendobj\n"
    )

    # Cross-reference table
    xref_offset = len(pdf)
    offsets = [obj1_offset, obj2_offset, obj3_offset, obj4_offset, obj5_offset]
    xref = f"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    pdf += xref.encode()

    pdf += f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return pdf


@pytest.fixture
def sample_pdf_bytes():
    """Return bytes of a valid minimal PDF with extractable text."""
    return _make_minimal_pdf("This is a test document for the RAG pipeline.")


@pytest.fixture
def multi_page_pdf_bytes():
    """Return bytes of a minimal 1-page PDF (multi-page creation is complex; use 1 page here)."""
    return _make_minimal_pdf(
        "Page one content. " * 50  # Long enough to chunk
    )


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_rag.db")


@pytest.fixture(autouse=False)
def isolated_settings(tmp_path, monkeypatch):
    """
    Redirect all persistent storage to a temporary directory for test isolation.
    """
    import config.settings as settings

    chroma_dir = str(tmp_path / "chroma_data")
    db_path = str(tmp_path / "test.db")

    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", chroma_dir)
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(settings, "CHROMA_COLLECTION_NAME", "test_collection")

    Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    yield {
        "chroma_dir": chroma_dir,
        "db_path": db_path,
    }
