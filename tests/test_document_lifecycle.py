"""
Tests for document lifecycle: upload, duplicate detection, deletion, persistence.
"""

import pytest
from datetime import datetime, timezone


def _make_pdf_bytes(seed: str = "default") -> bytes:
    """Generate distinct fake PDF bytes for each seed."""
    # Use conftest helper indirectly via a factory
    from tests.conftest import _make_minimal_pdf
    return _make_minimal_pdf(f"Document content: {seed}")


@pytest.fixture
def registry_env(isolated_settings):
    """Isolated SQLite registry."""
    from storage.document_registry import init_db
    import config.settings as s
    # Re-init with the isolated db path
    import storage.document_registry as dr
    dr.SQLITE_DB_PATH = s.SQLITE_DB_PATH if hasattr(s, 'SQLITE_DB_PATH') else s.SQLITE_DB_PATH
    init_db()
    yield


def test_new_document_registered(registry_env):
    from services.document_service import prepare_document_record
    from services.pdf_service import compute_sha256

    pdf = _make_pdf_bytes("unique_doc_A")
    record, is_dup = prepare_document_record("docA.pdf", pdf)
    assert not is_dup
    assert record.status == "pending"
    assert len(record.document_id) == 64


def test_duplicate_detected(registry_env):
    from services.document_service import prepare_document_record

    pdf = _make_pdf_bytes("duplicate_test")
    record1, is_dup1 = prepare_document_record("first.pdf", pdf)
    assert not is_dup1

    record2, is_dup2 = prepare_document_record("second_copy.pdf", pdf)
    assert is_dup2
    assert record2.document_id == record1.document_id


def test_different_files_different_ids(registry_env):
    from services.document_service import prepare_document_record

    pdf_a = _make_pdf_bytes("contentA")
    pdf_b = _make_pdf_bytes("contentB")

    rec_a, _ = prepare_document_record("a.pdf", pdf_a)
    rec_b, _ = prepare_document_record("b.pdf", pdf_b)
    assert rec_a.document_id != rec_b.document_id


def test_upload_validation_rejects_non_pdf(registry_env):
    from services.document_service import validate_upload
    ok, err = validate_upload("file.txt", b"some text content here")
    assert not ok
    assert "PDF" in err


def test_upload_validation_rejects_oversized(registry_env):
    from services.document_service import validate_upload
    big = b"X" * (60 * 1024 * 1024)  # 60 MB
    ok, err = validate_upload("big.pdf", big, max_mb=50)
    assert not ok
    assert "exceeds" in err.lower()


def test_upload_validation_accepts_valid(registry_env):
    from services.document_service import validate_upload
    pdf = _make_pdf_bytes("valid")
    ok, err = validate_upload("doc.pdf", pdf)
    assert ok
    assert err == ""


def test_document_persists_in_registry(registry_env):
    from services.document_service import prepare_document_record
    from storage.document_registry import get_document

    pdf = _make_pdf_bytes("persist_test")
    record, _ = prepare_document_record("persist.pdf", pdf)
    fetched = get_document(record.document_id)
    assert fetched is not None
    assert fetched.filename == "persist.pdf"


def test_delete_one_document_leaves_others(registry_env):
    from services.document_service import prepare_document_record
    from storage.document_registry import mark_deleted, list_documents

    pdf_a = _make_pdf_bytes("A")
    pdf_b = _make_pdf_bytes("B")

    rec_a, _ = prepare_document_record("a.pdf", pdf_a)
    rec_b, _ = prepare_document_record("b.pdf", pdf_b)

    mark_deleted(rec_a.document_id)

    docs = list_documents(include_deleted=False)
    doc_ids = [d.document_id for d in docs]
    assert rec_a.document_id not in doc_ids
    assert rec_b.document_id in doc_ids


def test_status_update(registry_env):
    from services.document_service import prepare_document_record
    from storage.document_registry import update_status, get_document

    pdf = _make_pdf_bytes("status_test")
    record, _ = prepare_document_record("s.pdf", pdf)
    update_status(record.document_id, "indexed", chunk_count=42)
    fetched = get_document(record.document_id)
    assert fetched.status == "indexed"
    assert fetched.chunk_count == 42
