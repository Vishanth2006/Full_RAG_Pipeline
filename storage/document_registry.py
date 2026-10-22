"""
SQLite-backed document registry.
Tracks all uploaded and processed PDF documents.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from config.settings import SQLITE_DB_PATH, STATUS_DELETED
from models.schemas import DocumentRecord

logger = logging.getLogger(__name__)

# ─── Schema ───────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    filename          TEXT NOT NULL,
    file_size_bytes   INTEGER NOT NULL,
    page_count        INTEGER NOT NULL DEFAULT 0,
    upload_time       TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
    chunk_count       INTEGER NOT NULL DEFAULT 0,
    chunk_size        INTEGER NOT NULL DEFAULT 800,
    chunk_overlap     INTEGER NOT NULL DEFAULT 120,
    embedding_model   TEXT NOT NULL DEFAULT '',
    error_message     TEXT NOT NULL DEFAULT '',
    processing_time_s REAL NOT NULL DEFAULT 0.0
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Create tables if they do not already exist."""
    with _connect() as conn:
        conn.executescript(_DDL)
    logger.info("Document registry initialised at %s", SQLITE_DB_PATH)


# ─── CRUD ─────────────────────────────────────────────────────────────────────

def upsert_document(doc: DocumentRecord) -> None:
    """Insert or replace a document record."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (document_id, filename, file_size_bytes, page_count, upload_time,
                 status, chunk_count, chunk_size, chunk_overlap,
                 embedding_model, error_message, processing_time_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename          = excluded.filename,
                file_size_bytes   = excluded.file_size_bytes,
                page_count        = excluded.page_count,
                upload_time       = excluded.upload_time,
                status            = excluded.status,
                chunk_count       = excluded.chunk_count,
                chunk_size        = excluded.chunk_size,
                chunk_overlap     = excluded.chunk_overlap,
                embedding_model   = excluded.embedding_model,
                error_message     = excluded.error_message,
                processing_time_s = excluded.processing_time_s
            """,
            (
                doc.document_id,
                doc.filename,
                doc.file_size_bytes,
                doc.page_count,
                doc.upload_time,
                doc.status,
                doc.chunk_count,
                doc.chunk_size,
                doc.chunk_overlap,
                doc.embedding_model,
                doc.error_message,
                doc.processing_time_s,
            ),
        )


def get_document(document_id: str) -> Optional[DocumentRecord]:
    """Fetch a single document by its hash ID."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
    return _row_to_record(row) if row else None


def document_exists(document_id: str) -> bool:
    return get_document(document_id) is not None


def list_documents(include_deleted: bool = False) -> list[DocumentRecord]:
    """Return all documents, optionally excluding deleted ones."""
    with _connect() as conn:
        if include_deleted:
            rows = conn.execute("SELECT * FROM documents ORDER BY upload_time DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM documents WHERE status != ? ORDER BY upload_time DESC",
                (STATUS_DELETED,),
            ).fetchall()
    return [_row_to_record(r) for r in rows]


def update_status(
    document_id: str,
    status: str,
    *,
    chunk_count: int = None,
    error_message: str = "",
    processing_time_s: float = 0.0,
) -> None:
    """Partial update for status and derived fields."""
    with _connect() as conn:
        if chunk_count is not None:
            conn.execute(
                """UPDATE documents
                   SET status = ?, chunk_count = ?, error_message = ?,
                       processing_time_s = ?
                   WHERE document_id = ?""",
                (status, chunk_count, error_message, processing_time_s, document_id),
            )
        else:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ? WHERE document_id = ?",
                (status, error_message, document_id),
            )


def mark_deleted(document_id: str) -> None:
    """Soft-delete a document (preserves history references)."""
    update_status(document_id, STATUS_DELETED)


def delete_document(document_id: str) -> None:
    """Hard-delete a document record."""
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def count_documents() -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status != ?", (STATUS_DELETED,)
        ).fetchone()[0]


def count_indexed() -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'indexed'"
        ).fetchone()[0]


# ─── Helper ───────────────────────────────────────────────────────────────────

def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        document_id=row["document_id"],
        filename=row["filename"],
        file_size_bytes=row["file_size_bytes"],
        page_count=row["page_count"],
        upload_time=row["upload_time"],
        status=row["status"],
        chunk_count=row["chunk_count"],
        chunk_size=row["chunk_size"],
        chunk_overlap=row["chunk_overlap"],
        embedding_model=row["embedding_model"],
        error_message=row["error_message"],
        processing_time_s=row["processing_time_s"],
    )
