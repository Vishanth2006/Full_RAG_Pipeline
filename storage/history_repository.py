"""
SQLite-backed query history repository.
Stores every query, retrieved context snapshot, and generated answer.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from config.settings import SQLITE_DB_PATH
from models.schemas import QueryHistoryRecord

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS query_history (
    query_id            TEXT PRIMARY KEY,
    user_question       TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    system_prompt       TEXT NOT NULL DEFAULT '',
    retrieved_chunk_ids TEXT NOT NULL DEFAULT '[]',
    retrieved_metadata  TEXT NOT NULL DEFAULT '[]',
    retrieved_scores    TEXT NOT NULL DEFAULT '[]',
    gemini_model        TEXT NOT NULL DEFAULT '',
    generated_answer    TEXT NOT NULL DEFAULT '',
    temperature         REAL NOT NULL DEFAULT 0.2,
    max_output_tokens   INTEGER NOT NULL DEFAULT 2048,
    top_k               INTEGER NOT NULL DEFAULT 4,
    document_filter     TEXT,
    workspace_doc_ids   TEXT NOT NULL DEFAULT '[]',
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL DEFAULT 0
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_history_db() -> None:
    """Create history table if it does not exist."""
    with _connect() as conn:
        conn.executescript(_DDL)
    logger.info("Query history store initialised at %s", SQLITE_DB_PATH)


def save_query(record: QueryHistoryRecord) -> None:
    """Persist a complete query history record."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO query_history
                (query_id, user_question, timestamp, system_prompt,
                 retrieved_chunk_ids, retrieved_metadata, retrieved_scores,
                 gemini_model, generated_answer, temperature, max_output_tokens,
                 top_k, document_filter, workspace_doc_ids,
                 prompt_tokens, completion_tokens, total_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.query_id,
                record.user_question,
                record.timestamp,
                record.system_prompt,
                json.dumps(record.retrieved_chunk_ids),
                json.dumps(record.retrieved_metadata),
                json.dumps(record.retrieved_scores),
                record.gemini_model,
                record.generated_answer,
                record.temperature,
                record.max_output_tokens,
                record.top_k,
                json.dumps(record.document_filter) if record.document_filter else None,
                json.dumps(record.workspace_doc_ids),
                record.prompt_tokens,
                record.completion_tokens,
                record.total_tokens,
            ),
        )


def get_query(query_id: str) -> Optional[QueryHistoryRecord]:
    """Fetch a single history record by ID."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM query_history WHERE query_id = ?", (query_id,)
        ).fetchone()
    return _row_to_record(row) if row else None


def list_queries(limit: int = 100) -> list[QueryHistoryRecord]:
    """Return recent queries, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM query_history ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def delete_query(query_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM query_history WHERE query_id = ?", (query_id,))


def clear_history() -> None:
    """Delete all query history records."""
    with _connect() as conn:
        conn.execute("DELETE FROM query_history")


def count_queries() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]


def generate_query_id() -> str:
    return str(uuid.uuid4())


def _row_to_record(row: sqlite3.Row) -> QueryHistoryRecord:
    doc_filter_raw = row["document_filter"]
    return QueryHistoryRecord(
        query_id=row["query_id"],
        user_question=row["user_question"],
        timestamp=row["timestamp"],
        system_prompt=row["system_prompt"],
        retrieved_chunk_ids=json.loads(row["retrieved_chunk_ids"]),
        retrieved_metadata=json.loads(row["retrieved_metadata"]),
        retrieved_scores=json.loads(row["retrieved_scores"]),
        gemini_model=row["gemini_model"],
        generated_answer=row["generated_answer"],
        temperature=row["temperature"],
        max_output_tokens=row["max_output_tokens"],
        top_k=row["top_k"],
        document_filter=json.loads(doc_filter_raw) if doc_filter_raw else None,
        workspace_doc_ids=json.loads(row["workspace_doc_ids"]),
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        total_tokens=row["total_tokens"],
    )
