"""
Document orchestration service.
Coordinates the full ingestion pipeline:
  upload validation → extraction → chunking → embedding → indexing
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    MAX_UPLOAD_SIZE_MB,
    STATUS_DUPLICATE,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_PENDING,
    STATUS_PROCESSING,
)
from models.schemas import DocumentRecord, PageExtractionResult, ChunkRecord
from services.chunking_service import chunk_pages
from services.embedding_service import embed_chunks
from services.pdf_service import (
    compute_sha256,
    extract_text_from_bytes,
    get_page_count,
    has_any_text,
)
from services.vector_store_service import upsert_chunks
from storage.document_registry import (
    document_exists,
    upsert_document,
    update_status,
)

logger = logging.getLogger(__name__)


def validate_upload(
    filename: str,
    file_bytes: bytes,
    max_mb: int = MAX_UPLOAD_SIZE_MB,
) -> tuple[bool, str]:
    """
    Basic upload validation.
    Returns (is_valid, error_message).
    """
    if not filename.lower().endswith(".pdf"):
        return False, f"'{filename}' is not a PDF file."
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"File size {size_mb:.1f} MB exceeds limit of {max_mb} MB."
    if len(file_bytes) < 5:
        return False, f"File '{filename}' appears to be empty or corrupt."
    return True, ""


def prepare_document_record(
    filename: str,
    file_bytes: bytes,
) -> tuple[DocumentRecord, bool]:
    """
    Compute document ID and build initial record.
    Returns (record, is_duplicate).
    """
    doc_id = compute_sha256(file_bytes)
    page_count = get_page_count(file_bytes)
    now_iso = datetime.now(timezone.utc).isoformat()

    existing = document_exists(doc_id)
    status = STATUS_DUPLICATE if existing else STATUS_PENDING

    record = DocumentRecord(
        document_id=doc_id,
        filename=filename,
        file_size_bytes=len(file_bytes),
        page_count=page_count,
        upload_time=now_iso,
        status=status,
    )

    if not existing:
        upsert_document(record)

    return record, existing


def ingest_document(
    filename: str,
    file_bytes: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    progress_callback=None,
) -> tuple[DocumentRecord, list[PageExtractionResult], list[ChunkRecord]]:
    """
    Run the full ingestion pipeline for a single document.

    progress_callback: optional callable(step: str, pct: float)
    Returns (final_record, pages, chunks)
    """

    def _progress(step: str, pct: float):
        if progress_callback:
            progress_callback(step, pct)

    doc_id = compute_sha256(file_bytes)
    page_count = get_page_count(file_bytes)
    now_iso = datetime.now(timezone.utc).isoformat()

    # --- Record ---
    record = DocumentRecord(
        document_id=doc_id,
        filename=filename,
        file_size_bytes=len(file_bytes),
        page_count=page_count,
        upload_time=now_iso,
        status=STATUS_PROCESSING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    upsert_document(record)
    _progress("Extracting text…", 0.1)

    t_start = time.time()

    try:
        # Step 1: Extract
        pages = extract_text_from_bytes(file_bytes, doc_id, filename)
        _progress("Chunking text…", 0.35)

        if not has_any_text(pages):
            update_status(
                doc_id,
                STATUS_EMPTY,
                error_message="No extractable text found in document.",
            )
            record.status = STATUS_EMPTY
            return record, pages, []

        # Step 2: Chunk
        chunks = chunk_pages(
            pages,
            document_id=doc_id,
            document_name=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )
        _progress("Generating embeddings…", 0.55)

        if not chunks:
            update_status(
                doc_id,
                STATUS_EMPTY,
                error_message="Chunking produced zero chunks.",
            )
            record.status = STATUS_EMPTY
            return record, pages, []

        # Step 3: Embed
        embeddings, embed_dur = embed_chunks(
            chunks, model_name=embedding_model
        )
        _progress("Indexing vectors…", 0.80)

        # Step 4: Store in ChromaDB
        upsert_chunks(chunks, embeddings)
        _progress("Done", 1.0)

        elapsed = time.time() - t_start
        update_status(
            doc_id,
            STATUS_INDEXED,
            chunk_count=len(chunks),
            processing_time_s=elapsed,
        )
        record.status = STATUS_INDEXED
        record.chunk_count = len(chunks)
        record.processing_time_s = elapsed
        upsert_document(record)

        return record, pages, chunks

    except Exception as exc:
        elapsed = time.time() - t_start
        logger.error("Ingestion failed for '%s': %s", filename, exc, exc_info=True)
        update_status(
            doc_id,
            STATUS_FAILED,
            error_message=str(exc),
            processing_time_s=elapsed,
        )
        record.status = STATUS_FAILED
        record.error_message = str(exc)
        raise
