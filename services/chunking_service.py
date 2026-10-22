"""
Text chunking service using LangChain's RecursiveCharacterTextSplitter.
Preserves page-level source information and generates deterministic chunk IDs.
"""

import hashlib
import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    CHUNK_SEPARATORS,
    EMBEDDING_MODEL_NAME,
)
from models.schemas import ChunkRecord, PageExtractionResult

logger = logging.getLogger(__name__)


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Deterministic chunk ID based on document hash and position."""
    raw = f"{document_id}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def chunk_pages(
    pages: list[PageExtractionResult],
    document_id: str,
    document_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> list[ChunkRecord]:
    """
    Split extracted page text into overlapping chunks while preserving page numbers.

    Strategy:
    1. Process each page independently.
    2. For pages whose text fits within chunk_size, create a single chunk.
    3. For pages whose text exceeds chunk_size, apply RecursiveCharacterTextSplitter.
    4. Each chunk records the exact source page it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
        length_function=len,
    )

    chunks: list[ChunkRecord] = []
    global_index = 0

    # Filter to pages that have extractable text
    ok_pages = [p for p in pages if p.status == "ok" and p.char_count > 0]

    for page in ok_pages:
        page_num = page.page_number
        texts = splitter.split_text(page.text)

        for text in texts:
            if not text.strip():
                continue

            chunk_id = _make_chunk_id(document_id, global_index)
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=document_name,
                    chunk_index=global_index,
                    page_start=page_num,
                    page_end=page_num,
                    source_page_numbers=[page_num],
                    char_count=len(text),
                    chunk_text=text,
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
            global_index += 1

    logger.info(
        "Chunked '%s' → %d chunks (size=%d, overlap=%d)",
        document_name, len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def get_chunk_stats(chunks: list[ChunkRecord]) -> dict:
    """Aggregate statistics for chunk explorer display."""
    if not chunks:
        return {"total": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0}
    char_counts = [c.char_count for c in chunks]
    return {
        "total": len(chunks),
        "avg_chars": int(sum(char_counts) / len(char_counts)),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
    }
