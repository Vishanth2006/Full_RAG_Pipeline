"""
Pydantic-style dataclasses and typed dictionaries for the RAG pipeline.
Using dataclasses for zero extra dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ─── Document ─────────────────────────────────────────────────────────────────

@dataclass
class DocumentRecord:
    """Persisted record for an uploaded and processed PDF."""
    document_id: str           # SHA-256 of file content
    filename: str
    file_size_bytes: int
    page_count: int
    upload_time: str           # ISO-8601 string
    status: str                # pending | processing | indexed | failed | duplicate | empty_text
    chunk_count: int = 0
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = ""
    error_message: str = ""
    processing_time_s: float = 0.0


# ─── Page extraction ──────────────────────────────────────────────────────────

@dataclass
class PageExtractionResult:
    """Text and metadata extracted from a single PDF page."""
    document_id: str
    document_name: str
    page_number: int           # 1-indexed
    text: str
    char_count: int
    status: str                # ok | empty | error
    warning: str = ""


# ─── Chunk ────────────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """A single text chunk produced by the chunking service."""
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page_start: int
    page_end: int
    source_page_numbers: list[int]
    char_count: int
    chunk_text: str
    embedding_model: str
    chunk_size: int = 800
    chunk_overlap: int = 120


# ─── Query / Retrieval ────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk returned from ChromaDB similarity search."""
    rank: int
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    source_page_numbers: list[int]
    chunk_text: str
    cosine_distance: float
    cosine_similarity: float   # 1 - cosine_distance


# ─── Query History ────────────────────────────────────────────────────────────

@dataclass
class QueryHistoryRecord:
    """Full record of a single user query stored in SQLite."""
    query_id: str
    user_question: str
    timestamp: str             # ISO-8601
    system_prompt: str
    retrieved_chunk_ids: list[str]
    retrieved_metadata: list[dict]   # serialised list of RetrievedChunk dicts
    retrieved_scores: list[float]
    gemini_model: str
    generated_answer: str
    temperature: float
    max_output_tokens: int
    top_k: int
    document_filter: Optional[list[str]]   # None = workspace-wide
    workspace_doc_ids: list[str]           # active doc IDs at query time
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
