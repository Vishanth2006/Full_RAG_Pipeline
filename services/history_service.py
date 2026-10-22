"""
History service: save and load query history records.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from models.schemas import QueryHistoryRecord, RetrievedChunk
from storage.history_repository import save_query, list_queries, get_query, count_queries

logger = logging.getLogger(__name__)


def record_query(
    user_question: str,
    system_prompt: str,
    retrieved_chunks: list[RetrievedChunk],
    gemini_model: str,
    generated_answer: str,
    temperature: float,
    max_output_tokens: int,
    top_k: int,
    document_filter: Optional[list[str]],
    workspace_doc_ids: list[str],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> QueryHistoryRecord:
    """Build and persist a QueryHistoryRecord, returning it."""
    query_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Serialise retrieved chunks as dicts for storage
    metadata_snapshots = [
        {
            "rank": c.rank,
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_name": c.document_name,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "source_page_numbers": c.source_page_numbers,
            "chunk_text": c.chunk_text,
            "cosine_distance": c.cosine_distance,
            "cosine_similarity": c.cosine_similarity,
        }
        for c in retrieved_chunks
    ]

    record = QueryHistoryRecord(
        query_id=query_id,
        user_question=user_question,
        timestamp=timestamp,
        system_prompt=system_prompt,
        retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
        retrieved_metadata=metadata_snapshots,
        retrieved_scores=[c.cosine_similarity for c in retrieved_chunks],
        gemini_model=gemini_model,
        generated_answer=generated_answer,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_k=top_k,
        document_filter=document_filter,
        workspace_doc_ids=workspace_doc_ids,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    save_query(record)
    logger.info("Saved query history record: %s", query_id)
    return record


def reconstruct_retrieved_chunks(record: QueryHistoryRecord) -> list[RetrievedChunk]:
    """
    Rebuild RetrievedChunk objects from stored metadata snapshot.
    Does NOT re-query the vector database.
    """
    chunks = []
    for meta in record.retrieved_metadata:
        chunks.append(
            RetrievedChunk(
                rank=meta.get("rank", 0),
                chunk_id=meta.get("chunk_id", ""),
                document_id=meta.get("document_id", ""),
                document_name=meta.get("document_name", ""),
                page_start=meta.get("page_start", 1),
                page_end=meta.get("page_end", 1),
                source_page_numbers=meta.get("source_page_numbers", []),
                chunk_text=meta.get("chunk_text", ""),
                cosine_distance=meta.get("cosine_distance", 0.0),
                cosine_similarity=meta.get("cosine_similarity", 0.0),
            )
        )
    return chunks
