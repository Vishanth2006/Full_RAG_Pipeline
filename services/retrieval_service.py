"""
Retrieval service: embeds the user query and fetches top-k similar chunks.
"""

import json
import logging
import time
from typing import Optional

from config.settings import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
from models.schemas import RetrievedChunk
from services.embedding_service import embed_query
from services.vector_store_service import query_collection

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    document_ids: Optional[list[str]] = None,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> tuple[list[RetrievedChunk], list[float], float]:
    """
    Embed the query and retrieve the top-k most relevant chunks.

    Returns:
        (retrieved_chunks, query_embedding_vector, retrieval_duration_seconds)
    """
    if not query_text.strip():
        raise ValueError("Query text must not be empty.")

    # Step 1: embed query
    query_vec = embed_query(query_text, model_name=embedding_model)

    # Step 2: search ChromaDB
    t0 = time.time()
    raw = query_collection(
        query_embedding=query_vec,
        top_k=top_k,
        document_ids=document_ids,
    )
    duration = time.time() - t0

    # Step 3: parse results
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: list[RetrievedChunk] = []
    for rank, (chunk_id, text, meta, dist) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        cosine_dist = float(dist)
        cosine_sim = max(0.0, 1.0 - cosine_dist)  # clamp to [0,1]

        source_pages_raw = meta.get("source_page_numbers", "[]")
        try:
            source_pages = json.loads(source_pages_raw)
        except (json.JSONDecodeError, TypeError):
            source_pages = [meta.get("page_start", 1)]

        results.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                document_name=meta.get("document_name", ""),
                page_start=int(meta.get("page_start", 1)),
                page_end=int(meta.get("page_end", 1)),
                source_page_numbers=source_pages,
                chunk_text=text,
                cosine_distance=cosine_dist,
                cosine_similarity=cosine_sim,
            )
        )

    logger.info(
        "Retrieved %d chunks for query in %.3fs", len(results), duration
    )
    return results, query_vec, duration
