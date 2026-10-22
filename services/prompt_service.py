"""
Prompt construction service.
Assembles the exact content sent to the Gemini API, separating:
  - System instruction (SDK system_instruction field)
  - Retrieved context + user question (SDK contents field)
"""

import json
from typing import Optional

from config.settings import DEFAULT_SYSTEM_PROMPT, GEMINI_MODEL
from models.schemas import RetrievedChunk

CONTEXT_DELIMITER_START = "===== DOCUMENT CONTEXT START ====="
CONTEXT_DELIMITER_END   = "===== DOCUMENT CONTEXT END ====="
CHUNK_SEPARATOR         = "--- [Chunk {rank}] ---"


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Build the formatted context string from retrieved chunks."""
    if not chunks:
        return "No relevant document context was retrieved."

    parts = []
    for chunk in chunks:
        pages = ", ".join(str(p) for p in chunk.source_page_numbers)
        header = (
            f"[Source: {chunk.document_name} | "
            f"Page(s): {pages} | "
            f"Chunk ID: {chunk.chunk_id[:8]}... | "
            f"Similarity: {chunk.cosine_similarity:.4f}]"
        )
        parts.append(f"{CHUNK_SEPARATOR.format(rank=chunk.rank)}\n{header}\n{chunk.chunk_text}")

    return (
        f"{CONTEXT_DELIMITER_START}\n"
        + "\n\n".join(parts)
        + f"\n{CONTEXT_DELIMITER_END}"
    )


def assemble_user_message(
    query_text: str,
    chunks: list[RetrievedChunk],
) -> str:
    """
    Build the user-turn message that includes context + the actual question.
    This is sent as the 'contents' part to the Gemini API.
    """
    context_block = build_context_block(chunks)
    return (
        f"Use the following document context to answer the question.\n\n"
        f"{context_block}\n\n"
        f"Question: {query_text}"
    )


def build_prompt_components(
    query_text: str,
    chunks: list[RetrievedChunk],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """
    Return the full prompt in an inspectable structure:
    {
        "system_instruction": str,
        "user_message": str,
        "context_block": str,
        "query_text": str,
        "assembled_readable": str  -- concatenation for display
    }
    """
    context_block = build_context_block(chunks)
    user_message = assemble_user_message(query_text, chunks)

    assembled = (
        f"[SYSTEM INSTRUCTION]\n{system_prompt}\n\n"
        f"[USER MESSAGE]\n{user_message}"
    )

    return {
        "system_instruction": system_prompt,
        "context_block": context_block,
        "user_message": user_message,
        "query_text": query_text,
        "assembled_readable": assembled,
    }


def build_source_citations(chunks: list[RetrievedChunk]) -> list[dict]:
    """Return a list of citation dicts for display in the answer view."""
    citations = []
    for chunk in chunks:
        pages = ", ".join(str(p) for p in chunk.source_page_numbers)
        citations.append(
            {
                "rank": chunk.rank,
                "document_name": chunk.document_name,
                "pages": pages,
                "chunk_id": chunk.chunk_id,
                "similarity": chunk.cosine_similarity,
                "chunk_text": chunk.chunk_text,
            }
        )
    return citations
