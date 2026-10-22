"""
Embedding service using sentence-transformers/all-MiniLM-L6-v2.
Loads the model once and caches it in Streamlit session state.
"""

import logging
import time
from typing import Optional

import numpy as np

from config.settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_NAME
from models.schemas import ChunkRecord

logger = logging.getLogger(__name__)

# Module-level cache so the model survives multiple calls within one process
_model_cache: dict = {}


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Load (or return cached) SentenceTransformer model."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        logger.info("Embedding model loaded successfully.")
        return model
    except Exception as exc:
        logger.error("Failed to load embedding model '%s': %s", model_name, exc)
        raise RuntimeError(
            f"Cannot load embedding model '{model_name}'. "
            "Ensure sentence-transformers is installed and the model name is correct."
        ) from exc


def embed_texts(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 64,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts in batches.
    Returns a list of 384-dimensional float vectors.
    """
    if not texts:
        return []

    model = load_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [emb.tolist() for emb in embeddings]


def embed_query(
    query_text: str,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> list[float]:
    """
    Embed a single query string.
    Uses the same model as document embedding for compatibility.
    """
    results = embed_texts([query_text], model_name=model_name)
    if not results:
        raise ValueError("Empty embedding returned for query.")
    return results[0]


def embed_chunks(
    chunks: list[ChunkRecord],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 64,
) -> tuple[list[list[float]], float]:
    """
    Embed all chunks and return (embeddings_list, duration_seconds).
    Embeddings are in the same order as input chunks.
    """
    texts = [c.chunk_text for c in chunks]
    start = time.time()
    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)
    duration = time.time() - start
    logger.info(
        "Embedded %d chunks in %.2fs using '%s'",
        len(chunks), duration, model_name,
    )
    return embeddings, duration


def validate_embedding_dimensions(
    embedding: list[float],
    expected: int = EMBEDDING_DIMENSIONS,
) -> bool:
    return len(embedding) == expected
