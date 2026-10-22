"""
ChromaDB vector store service.
Manages a persistent cosine-distance collection for all indexed chunks.
"""

import json
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

import config.settings as _settings
from models.schemas import ChunkRecord

logger = logging.getLogger(__name__)

_client_cache: dict = {}


def _get_client() -> chromadb.PersistentClient:
    """Return a singleton persistent ChromaDB client."""
    key = _settings.CHROMA_PERSIST_DIR
    if key not in _client_cache:
        _client_cache[key] = chromadb.PersistentClient(
            path=_settings.CHROMA_PERSIST_DIR,
        )
        logger.info("ChromaDB client initialised at %s", _settings.CHROMA_PERSIST_DIR)
    return _client_cache[key]


def get_collection(collection_name: str = None):
    """Get or create the cosine-distance collection."""
    if collection_name is None:
        collection_name = _settings.CHROMA_COLLECTION_NAME
    client = _get_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_chunks(
    chunks: list[ChunkRecord],
    embeddings: list[list[float]],
    collection_name: str = None,
) -> int:
    """
    Insert chunks and their embeddings into ChromaDB.
    Uses upsert to avoid duplicate insertion (idempotent).
    Returns number of records in collection after operation.
    """
    if not chunks or not embeddings:
        return 0

    if collection_name is None:
        collection_name = _settings.CHROMA_COLLECTION_NAME
    collection = get_collection(collection_name)

    ids = [c.chunk_id for c in chunks]
    texts = [c.chunk_text for c in chunks]
    metadatas = [
        {
            "document_id": c.document_id,
            "document_name": c.document_name,
            "chunk_index": c.chunk_index,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "source_page_numbers": json.dumps(c.source_page_numbers),
            "char_count": c.char_count,
            "embedding_model": c.embedding_model,
            "chunk_size": c.chunk_size,
            "chunk_overlap": c.chunk_overlap,
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    total = collection.count()
    logger.info("Upserted %d chunks. Total collection size: %d", len(chunks), total)
    return total


def query_collection(
    query_embedding: list[float],
    top_k: int = 4,
    document_ids: Optional[list[str]] = None,
    collection_name: str = None,
) -> dict:
    """
    Run a similarity search against the collection.

    Args:
        query_embedding: The query vector.
        top_k: Maximum number of results.
        document_ids: If provided, restrict search to these document IDs.
        collection_name: ChromaDB collection name.

    Returns raw ChromaDB query result dict.
    """
    if collection_name is None:
        collection_name = _settings.CHROMA_COLLECTION_NAME
    collection = get_collection(collection_name)
    total = collection.count()

    if total == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    effective_k = min(top_k, total)

    where_filter = None
    if document_ids:
        if len(document_ids) == 1:
            where_filter = {"document_id": {"$eq": document_ids[0]}}
        else:
            where_filter = {"document_id": {"$in": document_ids}}

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=effective_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    if where_filter:
        kwargs["where"] = where_filter

    try:
        result = collection.query(**kwargs)
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    return result


def delete_document_chunks(
    document_id: str,
    collection_name: str = None,
) -> int:
    """Delete all chunks belonging to a document. Returns remaining count."""
    if collection_name is None:
        collection_name = _settings.CHROMA_COLLECTION_NAME
    collection = get_collection(collection_name)
    results = collection.get(where={"document_id": {"$eq": document_id}})
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d chunks for document_id=%s", len(ids_to_delete), document_id
        )
    return collection.count()


def clear_collection(collection_name: str = None) -> None:
    """Drop and recreate the collection (complete reset)."""
    if collection_name is None:
        collection_name = _settings.CHROMA_COLLECTION_NAME
    client = _get_client()
    client.delete_collection(collection_name)
    get_collection(collection_name)
    logger.info("Collection '%s' cleared and recreated.", collection_name)


def get_collection_stats(collection_name: str = None) -> dict:
    """Return basic stats about the collection."""
    if collection_name is None:
        collection_name = _settings.CHROMA_COLLECTION_NAME
    try:
        collection = get_collection(collection_name)
        total = collection.count()
        return {"collection_name": collection_name, "total_records": total}
    except Exception as exc:
        logger.error("Cannot get collection stats: %s", exc)
        return {"collection_name": collection_name, "total_records": 0}


def get_all_records(
    document_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    collection_name: str = None,
) -> dict:
    """Fetch records for the vector database explorer."""
    if collection_name is None:
        collection_name = _settings.CHROMA_COLLECTION_NAME
    collection = get_collection(collection_name)
    kwargs = dict(include=["documents", "metadatas", "embeddings"])
    if document_id:
        kwargs["where"] = {"document_id": {"$eq": document_id}}

    all_data = collection.get(**kwargs)
    ids = all_data.get("ids", [])
    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])
    embs = all_data.get("embeddings", [])

    # Manual pagination
    sliced_ids = ids[offset : offset + limit]
    sliced_docs = docs[offset : offset + limit]
    sliced_metas = metas[offset : offset + limit] if metas is not None and len(metas) > 0 else [{}] * len(sliced_ids)
    sliced_embs = embs[offset : offset + limit] if embs is not None and len(embs) > 0 else [None] * len(sliced_ids)

    return {
        "ids": sliced_ids,
        "documents": sliced_docs,
        "metadatas": sliced_metas,
        "embeddings": sliced_embs,
        "total": len(ids),
    }


def chunk_id_exists(
    chunk_id: str,
    collection_name: str = None,
) -> bool:
    """Check whether a chunk ID is already stored."""
    if collection_name is None:
        collection_name = _settings.CHROMA_COLLECTION_NAME
    collection = get_collection(collection_name)
    result = collection.get(ids=[chunk_id])
    return len(result.get("ids", [])) > 0
