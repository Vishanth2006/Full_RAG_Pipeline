"""
Tests for the embedding service.
"""

import pytest
from services.embedding_service import (
    embed_query,
    embed_texts,
    validate_embedding_dimensions,
)
from config.settings import EMBEDDING_DIMENSIONS


def test_embed_single_text():
    vectors = embed_texts(["Hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIMENSIONS


def test_embed_multiple_texts():
    texts = ["First document.", "Second document.", "Third document."]
    vectors = embed_texts(texts)
    assert len(vectors) == len(texts)
    for v in vectors:
        assert len(v) == EMBEDDING_DIMENSIONS


def test_embedding_dimensions_are_384():
    vec = embed_query("What is the meaning of life?")
    assert len(vec) == 384


def test_embed_query_returns_list_of_floats():
    vec = embed_query("Test query")
    assert isinstance(vec, list)
    assert all(isinstance(v, float) for v in vec)


def test_embedding_is_normalised():
    """Normalised embeddings should have L2 norm ≈ 1.0."""
    import math
    vec = embed_query("Normalisation test")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 0.01


def test_embed_empty_list():
    result = embed_texts([])
    assert result == []


def test_validate_embedding_dimensions_ok():
    vec = [0.0] * 384
    assert validate_embedding_dimensions(vec, 384) is True


def test_validate_embedding_dimensions_fail():
    vec = [0.0] * 100
    assert validate_embedding_dimensions(vec, 384) is False


def test_same_text_same_embedding():
    """Deterministic: same input should produce the same output."""
    v1 = embed_query("Hello RAG")
    v2 = embed_query("Hello RAG")
    assert v1 == v2


def test_different_texts_different_embeddings():
    v1 = embed_query("cats are animals")
    v2 = embed_query("the stock market rose today")
    # Cosine similarity should be < 1
    dot = sum(a * b for a, b in zip(v1, v2))
    assert dot < 0.99


def test_embedding_model_caching():
    """Calling load_embedding_model twice should return the same object."""
    from services.embedding_service import load_embedding_model, _model_cache
    from config.settings import EMBEDDING_MODEL_NAME
    model_a = load_embedding_model(EMBEDDING_MODEL_NAME)
    model_b = load_embedding_model(EMBEDDING_MODEL_NAME)
    assert model_a is model_b
