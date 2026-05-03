import pytest
from src.rag.engine.embedding import Embedder

def test_embedder_init():
    embedder = Embedder()
    assert embedder.get_embeddings() is not None
