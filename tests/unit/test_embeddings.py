import pytest
from unittest.mock import patch, MagicMock
from src.rag.engine.embedding import Embedder

def test_embedder_init():
    # Mock HuggingFaceEmbeddings to avoid actual model loading
    # We patch it where it is IMPORTED in src/rag/engine/embedding.py
    with patch('src.rag.engine.embedding.HuggingFaceEmbeddings') as mock_hf:
        embedder = Embedder(model_name="test-model")
        assert embedder.get_embeddings() is not None
        mock_hf.assert_called_once_with(model_name="test-model")

def test_embedder_mock_behavior():
    # Verify the embedder returns what we expect
    mock_emb_obj = MagicMock()
    with patch('src.rag.engine.embedding.HuggingFaceEmbeddings', return_value=mock_emb_obj):
        embedder = Embedder()
        assert embedder.get_embeddings() == mock_emb_obj
