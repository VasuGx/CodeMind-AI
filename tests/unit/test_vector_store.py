import pytest
from unittest.mock import MagicMock
from src.rag.engine.vector_store import CodeVectorStore
from langchain_core.embeddings import Embeddings

class MockEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Return a list of fake 384-dim vectors
        return [[0.1] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 384

def test_vector_store_add_and_search():
    # Use mock embeddings to avoid network/downloads
    mock_emb = MockEmbeddings()
    store = CodeVectorStore(mock_emb)
    
    chunks = [
        {
            "content": "def foo(): pass",
            "metadata": {"name": "foo", "type": "function", "parent": None}
        },
        {
            "content": "class Bar: pass",
            "metadata": {"name": "Bar", "type": "class", "parent": None}
        }
    ]
    
    store.add_chunks(chunks)
    
    # Search
    results = store.search("function foo", top_k=1)
    
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "foo"
    assert "def foo()" in results[0]["content"]

def test_metadata_preservation():
    mock_emb = MockEmbeddings()
    store = CodeVectorStore(mock_emb)
    
    chunks = [
        {
            "content": "import sys",
            "metadata": {"name": "imports", "relevant_imports": ["import sys"]}
        }
    ]
    
    store.add_chunks(chunks)
    results = store.search("imports", top_k=1)
    
    assert results[0]["metadata"]["relevant_imports"] == ["import sys"]
