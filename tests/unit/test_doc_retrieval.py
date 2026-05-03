import pytest
from unittest.mock import MagicMock
from src.rag.retrieval.doc_retriever import GlobalRetriever
from src.schemas.rag_schemas import QueryAnalysis

def test_global_retrieval_merging():
    # Mock code retriever
    mock_code = MagicMock()
    mock_code.retrieve.return_value = [
        {"content": "code snippet", "metadata": {"name": "func", "file_path": "a.py"}}
    ]
    
    # Mock doc store
    mock_doc_store = MagicMock()
    mock_doc_doc = MagicMock()
    mock_doc_doc.page_content = "doc snippet"
    mock_doc_doc.metadata = {"source": "manual.md"}
    mock_doc_store.vector_store.similarity_search.return_value = [mock_doc_doc]
    
    retriever = GlobalRetriever(mock_code, mock_doc_store)
    analysis = QueryAnalysis(raw_query="help", keywords=[], error_type=None)
    
    # Weight 0.5/0.5
    results = retriever.retrieve_all(analysis, top_k=5, code_weight=0.5)
    
    assert len(results) == 2
    origins = {r["metadata"]["origin"] for r in results}
    assert "code" in origins
    assert "documentation" in origins

def test_weighting_logic():
    mock_code = MagicMock()
    mock_code.retrieve.return_value = [{"content": "code", "metadata": {"name": "f"}}]
    
    mock_doc_store = MagicMock()
    mock_doc_doc = MagicMock()
    mock_doc_doc.page_content = "doc"
    mock_doc_doc.metadata = {"source": "d"}
    mock_doc_store.vector_store.similarity_search.return_value = [mock_doc_doc]
    
    retriever = GlobalRetriever(mock_code, mock_doc_store)
    analysis = QueryAnalysis(raw_query="test", keywords=[], error_type=None)
    
    # 0.9 code weight -> Code should be first
    results = retriever.retrieve_all(analysis, top_k=2, code_weight=0.9)
    assert results[0]["metadata"]["origin"] == "code"
    
    # 0.1 code weight -> Docs should be first
    results = retriever.retrieve_all(analysis, top_k=2, code_weight=0.1)
    assert results[0]["metadata"]["origin"] == "documentation"
