import pytest
from unittest.mock import MagicMock
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.rag_schemas import QueryAnalysis

def test_hybrid_retrieval_merging():
    # 1. Setup mock vector store
    mock_vs = MagicMock()
    semantic_result = {
        "content": "semantic match",
        "metadata": {"file_path": "a.py", "name": "foo", "line_start": 1}
    }
    mock_vs.search.return_value = [semantic_result]
    
    # 2. Setup all chunks for keyword search
    keyword_match_chunk = {
        "content": "keyword match",
        "metadata": {"file_path": "b.py", "name": "bar", "line_start": 10, "keywords": ["debug"]}
    }
    all_chunks = [semantic_result, keyword_match_chunk]
    
    retriever = HybridRetriever(mock_vs, all_chunks)
    
    # 3. Setup QueryAnalysis
    analysis = QueryAnalysis(
        error_type=None,
        file_hint=None,
        function_hint="bar", # Should trigger keyword match for 'bar'
        keywords=["debug"],
        raw_query="semantic query"
    )
    
    results = retriever.retrieve(analysis, top_k=5)
    
    # Should contain both
    assert len(results) == 2
    names = {r["metadata"]["name"] for r in results}
    assert "foo" in names
    assert "bar" in names
    
    # Keyword match should be first (based on implementation order)
    assert results[0]["metadata"]["name"] == "bar"

def test_deduplication():
    mock_vs = MagicMock()
    shared_chunk = {
        "content": "shared content",
        "metadata": {"file_path": "c.py", "name": "baz", "line_start": 5, "keywords": ["test"]}
    }
    mock_vs.search.return_value = [shared_chunk]
    all_chunks = [shared_chunk]
    
    retriever = HybridRetriever(mock_vs, all_chunks)
    analysis = QueryAnalysis(
        error_type=None, file_hint=None, function_hint="baz", 
        keywords=["test"], raw_query="shared"
    )
    
    results = retriever.retrieve(analysis, top_k=5)
    
    # Should only have 1 result despite matching both paths
    assert len(results) == 1
