import pytest
from src.rag.retrieval.context_ranker import ContextRanker
from src.schemas.rag_schemas import QueryAnalysis

def test_ranking_logic():
    ranker = ContextRanker()
    
    chunks = [
        {"content": "generic code", "metadata": {"name": "other", "file_path": "other.py", "keywords": []}},
        {"content": "file match code", "metadata": {"name": "func1", "file_path": "target.py", "keywords": []}},
        {"content": "perfect match code", "metadata": {"name": "target_func", "file_path": "target.py", "keywords": ["error"]}},
    ]
    
    analysis = QueryAnalysis(
        error_type=None,
        file_hint="target.py",
        function_hint="target_func",
        keywords=["error"],
        raw_query="find error"
    )
    
    ranked = ranker.rank_and_filter(chunks, analysis)
    
    # Perfect match should be first
    assert ranked[0]["metadata"]["name"] == "target_func"
    # File match should be second
    assert ranked[1]["metadata"]["name"] == "func1"
    # Generic should be last
    assert ranked[2]["metadata"]["name"] == "other"

def test_token_filtering():
    ranker = ContextRanker()
    
    # Chunks of 1000 characters each (~250 tokens)
    chunks = [
        {"content": "A" * 1000, "metadata": {"name": "c1", "keywords": ["x"]}},
        {"content": "B" * 1000, "metadata": {"name": "c2", "keywords": ["x"]}},
        {"content": "C" * 1000, "metadata": {"name": "c3", "keywords": ["x"]}},
    ]
    
    analysis = QueryAnalysis(error_type=None, file_hint=None, function_hint=None, keywords=["x"], raw_query="...")
    
    # Limit to roughly 500 tokens (~2000 chars)
    filtered = ranker.rank_and_filter(chunks, analysis, max_tokens=500)
    
    assert len(filtered) == 2
    assert filtered[0]["content"].startswith("A")
    assert filtered[1]["content"].startswith("B")
