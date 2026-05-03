import pytest
import os
import json
from unittest.mock import MagicMock
from src.memory.project_memory import ProjectMemory

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    # Return simple deterministic vectors based on input string length or content
    def embed_side_effect(text):
        if "error" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]
    embedder.embed_query.side_effect = embed_side_effect
    return embedder

def test_memory_save_and_load(mock_embedder, tmp_path):
    storage = str(tmp_path / "memory.json")
    memory = ProjectMemory(mock_embedder, storage_path=storage)
    
    memory.add_fix("error 1", "fix 1", "context 1", "reason 1")
    
    # Reload
    new_memory = ProjectMemory(mock_embedder, storage_path=storage)
    assert len(new_memory.memories) == 1
    assert new_memory.memories[0].error_description == "error 1"

def test_similarity_matching(mock_embedder, tmp_path):
    storage = str(tmp_path / "sim.json")
    memory = ProjectMemory(mock_embedder, storage_path=storage)
    
    memory.add_fix("fatal error in system", "fix A", "ctx", "rsn")
    
    # Similar query (mock_embedder returns same vector for "error")
    match = memory.find_similar_fix("system error found", threshold=0.9)
    
    assert match is not None
    assert match.final_fix == "fix A"

def test_no_match(mock_embedder, tmp_path):
    storage = str(tmp_path / "none.json")
    memory = ProjectMemory(mock_embedder, storage_path=storage)
    
    memory.add_fix("error here", "fix", "ctx", "rsn")
    
    # Completely different query
    match = memory.find_similar_fix("something else", threshold=0.9)
    assert match is None
