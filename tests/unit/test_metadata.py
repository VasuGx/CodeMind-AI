import pytest
from src.rag.indexing.metadata_enricher import MetadataEnricher

def test_metadata_enrichment():
    # Mock chunks produced by SemanticChunker
    chunks = [
        {
            "content": "def helper_func(x):\n    return x + 1",
            "metadata": {
                "name": "helper_func",
                "type": "function",
                "imports": ["import sys", "import os"]
            }
        },
        {
            "content": "def main_func():\n    val = 10\n    return helper_func(val)",
            "metadata": {
                "name": "main_func",
                "type": "function",
                "imports": ["import sys", "import os"]
            }
        }
    ]
    
    enriched = MetadataEnricher.enrich_chunks(chunks)
    
    # Check main_func enrichment
    main = next(c for c in enriched if c["metadata"]["name"] == "main_func")
    
    # Keywords should include variable 'val' and params
    assert "val" in main["metadata"]["keywords"]
    assert "main_func" in main["metadata"]["keywords"]
    
    # Internal calls should detect 'helper_func'
    assert "helper_func" in main["metadata"]["internal_calls"]
    
    # Relevant imports: Since 'os' and 'sys' aren't in the content, they shouldn't be in relevant_imports
    # (Wait, my simple check uses 'in content'. Let's see.)
    assert len(main["metadata"]["relevant_imports"]) == 0

def test_relevant_imports():
    chunks = [
        {
            "content": "import math\ndef calculate(radius):\n    return math.pi * radius",
            "metadata": {
                "name": "calculate",
                "imports": ["import math", "import sys"]
            }
        }
    ]
    enriched = MetadataEnricher.enrich_chunks(chunks)
    calc = enriched[0]
    
    assert "import math" in calc["metadata"]["relevant_imports"]
    assert "import sys" not in calc["metadata"]["relevant_imports"]

def test_syntax_error_handling():
    # A chunk that is invalid python on its own (e.g. part of a class without indent)
    chunks = [
        {
            "content": "    def orphaned_method(self): pass",
            "metadata": {"name": "orphaned_method", "imports": []}
        }
    ]
    enriched = MetadataEnricher.enrich_chunks(chunks)
    # Should not crash, just provide empty metadata
    assert "keywords" in enriched[0]["metadata"]
