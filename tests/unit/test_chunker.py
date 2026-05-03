import pytest
import textwrap
from src.rag.indexing.semantic_chunker import SemanticChunker

@pytest.fixture
def chunker():
    return SemanticChunker(max_lines=50)

def test_basic_structure(chunker):
    code = textwrap.dedent("""
        import os
        
        class Database:
            \"\"\"Class docstring\"\"\"
            def connect(self):
                pass
                
        def global_func():
            pass
    """)
    chunks = chunker.chunk_code("test.py", code)
    
    assert len(chunks) == 3
    names = {c["metadata"]["name"] for c in chunks}
    assert "Database" in names
    assert "connect" in names
    assert "global_func" in names
    
    connect_chunk = next(c for c in chunks if c["metadata"]["name"] == "connect")
    assert connect_chunk["metadata"]["parent"] == "Database"
    
    db_chunk = next(c for c in chunks if c["metadata"]["name"] == "Database")
    assert db_chunk["metadata"]["parent"] is None
    # Verify hollowing (the actual implementation should be gone)
    assert "pass" not in db_chunk["content"]
    assert "Nested implementation" in db_chunk["content"]

def test_nested_functions(chunker):
    code = textwrap.dedent("""
        def outer():
            def inner():
                return 42
            return inner()
    """)
    chunks = chunker.chunk_code("nested.py", code)
    
    names = {c["metadata"]["name"] for c in chunks}
    assert "outer" in names
    assert "inner" in names
    
    inner_chunk = next(c for c in chunks if c["metadata"]["name"] == "inner")
    assert inner_chunk["metadata"]["parent"] == "outer"

def test_decorator_handling(chunker):
    code = textwrap.dedent("""
        def decorator(f): return f

        @decorator
        def decorated_func():
            pass
    """)
    chunks = chunker.chunk_code("deco.py", code)
    assert any(c["metadata"]["name"] == "decorated_func" for c in chunks)

def test_async_function(chunker):
    code = textwrap.dedent("""
        async def async_worker():
            pass
    """)
    chunks = chunker.chunk_code("async_test.py", code)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["name"] == "async_worker"

def test_import_extraction(chunker):
    code = textwrap.dedent("""
        import sys
        from os import path
        
        def foo(): pass
    """)
    chunks = chunker.chunk_code("imports.py", code)
    imports = chunks[0]["metadata"]["imports"]
    assert "import sys" in imports
    assert "from os import path" in imports

def test_docstring_extraction(chunker):
    code = textwrap.dedent("""
        def documented():
            \"\"\"This is a test docstring.\"\"\"
            pass
    """)
    chunks = chunker.chunk_code("docs.py", code)
    assert chunks[0]["metadata"]["docstring"] == "This is a test docstring."

def test_syntax_error_fallback(chunker):
    code = "def error_func("
    chunks = chunker.chunk_code("error.py", code)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["name"] == "unknown_due_to_syntax_error"

def test_empty_file(chunker):
    assert chunker.chunk_code("empty.py", "") == []
    assert chunker.chunk_code("empty.py", "  \n  ") == []

def test_no_duplicate_chunks(chunker):
    code = textwrap.dedent("""
        def unique(): pass
        class UniqueClass:
            def unique_method(self): pass
    """)
    chunks = chunker.chunk_code("unique.py", code)
    names = [c["metadata"]["name"] for c in chunks]
    assert len(names) == len(set(names))

def test_large_function_handling():
    # 60 lines
    body = "\n".join([f"    x = {i}" for i in range(60)])
    code = f"def huge_func():\n{body}"
    chunker = SemanticChunker(max_lines=50)
    chunks = chunker.chunk_code("huge.py", code)
    assert len(chunks) == 0

def test_large_class_with_small_methods():
    # Each method is 2 lines. 30 methods = 60 lines + class header
    method_template = "    def m{i}(self):\n        pass"
    methods = "\n".join([method_template.format(i=i) for i in range(30)])
    code = f"class HugeClass:\n{methods}"
    
    chunker = SemanticChunker(max_lines=50)
    chunks = chunker.chunk_code("huge_class.py", code)
    
    names = {c["metadata"]["name"] for c in chunks}
    assert "HugeClass" not in names
    assert "m0" in names
    assert "m29" in names
    assert len(chunks) == 30
