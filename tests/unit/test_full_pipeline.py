import pytest
from unittest.mock import MagicMock, patch
from src.rag.production_pipeline import ProductionRAG
from src.schemas.rag_schemas import QueryAnalysis
from langchain_core.embeddings import Embeddings

# Mock Embeddings to keep tests fast
class MockEmbeddings(Embeddings):
    def embed_documents(self, texts): return [[0.1] * 384 for _ in texts]
    def embed_query(self, text): return [0.1] * 384

@pytest.fixture
def mock_pipeline():
    with patch('src.rag.production_pipeline.Embedder') as mock_embedder_class:
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedder.get_embeddings.return_value = MockEmbeddings()
        mock_embedder_class.return_value = mock_embedder
        
        # Initialize pipeline with a mock LLM
        mock_llm = MagicMock()
        pipeline = ProductionRAG(mock_llm)
        
        # Mock the query parser chain
        pipeline.query_parser.chain = MagicMock()
        
        return pipeline

def test_full_pipeline_flow(mock_pipeline):
    # 1. Setup mock repo data
    repo_files = [
        {"path": "src/app.py", "content": "def run():\n    print('hello')"},
        {"path": "src/utils.py", "content": "def helper():\n    pass"}
    ]
    doc_files = [{"source": "readme.md", "content": "Welcome to the app."}]
    
    # 2. Initialize
    mock_pipeline.initialize_repo(repo_files, doc_files)
    
    # 3. Mock Query Analysis result
    mock_analysis = QueryAnalysis(
        error_type=None,
        file_hint="app.py",
        function_hint="run",
        keywords=["hello"],
        raw_query="how to run the app"
    )
    mock_pipeline.query_parser.chain.invoke.return_value = mock_analysis
    
    # 4. Execute Query
    results = mock_pipeline.query("how to run the app")
    
    # 5. Assertions
    assert len(results) > 0
    # The top result should be from app.py because of our ranker scoring (file/function match)
    assert "app.py" in results[0]["metadata"]["file_path"]
    assert results[0]["metadata"]["origin"] == "code"

def test_pipeline_not_initialized(mock_pipeline):
    with pytest.raises(ValueError, match="not initialized"):
        mock_pipeline.query("anything")
