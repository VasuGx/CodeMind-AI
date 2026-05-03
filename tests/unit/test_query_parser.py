import pytest
from unittest.mock import MagicMock
from src.rag.parsing.query_parser import QueryParser, QueryAnalysis

def test_query_parser_extraction():
    # Mock LLM and the result of the chain
    mock_llm = MagicMock()
    parser = QueryParser(mock_llm)
    
    expected = QueryAnalysis(
        error_type="ZeroDivisionError",
        file_hint="math_utils.py",
        function_hint="divide",
        keywords=["division", "zero"],
        raw_query="ZeroDivisionError in math_utils.py"
    )
    
    # Directly mock the chain's invoke method
    parser.chain = MagicMock()
    parser.chain.invoke.return_value = expected
    
    raw_input = "Traceback: ZeroDivisionError at math_utils.py line 10 in divide"
    result = parser.parse(raw_input)
    
    assert result.error_type == "ZeroDivisionError"
    assert result.file_hint == "math_utils.py"
    assert "division" in result.keywords

def test_query_parser_fallback():
    # Force an exception to test fallback logic
    mock_llm = MagicMock()
    parser = QueryParser(mock_llm)
    
    # Mock chain to raise exception
    parser.chain = MagicMock()
    parser.chain.invoke.side_effect = Exception("API Down")
    
    result = parser.parse("Something broke")
    
    assert result.error_type == "Unknown"
    assert result.raw_query == "Something broke"
