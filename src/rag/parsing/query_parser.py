from langchain_core.prompts import ChatPromptTemplate
from src.schemas.rag_schemas import QueryAnalysis

class QueryParser:
    def __init__(self, llm):
        """
        Initializes the QueryParser with an LLM that supports structured output.
        """
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior debugging assistant. Your goal is to take raw, messy input "
                       "(like stack traces, error logs, or user descriptions) and turn it into a structured "
                       "query for a RAG system. Extract filenames, function names, error types, and keywords."),
            ("user", "INPUT:\n{raw_input}")
        ])
        self.chain = self.prompt | self.llm.with_structured_output(QueryAnalysis)

    def parse(self, raw_input: str) -> QueryAnalysis:
        """
        Converts raw input into a structured QueryAnalysis object.
        """
        try:
            return self.chain.invoke({"raw_input": raw_input})
        except Exception as e:
            # Fallback for LLM failure
            return QueryAnalysis(
                error_type="Unknown",
                file_hint=None,
                function_hint=None,
                keywords=[],
                raw_query=raw_input[:200]
            )
