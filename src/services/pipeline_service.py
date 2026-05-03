"""
Module for routing user queries to the appropriate agent chain based on mode (Debug, Explain, Impact).
"""
from typing import Dict, Any, Union
from src.agents.management.orchestrator import DebuggingOrchestrator
from src.schemas.api_schemas import DebugResponse, ExplainResponse, ImpactResponse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Explanation(BaseModel):
    """Schema for architecture summary."""
    summary: str
    key_modules: list
    data_flow: str
    important_files: list

class Impact(BaseModel):
    """Schema for dependency impact analysis."""
    affected_files: list
    dependent_functions: list
    risk_level: str
    explanation: str

class PipelineService:
    """
    Acts as a high-level router for the AI system.
    Injects RAG context and selects the specialized agent logic for each request.
    """
    def __init__(self, orchestrator: DebuggingOrchestrator, llm):
        self.orchestrator = orchestrator
        self.llm = llm

    async def run_mode(self, mode: str, query: str, repo_id: str, rag_pipeline) -> Any:
        """Entry point for executing a query in a specific mode."""
        if mode == "debug":
            return await self._handle_debug(query, rag_pipeline)
        elif mode == "explain":
            return await self._handle_explain(query, rag_pipeline)
        elif mode == "impact":
            return await self._handle_impact(query, rag_pipeline)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    async def _handle_debug(self, query: str, rag_pipeline) -> DebugResponse:
        """Triggers the full LangGraph debugging orchestrator."""
        self.orchestrator.rag_pipeline = rag_pipeline
        
        state = {
            "input_type": "text",
            "raw_input": query,
            "available_files": [],
            "retry_count": 0,
            "current_model_tier": 0,
            "rag_top_k": 5
        }
        
        final_state = self.orchestrator.graph.invoke(state)
        fix_data = final_state.get("proposed_fix", {})
        
        return DebugResponse(
            root_cause=final_state.get("root_cause_hypothesis", "Unknown"),
            suspected_files=final_state.get("suspected_files", []),
            fix=fix_data.get("fixed_code", "N/A"),
            confidence=final_state.get("confidence_score", 0.0),
            attempts=final_state.get("retry_count", 0) + 1,
            reasoning=f"{fix_data.get('explanation', '')} | Security: {final_state.get('guardrail_report', {}).get('security', {}).get('risk_level', 'unknown')}"
        )

    async def _handle_explain(self, query: str, rag_pipeline) -> ExplainResponse:
        """Analyzes repository structure and modules."""
        context_chunks = rag_pipeline.query(query, top_k=10)
        context_text = "\n".join([c["content"] for c in context_chunks])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a software architect. Explain the repository based on the provided code context."),
            ("user", "QUERY: {query}\n\nCONTEXT:\n{context}")
        ])
        
        chain = prompt | self.llm.with_structured_output(Explanation)
        result = chain.invoke({"query": query, "context": context_text})
        
        return ExplainResponse(
            summary=result.summary,
            key_modules=result.key_modules,
            data_flow=result.data_flow,
            important_files=result.important_files
        )

    async def _handle_impact(self, query: str, rag_pipeline) -> ImpactResponse:
        """Traces dependencies and calculates risk of code changes."""
        context_chunks = rag_pipeline.query(query, top_k=10)
        context_text = "\n".join([c["content"] for c in context_chunks])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior developer. Perform a dependency impact analysis for the change described in the query."),
            ("user", "QUERY: {query}\n\nCONTEXT:\n{context}")
        ])
        
        chain = prompt | self.llm.with_structured_output(Impact)
        result = chain.invoke({"query": query, "context": context_text})
        
        return ImpactResponse(
            affected_files=result.affected_files,
            dependent_functions=result.dependent_functions,
            risk_level=result.risk_level,
            explanation=result.explanation
        )
