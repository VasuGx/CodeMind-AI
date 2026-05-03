"""
Core Orchestration Module using LangGraph to manage the multi-agent debugging workflow.
Integrates retrieval, fix generation, runtime validation, and security guardrails.
"""
from typing import TypedDict, List, Optional, Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from src.validation.guardrails import SecurityScanner, StyleEnforcer, ConfidenceCalculator

class FailureAnalysis(BaseModel):
    """Structured classification of a failed debugging attempt."""
    category: Literal["missing context", "bad reasoning", "syntax error"] = Field(description="The primary reason for the failure")
    explanation: str = Field(description="Detailed explanation of what the agent missed")

class AgentState(TypedDict):
    """The global state object passed between nodes in the LangGraph."""
    input_type: Literal["text", "image"]
    raw_input: str
    available_files: List[str]
    error_description: Optional[str] 
    suspected_files: Optional[List[str]] 
    root_cause_hypothesis: Optional[str] 
    rag_context: Optional[str] 
    doc_context: Optional[str]
    proposed_fix: Optional[Dict[str, Any]]
    validation_result: Optional[bool]
    is_memory_hit: Optional[bool]
    confidence_score: float
    guardrail_report: Dict[str, Any]
    retry_count: int
    last_execution_error: Optional[str]
    failure_category: Optional[str]
    current_model_tier: int
    rag_top_k: int

class DebuggingOrchestrator:
    """
    Main state machine for autonomous debugging.
    Orchestrates the flow from raw error input to a validated, secure code fix.
    """
    def __init__(self, models: Dict[str, Any], error_analysis_agent, code_fix_agent, rag_pipeline, execution_engine, memory=None):
        self.models = models
        self.error_analysis_agent = error_analysis_agent
        self.code_fix_agent = code_fix_agent
        self.rag_pipeline = rag_pipeline
        self.execution_engine = execution_engine
        self.memory = memory
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Constructs the recursive LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Define Nodes
        workflow.add_node("check_memory", self.node_check_memory)
        workflow.add_node("process_input", self.node_process_input)
        workflow.add_node("error_analysis", self.node_error_analysis)
        workflow.add_node("retrieve_context", self.node_retrieve_context)
        workflow.add_node("generate_fix", self.node_generate_fix)
        workflow.add_node("execute_and_verify", self.node_execute_and_verify)
        workflow.add_node("analyze_failure", self.node_analyze_failure)
        workflow.add_node("guardrail_check", self.node_guardrail_check)
        workflow.add_node("save_memory", self.node_save_memory)
        
        # Define Edges
        workflow.add_edge(START, "check_memory")
        workflow.add_conditional_edges("check_memory", lambda x: "hit" if x.get("is_memory_hit") else "miss", {"hit": END, "miss": "process_input"})
        workflow.add_edge("process_input", "error_analysis")
        workflow.add_edge("error_analysis", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_fix")
        workflow.add_edge("generate_fix", "execute_and_verify")
        
        workflow.add_conditional_edges(
            "execute_and_verify",
            lambda x: "success" if x.get("validation_result") else "fail",
            {"success": "guardrail_check", "fail": "analyze_failure"}
        )
        
        workflow.add_conditional_edges("analyze_failure", lambda x: "retry" if x.get("retry_count", 0) < 3 else "give_up", {"retry": "retrieve_context", "give_up": "guardrail_check"})
        workflow.add_edge("guardrail_check", "save_memory")
        workflow.add_edge("save_memory", END)
        
        return workflow.compile()

    # Node Implementations
    def node_check_memory(self, state: AgentState):
        """Short-circuits the pipeline if a similar bug/fix exists in memory."""
        state.setdefault("retry_count", 0)
        state.setdefault("current_model_tier", 0)
        state.setdefault("rag_top_k", 5)
        if not self.memory or state["input_type"] != "text": return {"is_memory_hit": False}
            
        match = self.memory.find_similar_fix(state["raw_input"])
        if match:
            return {
                "is_memory_hit": True,
                "proposed_fix": {"fixed_code": match.final_fix, "explanation": f"REUSED FIX: {match.reasoning}"},
                "validation_result": True,
                "confidence_score": 1.0
            }
        return {"is_memory_hit": False}

    def node_process_input(self, state: AgentState): 
        """Pre-processes the raw error log or description."""
        return {"error_description": state["raw_input"]}

    def node_error_analysis(self, state: AgentState):
        """Hypothesizes the root cause and identifies suspicious files."""
        analysis = self.error_analysis_agent.analyze(state["error_description"], state["available_files"])
        return {"suspected_files": analysis.suspected_files, "root_cause_hypothesis": analysis.root_cause}
        
    def node_retrieve_context(self, state: AgentState):
        """Fetches code and documentation context using hybrid RAG."""
        results = self.rag_pipeline.query(state["error_description"], top_k=state["rag_top_k"])
        code_ctx = ""
        for res in results:
            code_ctx += f"--- Source: {res['metadata'].get('file_path')} ---\n{res['content']}\n\n"
        return {"rag_context": code_ctx, "doc_context": ""}

    def node_generate_fix(self, state: AgentState):
        """Generates a code fix using a model tier selected by the escalation logic."""
        tier_map = {0: "fast", 1: "medium", 2: "expert"}
        llm = self.models[tier_map.get(state["current_model_tier"])]
        proposal = self.code_fix_agent.analyze(
            llm=llm, error_description=state["error_description"],
            suspected_files=state["suspected_files"], rag_context=state["rag_context"],
            previous_error=state.get("last_execution_error", "None"),
            failure_analysis=state.get("failure_category", "Initial attempt.")
        )
        return {"proposed_fix": proposal.model_dump()}

    def node_execute_and_verify(self, state: AgentState):
        """Validates the fix in a sandbox environment."""
        result = self.execution_engine.validate_fix_runtime(state["proposed_fix"]["file_path"], state["proposed_fix"]["fixed_code"])
        if result["is_valid"]: return {"validation_result": True}
        return {"validation_result": False, "last_execution_error": result["error_message"]}

    def node_analyze_failure(self, state: AgentState):
        """Determines if the system needs more context or a smarter model after a failure."""
        llm = self.models["fast"]
        prompt = f"Analyze this error: {state['last_execution_error']}\nFix was: {state['proposed_fix']['fixed_code']}\nClassify as 'missing context', 'bad reasoning', or 'syntax error'."
        analysis = llm.with_structured_output(FailureAnalysis).invoke(prompt)
        new_tier = state["current_model_tier"]
        new_k = state["rag_top_k"]
        if analysis.category == "missing context": new_k += 5
        elif analysis.category == "bad reasoning": new_tier = min(new_tier + 1, 2)
        return {"failure_category": analysis.explanation, "current_model_tier": new_tier, "rag_top_k": new_k, "retry_count": state["retry_count"] + 1}

    def node_guardrail_check(self, state: AgentState):
        """Performs final security audit and style compliance check."""
        fix_code = state["proposed_fix"]["fixed_code"]
        security_report = SecurityScanner.scan(fix_code)
        style_report = StyleEnforcer.check_conventions(fix_code, state.get("doc_context", ""))
        confidence = ConfidenceCalculator.calculate(
            consensus=state["current_model_tier"] > 0,
            retry_count=state["retry_count"],
            rag_top_k=state["rag_top_k"],
            validation_passed=state.get("validation_result", False)
        )
        return {
            "confidence_score": confidence,
            "guardrail_report": {
                "security": security_report,
                "style": style_report
            }
        }

    def node_save_memory(self, state: AgentState):
        """Persists high-confidence fixes to the project memory layer."""
        if self.memory and state.get("validation_result") and state.get("confidence_score", 0) > 0.7:
            self.memory.add_fix(state["error_description"], state["proposed_fix"]["fixed_code"], state["rag_context"], state["proposed_fix"]["explanation"])
        return {}
