from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    # Inputs
    input_type: Literal["text", "image"]
    raw_input: str   # The text error OR the base64 image string
    available_files: List[str] # List of all repo files
    
    # Internal State
    error_description: Optional[str] # The parsed error string
    suspected_files: Optional[List[str]] # From ErrorAnalysisAgent
    root_cause_hypothesis: Optional[str] # From ErrorAnalysisAgent
    rag_context: Optional[str] # From VectorStore retrieval

class DebuggingOrchestrator:
    def __init__(self, vision_agent, error_analysis_agent, vector_store=None):
        self.vision_agent = vision_agent
        self.error_analysis_agent = error_analysis_agent
        self.vector_store = vector_store
        
        # Initialize graph
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("process_input", self.node_process_input)
        workflow.add_node("vision_analysis", self.node_vision_analysis)
        workflow.add_node("error_analysis", self.node_error_analysis)
        workflow.add_node("retrieve_context", self.node_retrieve_context)
        
        # Define Edges
        workflow.add_edge(START, "process_input")
        
        # Conditional routing based on input type
        workflow.add_conditional_edges(
            "process_input",
            self.route_input,
            {
                "image": "vision_analysis",
                "text": "error_analysis"
            }
        )
        
        workflow.add_edge("vision_analysis", "error_analysis")
        workflow.add_edge("error_analysis", "retrieve_context")
        workflow.add_edge("retrieve_context", END)
        
        return workflow.compile()
        
    def node_process_input(self, state: AgentState):
        """Initial check of input type. If text, map raw_input to error_description."""
        if state["input_type"] == "text":
            return {"error_description": state["raw_input"]}
        return {} # If image, do nothing here, Vision node handles it.
        
    def route_input(self, state: AgentState):
        return state["input_type"]
        
    def node_vision_analysis(self, state: AgentState):
        """Takes base64 image and extracts error text."""
        print("   -> [Vision Node] Processing image using multimodal LLM...")
        description = self.vision_agent.analyze_image(state["raw_input"])
        return {"error_description": description}
        
    def node_error_analysis(self, state: AgentState):
        """Identifies suspected files from the error text."""
        print("   -> [Error Analysis Node] Pinpointing suspected files...")
        analysis = self.error_analysis_agent.analyze(
            stack_trace=state["error_description"],
            files=state["available_files"]
        )
        return {
            "suspected_files": analysis.suspected_files,
            "root_cause_hypothesis": analysis.root_cause
        }
        
    def node_retrieve_context(self, state: AgentState):
        """Retrieves actual code snippets for the suspected files and error."""
        print("   -> [RAG Node] Fetching relevant code from Vector Database...")
        if not self.vector_store:
            return {"rag_context": "VectorStore not provided. Skipping RAG."}
            
        # We query the vector store using the error description
        query = state["error_description"]
        results = self.vector_store.search(query, top_k=3)
        
        # Format results into a string
        context_str = ""
        for i, res in enumerate(results):
            context_str += f"--- Result {i+1} ---\nSource: {res['source']}\nSnippet: {res['content_snippet']}\n\n"
            
        return {"rag_context": context_str.strip()}
