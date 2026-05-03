from typing import Dict, Any

class ContextBuilder:
    def __init__(self, code_vector_store, doc_vector_store, standards_enforcer):
        self.code_vs = code_vector_store
        self.doc_vs = doc_vector_store
        self.enforcer = standards_enforcer
        self.project_standards = None

    def build_context(self, error_description: str) -> Dict[str, Any]:
        """
        Combines code context, error context, and documentation context.
        """
        # 1. Fetch Code Context
        code_results = self.code_vs.search(error_description, top_k=3) if self.code_vs else []
        code_context_str = ""
        for i, res in enumerate(code_results):
            code_context_str += f"--- Result {i+1} ---\nSource: {res['source']}\nSnippet: {res['content_snippet']}\n\n"
            
        # 2. Fetch Documentation Context
        doc_context_str = self.doc_vs.search(error_description, top_k=2) if self.doc_vs else "No documentation found."
        
        # 3. Retrieve Enforced Standards (usually computed once at startup)
        # If not pre-computed, we just pass None and agent ignores it.
        standards_str = ""
        if self.project_standards:
            standards_str = self.project_standards.model_dump_json(indent=2)
            
        return {
            "code_context": code_context_str.strip(),
            "doc_context": doc_context_str.strip(),
            "standards": standards_str
        }
