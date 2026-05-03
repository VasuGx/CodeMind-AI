from pydantic import BaseModel, Field
from typing import List, Union, Optional
from langchain_core.prompts import ChatPromptTemplate
from src.validation.validation import CodeValidator

class CodeFixProposal(BaseModel):
    file_path: str = Field(description="The path of the file to fix")
    explanation: str = Field(description="Explanation of the fix")
    fixed_code: str = Field(description="The complete fixed code snippet for the function/class")

class ConflictReport(BaseModel):
    has_conflict: bool = Field(default=True, description="True if models disagreed")
    model_1_proposal: CodeFixProposal
    model_2_proposal: CodeFixProposal

class ConsensusCheck(BaseModel):
    is_equivalent: bool = Field(description="Are the two code snippets functionally equivalent?")

class CodeFixAgent:
    def __init__(self, default_llm):
        """Initializes with a default LLM, but allows dynamic escalation."""
        self.default_llm = default_llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert python developer. Generate a fix for the buggy code.\n"
                       "CRITICAL: Obey the provided documentation context and coding standards.\n"
                       "Include ALL necessary imports in your fixed_code snippet."),
            ("user", "ERROR DESCRIPTION:\n{error}\n\n"
                     "SUSPECTED FILES:\n{suspected_files}\n\n"
                     "PREVIOUS ATTEMPT ERROR:\n{previous_error}\n\n"
                     "REASON FOR PREVIOUS FAILURE:\n{failure_analysis}\n\n"
                     "CODE CONTEXT:\n{rag_context}\n\n"
                     "DOC CONTEXT:\n{doc_context}")
        ])

    def analyze(self, 
                llm, 
                error_description: str, 
                suspected_files: List[str], 
                rag_context: str, 
                doc_context: str = "", 
                previous_error: str = "None",
                failure_analysis: str = "First attempt.") -> Union[CodeFixProposal, ConflictReport]:
        
        # We create a dynamic chain using the passed-in LLM (escalated or default)
        chain = self.prompt | llm.with_structured_output(CodeFixProposal)
        
        inputs = {
            "error": error_description,
            "previous_error": previous_error,
            "failure_analysis": failure_analysis,
            "suspected_files": ", ".join(suspected_files),
            "rag_context": rag_context,
            "doc_context": doc_context
        }
        
        return chain.invoke(inputs)
