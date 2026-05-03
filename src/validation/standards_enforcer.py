from pydantic import BaseModel, Field
from typing import List

class CodingStandards(BaseModel):
    naming_conventions: List[str] = Field(description="Rules about variable/function naming found in docs.")
    architecture_patterns: List[str] = Field(description="Required architectural or design patterns found in docs.")
    banned_practices: List[str] = Field(description="Things the docs explicitly say NOT to do.")

class StandardsEnforcer:
    def __init__(self, llm):
        self.llm = llm
        
    def extract_standards(self, full_doc_text: str) -> CodingStandards:
        """
        Reads documentation and extracts coding standards to inject into the CodeFix Agent.
        """
        # If docs are too huge, truncate. In a real system, we might use a map-reduce chain.
        if len(full_doc_text) > 4000:
            full_doc_text = full_doc_text[:4000]
            
        prompt = f"Analyze this documentation and extract coding standards.\n\nDOCS:\n{full_doc_text}"
        
        chain = self.llm.with_structured_output(CodingStandards)
        try:
            return chain.invoke(prompt)
        except Exception:
            # Fallback if LLM fails
            return CodingStandards(naming_conventions=[], architecture_patterns=[], banned_practices=[])
