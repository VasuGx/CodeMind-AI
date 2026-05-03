from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate

class ErrorAnalysis(BaseModel):
    suspected_files: List[str] = Field(description="Paths to files likely causing the error based on the stack trace")
    root_cause: str = Field(description="Concise hypothesis of the root cause. Max 2 sentences.")
    recommended_action: str = Field(description="Next step to fix or investigate the issue.")

class ErrorAnalysisAgent:
    def __init__(self, llm):
        self.llm = llm
        
        # Highly optimized prompt to save input/output tokens
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert debugger. Given a stack trace and a list of available project files, "
                       "identify the root cause and suspected files. Be concise."),
            ("user", "AVAILABLE FILES:\n{file_list}\n\nSTACK TRACE:\n{stack_trace}")
        ])
        
        self.chain = self.prompt | self.llm.with_structured_output(ErrorAnalysis)

    def analyze(self, stack_trace: str, files: List[str]) -> ErrorAnalysis:
        # Join files cleanly to save tokens (one per line)
        file_list_str = "\n".join(files)
        
        # Smart Truncation: Keep top and bottom of stack trace if monstrously huge 
        # (Usually the origin and the final crash point are most important)
        if len(stack_trace) > 3000:
            stack_trace = stack_trace[:1500] + "\n...[TRUNCATED TO SAVE TOKENS]...\n" + stack_trace[-1500:]
            
        return self.chain.invoke({
            "file_list": file_list_str,
            "stack_trace": stack_trace
        })
