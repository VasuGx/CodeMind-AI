import json
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate

class ModuleInfo(BaseModel):
    name: str = Field(description="Name of the module or directory")
    description: str = Field(description="What this module does")

class RepoArchitecture(BaseModel):
    modules: List[ModuleInfo] = Field(description="List of identified modules")
    dependencies: List[str] = Field(description="List of core external or internal dependencies")
    summary: str = Field(description="A high-level architecture summary of the repository")

class RepoAnalysisAgent:
    def __init__(self, llm):
        """
        Initialize with a LangChain chat model.
        Uses native with_structured_output which is fully supported by Groq.
        """
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior AI systems engineer analyzing a codebase. "
                       "Based on the provided repository structure, identify the architecture, "
                       "core modules, and dependencies. Return your analysis strictly as structured data."),
            ("user", "Repository structure:\n{repo_summary}")
        ])
        
        # Native structured output support
        self.chain = self.prompt | self.llm.with_structured_output(RepoArchitecture)

    def analyze(self, repo_summary: dict) -> RepoArchitecture:
        # Filter the summary to remove empty classes/functions to save tokens
        optimized_summary = []
        for f in repo_summary.get("files", []):
            opt_file = {"file": f["file"]}
            if f.get("classes"): opt_file["classes"] = f["classes"]
            if f.get("functions"): opt_file["functions"] = f["functions"]
            optimized_summary.append(opt_file)

        summary_str = json.dumps(optimized_summary, indent=2)
        # Aggressively limit context to save tokens
        if len(summary_str) > 4000:
             summary_str = summary_str[:4000] + "\n...[truncated to save tokens]..."
             
        return self.chain.invoke({"repo_summary": summary_str})
