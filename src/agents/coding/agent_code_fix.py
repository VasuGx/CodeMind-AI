from pydantic import BaseModel, Field
from typing import List, Union
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
    is_equivalent: bool = Field(description="Are the two code snippets functionally equivalent despite minor syntax differences?")

class CodeFixAgent:
    def __init__(self, llm_1, llm_2, arbiter_llm):
        """Initializes the dual-model agent architecture with an Arbiter for near-consensus."""
        self.llm_1 = llm_1
        self.llm_2 = llm_2
        self.arbiter_llm = arbiter_llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert python developer. Generate a fix for the buggy code. "
                       "CRITICAL: Obey the provided documentation context and coding standards.\n"
                       "Return the fixed code snippet, file path, and an explanation."),
            ("user", "ERROR DESCRIPTION:\n{error}\n\n"
                     "SUSPECTED FILES:\n{suspected_files}\n\n"
                     "CODE CONTEXT:\n{rag_context}\n\n"
                     "DOC CONTEXT:\n{doc_context}\n\n"
                     "CODING STANDARDS:\n{standards}")
        ])
        
        self.chain_1 = self.prompt | self.llm_1.with_structured_output(CodeFixProposal)
        self.chain_2 = self.prompt | self.llm_2.with_structured_output(CodeFixProposal)

        self.arbiter_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a code equivalence judge. Do these two Python code snippets perform the exact same functional logic?"),
            ("user", "CODE 1:\n{code_1}\n\nCODE 2:\n{code_2}")
        ])
        self.arbiter_chain = self.arbiter_prompt | self.arbiter_llm.with_structured_output(ConsensusCheck)

    def analyze(self, error_description: str, suspected_files: List[str], rag_context: str, doc_context: str = "", standards: str = "") -> Union[CodeFixProposal, ConflictReport]:
        suspected_files_str = ", ".join(suspected_files)
        
        inputs = {
            "error": error_description,
            "suspected_files": suspected_files_str,
            "rag_context": rag_context,
            "doc_context": doc_context,
            "standards": standards
        }
        
        proposal_1 = self.chain_1.invoke(inputs)
        proposal_2 = self.chain_2.invoke(inputs)

        # Validation Layer Injection
        val_1 = CodeValidator.validate(proposal_1.fixed_code)
        val_2 = CodeValidator.validate(proposal_2.fixed_code)

        if val_1.is_valid and not val_2.is_valid:
            return proposal_1
        elif val_2.is_valid and not val_1.is_valid:
            return proposal_2

        # Near-Consensus Logic
        consensus = self.arbiter_chain.invoke({
            "code_1": proposal_1.fixed_code,
            "code_2": proposal_2.fixed_code
        })
        
        if consensus.is_equivalent:
            return proposal_1
        else:
            return ConflictReport(
                model_1_proposal=proposal_1,
                model_2_proposal=proposal_2
            )
