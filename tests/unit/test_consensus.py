import pytest
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.agents.coding.agent_code_fix import CodeFixAgent

@pytest.fixture
def groq_llms():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY_1")
    if not key: pytest.skip("No API Key")
    # Use ultra-fast models for testing consensus
    llm1 = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=key)
    llm2 = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, api_key=key)
    arbiter = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=key)
    return llm1, llm2, arbiter

def test_code_fix_agent(groq_llms):
    llm1, llm2, arbiter = groq_llms
    agent = CodeFixAgent(llm1, llm2, arbiter)
    assert agent is not None
