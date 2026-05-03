"""
Main Entry Point for the CodeMind AI FastAPI Backend.
Wires together sessions, repository management, and multi-agent orchestration.
"""
import os
from fastapi import FastAPI, HTTPException, Body
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Schemas
from src.schemas.api_schemas import (
    RepoLoadRequest, RepoLoadResponse, RunRequest, 
    SessionResponse, HistoryResponse
)

# Services & Managers
from src.services.repo_manager import RepoManager
from src.services.pipeline_service import PipelineService
from src.memory.session_manager import SessionManager
from src.memory.chat_memory import ChatMemory

# Original System Components
from src.agents.management.orchestrator import DebuggingOrchestrator
from src.agents.analysis.agent_error_analysis import ErrorAnalysisAgent
from src.agents.coding.agent_code_fix import CodeFixAgent
from src.validation.execution_engine import ExecutionEngine
from src.memory.project_memory import ProjectMemory
from src.rag.engine.embedding import Embedder

load_dotenv()

app = FastAPI(title="CodeMind AI API", version="1.0.0")

# Global State Managers
session_manager = SessionManager()
chat_memory = ChatMemory()

# Model Factory
key1 = os.getenv("GROQ_API_KEY_1")
key2 = os.getenv("GROQ_API_KEY_2")
MODELS = {
    "fast": ChatGroq(groq_api_key=key1, model_name="llama-3.1-8b-instant"),
    "medium": ChatGroq(groq_api_key=key2, model_name="mixtral-8x7b-32768"),
    "expert": ChatGroq(groq_api_key=key1, model_name="llama-3.3-70b-versatile")
}

# Core Engines
repo_manager = RepoManager(llm=MODELS["fast"])
exec_engine = ExecutionEngine()
memory_layer = ProjectMemory(embedder=Embedder().get_embeddings())

# Orchestrator
orchestrator = DebuggingOrchestrator(
    models=MODELS,
    error_analysis_agent=ErrorAnalysisAgent(llm=MODELS["fast"]),
    code_fix_agent=CodeFixAgent(default_llm=MODELS["fast"]),
    rag_pipeline=None,
    execution_engine=exec_engine,
    memory=memory_layer
)

pipeline_service = PipelineService(orchestrator, llm=MODELS["expert"])

@app.post("/start-session", response_model=SessionResponse)
async def start_session():
    """Initializes a new user session."""
    session_id = session_manager.create_session()
    return {"session_id": session_id}

@app.post("/upload-doc")
async def upload_doc(repo_id: str = Body(...), name: str = Body(...), content: str = Body(...)):
    """Attaches external documentation to an existing repo index."""
    try:
        repo_manager.load_external_docs(repo_id, name, content)
        return {"status": "doc_attached"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/load-repo", response_model=RepoLoadResponse)
async def load_repo(request: RepoLoadRequest):
    """Indexes a repository from a local path or GitHub URL."""
    try:
        repo_id = repo_manager.load_repo(
            repo_url=request.repo_url, 
            local_path=request.local_path
        )
        return {
            "repo_id": repo_id,
            "status": "indexed",
            "files_indexed": repo_manager.get_file_count(repo_id)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/run")
async def run_pipeline(request: RunRequest):
    """Executes a query in debug, explain, or impact mode."""
    if not session_manager.validate_session(request.session_id):
        raise HTTPException(status_code=404, detail="Invalid session_id")
    
    rag = repo_manager.get_rag(request.repo_id)
    if not rag:
        raise HTTPException(status_code=404, detail="Repo not found or not indexed")

    chat_memory.add_message(request.session_id, "user", request.query)

    # 4. Execute Mode
    response = await pipeline_service.run_mode(
        mode=request.mode,
        query=request.query,
        repo_id=request.repo_id,
        rag_pipeline=rag
    )
    
    # 5. Store System Response
    chat_memory.add_message(request.session_id, "system", str(response))
    
    return response

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Retrieves conversation history for a session."""
    if not session_manager.validate_session(session_id):
        raise HTTPException(status_code=404, detail="Invalid session_id")
    
    messages = chat_memory.get_history(session_id)
    return {"session_id": session_id, "messages": messages}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Cleanup session data."""
    session_manager.delete_session(session_id)
    chat_memory.delete_history(session_id)
    return {"status": "deleted"}
