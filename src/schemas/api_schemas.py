from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class RepoLoadRequest(BaseModel):
    repo_url: Optional[str] = None
    local_path: Optional[str] = None

class RepoLoadResponse(BaseModel):
    repo_id: str
    status: str
    files_indexed: int

class RunRequest(BaseModel):
    session_id: str
    repo_id: str
    mode: Literal["debug", "explain", "impact"]
    query: str

class DebugResponse(BaseModel):
    mode: str = "debug"
    root_cause: str
    suspected_files: List[str]
    fix: str
    confidence: float
    attempts: int
    reasoning: str

class ExplainResponse(BaseModel):
    mode: str = "explain"
    summary: str
    key_modules: List[str]
    data_flow: str
    important_files: List[str]

class ImpactResponse(BaseModel):
    mode: str = "impact"
    affected_files: List[str]
    dependent_functions: List[str]
    risk_level: Literal["low", "medium", "high"]
    explanation: str

class SessionResponse(BaseModel):
    session_id: str

class Message(BaseModel):
    role: str
    content: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Message]
