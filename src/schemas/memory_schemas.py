from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class MemoryItem(BaseModel):
    error_id: str = Field(description="Unique hash or ID of the error")
    error_description: str
    retrieved_context: str
    final_fix: str
    reasoning: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
