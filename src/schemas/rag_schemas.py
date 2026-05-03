from pydantic import BaseModel, Field
from typing import List, Optional

class QueryAnalysis(BaseModel):
    error_type: Optional[str] = Field(default=None, description="The type of error found (e.g., ValueError, ZeroDivisionError).")
    file_hint: Optional[str] = Field(default=None, description="Potential filename found in the logs or query.")
    function_hint: Optional[str] = Field(default=None, description="Potential function/method name found in the logs or query.")
    keywords: List[str] = Field(default_factory=list, description="A list of relevant search keywords extracted from the input.")
    raw_query: str = Field(description="A cleaned version of the original user query or stack trace summary.")
