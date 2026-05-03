"""
Module for managing in-memory chat history for sessions.
"""
from typing import Dict, List
from src.schemas.api_schemas import Message

class ChatMemory:
    """
    Handles persistence of user and assistant messages per session.
    Limits history to a configurable maximum to optimize context.
    """
    def __init__(self, max_history: int = 20):
        self.history: Dict[str, List[Message]] = {}
        self.max_history = max_history

    def add_message(self, session_id: str, role: str, content: str):
        """Appends a new message to the session history."""
        if session_id not in self.history:
            self.history[session_id] = []
        
        self.history[session_id].append(Message(role=role, content=content))
        
        if len(self.history[session_id]) > self.max_history:
            self.history[session_id] = self.history[session_id][-self.max_history:]

    def get_history(self, session_id: str) -> List[Message]:
        """Retrieves all messages for a specific session."""
        return self.history.get(session_id, [])

    def delete_history(self, session_id: str):
        """Clears history for a session."""
        if session_id in self.history:
            del self.history[session_id]
