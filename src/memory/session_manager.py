"""
Module for managing unique user sessions using UUIDs.
"""
import uuid
from typing import Dict

class SessionManager:
    """
    Tracks active sessions and provides validation for API requests.
    """
    def __init__(self):
        self.sessions: Dict[str, bool] = {}

    def create_session(self) -> str:
        """Generates a new UUID session ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = True
        return session_id

    def validate_session(self, session_id: str) -> bool:
        """Checks if a session ID is currently active."""
        return session_id in self.sessions

    def delete_session(self, session_id: str):
        """Invalidates a session ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]
