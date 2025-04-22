from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class ChatSession(BaseModel):
    """Information about a chat session."""
    session_id: str
    message_count: int

class ChatSessionList(BaseModel):
    """List of chat sessions."""
    sessions: List[ChatSession] = []

class ChatRequest(BaseModel):
    """Request payload for chat API."""
    query: str
    session_id: str = Field(default_factory=lambda: f"session_{datetime.now().timestamp()}")
    case_file_id: Optional[str] = None
    num_results: Optional[int] = 5

class Source(BaseModel):
    """Source of information used in a response."""
    source_type: str  # "law", "document", "case"
    title: str
    content: str
    citation: Optional[str] = None
    relevance_score: Optional[float] = None
    document_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Response payload for chat API."""
    message: str
    session_id: str
    case_file_id: Optional[str] = None
    sources: List[Source] = []
    next_questions: List[str] = [] 