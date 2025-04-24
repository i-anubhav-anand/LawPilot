from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class Source(BaseModel):
    source_type: str  # "law", "document", "case"
    title: str
    content: str
    citation: Optional[str] = None
    relevance_score: Optional[float] = None
    document_id: Optional[str] = None
    
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()
    
class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    suggested_questions: List[str] = []
    extracted_facts: Optional[Dict[str, Any]] = None
    
class ChatResponse(BaseModel):
    message: str
    session_id: str
    case_file_id: Optional[str] = None
    sources: List[Source] = []
    next_questions: List[str] = [] 