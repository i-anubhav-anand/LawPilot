from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class CaseFileCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    case_file_id: Optional[str] = None
    session_id: Optional[str] = None
    
class CaseFile(BaseModel):
    case_file_id: str
    title: str
    description: Optional[str] = None
    session_id: str
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    facts: Dict[str, Any] = {}
    documents: List[str] = []  # List of document IDs
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        } 