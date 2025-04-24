from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class CaseFileBase(BaseModel):
    """Base class for case file schemas."""
    title: str
    description: Optional[str] = None
    session_id: Optional[str] = None

class CaseFileCreate(CaseFileBase):
    """Schema for creating a case file."""
    case_file_id: Optional[str] = None

class CaseFile(CaseFileBase):
    """Schema for a case file."""
    case_file_id: str
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    facts: Dict[str, Any] = {}
    documents: List[str] = []  # List of document IDs
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        } 