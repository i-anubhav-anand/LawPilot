from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

class CaseSummarySection(BaseModel):
    """A section in a structured case summary."""
    title: str
    key_details: Dict[str, str]
    
class CaseSummary(BaseModel):
    """A structured case summary with sections and key details."""
    case_file_id: str
    session_id: str
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    sections: List[CaseSummarySection] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        } 