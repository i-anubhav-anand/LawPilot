from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    session_id: Optional[str] = None
    case_file_id: Optional[str] = None
    status: str  # "processing", "processed", "failed"
    created_at: datetime = datetime.now()
    processed_at: Optional[datetime] = None
    error: Optional[str] = None
    is_global: bool = False  # Added to distinguish global vs session documents
    
class DocumentAnalysis(BaseModel):
    summary: str
    key_points: List[str]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    relevant_laws: List[Dict[str, Any]]
    
class DocumentAnalysisResponse(BaseModel):
    document_id: str
    summary: str
    key_points: List[str]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    relevant_laws: List[Dict[str, Any]] 