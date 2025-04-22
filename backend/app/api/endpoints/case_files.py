from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid

from app.core.case_file_manager import CaseFileManager
from app.models.case_file import CaseFile, CaseFileCreateRequest

router = APIRouter()
case_file_manager = CaseFileManager()

@router.post("/", response_model=CaseFile)
async def create_case_file(request: CaseFileCreateRequest):
    """
    Create a new case file.
    """
    case_file_id = request.case_file_id or str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    case_file = case_file_manager.create_case_file(
        case_file_id=case_file_id,
        title=request.title,
        description=request.description,
        session_id=session_id
    )
    
    return case_file

@router.get("/{case_file_id}", response_model=CaseFile)
async def get_case_file(case_file_id: str):
    """
    Get a case file by ID.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    return case_file

@router.put("/{case_file_id}", response_model=CaseFile)
async def update_case_file(case_file_id: str, facts: Dict[str, Any]):
    """
    Update a case file with new facts.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    updated_case_file = case_file_manager.update_case_file(
        case_file_id=case_file_id,
        facts=facts
    )
    
    return updated_case_file

@router.get("/", response_model=List[CaseFile])
async def list_case_files(session_id: Optional[str] = None):
    """
    List all case files, optionally filtered by session ID.
    """
    case_files = case_file_manager.list_case_files(session_id)
    return case_files

@router.delete("/{case_file_id}", response_model=Dict[str, str])
async def delete_case_file(case_file_id: str):
    """
    Delete a case file.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    case_file_manager.delete_case_file(case_file_id)
    
    return {"status": "deleted"} 