from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid

from app.core.case_file_manager import CaseFileManager
from app.models.case_file import CaseFile, CaseFileCreateRequest
from app.core.case_summary_manager import CaseSummaryManager
from app.models.case_summary import CaseSummary, CaseSummarySection

router = APIRouter()
case_file_manager = CaseFileManager()
case_summary_manager = CaseSummaryManager()

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
    
    case_summary_manager.create_summary(case_file_id, session_id)
    
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

class SectionUpdateRequest(BaseModel):
    """Request model for updating a section."""
    section_title: str
    key_details: Dict[str, str]

@router.get("/{case_file_id}/summary", response_model=CaseSummary)
async def get_case_summary(case_file_id: str):
    """
    Get the structured case summary for a case file.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    summary = case_summary_manager.get_summary(case_file_id)
    if not summary:
        summary = case_summary_manager.create_summary(case_file_id, case_file.session_id)
    
    return summary

@router.put("/{case_file_id}/summary/section", response_model=CaseSummary)
async def update_case_summary_section(case_file_id: str, update: SectionUpdateRequest):
    """
    Update a section in the case summary.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    summary = case_summary_manager.update_section(
        case_file_id=case_file_id,
        section_title=update.section_title,
        key_details=update.key_details
    )
    
    if not summary:
        raise HTTPException(status_code=500, detail="Failed to update case summary")
    
    return summary

@router.get("/{case_file_id}/summary/formatted", response_model=Dict[str, str])
async def get_formatted_summary(case_file_id: str):
    """
    Get a formatted version of the case summary for display.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    formatted_summary = case_summary_manager.format_summary_for_display(case_file_id)
    return {"summary": formatted_summary}

class ChatHistoryUpdateRequest(BaseModel):
    """Request model for updating summary from chat history."""
    chat_history: List[Dict[str, Any]]
    force_update: bool = False

@router.post("/{case_file_id}/update-summary-from-chat", response_model=Dict[str, str])
async def update_summary_from_chat(
    case_file_id: str,
    request: ChatHistoryUpdateRequest,
    background_tasks: BackgroundTasks
):
    """
    Update the case summary based on chat history.
    This endpoint accepts chat history and updates the case summary.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    try:
        # If force_update is False, run in background
        if not request.force_update:
            async def update_summary_background():
                await case_summary_manager.generate_summary_from_chat(
                    case_file_id=case_file_id,
                    session_id=case_file.session_id,
                    chat_history=request.chat_history
                )
            
            background_tasks.add_task(update_summary_background)
            return {"status": "Case summary update scheduled in background"}
        else:
            # Immediate update
            summary = await case_summary_manager.generate_summary_from_chat(
                case_file_id=case_file_id,
                session_id=case_file.session_id,
                chat_history=request.chat_history
            )
            
            if not summary:
                raise HTTPException(status_code=500, detail="Failed to generate summary")
            
            return {"status": "Case summary updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating summary: {str(e)}") 