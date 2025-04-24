import logging
import time
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from app.core.case_summary_manager import CaseSummaryManager
from app.core.case_file_manager import CaseFileManager
from app.models.case_summary import CaseSummary
from app.core.rag_engine import RAGEngine
from app.api.deps import get_rag_engine

# Configure logging
logger = logging.getLogger("summaries_endpoint")

# Create the router with explicit tags
router = APIRouter(tags=["Summaries"])
case_summary_manager = CaseSummaryManager()
case_file_manager = CaseFileManager()

class ChatHistoryRequest(BaseModel):
    """Request model for chat history to generate summary from."""
    chat_history: List[Dict[str, Any]]

class ChatMessageRequest(BaseModel):
    """Request model for adding a single chat message and updating the summary."""
    case_file_id: str
    message_content: str
    message_role: str = "user"
    recent_chat_history: Optional[List[Dict[str, Any]]] = None

# New endpoint to directly summarize from a session ID
@router.get("/from-session/{session_id}", response_model=Dict[str, str])
async def generate_summary_from_session(
    session_id: str,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Generate a summary directly from a chat session without requiring a case file.
    This endpoint retrieves the chat history from the given session and generates a summary.
    """
    start_time = time.time()
    logger.info(f"🔄 SUMMARY REQUEST RECEIVED: session_id={session_id}")
    
    # Retrieve chat history for the session
    chat_history = rag_engine.get_chat_history(session_id)
    
    if not chat_history or len(chat_history) == 0:
        logger.warning(f"⚠️ EMPTY CHAT HISTORY: session_id={session_id}")
        raise HTTPException(status_code=404, detail="Chat session not found or empty")
    
    # Format the chat history for the summarizer
    formatted_history = [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else None
        }
        for msg in chat_history
    ]
    
    logger.info(f"🔄 GENERATING SUMMARY: session_id={session_id}, message_count={len(formatted_history)}")
    
    try:
        # Generate a plain text summary using the existing summarization logic
        summary_text = await case_summary_manager.generate_summary_text_only(
            session_id=session_id,
            chat_history=formatted_history
        )
        
        if not summary_text:
            logger.error(f"❌ SUMMARY GENERATION FAILED: session_id={session_id}")
            raise HTTPException(status_code=500, detail="Failed to generate summary")
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ SUMMARY GENERATED SUCCESSFULLY: session_id={session_id}, time={elapsed_time:.2f}s")
        
        return {"summary": summary_text}
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ SUMMARY GENERATION ERROR: session_id={session_id}, error={str(e)}, time={elapsed_time:.2f}s")
        import traceback
        logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@router.post("/generate/{case_file_id}", response_model=Dict[str, str])
async def generate_summary_from_chat(
    case_file_id: str,
    request: ChatHistoryRequest
):
    """
    Generate a case summary from chat history.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    # Generate summary from chat history
    summary = await case_summary_manager.generate_summary_from_chat(
        case_file_id=case_file_id,
        session_id=case_file.session_id,
        chat_history=request.chat_history
    )
    
    if not summary:
        raise HTTPException(status_code=500, detail="Failed to generate summary")
    
    return {"summary": summary}

@router.post("/update-from-message", response_model=Dict[str, str])
async def update_summary_from_message(
    request: ChatMessageRequest,
    background_tasks: BackgroundTasks
):
    """
    Update the case summary after a new chat message.
    This endpoint adds the new message to the history and regenerates the summary.
    The update is done in the background to avoid blocking the response.
    """
    case_file = case_file_manager.get_case_file(request.case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    # Create a minimal chat history if none provided
    chat_history = request.recent_chat_history or []
    
    # Add the new message to the history
    chat_history.append({
        "role": request.message_role,
        "content": request.message_content
    })
    
    # Schedule the summary update in the background
    async def update_summary_background():
        await case_summary_manager.generate_summary_from_chat(
            case_file_id=request.case_file_id,
            session_id=case_file.session_id,
            chat_history=chat_history
        )
    
    background_tasks.add_task(update_summary_background)
    
    return {"status": "Summary update scheduled", "message": "The case summary will be updated in the background"}

@router.get("/formatted/{case_file_id}", response_model=Dict[str, str])
async def get_formatted_summary(case_file_id: str):
    """
    Get a formatted version of the case summary for display.
    """
    case_file = case_file_manager.get_case_file(case_file_id)
    if not case_file:
        raise HTTPException(status_code=404, detail="Case file not found")
    
    formatted_summary = case_summary_manager.format_summary_for_display(case_file_id)
    return {"summary": formatted_summary} 