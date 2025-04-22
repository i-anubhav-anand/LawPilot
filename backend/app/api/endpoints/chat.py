import logging
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_rag_engine, get_case_file_service
from app.core.rag_engine import RAGEngine
from app.core.case_file import CaseFileService
from app.schemas.chat import ChatRequest, ChatResponse, ChatSession, ChatSessionList
from app.schemas.case_file import CaseFile

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def process_chat(
    chat_request: ChatRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    case_file_service: CaseFileService = Depends(get_case_file_service)
):
    """
    Process a chat message using RAG.
    """
    logger.info(f"🔄 RECEIVED CHAT REQUEST: session_id={chat_request.session_id}, query='{chat_request.query}'")
    
    # If case_file_id is provided, get the case file
    case_file: Optional[CaseFile] = None
    if chat_request.case_file_id:
        logger.info(f"🔄 RETRIEVING CASE FILE: id={chat_request.case_file_id}")
        case_file = case_file_service.get_case_file(chat_request.case_file_id)
        if not case_file:
            logger.warning(f"⚠️ CASE FILE NOT FOUND: id={chat_request.case_file_id}")
            raise HTTPException(status_code=404, detail="Case file not found")
        logger.info(f"✅ CASE FILE RETRIEVED: id={chat_request.case_file_id}")

    try:
        # Process the query
        response = await rag_engine.process_query(
            query=chat_request.query,
            session_id=chat_request.session_id,
            case_file=case_file,
            num_results=chat_request.num_results if chat_request.num_results else 5
        )
        logger.info(f"✅ CHAT PROCESSING COMPLETED: session_id={chat_request.session_id}")
        return response
    except Exception as e:
        logger.error(f"❌ ERROR PROCESSING CHAT: session_id={chat_request.session_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.get("/chat/sessions", response_model=ChatSessionList)
async def list_chat_sessions(
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    List all chat sessions.
    """
    logger.info("🔄 LISTING CHAT SESSIONS")
    sessions = [
        ChatSession(session_id=session_id, message_count=len(history))
        for session_id, history in rag_engine.chat_histories.items()
    ]
    logger.info(f"✅ CHAT SESSIONS LISTED: count={len(sessions)}")
    return ChatSessionList(sessions=sessions)

@router.get("/chat/{session_id}/history", response_model=List[Dict])
async def get_chat_history(
    session_id: str,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Get chat history for a specific session.
    """
    logger.info(f"🔄 RETRIEVING CHAT HISTORY: session_id={session_id}")
    
    if session_id not in rag_engine.chat_histories:
        logger.warning(f"⚠️ CHAT SESSION NOT FOUND: session_id={session_id}")
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    history = [msg.dict() for msg in rag_engine.chat_histories[session_id]]
    logger.info(f"✅ CHAT HISTORY RETRIEVED: session_id={session_id}, message_count={len(history)}")
    return history 