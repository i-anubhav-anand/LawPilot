import logging
from typing import Dict, List, Optional, Union
from uuid import uuid4
import uuid
from datetime import datetime
import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from app.api.deps import get_rag_engine, get_case_file_service, get_document_processor, get_direct_text_processor, get_vision_service
from app.core.rag_engine import RAGEngine
from app.core.case_file import CaseFileService
from app.core.document_processor import DocumentProcessor
from app.core.direct_processor import DirectTextProcessor
from app.core.vision_service import VisionService
from app.schemas.chat import ChatRequest, ChatResponse, ChatSession, ChatSessionList
from app.schemas.case_file import CaseFile
from app.models.documents import DocumentResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=ChatResponse)
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
        rag_response = await rag_engine.process_query(
            query=chat_request.query,
            session_id=chat_request.session_id,
            case_file=case_file,
            num_results=chat_request.num_results if chat_request.num_results else 5
        )
        logger.info(f"✅ CHAT PROCESSING COMPLETED: session_id={chat_request.session_id}")
        
        # Log the RAG response for debugging
        logger.info(f"✅ RAG RESPONSE: answer_length={len(rag_response.answer)}, sources={len(rag_response.sources)}, suggested_questions={len(rag_response.suggested_questions)}")
        if rag_response.extracted_facts:
            logger.info(f"✅ EXTRACTED FACTS: {rag_response.extracted_facts}")
        
        # Transform RAGResponse to ChatResponse
        try:
            # Convert Source objects to dict and then back to new Source objects
            # This resolves the validation error when passing Source objects directly
            sources = []
            for source in rag_response.sources:
                source_dict = {
                    "source_type": source.source_type,
                    "title": source.title,
                    "content": source.content,
                    "citation": source.citation,
                    "relevance_score": source.relevance_score,
                    "document_id": source.document_id
                }
                sources.append(source_dict)
            
            chat_response = ChatResponse(
                message=rag_response.answer,
                session_id=chat_request.session_id,
                case_file_id=chat_request.case_file_id,
                sources=sources,
                next_questions=rag_response.suggested_questions
            )
            
            logger.info(f"✅ CHAT RESPONSE CREATED SUCCESSFULLY")
            return chat_response
            
        except Exception as validation_error:
            # Handle validation errors in the response transformation
            logger.error(f"❌ VALIDATION ERROR WHEN CREATING CHAT RESPONSE: {str(validation_error)}")
            
            # Create a fallback response
            chat_response = ChatResponse(
                message=f"I processed your query but encountered an error formatting the response. Here's what I found: {rag_response.answer}",
                session_id=chat_request.session_id,
                case_file_id=chat_request.case_file_id
            )
            
            return chat_response
            
    except Exception as e:
        logger.error(f"❌ ERROR PROCESSING CHAT: session_id={chat_request.session_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.post("/create-session", response_model=ChatSession)
async def create_chat_session(
    case_file_id: Optional[str] = None,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Create a new chat session and return the session ID.
    This endpoint makes it easy for frontends to explicitly create a new session.
    
    Args:
        case_file_id: Optional case file to associate with this session
        
    Returns:
        The newly created session information
    """
    # Generate a new unique session ID
    session_id = f"session_{uuid4()}"
    
    # Initialize an empty chat history for this session
    if session_id not in rag_engine.chat_histories:
        rag_engine.chat_histories[session_id] = []
    
    logger.info(f"✅ NEW CHAT SESSION CREATED: session_id={session_id}")
    
    # Return the new session information
    return ChatSession(
        session_id=session_id,
        message_count=0,
        created_at=datetime.now().isoformat()
    )

@router.get("/sessions", response_model=ChatSessionList)
@router.get("/sessions/", response_model=ChatSessionList)  # Add endpoint with trailing slash too
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

@router.get("/session/{session_id}", response_model=ChatSession)
@router.get("/session/{session_id}/", response_model=ChatSession)
async def get_chat_session(
    session_id: str,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Get details for a specific chat session.
    """
    logger.info(f"🔄 RETRIEVING CHAT SESSION: session_id={session_id}")
    
    if session_id not in rag_engine.chat_histories:
        logger.warning(f"⚠️ CHAT SESSION NOT FOUND: session_id={session_id}")
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    history = rag_engine.chat_histories[session_id]
    session = ChatSession(
        session_id=session_id,
        message_count=len(history)
    )
    
    logger.info(f"✅ CHAT SESSION RETRIEVED: session_id={session_id}, message_count={len(history)}")
    return session

@router.get("/{session_id}/history", response_model=List[Dict])
@router.get("/{session_id}/history/", response_model=List[Dict])  # Add endpoint with trailing slash too
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

@router.post("/with-document-text", response_model=ChatResponse)
async def process_chat_with_document_text(
    request: dict,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    case_file_service: CaseFileService = Depends(get_case_file_service),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    direct_processor: DirectTextProcessor = Depends(get_direct_text_processor)
):
    """
    Process a chat message with extracted document text using the dual-path approach.
    
    This endpoint enables real-time responses by:
    1. Processing the document text directly without waiting for embedding
    2. Initiating background indexing for future queries
    3. Combining direct text processing with vector search for comprehensive responses
    
    The document text will be processed immediately for the current query and 
    made available for future RAG queries through background processing.
    """
    logger.info(f"🔄 RECEIVED CHAT WITH DOCUMENT REQUEST: session_id={request.get('session_id')}")
    
    # Extract components from request
    query = request.get("query", "")
    session_id = request.get("session_id")
    case_file_id = request.get("case_file_id")
    document_text = request.get("document_text", "")
    document_name = request.get("document_name", "Extracted Text")
    num_results = request.get("num_results", 5)
    
    if not query:
        logger.warning(f"⚠️ EMPTY QUERY RECEIVED")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not session_id:
        logger.warning(f"⚠️ NO SESSION ID PROVIDED")
        raise HTTPException(status_code=400, detail="Session ID must be provided")
    
    # If case_file_id is provided, get the case file
    case_file: Optional[CaseFile] = None
    if case_file_id:
        logger.info(f"🔄 RETRIEVING CASE FILE: id={case_file_id}")
        case_file = case_file_service.get_case_file(case_file_id)
        if not case_file:
            logger.warning(f"⚠️ CASE FILE NOT FOUND: id={case_file_id}")
            raise HTTPException(status_code=404, detail="Case file not found")
        logger.info(f"✅ CASE FILE RETRIEVED: id={case_file_id}")
    
    try:
        # Check if we have document text to process
        if document_text and len(document_text.strip()) > 0:
            logger.info(f"🔄 PROCESSING WITH DIRECT TEXT APPROACH: length={len(document_text)}")
            
            # Use the dual-path processing method
            rag_response = await rag_engine.process_query_with_direct_text(
                query=query,
                text=document_text,
                document_name=document_name,
                session_id=session_id,
                case_file=case_file,
                num_results=num_results
            )
            
            logger.info(f"✅ DIRECT TEXT PROCESSING COMPLETED")
        else:
            # Standard RAG processing without direct text
            logger.info(f"🔄 PROCESSING WITH STANDARD RAG APPROACH")
            rag_response = await rag_engine.process_query(
                query=query,
                session_id=session_id,
                case_file=case_file,
                num_results=num_results
            )
            logger.info(f"✅ STANDARD RAG PROCESSING COMPLETED")
        
        logger.info(f"✅ CHAT PROCESSING COMPLETED: session_id={session_id}")
        
        # Log the RAG response for debugging
        logger.info(f"✅ RAG RESPONSE: answer_length={len(rag_response.answer)}, sources={len(rag_response.sources)}, suggested_questions={len(rag_response.suggested_questions)}")
        if rag_response.extracted_facts:
            logger.info(f"✅ EXTRACTED FACTS: {rag_response.extracted_facts}")
        
        # Transform RAGResponse to ChatResponse
        try:
            # Convert Source objects to dict and then back to new Source objects
            sources = []
            for source in rag_response.sources:
                source_dict = {
                    "source_type": source.source_type,
                    "title": source.title,
                    "content": source.content,
                    "citation": source.citation,
                    "relevance_score": source.relevance_score,
                    "document_id": source.document_id
                }
                sources.append(source_dict)
            
            chat_response = ChatResponse(
                message=rag_response.answer,
                session_id=session_id,
                case_file_id=case_file_id,
                sources=sources,
                next_questions=rag_response.suggested_questions
            )
            
            logger.info(f"✅ CHAT RESPONSE CREATED SUCCESSFULLY")
            return chat_response
            
        except Exception as validation_error:
            # Handle validation errors in the response transformation
            logger.error(f"❌ VALIDATION ERROR WHEN CREATING CHAT RESPONSE: {str(validation_error)}")
            
            # Create a fallback response
            chat_response = ChatResponse(
                message=f"I processed your query but encountered an error formatting the response. Here's what I found: {rag_response.answer}",
                session_id=session_id,
                case_file_id=case_file_id
            )
            
            return chat_response
            
    except Exception as e:
        logger.error(f"❌ ERROR PROCESSING CHAT: session_id={session_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.post("/with-file", response_model=ChatResponse)
async def process_chat_with_file(
    query: str = Form(...),
    session_id: str = Form(...),
    case_file_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    rag_engine: RAGEngine = Depends(get_rag_engine),
    case_file_service: CaseFileService = Depends(get_case_file_service),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Process a chat message with an attached file.
    
    This endpoint allows sending both a text message and a file in a single request.
    The file will be processed and indexed, and the message will be processed together.
    
    Args:
        query: The user's query text
        session_id: The session ID for this conversation
        case_file_id: Optional case file ID to associate
        file: Optional file attachment to process
        
    Returns:
        Chat response with the assistant's reply
    """
    logger.info(f"🔄 RECEIVED CHAT WITH FILE REQUEST: session_id={session_id}, query='{query}'")
    
    # Process the file if it was uploaded
    document_id = None
    if file:
        try:
            logger.info(f"🔄 PROCESSING ATTACHED FILE: {file.filename}")
            
            # Generate a unique ID for the file
            document_id = str(uuid.uuid4())
            
            # Save the uploaded file
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            
            file_extension = Path(file.filename).suffix.lower()
            file_path = upload_dir / f"{document_id}{file_extension}"
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"✅ FILE SAVED: path={file_path}")
            
            # Create a document response object
            document = DocumentResponse(
                document_id=document_id,
                filename=file.filename,
                session_id=session_id,
                case_file_id=case_file_id,
                status="processing"
            )
            
            # Start processing the document in the background
            asyncio.create_task(document_processor.process_document(
                str(file_path),
                document_id,
                session_id,
                case_file_id
            ))
            
            logger.info(f"✅ DOCUMENT PROCESSING TASK INITIATED: id={document_id}")
            
            # Add a note to the query about the file
            query_with_file = f"{query}\n\n[User uploaded a file: {file.filename}]"
            
        except Exception as e:
            logger.error(f"❌ ERROR PROCESSING ATTACHED FILE: {str(e)}")
            # Continue with the original query if file processing failed
            query_with_file = query
    else:
        # No file uploaded, just use the original query
        query_with_file = query
    
    # If case_file_id is provided, get the case file
    case_file: Optional[CaseFile] = None
    if case_file_id:
        logger.info(f"🔄 RETRIEVING CASE FILE: id={case_file_id}")
        case_file = case_file_service.get_case_file(case_file_id)
        if not case_file:
            logger.warning(f"⚠️ CASE FILE NOT FOUND: id={case_file_id}")
            raise HTTPException(status_code=404, detail="Case file not found")
        logger.info(f"✅ CASE FILE RETRIEVED: id={case_file_id}")

    try:
        # Process the query
        rag_response = await rag_engine.process_query(
            query=query_with_file,
            session_id=session_id,
            case_file=case_file,
            num_results=5
        )
        logger.info(f"✅ CHAT PROCESSING COMPLETED: session_id={session_id}")
        
        # Log the RAG response for debugging
        logger.info(f"✅ RAG RESPONSE: answer_length={len(rag_response.answer)}, sources={len(rag_response.sources)}, suggested_questions={len(rag_response.suggested_questions)}")
        
        # Transform RAGResponse to ChatResponse
        try:
            # Convert Source objects to dict and then back to new Source objects
            sources = []
            for source in rag_response.sources:
                source_dict = {
                    "source_type": source.source_type,
                    "title": source.title,
                    "content": source.content,
                    "citation": source.citation,
                    "relevance_score": source.relevance_score,
                    "document_id": source.document_id
                }
                sources.append(source_dict)
            
            chat_response = ChatResponse(
                message=rag_response.answer,
                session_id=session_id,
                case_file_id=case_file_id,
                sources=sources,
                next_questions=rag_response.suggested_questions
            )
            
            logger.info(f"✅ CHAT RESPONSE CREATED SUCCESSFULLY")
            return chat_response
            
        except Exception as validation_error:
            # Handle validation errors in the response transformation
            logger.error(f"❌ VALIDATION ERROR WHEN CREATING CHAT RESPONSE: {str(validation_error)}")
            
            # Create a fallback response
            chat_response = ChatResponse(
                message=f"I processed your query but encountered an error formatting the response. Here's what I found: {rag_response.answer}",
                session_id=session_id,
                case_file_id=case_file_id
            )
            
            return chat_response
            
    except Exception as e:
        logger.error(f"❌ ERROR PROCESSING CHAT: session_id={session_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.post("/with-image", response_model=ChatResponse)
async def process_chat_with_image(
    query: str = Form(...),
    session_id: str = Form(...),
    case_file_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    rag_engine: RAGEngine = Depends(get_rag_engine),
    case_file_service: CaseFileService = Depends(get_case_file_service),
    vision_service: VisionService = Depends(get_vision_service)
):
    """
    Process a chat message with an image attachment using vision model analysis.
    
    This endpoint enables analysis of image content using a vision-capable model,
    providing insights based on the visual content of the image.
    
    The image will be processed immediately for the current query and
    its text content will also be made available for future RAG queries.
    """
    logger.info(f"🔄 RECEIVED CHAT WITH IMAGE REQUEST: session_id={session_id}, image={file.filename}")
    
    if not query:
        logger.warning(f"⚠️ EMPTY QUERY RECEIVED")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not session_id:
        logger.warning(f"⚠️ NO SESSION ID PROVIDED")
        raise HTTPException(status_code=400, detail="Session ID must be provided")
    
    # If case_file_id is provided, get the case file
    case_file: Optional[CaseFile] = None
    if case_file_id:
        logger.info(f"🔄 RETRIEVING CASE FILE: id={case_file_id}")
        case_file = case_file_service.get_case_file(case_file_id)
        if not case_file:
            logger.warning(f"⚠️ CASE FILE NOT FOUND: id={case_file_id}")
            raise HTTPException(status_code=404, detail="Case file not found")
    
    try:
        # Get conversation state
        conversation_state = rag_engine._get_conversation_state(session_id)
        
        # Save the uploaded file
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        
        if file_extension not in allowed_extensions:
            logger.warning(f"❌ UNSUPPORTED IMAGE FORMAT: {file_extension}")
            raise HTTPException(status_code=400, detail="Unsupported image format. Please upload a JPG, PNG, GIF, or WebP image.")
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Generate a unique file ID
        image_id = f"image_{uuid.uuid4()}"
        file_path = upload_dir / f"{image_id}{file_extension}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"✅ IMAGE SAVED: path={file_path}")
        
        # First, analyze the image using the vision service
        # We use "Query: {query}" to make the analysis more relevant to the user's question
        vision_prompt = f"Analyze this image in detail, focusing on any legal documents, notices, or relevant information it contains. Query: {query}"
        vision_result = await vision_service.analyze_image(str(file_path), prompt=vision_prompt)
        
        if "error" in vision_result:
            logger.error(f"❌ VISION ANALYSIS ERROR: {vision_result.get('error')}")
            raise HTTPException(status_code=500, detail=f"Error analyzing image: {vision_result.get('error')}")
        
        image_analysis = vision_result.get("analysis", "")
        logger.info(f"✅ IMAGE ANALYSIS COMPLETE: length={len(image_analysis)}")
        
        # Add the image analysis to the chat history
        rag_engine.chat_histories[session_id].append(ChatMessage(
            role="user",
            content=f"{query}\n\n[Attached image: {file.filename}]"
        ))
        
        # Also add context about the image analysis as a special system message
        # This won't be shown to the user but will be included in the context for the LLM
        system_note = f"IMAGE ANALYSIS: The user uploaded an image ({file.filename}). Here's what the vision model detected:\n\n{image_analysis}"
        
        # Prepare enhanced query for RAG processing
        enhanced_query = f"{query}\n\nIMAGE CONTEXT: {image_analysis}"
        
        # Process with RAG engine
        rag_response = await rag_engine.process_query(
            query=enhanced_query,
            session_id=session_id,
            case_file=case_file
        )
        
        # Start background processing to add the image text to the vector store
        # This allows the content to be available for future searches
        background_tasks = BackgroundTasks()
        
        # Create a temporary document processor task
        async def process_image_text_wrapper():
            try:
                await asyncio.sleep(1)  # Small delay to avoid conflicts
                doc_processor = DocumentProcessor()
                await doc_processor.process_document(
                    str(file_path),
                    document_id=image_id,
                    session_id=session_id,
                    case_file_id=case_file_id
                )
            except Exception as e:
                logger.error(f"❌ BACKGROUND IMAGE PROCESSING ERROR: {str(e)}")
        
        background_tasks.add_task(process_image_text_wrapper)
        
        # Transform RAGResponse to ChatResponse
        try:
            # Convert Source objects to dict and then back to new Source objects
            sources = []
            for source in rag_response.sources:
                source_dict = {
                    "source_type": source.source_type,
                    "title": source.title,
                    "content": source.content,
                    "citation": source.citation,
                    "relevance_score": source.relevance_score,
                    "document_id": source.document_id
                }
                sources.append(source_dict)
            
            # Add a special source for the image analysis
            sources.append({
                "source_type": "image_analysis",
                "title": f"Analysis of {file.filename}",
                "content": image_analysis[:500] + ("..." if len(image_analysis) > 500 else ""),
                "citation": None,
                "relevance_score": 0.99,  # High relevance since it's direct analysis
                "document_id": image_id
            })
            
            chat_response = ChatResponse(
                message=rag_response.answer,
                session_id=session_id,
                case_file_id=case_file_id,
                sources=sources,
                next_questions=rag_response.suggested_questions
            )
            
            logger.info(f"✅ CHAT RESPONSE WITH IMAGE ANALYSIS CREATED SUCCESSFULLY")
            return chat_response
            
        except Exception as validation_error:
            # Handle validation errors in the response transformation
            logger.error(f"❌ VALIDATION ERROR WHEN CREATING CHAT RESPONSE: {str(validation_error)}")
            
            # Create a fallback response
            chat_response = ChatResponse(
                message=f"I analyzed the image you provided and found the following: {image_analysis[:500]}... Based on this and your query, {rag_response.answer}",
                session_id=session_id,
                case_file_id=case_file_id
            )
            
            return chat_response
            
    except Exception as e:
        logger.error(f"❌ ERROR PROCESSING CHAT WITH IMAGE: session_id={session_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat with image: {str(e)}") 