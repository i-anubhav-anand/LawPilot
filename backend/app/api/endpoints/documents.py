from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid
import os
from pathlib import Path
import logging
from datetime import datetime

from app.core.document_processor import DocumentProcessor
from app.core.rag_engine import RAGEngine
from app.models.documents import DocumentResponse, DocumentAnalysisResponse

# Set up logging
logger = logging.getLogger("documents_api")

router = APIRouter()
document_processor = DocumentProcessor()
rag_engine = RAGEngine()

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(session_id: Optional[str] = None):
    """
    List all documents and their processing status.
    Optionally filter by session ID.
    """
    logger.info(f"🔍 LIST DOCUMENTS: session_id={session_id}")
    all_docs = document_processor.get_all_documents()
    
    # Filter by session_id if provided
    if session_id:
        all_docs = [doc for doc in all_docs if doc.session_id == session_id]
    
    logger.info(f"✅ RETURNED {len(all_docs)} DOCUMENTS")
    return all_docs

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    case_file_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a document for processing and analysis.
    """
    logger.info(f"📤 DOCUMENT UPLOAD: filename={file.filename}, session_id={session_id}")
    
    # Generate IDs if not provided
    session_id = session_id or str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_extension = Path(file.filename).suffix.lower()
    allowed_extensions = ['.pdf', '.txt', '.docx', '.doc', '.jpg', '.jpeg', '.png']
    
    if file_extension not in allowed_extensions:
        logger.warning(f"❌ UNSUPPORTED FILE FORMAT: {file_extension}")
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Create uploads directory if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{document_id}{file_extension}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"✅ FILE SAVED: path={file_path}")
    
    # Process document in background
    logger.info(f"🔄 STARTING BACKGROUND DOCUMENT PROCESSING: id={document_id}")
    
    # Wrap async function for background tasks
    async def process_document_wrapper():
        await document_processor.process_document(
            str(file_path),
            document_id,
            session_id,
            case_file_id
        )
    
    background_tasks.add_task(process_document_wrapper)
    
    logger.info(f"✅ DOCUMENT PROCESSING TASK INITIATED: id={document_id}")
    
    return DocumentResponse(
        document_id=document_id,
        filename=file.filename,
        session_id=session_id,
        case_file_id=case_file_id,
        status="processing"
    )

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_status(document_id: str):
    """
    Get the status of a processed document.
    """
    logger.info(f"🔍 GET DOCUMENT STATUS: id={document_id}")
    document = document_processor.get_document(document_id)
    if not document:
        logger.warning(f"❌ DOCUMENT NOT FOUND: id={document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    logger.info(f"✅ DOCUMENT STATUS: id={document_id}, status={document.status}")
    return document

@router.post("/{document_id}/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(document_id: str):
    """
    Analyze a document for legal issues and extract relevant information.
    """
    logger.info(f"🔍 ANALYZE DOCUMENT: id={document_id}")
    document = document_processor.get_document(document_id)
    if not document:
        logger.warning(f"❌ DOCUMENT NOT FOUND: id={document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.status != "processed":
        logger.warning(f"❌ DOCUMENT NOT PROCESSED: id={document_id}, status={document.status}")
        raise HTTPException(status_code=400, detail="Document is still processing")
    
    # Analyze the document using RAG
    logger.info(f"🔄 STARTING DOCUMENT ANALYSIS: id={document_id}")
    analysis = await rag_engine.analyze_document(document_id)
    logger.info(f"✅ DOCUMENT ANALYSIS COMPLETED: id={document_id}")
    
    return DocumentAnalysisResponse(
        document_id=document_id,
        summary=analysis.summary,
        key_points=analysis.key_points,
        issues=analysis.issues,
        recommendations=analysis.recommendations,
        relevant_laws=analysis.relevant_laws
    )

@router.get("/processing-status", response_model=Dict[str, Any])
async def document_processing_status():
    """
    Get the current document processing status including stats.
    """
    logger.info(f"🔍 GET DOCUMENT PROCESSING STATUS")
    
    # Get all documents by status
    all_docs = document_processor.get_all_documents()
    
    processing_count = len([doc for doc in all_docs if doc.status == "processing"])
    processed_count = len([doc for doc in all_docs if doc.status == "processed"])
    failed_count = len([doc for doc in all_docs if doc.status == "failed"])
    
    # Get vector store stats if available
    vector_stats = {}
    try:
        vector_count = document_processor.vector_store.index.ntotal if hasattr(document_processor.vector_store, 'index') else 0
        document_count = len(document_processor.vector_store.document_store)
        chunk_count = len(document_processor.vector_store.chunk_ids)
        
        vector_stats = {
            "vector_count": vector_count,
            "document_count": document_count,
            "chunk_count": chunk_count
        }
    except Exception as e:
        logger.warning(f"⚠️ ERROR GETTING VECTOR STATS: {str(e)}")
        vector_stats = {"error": str(e)}
    
    # Get recently processed documents
    recent_documents = []
    for doc in sorted(all_docs, key=lambda x: x.created_at, reverse=True)[:5]:
        recent_documents.append({
            "document_id": doc.document_id,
            "filename": doc.filename,
            "status": doc.status,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "processed_at": doc.processed_at.isoformat() if doc.processed_at else None
        })
    
    return {
        "status_counts": {
            "processing": processing_count,
            "processed": processed_count,
            "failed": failed_count,
            "total": len(all_docs)
        },
        "vector_store": vector_stats,
        "recent_documents": recent_documents,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/upload/global", response_model=DocumentResponse)
async def upload_global_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a document to the global RAG knowledge base.
    These documents will be available to all chat sessions.
    
    This endpoint should be used for uploading reference materials like the 'California Tenant Guide'
    that should be accessible to all users.
    """
    logger.info(f"📤 GLOBAL DOCUMENT UPLOAD: filename={file.filename}")
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_extension = Path(file.filename).suffix.lower()
    allowed_extensions = ['.pdf', '.txt', '.docx', '.doc', '.jpg', '.jpeg', '.png']
    
    if file_extension not in allowed_extensions:
        logger.warning(f"❌ UNSUPPORTED FILE FORMAT: {file_extension}")
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Create uploads directory if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{document_id}{file_extension}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"✅ GLOBAL FILE SAVED: path={file_path}")
    
    # Process document in background
    logger.info(f"🔄 STARTING BACKGROUND GLOBAL DOCUMENT PROCESSING: id={document_id}")
    
    # Wrap async function for background tasks
    async def process_global_document_wrapper():
        await document_processor.process_document(
            str(file_path),
            document_id,
            is_global=True  # Mark as global document
        )
    
    background_tasks.add_task(process_global_document_wrapper)
    
    logger.info(f"✅ GLOBAL DOCUMENT PROCESSING TASK INITIATED: id={document_id}")
    
    return DocumentResponse(
        document_id=document_id,
        filename=file.filename,
        status="processing",
        is_global=True
    )

@router.get("/global", response_model=List[DocumentResponse])
async def list_global_documents():
    """
    List all global documents and their processing status.
    These are documents available to all chat sessions.
    """
    logger.info(f"🔍 LIST GLOBAL DOCUMENTS")
    all_docs = document_processor.get_all_documents()
    
    # Filter to only include global documents
    global_docs = [doc for doc in all_docs if getattr(doc, "is_global", False)]
    
    logger.info(f"✅ RETURNED {len(global_docs)} GLOBAL DOCUMENTS")
    return global_docs

@router.post("/{document_id}/toggle-global", response_model=DocumentResponse)
async def toggle_document_global_status(
    document_id: str
):
    """
    Toggle a document's global status (make it available to all sessions or restrict it).
    This allows changing a document's global flag without reprocessing it.
    """
    logger.info(f"🔄 TOGGLING DOCUMENT GLOBAL STATUS: id={document_id}")
    
    # Check if document exists
    document = document_processor.get_document(document_id)
    if not document:
        logger.warning(f"❌ DOCUMENT NOT FOUND: id={document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if document is processed
    if document.status != "processed":
        logger.warning(f"❌ DOCUMENT NOT READY: id={document_id}, status={document.status}")
        raise HTTPException(status_code=400, detail="Document is not in 'processed' state")
    
    try:
        # Toggle the global flag
        new_global_status = not document.is_global
        
        # Update in document store
        updated_document = DocumentResponse(
            document_id=document.document_id,
            filename=document.filename,
            session_id=document.session_id,
            case_file_id=document.case_file_id,
            status=document.status,
            created_at=document.created_at,
            processed_at=document.processed_at,
            is_global=new_global_status
        )
        document_processor.document_store[document_id] = updated_document
        
        # Save the updated metadata to disk
        document_processor._save_document_metadata(document_id)
        
        # Update in vector store metadata
        vector_document = await document_processor.vector_store.get_document(document_id)
        if vector_document:
            # Update the metadata in the vector store
            vector_document["metadata"]["is_global"] = new_global_status
            
            # Save the updated metadata
            await document_processor.vector_store._save_data()
            
            logger.info(f"✅ DOCUMENT GLOBAL STATUS UPDATED: id={document_id}, is_global={new_global_status}")
        else:
            logger.warning(f"⚠️ DOCUMENT NOT FOUND IN VECTOR STORE: id={document_id}")
        
        return updated_document
    except Exception as e:
        logger.error(f"❌ ERROR UPDATING DOCUMENT GLOBAL STATUS: id={document_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating document global status: {str(e)}") 