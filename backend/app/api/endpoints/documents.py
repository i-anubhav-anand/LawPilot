from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends
from typing import List, Optional, Dict, Any
import uuid
import os
from pathlib import Path
import logging
from datetime import datetime
import time
import asyncio

from app.core.document_processor import DocumentProcessor
from app.core.rag_engine import RAGEngine
from app.models.documents import DocumentResponse, DocumentAnalysisResponse
from app.api.deps import get_document_processor, get_rag_engine

# Set up logging
logger = logging.getLogger("documents_api")

router = APIRouter()
# Remove global document_processor and rag_engine instances
# document_processor = DocumentProcessor()
# rag_engine = RAGEngine()

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    session_id: Optional[str] = None,
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    List all documents and their processing status, optionally filtered by session ID.
    """
    logger.info(f"🔍 LIST DOCUMENTS: session_id={session_id}")
    all_docs = document_processor.get_all_documents()
    
    # Filter by session ID if provided
    if session_id:
        session_docs = [doc for doc in all_docs if doc.session_id == session_id]
        logger.info(f"✅ RETURNED {len(session_docs)} DOCUMENTS FOR SESSION: {session_id}")
        return session_docs
    
    logger.info(f"✅ RETURNED {len(all_docs)} DOCUMENTS")
    return all_docs

@router.get("/list", response_model=List[DocumentResponse])
async def list_documents_alternative(
    session_id: Optional[str] = None,
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Alternative endpoint to list all documents (compatible with frontend calls).
    """
    logger.info(f"🔍 LIST DOCUMENTS (/list ENDPOINT): session_id={session_id}")
    return await list_documents(session_id, document_processor)

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    case_file_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    use_vision: bool = Form(True),  # Add a flag to control vision analysis
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Upload a document for processing.
    
    This endpoint allows uploading documents like PDFs, text files, Word docs, and images.
    The document will be processed in the background and made available for RAG.
    """
    logger.info(f"📤 DOCUMENT UPLOAD: filename={file.filename}, session_id={session_id}, case_file_id={case_file_id}")
    
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
    
    logger.info(f"✅ FILE SAVED: path={file_path}")
    
    # Check if this is an image
    is_image = file_extension.lower() in ['.jpg', '.jpeg', '.png']
    logger.info(f"📝 FILE INFO: is_image={is_image}, use_vision={use_vision}")
    
    # Process document in background
    logger.info(f"🔄 STARTING BACKGROUND DOCUMENT PROCESSING: id={document_id}")
    
    # Wrap async function for background tasks
    async def process_document_wrapper():
        # Initialize async components if needed
        await document_processor.async_initialize()
        
        # For image files, we add a special metadata flag for vision analysis
        metadata = {
            "is_image": is_image,
            "use_vision": use_vision and is_image
        }
        
        await document_processor.process_document(
            str(file_path),
            document_id,
            session_id,
            case_file_id,
            metadata=metadata
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

@router.get("/processing-status", response_model=Dict[str, Any])
async def document_processing_status(
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
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
    background_tasks: BackgroundTasks = None,
    document_processor: DocumentProcessor = Depends(get_document_processor)
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
    
    # Wrap async function for background tasks - use the existing document_processor instance
    async def process_global_document_wrapper():
        # Initialize async components if needed
        await document_processor.async_initialize()
        
        # Process the document
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
async def list_global_documents(
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
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

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
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

@router.get("/{document_id}/status", response_model=DocumentResponse)
async def get_document_status_with_suffix(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Get the status of a processed document (alternative endpoint with /status suffix).
    This endpoint exists for compatibility with frontend calls.
    """
    logger.info(f"🔍 GET DOCUMENT STATUS (WITH /status SUFFIX): id={document_id}")
    return await get_document_status(document_id, document_processor)

@router.post("/{document_id}/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor),
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
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

@router.post("/{document_id}/toggle-global", response_model=DocumentResponse)
async def toggle_document_global_status(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor)
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

@router.post("/extract-text", response_model=Dict[str, Any])
async def extract_document_text(
    file: UploadFile = File(...),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Extract text from a document without storing it in the vector database.
    
    This endpoint extracts text from documents (PDF, images, etc.) and returns the raw text.
    It's designed for immediate client-side use without waiting for indexing.
    
    The extracted text can be sent to the /chat/with-document-text endpoint for dual-path processing.
    """
    logger.info(f"🔄 EXTRACTING TEXT FROM DOCUMENT: filename={file.filename}")
    
    try:
        # Create a unique temporary file
        temp_file_id = f"temp_{uuid.uuid4()}"
        
        # Create a temporary storage directory
        upload_dir = Path("uploads/temp")
        upload_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the uploaded file
        temp_file_path = upload_dir / f"{temp_file_id}_{file.filename}"
        with open(temp_file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Extract text based on file type
        logger.info(f"✅ FILE SAVED: path={temp_file_path}")
        
        # Use the document processor to extract text
        start_time = time.time()
        
        # Call the extraction method directly - this method is synchronous and safe
        extracted_text = document_processor._extract_text(str(temp_file_path))
        
        extraction_time = time.time() - start_time
        logger.info(f"✅ TEXT EXTRACTED: {len(extracted_text)} characters in {extraction_time:.2f}s")
        
        # Clean up the temporary file
        try:
            os.remove(temp_file_path)
            logger.info(f"✅ TEMPORARY FILE REMOVED: {temp_file_path}")
        except Exception as e:
            logger.warning(f"⚠️ COULD NOT REMOVE TEMP FILE: {str(e)}")
        
        return {
            "filename": file.filename,
            "text": extracted_text,
            "extraction_time": extraction_time,
            "text_length": len(extracted_text)
        }
        
    except Exception as e:
        logger.error(f"❌ ERROR EXTRACTING TEXT: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

@router.post("/{document_id}/retry", response_model=DocumentResponse)
async def retry_document_processing(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor),
    background_tasks: BackgroundTasks = None
):
    """
    Retry processing a failed document. Useful when document failed processing for transient reasons.
    """
    logger.info(f"🔄 RETRY DOCUMENT PROCESSING: id={document_id}")
    
    # Check if document exists
    document = document_processor.get_document(document_id)
    if not document:
        logger.warning(f"❌ DOCUMENT NOT FOUND: id={document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Mark document for retry
    document_processor.mark_document_for_retry(document_id)
    
    # Retry processing in background
    async def retry_wrapper():
        # Initialize async components if needed
        await document_processor.async_initialize()
        
        # Retry processing
        await document_processor.retry_failed_document(document_id)
    
    background_tasks.add_task(retry_wrapper)
    
    logger.info(f"✅ DOCUMENT RETRY QUEUED: id={document_id}")
    
    return document_processor.get_document(document_id)

@router.post("/recover-failed", response_model=Dict[str, Any])
async def recover_failed_documents(
    document_processor: DocumentProcessor = Depends(get_document_processor),
    background_tasks: BackgroundTasks = None
):
    """
    Attempt to recover all failed documents.
    
    This endpoint will scan for all documents with a 'failed' status and attempt to reprocess them.
    """
    logger.info(f"🔄 RECOVER ALL FAILED DOCUMENTS REQUEST")
    
    # Get all failed documents
    failed_documents = [doc for doc in document_processor.document_store.values() if doc.status == "failed"]
    failed_count = len(failed_documents)
    
    if failed_count == 0:
        logger.info("✅ NO FAILED DOCUMENTS TO RECOVER")
        return {"status": "success", "message": "No failed documents to recover", "count": 0}
    
    logger.info(f"🔄 FOUND {failed_count} FAILED DOCUMENTS TO RECOVER")
    
    # Start recovery in the background
    if background_tasks:
        background_tasks.add_task(document_processor.recover_all_failed_documents)
        logger.info(f"✅ RECOVERY SCHEDULED IN BACKGROUND FOR {failed_count} DOCUMENTS")
    else:
        # If background_tasks is not available, create our own task
        asyncio.create_task(document_processor.recover_all_failed_documents())
        logger.info(f"✅ RECOVERY SCHEDULED AS TASK FOR {failed_count} DOCUMENTS")
    
    return {
        "status": "success",
        "message": f"Recovery started for {failed_count} failed documents",
        "count": failed_count,
        "document_ids": [doc.document_id for doc in failed_documents]
    }

@router.delete("/{document_id}", response_model=Dict[str, str])
async def delete_document(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Delete a document from the system.
    This removes the document from both the document store and vector store.
    """
    logger.info(f"🔄 DELETE DOCUMENT REQUEST: id={document_id}")
    
    # Initialize async components
    await document_processor.async_initialize()
    
    # Check if document exists
    document = document_processor.get_document(document_id)
    if not document:
        logger.warning(f"❌ DOCUMENT NOT FOUND: id={document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete from vector store first
        vector_store_deleted = await document_processor.vector_store.delete_document(document_id)
        
        if not vector_store_deleted:
            logger.warning(f"⚠️ DOCUMENT NOT FOUND IN VECTOR STORE: id={document_id}")
        
        # Remove metadata file
        metadata_path = Path("documents_metadata") / f"{document_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"✅ DOCUMENT METADATA DELETED: id={document_id}")
        
        # Remove from document store
        if document_id in document_processor.document_store:
            del document_processor.document_store[document_id]
            logger.info(f"✅ DOCUMENT REMOVED FROM DOCUMENT STORE: id={document_id}")
        
        logger.info(f"✅ DOCUMENT DELETED SUCCESSFULLY: id={document_id}")
        return {"status": "deleted"}
        
    except Exception as e:
        logger.error(f"❌ ERROR DELETING DOCUMENT: id={document_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}") 