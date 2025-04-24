import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import pypdf
import docx2txt
import pytesseract
from PIL import Image
import json
import logging
import asyncio
import time
import concurrent.futures
import shutil
import traceback

from app.models.documents import DocumentResponse
from app.core.vector_store import VectorStore
from app.core.text_chunker import TextChunker
from app.core.vision_service import VisionService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document_processor")

class EmbeddingProgressReporter:
    """Helper class to report embedding generation progress."""
    
    def __init__(self, total_chunks: int):
        self.total_chunks = total_chunks
        self.processed_chunks = 0
        self.last_report_time = time.time()
        self.report_interval = 5  # Report every 5 seconds
    
    def report_progress(self, chunk_index: int = None):
        """Report progress of embedding generation."""
        self.processed_chunks += 1
        current_time = time.time()
        
        # Report on first, every 5th, and last chunk. Also report periodically by time.
        if (self.processed_chunks == 1 or 
            self.processed_chunks % 5 == 0 or 
            self.processed_chunks == self.total_chunks or
            current_time - self.last_report_time >= self.report_interval):
            
            percent_complete = (self.processed_chunks / self.total_chunks) * 100
            logger.info(f"⏳ EMBEDDING PROGRESS: {self.processed_chunks}/{self.total_chunks} chunks ({percent_complete:.1f}%)")
            self.last_report_time = current_time

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor."""
        self.document_store = {}  # In-memory store, will be persisted to disk
        self.vector_store = VectorStore()
        self.text_chunker = TextChunker(chunk_size=300, chunk_overlap=50, max_chunk_time=60)
        self.vision_service = VisionService()
        self._initialized = False
        
        # Create a dedicated thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("processed", exist_ok=True)
        os.makedirs("documents_metadata", exist_ok=True)  # New directory for document metadata persistence
        logger.info("🚀 Document processor initialized. Directories created.")
        
        # Load document metadata from disk
        self._load_document_metadata()
        
        # Don't start the async task in __init__ - will be done in async_initialize
        # This avoids the "no running event loop" error

    async def async_initialize(self):
        """Perform async initialization tasks. Call this method in an async context."""
        if not self._initialized:
            # Make sure document metadata is loaded
            self._load_document_metadata()
            
            # Synchronize with vector store
            await self._sync_with_vector_store()
            
            self._initialized = True
            logger.info("✅ Document processor async initialization complete")
        return self
        
    async def _sync_with_vector_store(self):
        """Synchronize the document store with the vector store to ensure consistency."""
        try:
            logger.info("🔄 SYNCING DOCUMENT STORE WITH VECTOR STORE")
            
            # Get all documents from vector store
            vector_docs = await self.vector_store.get_all_documents()
            vector_doc_ids = {doc["document_id"] for doc in vector_docs}
            
            # Get all documents from document store
            doc_store_ids = set(self.document_store.keys())
            
            # Find documents in vector store but not in document store
            missing_in_doc_store = vector_doc_ids - doc_store_ids
            if missing_in_doc_store:
                logger.warning(f"⚠️ FOUND {len(missing_in_doc_store)} DOCUMENTS IN VECTOR STORE BUT NOT IN DOCUMENT STORE")
                for doc_id in missing_in_doc_store:
                    # If we have the processed file, try to restore the document
                    processed_path = Path("processed") / f"{doc_id}.json"
                    if processed_path.exists():
                        try:
                            with open(processed_path, "r") as f:
                                processed_data = json.load(f)
                                
                            # Create document entry
                            self.document_store[doc_id] = DocumentResponse(
                                document_id=doc_id,
                                filename=processed_data.get("filename", f"{doc_id}.pdf"),
                                session_id=processed_data.get("session_id"),
                                case_file_id=processed_data.get("case_file_id"),
                                status="processed",
                                created_at=datetime.now(),
                                processed_at=datetime.now(),
                                is_global=processed_data.get("is_global", False)
                            )
                            
                            # Save the restored metadata
                            self._save_document_metadata(doc_id)
                            logger.info(f"✅ RESTORED DOCUMENT METADATA: id={doc_id}")
                        except Exception as e:
                            logger.error(f"❌ ERROR RESTORING DOCUMENT METADATA: id={doc_id}, error={str(e)}")
            
            # Find documents in document store but not in vector store
            missing_in_vector_store = doc_store_ids - vector_doc_ids
            if missing_in_vector_store:
                logger.warning(f"⚠️ FOUND {len(missing_in_vector_store)} DOCUMENTS IN DOCUMENT STORE BUT NOT IN VECTOR STORE")
                for doc_id in missing_in_vector_store:
                    if self.document_store[doc_id].status == "processed":
                        # Mark as failed since it's missing from vector store
                        self.document_store[doc_id] = DocumentResponse(
                            document_id=doc_id,
                            filename=self.document_store[doc_id].filename,
                            session_id=self.document_store[doc_id].session_id,
                            case_file_id=self.document_store[doc_id].case_file_id,
                            status="failed",
                            created_at=self.document_store[doc_id].created_at,
                            processed_at=self.document_store[doc_id].processed_at,
                            error="Document missing from vector store - reprocessing required",
                            is_global=self.document_store[doc_id].is_global
                        )
                        # Save the updated metadata
                        self._save_document_metadata(doc_id)
                        logger.info(f"⚠️ MARKED DOCUMENT AS FAILED DUE TO MISSING VECTOR DATA: id={doc_id}")
            
            logger.info("✅ DOCUMENT STORE SYNCED WITH VECTOR STORE")
            
        except Exception as e:
            logger.error(f"❌ ERROR SYNCING DOCUMENT STORE WITH VECTOR STORE: {str(e)}", exc_info=True)
            
    def _load_document_metadata(self):
        """
        Load document metadata from disk.
        This ensures we maintain document information between restarts.
        """
        try:
            metadata_dir = Path("documents_metadata")
            if not metadata_dir.exists():
                metadata_dir.mkdir(exist_ok=True)
                logger.info("✅ Created documents_metadata directory")
                return
            
            loaded_count = 0
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        document_id = metadata_file.stem
                        metadata = json.load(f)
                        
                        # Convert ISO format strings back to datetime objects if needed
                        created_at = datetime.now()
                        processed_at = None
                        
                        if "created_at" in metadata and metadata["created_at"]:
                            try:
                                created_at = datetime.fromisoformat(metadata["created_at"])
                            except:
                                pass
                        
                        if "processed_at" in metadata and metadata["processed_at"]:
                            try:
                                processed_at = datetime.fromisoformat(metadata["processed_at"])
                            except:
                                pass
                        
                        # Recreate DocumentResponse object
                        self.document_store[document_id] = DocumentResponse(
                            document_id=document_id,
                            filename=metadata.get("filename", "unknown.pdf"),
                            session_id=metadata.get("session_id"),
                            case_file_id=metadata.get("case_file_id"),
                            status=metadata.get("status", "processed"),
                            created_at=created_at,
                            processed_at=processed_at,
                            error=metadata.get("error"),
                            is_global=metadata.get("is_global", False)
                        )
                        loaded_count += 1
                except Exception as e:
                    logger.error(f"❌ ERROR LOADING DOCUMENT METADATA: {metadata_file} - {str(e)}")
            
            logger.info(f"✅ LOADED {loaded_count} DOCUMENT METADATA RECORDS")
        except Exception as e:
            logger.error(f"❌ ERROR LOADING DOCUMENT METADATA: {str(e)}")

    def _save_document_metadata(self, document_id: str):
        """Save document metadata to disk."""
        document = self.document_store.get(document_id)
        if not document:
            return
            
        metadata_path = Path("documents_metadata") / f"{document_id}.json"
        
        try:
            # Convert document to dict
            doc_dict = document.dict()
            
            # Convert datetime objects to strings
            if doc_dict["created_at"]:
                doc_dict["created_at"] = doc_dict["created_at"].isoformat()
            if doc_dict["processed_at"]:
                doc_dict["processed_at"] = doc_dict["processed_at"].isoformat()
                
            # Save to file
            with open(metadata_path, "w") as f:
                json.dump(doc_dict, f, indent=2)
                
            logger.info(f"✅ DOCUMENT METADATA SAVED: id={document_id}")
        except Exception as e:
            logger.error(f"❌ ERROR SAVING DOCUMENT METADATA: id={document_id}, error={str(e)}")
        
    def get_document(self, document_id: str) -> Optional[DocumentResponse]:
        """
        Get document metadata by ID.
        """
        return self.document_store.get(document_id)
    
    def get_all_documents(self) -> List[DocumentResponse]:
        """
        Get all documents and their processing status.
        
        Returns:
            List of all document responses.
        """
        return list(self.document_store.values())
    
    async def process_document(
        self, 
        file_path: str, 
        document_id: str, 
        session_id: Optional[str] = None,
        case_file_id: Optional[str] = None,
        is_global: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Process a document: extract text, chunk it, and add to vector store.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            session_id: Optional session ID to associate with document
            case_file_id: Optional case file ID to associate with document
            is_global: Whether this document should be available to all sessions
            metadata: Additional metadata to store with the document
        """
        try:
            logger.info(f"🔄 PROCESSING DOCUMENT: id={document_id}, path={file_path}")
            
            # Get or create document record
            document = DocumentResponse(
                document_id=document_id,
                filename=os.path.basename(file_path),
                session_id=session_id,
                case_file_id=case_file_id,
                status="processing",
                is_global=is_global
            )
            
            # Update document store
            self.document_store[document_id] = document
            
            # Extract text from file
            start_time = time.time()
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if this is an image requiring vision analysis
            use_vision = False
            if metadata and isinstance(metadata, dict):
                is_image = metadata.get("is_image", False)
                use_vision_flag = metadata.get("use_vision", False)
                use_vision = is_image and use_vision_flag
            
            logger.info(f"🔄 EXTRACTING TEXT: use_vision={use_vision}")
            
            if use_vision:
                # For vision processing, we'll extract text using OCR and also analyze with vision model
                # This async function will handle both tasks
                extracted_text = await self._process_with_vision(file_path, document_id)
            else:
                # For regular documents, extract text directly
                extracted_text = self._extract_text(file_path)
            
            extraction_time = time.time() - start_time
            logger.info(f"✅ TEXT EXTRACTED: {len(extracted_text)} characters in {extraction_time:.2f}s")
            
            # Create a copy of the file in uploads with document_id as the name
            target_ext = os.path.splitext(file_path)[1].lower()
            target_dir = Path("uploads")
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / f"{document_id}{target_ext}"
            
            # Only copy if not already in the target path
            if file_path != str(target_path):
                shutil.copy2(file_path, target_path)
                logger.info(f"✅ FILE COPIED: {file_path} → {target_path}")
            
            # If this is a temporary upload, clean it up
            if "/temp/" in file_path:
                try:
                    os.remove(file_path)
                    logger.info(f"✅ TEMP FILE REMOVED: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ FAILED TO REMOVE TEMP FILE: {file_path} - {str(e)}")
            
            # Process the extracted text
            start_time = time.time()
            
            # Initialize the chunker with appropriate settings
            # Use smaller chunks for large documents to improve retrieval precision
            document_size = len(extracted_text)
            
            # Adapt chunk size for document length - smaller chunks for larger documents
            chunk_size = 300 if document_size < 50000 else 200
            
            # Create the text chunker with appropriate settings
            chunker = TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=50,
                max_chunk_time=30 + int(document_size / 100000) * 10  # 30s base + 10s per MB
            )
            
            # Chunk the text
            try:
                logger.info(f"🔄 CHUNKING TEXT: size={chunk_size}, overlap=50")
                chunks = chunker.chunk_text(extracted_text)
                chunking_time = time.time() - start_time
                logger.info(f"✅ TEXT CHUNKED: {len(chunks)} chunks in {chunking_time:.2f}s")
            except Exception as e:
                logger.error(f"❌ CHUNKING ERROR: {str(e)}")
                logger.info(f"🔄 FALLING BACK TO EMERGENCY CHUNKING")
                chunks = chunker._emergency_chunking(extracted_text)
                logger.info(f"✅ EMERGENCY CHUNKING COMPLETE: {len(chunks)} chunks")
            
            # Initialize progress monitoring
            progress_reporter = EmbeddingProgressReporter(len(chunks))
            
            # Document metadata to store
            doc_metadata = {
                "filename": os.path.basename(file_path),
                "document_id": document_id,
                "session_id": session_id,
                "case_file_id": case_file_id,
                "is_global": is_global,
                "upload_time": datetime.now().isoformat(),
                "file_type": os.path.splitext(file_path)[1].lower(),
                "text_size": len(extracted_text),
                "chunk_count": len(chunks)
            }
            
            # Update with additional metadata if provided
            if metadata:
                doc_metadata.update(metadata)
            
            # If document has many chunks, process in batches to avoid memory issues
            # This can happen with very large documents
            if len(chunks) > 200:
                logger.info(f"⚠️ DOCUMENT HAS MANY CHUNKS: {len(chunks)}. Processing in batches of 500")
                
                # Add document to vector store with improved batch processing
                await self.vector_store.add_document(
                    document_id=document_id,
                    chunks=chunks,
                    metadata=doc_metadata,
                    progress_callback=progress_reporter.report_progress
                )
            else:
                # For smaller documents, process all at once
                await self.vector_store.add_document(
                    document_id=document_id,
                    chunks=chunks,
                    metadata=doc_metadata,
                    progress_callback=progress_reporter.report_progress
                )
            
            # Mark document as processed
            document.status = "processed"
            document.processed_at = datetime.now()
            self.document_store[document_id] = document
            
            # Save document metadata
            self._save_document_metadata(document_id)
            
            # If this document is part of a case file, update the case file
            if case_file_id:
                try:
                    # Locate the case file manager
                    case_file_manager = CaseFileManager()
                    
                    # Update the case file with this document
                    case_file = case_file_manager.get_case_file(case_file_id)
                    if case_file:
                        # Add document to case file's document list
                        documents = case_file.documents.copy()
                        if document_id not in documents:
                            documents.append(document_id)
                        
                        # Update case file
                        case_file_manager.update_case_file(
                            case_file_id=case_file_id,
                            documents=documents
                        )
                        logger.info(f"✅ DOCUMENT ADDED TO CASE FILE: case_file_id={case_file_id}")
                    else:
                        logger.warning(f"⚠️ CASE FILE NOT FOUND: case_file_id={case_file_id}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO UPDATE CASE FILE: {str(e)}")
            
            processing_time = time.time() - start_time
            logger.info(f"✅ DOCUMENT PROCESSED: id={document_id}, chunks={len(chunks)}, time={processing_time:.2f}s")
            
            return document
            
        except Exception as e:
            logger.error(f"❌ DOCUMENT PROCESSING FAILED: {str(e)}")
            traceback.print_exc()
            
            # Update document status
            document = DocumentResponse(
                document_id=document_id,
                filename=os.path.basename(file_path),
                session_id=session_id,
                case_file_id=case_file_id,
                status="failed",
                error=str(e),
                is_global=is_global
            )
            self.document_store[document_id] = document
            
            # Save metadata
            self._save_document_metadata(document_id)
            
            return document
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from various document formats.
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif ext == '.txt':
            return self._extract_from_txt(file_path)
        elif ext in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self._extract_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        logger.info(f"🔄 EXTRACTING TEXT FROM PDF: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                pdf = pypdf.PdfReader(f)
                text = ""
                total_pages = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    if i % 5 == 0:  # Log progress every 5 pages
                        logger.info(f"📄 PDF EXTRACTION PROGRESS: page {i+1}/{total_pages}")
                    text += page.extract_text() + "\n"
                
                logger.info(f"✅ PDF EXTRACTION COMPLETED: {total_pages} pages, {len(text)} characters")
                return text
        except Exception as e:
            logger.error(f"❌ PDF EXTRACTION ERROR: {str(e)}")
            raise
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from a text file."""
        logger.info(f"🔄 EXTRACTING TEXT FROM TXT: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            logger.info(f"✅ TXT EXTRACTION COMPLETED: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"❌ TXT EXTRACTION ERROR: {str(e)}")
            raise
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from a Word document."""
        logger.info(f"🔄 EXTRACTING TEXT FROM DOCX: {file_path}")
        try:
            text = docx2txt.process(file_path)
            logger.info(f"✅ DOCX EXTRACTION COMPLETED: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"❌ DOCX EXTRACTION ERROR: {str(e)}")
            raise
    
    def _extract_from_image(self, file_path: str) -> str:
        """
        Extract text from an image using Tesseract OCR.
        This is a fallback if vision analysis isn't available.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Extracted text
        """
        try:
            logger.info(f"🔄 EXTRACTING TEXT FROM IMAGE: {file_path}")
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            logger.info(f"✅ TEXT EXTRACTED FROM IMAGE: length={len(text)}")
            return text
        except Exception as e:
            logger.error(f"❌ ERROR EXTRACTING TEXT FROM IMAGE: {str(e)}")
            return f"[Error extracting text from image: {str(e)}]"

    async def retry_failed_document(self, document_id: str, max_attempts: int = 3) -> Optional[DocumentResponse]:
        """
        Retry processing a failed document.
        
        Args:
            document_id: ID of the document to retry
            max_attempts: Maximum number of retry attempts
            
        Returns:
            Updated document response or None if retry failed
        """
        logger.info(f"🔄 RETRYING FAILED DOCUMENT: id={document_id}, max_attempts={max_attempts}")
        
        # Get the document
        document = self.get_document(document_id)
        if not document:
            logger.error(f"❌ DOCUMENT NOT FOUND FOR RETRY: id={document_id}")
            return None
        
        # Only retry if the document failed or is marked for retry
        if document.status not in ["failed", "retry_requested"]:
            logger.warning(f"⚠️ DOCUMENT NOT FAILED OR MARKED FOR RETRY: id={document_id}, status={document.status}")
            return document
        
        # Check if we have the original file
        original_file_path = None
        for dir_path in ["uploads", "direct_text_cache"]:
            # Check both the original name format and text file format
            potential_paths = [
                Path(f"{dir_path}/{document_id}_{document.filename}"),
                Path(f"{dir_path}/{document_id}.txt"),
                Path(f"{dir_path}/{document_id}")
            ]
            
            for path in potential_paths:
                if path.exists():
                    original_file_path = path
                    break
            
            if original_file_path:
                break
        
        if not original_file_path:
            logger.error(f"❌ ORIGINAL FILE NOT FOUND FOR RETRY: id={document_id}")
            return None
        
        logger.info(f"✅ ORIGINAL FILE FOUND: path={original_file_path}")
        
        # Update document status to indicate retry
        document.status = "processing"
        document.error = None
        self._save_document_metadata(document_id)
        
        # Try to process the document again
        retry_count = 0
        last_error = None
        
        while retry_count < max_attempts:
            retry_count += 1
            logger.info(f"🔄 RETRY ATTEMPT {retry_count}/{max_attempts}")
            
            try:
                # Process the document
                await self.process_document(
                    file_path=str(original_file_path),
                    document_id=document_id,
                    session_id=document.session_id,
                    case_file_id=document.case_file_id,
                    is_global=document.is_global
                )
                
                # Check if processing was successful
                updated_document = self.get_document(document_id)
                if updated_document and updated_document.status == "processed":
                    logger.info(f"✅ DOCUMENT RETRY SUCCESSFUL: id={document_id}")
                    return updated_document
                
                # Allow a short sleep between retries
                await asyncio.sleep(0.5)
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"❌ RETRY ATTEMPT {retry_count} FAILED: id={document_id}, error={last_error}")
                
                # Increment back-off delay for next retry
                await asyncio.sleep(retry_count * 0.5)
        
        # All retries failed
        document = self.get_document(document_id)
        if document:
            document.status = "failed"
            document.error = f"Retry failed after {max_attempts} attempts. Last error: {last_error or 'Unknown error'}"
            document.processed_at = datetime.now()
            self._save_document_metadata(document_id)
        
        logger.error(f"❌ ALL RETRY ATTEMPTS FAILED: id={document_id}, attempts={max_attempts}")
        return document

    async def recover_all_failed_documents(self, max_attempts: int = 2) -> Dict[str, str]:
        """
        Try to recover all failed documents.
        
        Args:
            max_attempts: Maximum number of retry attempts per document
            
        Returns:
            Dictionary mapping document IDs to their recovery status
        """
        logger.info(f"🔄 RECOVERING ALL FAILED DOCUMENTS: max_attempts={max_attempts}")
        
        # Get all failed documents
        failed_documents = [doc for doc in self.document_store.values() if doc.status == "failed"]
        
        if not failed_documents:
            logger.info("✅ NO FAILED DOCUMENTS TO RECOVER")
            return {}
        
        logger.info(f"🔄 FOUND {len(failed_documents)} FAILED DOCUMENTS TO RECOVER")
        
        # Try to recover each document
        results = {}
        for document in failed_documents:
            try:
                logger.info(f"🔄 ATTEMPTING TO RECOVER DOCUMENT: id={document.document_id}")
                recovered_document = await self.retry_failed_document(document.document_id, max_attempts)
                
                if recovered_document and recovered_document.status == "processed":
                    results[document.document_id] = "recovered"
                    logger.info(f"✅ DOCUMENT RECOVERED: id={document.document_id}")
                else:
                    results[document.document_id] = "failed"
                    logger.warning(f"⚠️ DOCUMENT RECOVERY FAILED: id={document.document_id}")
                
                # Allow a delay between processing documents
                await asyncio.sleep(0.5)
                
            except Exception as e:
                results[document.document_id] = f"error: {str(e)}"
                logger.error(f"❌ ERROR RECOVERING DOCUMENT: id={document.document_id}, error={str(e)}")
        
        logger.info(f"✅ RECOVERY PROCESS COMPLETED: {results}")
        return results

    # Add a method to mark a document for retry
    def mark_document_for_retry(self, document_id: str) -> Optional[DocumentResponse]:
        """
        Mark a document for retry processing.
        
        Args:
            document_id: ID of the document to mark for retry
            
        Returns:
            Updated document response or None if document not found
        """
        document = self.get_document(document_id)
        if not document:
            logger.error(f"❌ DOCUMENT NOT FOUND FOR RETRY MARKING: id={document_id}")
            return None
        
        # Mark for retry
        document.status = "retry_requested"
        document.error = None
        self._save_document_metadata(document_id)
        
        logger.info(f"✅ DOCUMENT MARKED FOR RETRY: id={document_id}")
        return document

    def _is_image_file(self, file_path: str) -> bool:
        """
        Check if the file is an image based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if this is an image file, False otherwise
        """
        file_extension = Path(file_path).suffix.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        return file_extension in image_extensions
    
    def _should_use_vision(self, file_path: str) -> bool:
        """
        Determine if we should use vision analysis for this file.
        Currently, we use vision for all image files, but this could
        be made more sophisticated based on file size, content, etc.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if vision analysis should be used
        """
        return self._is_image_file(file_path)

    async def _process_with_vision(self, file_path: str, document_id: str) -> str:
        """
        Process an image file using both OCR and vision model analysis.
        
        Args:
            file_path: Path to the image file
            document_id: Unique ID for the document
            
        Returns:
            Extracted text from the image
        """
        logger.info(f"🔄 PROCESSING IMAGE WITH VISION: id={document_id}")
        
        try:
            # First extract text using OCR for baseline content
            ocr_text = self._extract_from_image(file_path)
            logger.info(f"✅ OCR TEXT EXTRACTED: {len(ocr_text)} characters")
            
            # Then use vision model for enhanced understanding
            vision_prompt = """
            Analyze this image carefully. Focus on any text content, legal documents, notices, forms, 
            or other relevant information. Provide a detailed description of what you see, 
            making sure to capture all text content as accurately as possible.
            If there are any legal terms, clauses, or important information, please be extra
            careful to reproduce them exactly.
            """
            
            # Initialize vision service if not available
            if not hasattr(self, 'vision_service'):
                from app.core.vision_service import VisionService
                self.vision_service = VisionService()
            
            # Call vision API
            vision_result = await self.vision_service.analyze_image(
                file_path, 
                prompt=vision_prompt
            )
            
            vision_text = vision_result.get("analysis", "")
            logger.info(f"✅ VISION ANALYSIS COMPLETE: {len(vision_text)} characters")
            
            # Combine both sources, with vision analysis first (it's usually more comprehensive)
            # If vision analysis failed, use OCR text only
            if len(vision_text) > 50:  # Only use vision text if it's substantial
                combined_text = f"{vision_text}\n\n--- OCR EXTRACTED TEXT ---\n\n{ocr_text}"
            else:
                combined_text = ocr_text
                logger.warning(f"⚠️ VISION ANALYSIS PRODUCED LIMITED TEXT. USING OCR TEXT PRIMARILY.")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"❌ ERROR IN VISION PROCESSING: {str(e)}")
            # Fall back to OCR-only if vision fails
            return self._extract_from_image(file_path) 