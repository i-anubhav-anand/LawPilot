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
        
        # Create a dedicated thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("processed", exist_ok=True)
        os.makedirs("documents_metadata", exist_ok=True)  # New directory for document metadata persistence
        logger.info("🚀 Document processor initialized. Directories created.")
        
        # Load document metadata from disk
        self._load_document_metadata()
        
        # Sync document store with vector store
        asyncio.create_task(self._sync_with_vector_store())
        
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
        """Load document metadata from disk."""
        metadata_dir = Path("documents_metadata")
        if not metadata_dir.exists():
            return
            
        logger.info("🔄 LOADING DOCUMENT METADATA FROM DISK")
        count = 0
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    doc_data = json.load(f)
                    
                # Convert string dates back to datetime objects
                if "created_at" in doc_data and doc_data["created_at"]:
                    doc_data["created_at"] = datetime.fromisoformat(doc_data["created_at"])
                if "processed_at" in doc_data and doc_data["processed_at"]:
                    doc_data["processed_at"] = datetime.fromisoformat(doc_data["processed_at"])
                    
                # Create DocumentResponse object and add to document store
                doc_response = DocumentResponse(**doc_data)
                self.document_store[doc_response.document_id] = doc_response
                count += 1
            except Exception as e:
                logger.error(f"❌ ERROR LOADING DOCUMENT METADATA: file={metadata_file}, error={str(e)}")
                
        logger.info(f"✅ LOADED {count} DOCUMENT METADATA RECORDS FROM DISK")
        
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
        Process a document by extracting text, chunking, and storing in vector DB.
        
        Args:
            file_path: Path to the document file
            document_id: Unique ID for the document
            session_id: Optional session ID to associate with this document
            case_file_id: Optional case file ID to associate with this document
            is_global: Whether this document should be available to all sessions
            metadata: Optional additional metadata about the document
        
        Returns:
            None - updates are made to the document store
        """
        # Create a document record
        document = DocumentResponse(
            document_id=document_id,
            filename=os.path.basename(file_path),
            session_id=session_id,
            case_file_id=case_file_id,
            status="processing",
            is_global=is_global
        )
        
        # Add to document store
        self.document_store[document_id] = document
        
        # Save initial document metadata
        self._save_document_metadata(document_id)
        
        try:
            logger.info(f"🔄 PROCESSING DOCUMENT: id={document_id}, file={file_path}")
            self.last_processing_time = time.time()
            self.is_processing = True
            self.processing_document_id = document_id
            
            # Initialize metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Check if this is an image file that needs vision analysis
            is_image = self._is_image_file(file_path)
            
            # Use vision if specified in metadata or if image and should use vision
            use_vision = metadata.get('use_vision', False) or (is_image and self._should_use_vision(file_path))
            
            # Extract text from the document
            if use_vision and is_image:
                logger.info(f"🔄 USING VISION MODEL FOR IMAGE ANALYSIS: {file_path}")
                vision_result = await self.vision_service.analyze_image(file_path)
                text = vision_result.get("analysis", "")
                
                # If there was an error with vision analysis, fall back to OCR
                if "error" in vision_result:
                    logger.warning(f"⚠️ VISION ANALYSIS FAILED, FALLING BACK TO OCR: {vision_result.get('error')}")
                    text = self._extract_text(file_path)
            else:
                # Use regular text extraction
                text = self._extract_text(file_path)
                
            if not text or len(text.strip()) == 0:
                logger.warning(f"⚠️ NO TEXT EXTRACTED FROM DOCUMENT: id={document_id}")
                raise ValueError("No text could be extracted from the document")
                
            logger.info(f"✅ TEXT EXTRACTED: id={document_id}, length={len(text)}")
            
            # Chunk the text with progress reporting
            logger.info(f"🔄 CHUNKING TEXT: document={document_id}")
            chunk_start_time = time.time()
            
            # Adaptive chunk size based on document size
            is_large_document = os.path.getsize(file_path) > 2 * (1024 * 1024)  # Consider files > 2MB as large
            
            # Create a special chunker instance with appropriate settings for this document
            document_chunker = TextChunker(
                chunk_size=300 if not is_large_document else 200, 
                chunk_overlap=50,
                max_chunk_time=60  # 1 minute max for chunking
            )
            
            try:
                chunking_task = self.thread_pool.submit(lambda: document_chunker.chunk_text(text))
                # Set timeout for chunking (30 seconds + 10 seconds per MB)
                chunking_timeout = 30 + (os.path.getsize(file_path) / (1024 * 1024) * 10)
                chunks = await asyncio.wait_for(chunking_task, timeout=chunking_timeout)
            except asyncio.TimeoutError:
                logger.error(f"⚠️ CHUNKING TIMEOUT: Text chunking took longer than {chunking_timeout} seconds")
                # If chunking times out, use emergency chunking
                logger.info("🆘 Using emergency chunking method")
                chunks = document_chunker._emergency_chunking(text)
            
            chunking_time = time.time() - chunk_start_time
            logger.info(f"✅ TEXT CHUNKED: {len(chunks)} chunks created in {chunking_time:.2f}s")
            
            # Log chunk details
            for i, chunk in enumerate(chunks):
                if i % 10 == 0 or i == len(chunks) - 1:  # Log every 10th chunk and the last one
                    logger.info(f"🔹 CHUNK {i+1}/{len(chunks)}: {len(chunk)} characters")
            
            # Yield control back to event loop
            await asyncio.sleep(0.1)
            
            # For extremely large documents or too many chunks, split processing into batches
            max_chunks_per_batch = 25  # Smaller batch size to avoid memory issues
            if len(chunks) > max_chunks_per_batch or is_large_document:
                logger.info(f"⚠️ DOCUMENT HAS MANY CHUNKS: {len(chunks)}. Processing in batches of {max_chunks_per_batch}")
                
                all_chunks = chunks
                processed_chunks = 0
                
                while processed_chunks < len(all_chunks):
                    # Process the next batch
                    batch = all_chunks[processed_chunks:processed_chunks + max_chunks_per_batch]
                    batch_end = min(processed_chunks + max_chunks_per_batch, len(all_chunks))
                    logger.info(f"🔄 PROCESSING CHUNK BATCH: {processed_chunks+1}-{batch_end} of {len(all_chunks)}")
                    
                    # Store chunks in vector store
                    logger.info(f"🔄 ADDING BATCH TO VECTOR STORE: chunks={len(batch)}")
                    start_embedding_time = time.time()
                    
                    # Add progress report callback
                    progress_reporter = EmbeddingProgressReporter(len(batch))
                    
                    try:
                        # Set timeout for embedding generation (30 seconds + 5 seconds per chunk)
                        embedding_timeout = 30 + (len(batch) * 5)
                        embedding_task = self.vector_store.add_document(
                            document_id=document_id,
                            chunks=batch,
                            metadata={
                                "filename": os.path.basename(file_path),
                                "session_id": session_id,
                                "case_file_id": case_file_id,
                                "batch": f"{processed_chunks+1}-{batch_end}/{len(all_chunks)}",
                                "is_global": is_global
                            },
                            progress_callback=progress_reporter.report_progress
                        )
                        await asyncio.wait_for(embedding_task, timeout=embedding_timeout)
                    except asyncio.TimeoutError:
                        logger.error(f"⚠️ EMBEDDING TIMEOUT: Batch {processed_chunks+1}-{batch_end} took too long")
                        # Continue with the next batch even if this one timed out
                    except Exception as e:
                        logger.error(f"⚠️ EMBEDDING ERROR: {str(e)}")
                        # Continue with the next batch even if there was an error
                    
                    embedding_time = time.time() - start_embedding_time
                    logger.info(f"✅ BATCH EMBEDDED: {len(batch)} chunks, time={embedding_time:.2f}s")
                    
                    # Update processed count
                    processed_chunks += len(batch)
                    
                    # Allow other tasks to run - important for API responsiveness
                    await asyncio.sleep(0.5)
                
                logger.info(f"✅ ALL BATCHES PROCESSED: {len(all_chunks)} chunks in total")
            else:
                # Process all chunks at once for smaller documents
                # Store chunks in vector store
                logger.info(f"🔄 ADDING DOCUMENT TO VECTOR STORE: id={document_id}, chunks={len(chunks)}")
                start_embedding_time = time.time()
                
                logger.info(f"⏳ EMBEDDING GENERATION STARTING: This may take some time...")
                
                # Add progress report callback
                progress_reporter = EmbeddingProgressReporter(len(chunks))
                
                try:
                    # Set timeout for embedding generation (30 seconds + 2 seconds per chunk)
                    embedding_timeout = 30 + (len(chunks) * 2)
                    await asyncio.wait_for(
                        self.vector_store.add_document(
                            document_id=document_id,
                            chunks=chunks,
                            metadata={
                                "filename": os.path.basename(file_path),
                                "session_id": session_id,
                                "case_file_id": case_file_id,
                                "is_global": is_global
                            },
                            progress_callback=progress_reporter.report_progress
                        ),
                        timeout=embedding_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⚠️ EMBEDDING TIMEOUT: Embedding generation took longer than {embedding_timeout} seconds")
                    # Continue even if embedding times out
                
                embedding_time = time.time() - start_embedding_time
                logger.info(f"✅ DOCUMENT EMBEDDED: id={document_id}, time={embedding_time:.2f}s")
            
            # Save processed text
            logger.info(f"🔄 SAVING PROCESSED DOCUMENT: id={document_id}")
            processed_path = Path("processed") / f"{document_id}.json"
            
            processed_data = {
                "document_id": document_id,
                "filename": os.path.basename(file_path),
                "text": text,
                "chunks": chunks,
                "session_id": session_id,
                "case_file_id": case_file_id,
                "processing_stats": {
                    "text_extraction_time": time.time() - self.last_processing_time,
                    "chunking_time": chunking_time,
                    "embedding_time": time.time() - self.last_processing_time - chunking_time,
                    "num_chunks": len(chunks),
                    "total_characters": len(text)
                }
            }
            
            # Write to file in thread pool
            await asyncio.run_in_executor(
                self.thread_pool, 
                lambda: json.dump(processed_data, open(processed_path, "w"))
            )
            
            logger.info(f"✅ PROCESSED DOCUMENT SAVED: {processed_path}")
            
            # Update document status
            self.document_store[document_id] = DocumentResponse(
                document_id=document_id,
                filename=os.path.basename(file_path),
                session_id=session_id,
                case_file_id=case_file_id,
                status="processed",
                created_at=self.document_store[document_id].created_at,
                processed_at=datetime.now(),
                is_global=is_global
            )
            
            # Save updated metadata
            self._save_document_metadata(document_id)
            
            total_processing_time = time.time() - self.last_processing_time
            logger.info(f"✅ DOCUMENT PROCESSING COMPLETED: id={document_id}, total_time={total_processing_time:.2f}s")
            
            # If document is part of a case file, update case file
            if case_file_id:
                try:
                    from app.core.case_file_manager import CaseFileManager
                    case_file_manager = CaseFileManager()
                    case_file = case_file_manager.get_case_file(case_file_id)
                    if case_file:
                        documents = case_file.documents.copy()
                        if document_id not in documents:
                            documents.append(document_id)
                        case_file_manager.update_case_file(case_file_id, documents=documents)
                        logger.info(f"✅ CASE FILE UPDATED: case_file_id={case_file_id}, document_id={document_id}")
                except Exception as e:
                    logger.error(f"⚠️ ERROR UPDATING CASE FILE: {str(e)}")
            
        except TimeoutError as e:
            logger.error(f"❌ TIMEOUT ERROR PROCESSING DOCUMENT: id={document_id}, error={str(e)}")
            # Update document with error
            self.document_store[document_id] = DocumentResponse(
                document_id=document_id,
                filename=os.path.basename(file_path),
                session_id=session_id,
                case_file_id=case_file_id,
                status="failed",
                created_at=self.document_store[document_id].created_at,
                error=f"Processing timed out: {str(e)}"
            )
            # Save error metadata
            self._save_document_metadata(document_id)
        except Exception as e:
            logger.error(f"❌ ERROR PROCESSING DOCUMENT: id={document_id}, error={str(e)}", exc_info=True)
            # Update document with error
            self.document_store[document_id] = DocumentResponse(
                document_id=document_id,
                filename=os.path.basename(file_path),
                session_id=session_id,
                case_file_id=case_file_id,
                status="failed",
                created_at=self.document_store[document_id].created_at,
                error=str(e)
            )
            # Save error metadata
            self._save_document_metadata(document_id)
    
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