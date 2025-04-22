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
        self.document_store = {}  # In-memory store, replace with database in production
        self.vector_store = VectorStore()
        # Use smaller chunk size by default to avoid getting stuck
        self.text_chunker = TextChunker(chunk_size=300, chunk_overlap=50, max_chunk_time=60)
        
        # Create a dedicated thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("processed", exist_ok=True)
        logger.info("🚀 Document processor initialized. Uploads and processed directories created.")
        
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
        is_global: bool = False
    ):
        """
        Process a document: extract text, chunk it, and store in vector DB.
        
        Args:
            file_path: Path to the document file
            document_id: Unique ID for the document
            session_id: Optional session ID for associating with a chat session
            case_file_id: Optional case file ID for associating with a case file
            is_global: Whether this document should be available to all sessions
        """
        try:
            # Initialize document in store
            self.document_store[document_id] = DocumentResponse(
                document_id=document_id,
                filename=Path(file_path).name,
                session_id=session_id,
                case_file_id=case_file_id,
                status="processing",
                is_global=is_global
            )
            
            logger.info(f"🔄 STARTED PROCESSING DOCUMENT: id={document_id}, file={Path(file_path).name}, global={is_global}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"📊 DOCUMENT SIZE: {file_size_mb:.2f} MB")
            
            # For large documents, use a different chunking strategy
            is_large_document = file_size_mb > 2  # Consider files > 2MB as large
            
            # Extract text from document
            start_time = time.time()
            logger.info(f"🔄 EXTRACTING TEXT: file={file_path}")
            
            # Run text extraction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            try:
                extraction_task = loop.run_in_executor(self.thread_pool, lambda: self._extract_text(file_path))
                # Set a timeout for text extraction based on file size (30 seconds + 5 seconds per MB)
                extraction_timeout = 30 + (file_size_mb * 5)
                text = await asyncio.wait_for(extraction_task, timeout=extraction_timeout)
            except asyncio.TimeoutError:
                logger.error(f"⚠️ TEXT EXTRACTION TIMEOUT: File processing took longer than {extraction_timeout} seconds")
                raise TimeoutError(f"Text extraction timed out after {extraction_timeout} seconds")
            
            text_extraction_time = time.time() - start_time
            logger.info(f"✅ TEXT EXTRACTED: {len(text)} characters, time={text_extraction_time:.2f}s")
            
            # Chunk the text with progress reporting
            logger.info(f"🔄 CHUNKING TEXT: document={document_id}")
            chunk_start_time = time.time()
            
            # Adaptive chunk size based on document size
            if is_large_document:
                logger.info(f"⚠️ LARGE DOCUMENT DETECTED: Using smaller chunk size")
                chunk_size = 200
                chunk_overlap = 50
            else:
                chunk_size = 300
                chunk_overlap = 50
            
            # Create a special chunker instance with appropriate settings for this document
            document_chunker = TextChunker(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                max_chunk_time=60  # 1 minute max for chunking
            )
            
            try:
                chunking_task = loop.run_in_executor(self.thread_pool, lambda: document_chunker.chunk_text(text))
                # Set timeout for chunking (30 seconds + 10 seconds per MB)
                chunking_timeout = 30 + (file_size_mb * 10)
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
                                "filename": Path(file_path).name,
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
                                "filename": Path(file_path).name,
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
                "filename": Path(file_path).name,
                "text": text,
                "chunks": chunks,
                "session_id": session_id,
                "case_file_id": case_file_id,
                "processing_stats": {
                    "text_extraction_time": text_extraction_time,
                    "chunking_time": chunking_time,
                    "embedding_time": time.time() - start_time - text_extraction_time - chunking_time,
                    "num_chunks": len(chunks),
                    "total_characters": len(text)
                }
            }
            
            # Write to file in thread pool
            await loop.run_in_executor(
                self.thread_pool, 
                lambda: json.dump(processed_data, open(processed_path, "w"))
            )
            
            logger.info(f"✅ PROCESSED DOCUMENT SAVED: {processed_path}")
            
            # Update document status
            self.document_store[document_id] = DocumentResponse(
                document_id=document_id,
                filename=Path(file_path).name,
                session_id=session_id,
                case_file_id=case_file_id,
                status="processed",
                created_at=self.document_store[document_id].created_at,
                processed_at=datetime.now()
            )
            
            total_processing_time = time.time() - start_time
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
                filename=Path(file_path).name,
                session_id=session_id,
                case_file_id=case_file_id,
                status="failed",
                created_at=self.document_store[document_id].created_at,
                error=f"Processing timed out: {str(e)}"
            )
        except Exception as e:
            logger.error(f"❌ ERROR PROCESSING DOCUMENT: id={document_id}, error={str(e)}", exc_info=True)
            # Update document with error
            self.document_store[document_id] = DocumentResponse(
                document_id=document_id,
                filename=Path(file_path).name,
                session_id=session_id,
                case_file_id=case_file_id,
                status="failed",
                created_at=self.document_store[document_id].created_at,
                error=str(e)
            )
    
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
        """Extract text from an image using OCR."""
        logger.info(f"🔄 EXTRACTING TEXT FROM IMAGE USING OCR: {file_path}")
        try:
            text = pytesseract.image_to_string(Image.open(file_path))
            logger.info(f"✅ OCR EXTRACTION COMPLETED: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"❌ OCR EXTRACTION ERROR: {str(e)}")
            raise 