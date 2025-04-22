import asyncio
import logging
import time
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
import os

from app.models.documents import DocumentResponse
from app.core.text_chunker import TextChunker

# Configure logging
logger = logging.getLogger("direct_processor")

class DirectTextProcessor:
    """
    Processor for handling direct text input without waiting for indexing.
    This enables the dual-path approach for real-time document processing.
    """
    
    def __init__(self):
        """Initialize the direct text processor."""
        self.text_chunker = TextChunker()
        
        # Ensure temp directory exists
        os.makedirs("direct_text_cache", exist_ok=True)
    
    async def process_text(
        self,
        text: str,
        document_name: str,
        query: str,
        session_id: Optional[str] = None,
        case_file_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process raw text directly for immediate use in chat responses.
        This bypasses the embedding and indexing process for instant response.
        
        Args:
            text: The raw text to process
            document_name: Name of the document or text source
            query: The user's query to contextualize the processing
            session_id: Optional session ID
            case_file_id: Optional case file ID
            
        Returns:
            Dictionary containing processed chunks and metadata
        """
        logger.info(f"🔄 DIRECT TEXT PROCESSING: length={len(text)}, query='{query}'")
        
        # Generate a unique ID for this text
        text_id = f"direct_{str(uuid.uuid4())}"
        
        # Cache the text for potential later indexing
        cache_path = Path(f"direct_text_cache/{text_id}.txt")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        logger.info(f"✅ TEXT CACHED: path={cache_path}")
        
        # Chunk the text
        start_time = time.time()
        chunks = self.text_chunker.chunk_text(text)
        chunking_time = time.time() - start_time
        
        logger.info(f"✅ TEXT CHUNKED: {len(chunks)} chunks in {chunking_time:.2f}s")
        
        # If the query is short, use it for relevance filtering
        # This helps focus on the most relevant parts of the document
        relevant_chunks = chunks
        if query and len(query) > 10 and len(chunks) > 3:
            relevant_chunks = self._filter_chunks_by_relevance(chunks, query)
            logger.info(f"✅ CHUNKS FILTERED: {len(relevant_chunks)}/{len(chunks)} chunks kept")
        
        # Create a document response object
        document_response = DocumentResponse(
            document_id=text_id,
            filename=document_name,
            session_id=session_id,
            case_file_id=case_file_id,
            status="direct_processed",  # Special status for direct processing
            is_global=False
        )
        
        return {
            "document": document_response,
            "chunks": relevant_chunks,
            "all_chunks": chunks,
            "cache_path": str(cache_path),
            "processing_time": chunking_time
        }
    
    def _filter_chunks_by_relevance(self, chunks: List[str], query: str, max_chunks: int = 10) -> List[str]:
        """
        Filter chunks based on simple keyword relevance to the query.
        This is a basic filtering method that doesn't require embeddings.
        
        Args:
            chunks: List of text chunks
            query: The query to filter by
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of most relevant chunks
        """
        # Simple relevance scoring based on keyword matching
        # For a production system, you would want a more sophisticated approach
        query_terms = set(query.lower().split())
        
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            score = 0
            chunk_lower = chunk.lower()
            
            # Score based on term frequency
            for term in query_terms:
                if term in chunk_lower:
                    score += chunk_lower.count(term)
            
            # Boost score for exact phrase matches
            if query.lower() in chunk_lower:
                score += 10
                
            chunk_scores.append((i, score))
        
        # Sort by score and get top chunks
        top_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:max_chunks]
        
        # Sort by original order to maintain document flow
        top_chunks.sort(key=lambda x: x[0])
        
        # Return the filtered chunks
        return [chunks[i] for i, _ in top_chunks]
    
    async def schedule_background_indexing(
        self,
        text_id: str,
        document_processor,
        session_id: Optional[str] = None,
        case_file_id: Optional[str] = None
    ):
        """
        Schedule the text for background indexing in the vector store.
        
        Args:
            text_id: ID of the cached text
            document_processor: Document processor instance
            session_id: Optional session ID
            case_file_id: Optional case file ID
        """
        logger.info(f"🔄 SCHEDULING BACKGROUND INDEXING: text_id={text_id}")
        
        # Check if the text file exists in cache
        cache_path = Path(f"direct_text_cache/{text_id}.txt")
        if not cache_path.exists():
            logger.error(f"❌ CACHED TEXT NOT FOUND: path={cache_path}")
            return
        
        # Create a background task for indexing
        try:
            # Start processing as a background task
            asyncio.create_task(document_processor.process_document(
                str(cache_path),
                text_id,
                session_id,
                case_file_id,
                is_global=False
            ))
            
            logger.info(f"✅ BACKGROUND INDEXING STARTED: text_id={text_id}")
        except Exception as e:
            logger.error(f"❌ ERROR SCHEDULING BACKGROUND INDEXING: {str(e)}") 