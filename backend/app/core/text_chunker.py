from typing import List, Dict, Any, Optional
import re
import logging
import time
import concurrent.futures
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_chunker")

class TextChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50, max_chunk_time: int = 30):
        """
        Initialize a text chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            max_chunk_time: Maximum time in seconds to spend on chunking before using a simpler method.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_time = max_chunk_time
        logger.info(f"TextChunker initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, max_chunk_time={max_chunk_time}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap, with timeout protection.
        
        Args:
            text: The text to chunk.
            
        Returns:
            List of text chunks.
        """
        logger.info(f"Chunking text of length {len(text)} characters")
        
        # For very large texts or small texts, use simple chunking immediately
        if len(text) > 100000 or len(text) < 1000:
            logger.info(f"Text length ({len(text)} chars) triggers immediate simple chunking")
            return self._failsafe_chunking(text)
        
        # Create a thread pool for chunking with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Start timer
            start_time = time.time()
            
            # Submit semantic chunking task
            try:
                # Try semantic chunking with a timeout
                semantic_task = executor.submit(self._semantic_chunking, text)
                semantic_chunks = semantic_task.result(timeout=self.max_chunk_time / 2)
                
                if semantic_chunks:
                    logger.info(f"Semantic chunking successful, found {len(semantic_chunks)} semantic chunks")
                    # Process large semantic chunks
                    result = []
                    for i, chunk in enumerate(semantic_chunks):
                        if i % 5 == 0 or i == len(semantic_chunks) - 1:
                            logger.info(f"Processing semantic chunk {i+1}/{len(semantic_chunks)}")
                        
                        if len(chunk) <= self.chunk_size:
                            result.append(chunk)
                        else:
                            # Use simple chunking for large semantic chunks
                            local_chunks = self._simple_chunk_by_size(chunk)
                            result.extend(local_chunks)
                            
                    logger.info(f"Chunking completed in {time.time() - start_time:.2f}s, {len(result)} chunks created")
                    return result
                else:
                    logger.info("No semantic chunks found, falling back to simple chunking")
            except concurrent.futures.TimeoutError:
                logger.warning(f"Semantic chunking timed out after {self.max_chunk_time/2:.1f}s, falling back to simple chunking")
            except Exception as e:
                logger.error(f"Error during semantic chunking: {str(e)}, falling back to simple chunking")
        
        # Fall back to simple chunking
        return self._failsafe_chunking(text)
    
    def _failsafe_chunking(self, text: str) -> List[str]:
        """Simple, reliable chunking method that won't get stuck."""
        logger.info("Using failsafe chunking method")
        start_time = time.time()
        
        try:
            # Use the simplest, most reliable chunking method
            chunks = self._simple_chunk_by_size(text)
            logger.info(f"Failsafe chunking completed in {time.time() - start_time:.2f}s, {len(chunks)} chunks created")
            return chunks
        except Exception as e:
            logger.error(f"Error during failsafe chunking: {str(e)}, using emergency chunking")
            # Emergency chunking - just split by a fixed size with no fancy processing
            return self._emergency_chunking(text)
    
    def _simple_chunk_by_size(self, text: str) -> List[str]:
        """Very simple chunking that won't get stuck in loops."""
        chunks = []
        start = 0
        chunk_count = 0
        max_chunk_count = (len(text) // (self.chunk_size - self.chunk_overlap)) + 2  # Safety upper bound
        
        while start < len(text) and chunk_count < max_chunk_count:
            chunk_count += 1
            
            # Calculate end position, ensuring we don't go beyond text length
            end = min(start + self.chunk_size, len(text))
            
            # Only try to find a better boundary if we're not at the end of the text
            if end < len(text):
                # Try to find paragraph, sentence, or word boundaries
                paragraph_end = text.find('\n\n', start, end)
                if paragraph_end != -1:
                    end = paragraph_end + 2
                else:
                    sentence_end = text.rfind('. ', start, end)
                    if sentence_end != -1:
                        end = sentence_end + 2
                    else:
                        space = text.rfind(' ', start, end)
                        if space != -1:
                            end = space + 1
            
            # Extract the chunk and add it
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Log progress for large texts
            if len(text) > 50000 and (chunk_count % 10 == 0 or chunk_count == 1):
                percent_complete = (end / len(text)) * 100
                logger.info(f"Chunking progress: {percent_complete:.1f}% ({chunk_count} chunks created)")
            
            # Move to next chunk, handling overlap
            start = end - self.chunk_overlap
            # Ensure we're making progress
            if start >= end:
                start = end
            # Safety check to prevent infinite loops
            if start == end and end < len(text):
                start += 1
        
        if chunk_count >= max_chunk_count:
            logger.warning(f"Reached maximum chunk count ({max_chunk_count}), chunking may be incomplete")
        
        return chunks
    
    def _emergency_chunking(self, text: str) -> List[str]:
        """Absolute last resort chunking that just splits by fixed size."""
        logger.warning("Using emergency chunking - splitting text at fixed intervals")
        chunks = []
        total_chunks = (len(text) + self.chunk_size - 1) // self.chunk_size  # Ceiling division
        
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i+self.chunk_size]
            chunks.append(chunk)
            
            # Log progress
            if i % (10 * self.chunk_size) == 0:
                current_chunk = i // self.chunk_size
                logger.info(f"Emergency chunking progress: {current_chunk}/{total_chunks} chunks")
        
        logger.info(f"Emergency chunking completed: {len(chunks)} chunks created")
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Try to chunk text based on semantic structure like sections, paragraphs, etc.
        Uses a faster, more efficient algorithm with better boundary detection.
        
        Returns empty list if no clear structure is detected.
        """
        # First look for standard section headers - very common in legal/technical documents
        section_patterns = [
            # Standard section headers (Section 1, Section 1.2, etc.)
            r'(?:^|\n)(?:Section|§)\s+\d+(?:\.\d+)*(?:[a-z])?(?:\s+[A-Z][^.]*)?',
            # Numbered sections (1., 1.1., I., A., etc.)
            r'(?:^|\n)(?:\d+\.|\d+\.\d+\.|\([a-z0-9]+\)|[IVXLCDM]+\.|\w\.)(?:\s+[A-Z][^.]*)?',
            # Title case headers
            r'(?:^|\n)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}:)',
            # ALL CAPS headers
            r'(?:^|\n)(?:[A-Z]{2,}(?:\s+[A-Z]+){0,4})(?:\n)'
        ]
        
        # Try each pattern to find a structured document
        for pattern in section_patterns:
            try:
                start_time = time.time()
                sections = self._find_sections(text, pattern)
                
                if sections and len(sections) > 1:
                    duration = time.time() - start_time
                    logger.info(f"Found {len(sections)} sections using pattern in {duration:.2f}s")
                    return sections
            except Exception as e:
                logger.warning(f"Error while trying section pattern: {str(e)}")
        
        # If no sections found, try splitting by paragraphs
        try:
            # Split by double newlines - common paragraph separator
            paragraphs = re.split(r'\n\s*\n', text)
            
            if len(paragraphs) > 1:
                logger.info(f"Found {len(paragraphs)} paragraphs")
                return self._group_paragraphs(paragraphs)
        except Exception as e:
            logger.warning(f"Error while splitting by paragraphs: {str(e)}")
        
        # No clear structure detected
        logger.info("No clear document structure detected")
        return []
    
    def _find_sections(self, text: str, pattern: str) -> List[str]:
        """Find sections in text using a regex pattern."""
        section_matches = list(re.finditer(pattern, text, re.MULTILINE))
        
        if not section_matches:
            return []
        
        sections = []
        
        # Process each section
        for i, match in enumerate(section_matches):
            start = match.start()
            
            # Find the end of this section (start of next section or end of text)
            if i < len(section_matches) - 1:
                end = section_matches[i+1].start()
            else:
                end = len(text)
            
            # Extract the section content
            section_content = text[start:end].strip()
            if section_content:
                sections.append(section_content)
        
        # Check for content before the first section
        if section_matches and section_matches[0].start() > 0:
            preamble = text[:section_matches[0].start()].strip()
            if preamble:
                sections.insert(0, preamble)
        
        return sections
    
    def _group_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Group small paragraphs into chunks of appropriate size."""
        result = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            stripped_paragraph = paragraph.strip()
            
            # Skip empty paragraphs
            if not stripped_paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size, start a new chunk
            if len(current_chunk) + len(stripped_paragraph) > self.chunk_size and current_chunk:
                result.append(current_chunk)
                current_chunk = stripped_paragraph
            else:
                # Add to current chunk with a separator if not empty
                if current_chunk:
                    current_chunk += "\n\n" + stripped_paragraph
                else:
                    current_chunk = stripped_paragraph
        
        # Add the last chunk if not empty
        if current_chunk:
            result.append(current_chunk)
        
        return result 