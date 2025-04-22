import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re

from app.core.text_chunker import TextChunker
from app.core.vector_store import VectorStore

class CorpusLoader:
    def __init__(self):
        """Initialize the corpus loader utility."""
        self.text_chunker = TextChunker()
        self.vector_store = VectorStore()
        
        # Create directory for storing legal corpus
        os.makedirs("legal_corpus", exist_ok=True)
    
    async def load_corpus_file(self, file_path: str, corpus_type: str = "law") -> str:
        """
        Load and index a legal corpus file.
        
        Args:
            file_path: Path to the corpus file.
            corpus_type: Type of corpus (law, ordinance, etc.).
            
        Returns:
            ID of the indexed corpus.
        """
        corpus_id = f"{corpus_type}_{Path(file_path).stem}"
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Chunk the content
        chunks = self.text_chunker.chunk_text(content)
        
        # Index in vector store
        self.vector_store.add_document(
            document_id=corpus_id,
            chunks=chunks,
            metadata={
                "filename": Path(file_path).name,
                "corpus_type": corpus_type,
                "path": file_path
            }
        )
        
        # Save a copy in the legal_corpus directory
        corpus_path = Path("legal_corpus") / f"{corpus_id}.txt"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return corpus_id
    
    async def load_corpus_directory(self, directory_path: str, corpus_type: str = "law") -> List[str]:
        """
        Load and index all text files in a directory as legal corpus.
        
        Args:
            directory_path: Path to the directory containing corpus files.
            corpus_type: Type of corpus (law, ordinance, etc.).
            
        Returns:
            List of corpus IDs indexed.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        corpus_ids = []
        for file_path in dir_path.glob("*.txt"):
            corpus_id = await self.load_corpus_file(str(file_path), corpus_type)
            corpus_ids.append(corpus_id)
        
        return corpus_ids
    
    async def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from legal text for better organization.
        
        Args:
            text: The legal text to process.
            
        Returns:
            Dictionary mapping section identifiers to their content.
        """
        # Pattern for matching section identifiers
        # This pattern aims to match common legal section formats
        section_pattern = r'(?:Section|§)\s+(\d+[a-z]?(?:\.\d+)?(?:\([a-z0-9]+\))?)'
        
        # Find all section identifiers
        sections = {}
        current_section = None
        current_text = []
        
        for line in text.split('\n'):
            section_match = re.search(section_pattern, line)
            if section_match:
                # If we have a previous section, save it
                if current_section is not None:
                    sections[current_section] = '\n'.join(current_text)
                
                # Start a new section
                current_section = section_match.group(1)
                current_text = [line]
            elif current_section is not None:
                current_text.append(line)
        
        # Add the last section
        if current_section is not None:
            sections[current_section] = '\n'.join(current_text)
        
        return sections
    
    async def format_legal_text(self, text: str, title: str) -> str:
        """
        Format legal text with proper metadata for better indexing.
        
        Args:
            text: The legal text to format.
            title: Title of the legal document.
            
        Returns:
            Formatted text with metadata.
        """
        header = f"TITLE: {title}\n\n"
        
        # Try to extract sections
        sections = await self.extract_sections(text)
        
        if sections:
            # Format with sections
            formatted_text = header
            for section_id, section_text in sections.items():
                formatted_text += f"SECTION: {section_id}\n\n{section_text}\n\n"
            return formatted_text
        else:
            # No clear sections found, just add the header
            return header + text 