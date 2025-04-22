from typing import Callable
from functools import lru_cache
from fastapi import Depends

from app.core.rag_engine import RAGEngine
from app.core.case_file import CaseFileService
from app.core.document_processor import DocumentProcessor
from app.core.direct_processor import DirectTextProcessor
from app.core.vision_service import VisionService

# Use lru_cache to create singleton instances of services
@lru_cache()
def get_rag_engine() -> RAGEngine:
    """
    Returns a singleton instance of the RAG Engine.
    This ensures we reuse the same engine instance across requests.
    """
    return RAGEngine()

@lru_cache()
def get_case_file_service() -> CaseFileService:
    """
    Returns a singleton instance of the Case File Service.
    This ensures we reuse the same service instance across requests.
    """
    return CaseFileService()

@lru_cache()
def get_document_processor() -> DocumentProcessor:
    """
    Returns a singleton instance of the Document Processor.
    This ensures we reuse the same processor instance across requests.
    """
    return DocumentProcessor()

@lru_cache()
def get_direct_text_processor() -> DirectTextProcessor:
    """
    Returns a singleton instance of the Direct Text Processor.
    This enables immediate document processing without waiting for indexing.
    """
    return DirectTextProcessor()

@lru_cache()
def get_vision_service() -> VisionService:
    """
    Returns a singleton instance of the Vision Service.
    This provides image analysis capabilities.
    """
    return VisionService() 