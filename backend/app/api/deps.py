from typing import Callable
from functools import lru_cache
from fastapi import Depends

from app.core.rag_engine import RAGEngine
from app.core.case_file import CaseFileService

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