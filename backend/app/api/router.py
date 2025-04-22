from fastapi import APIRouter
from .endpoints import chat, documents, case_files

router = APIRouter()

# Include all API endpoint routers
router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(case_files.router, prefix="/case-files", tags=["Case Files"]) 