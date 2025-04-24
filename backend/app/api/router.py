from fastapi import APIRouter
from .endpoints import chat, documents, case_files, summary
from datetime import datetime

router = APIRouter()

# Include all API endpoint routers
router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(case_files.router, prefix="/case-files", tags=["Case Files"])
router.include_router(summary.router, prefix="/summaries", tags=["Summaries"])

# Add API-specific health check endpoint
@router.get("/health", tags=["Health"])
async def api_health_check():
    """
    API-specific health check that always responds immediately.
    This endpoint is designed to be extremely fast for API health monitoring.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()} 