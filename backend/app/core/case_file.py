from typing import List, Optional
from app.core.case_file_manager import CaseFileManager
from app.models.case_file import CaseFile

class CaseFileService:
    """
    Service wrapper for CaseFileManager to ensure consistent interface.
    """
    def __init__(self):
        self.manager = CaseFileManager()
    
    def get_case_file(self, case_file_id: str) -> Optional[CaseFile]:
        """Get a case file by ID."""
        return self.manager.get_case_file(case_file_id)
    
    def create_case_file(self, case_file_id: str, title: str, description: Optional[str] = None, session_id: str = "") -> CaseFile:
        """Create a new case file."""
        return self.manager.create_case_file(case_file_id, title, description, session_id)
    
    def update_case_file(self, case_file_id: str, **kwargs) -> Optional[CaseFile]:
        """Update a case file."""
        return self.manager.update_case_file(case_file_id, **kwargs)
    
    def list_case_files(self, session_id: Optional[str] = None) -> List[CaseFile]:
        """List all case files, optionally filtered by session ID."""
        return self.manager.list_case_files(session_id)
    
    def delete_case_file(self, case_file_id: str) -> bool:
        """Delete a case file."""
        return self.manager.delete_case_file(case_file_id) 