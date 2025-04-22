import yaml
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from app.models.case_file import CaseFile

class CaseFileManager:
    def __init__(self):
        """Initialize the case file manager."""
        self.case_files = {}
        
        # Create directory for storing case files
        os.makedirs("case_files", exist_ok=True)
        
        # Load existing case files
        self._load_case_files()
    
    def _load_case_files(self):
        """Load existing case files from disk."""
        case_files_dir = Path("case_files")
        for file_path in case_files_dir.glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    case_file_data = yaml.safe_load(f)
                    
                case_file_id = case_file_data.get("case_file_id")
                if case_file_id:
                    created_at = case_file_data.get("created_at")
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at)
                        
                    updated_at = case_file_data.get("updated_at")
                    if isinstance(updated_at, str):
                        updated_at = datetime.fromisoformat(updated_at)
                    
                    self.case_files[case_file_id] = CaseFile(
                        case_file_id=case_file_id,
                        title=case_file_data.get("title", "Untitled Case"),
                        description=case_file_data.get("description"),
                        session_id=case_file_data.get("session_id", ""),
                        created_at=created_at or datetime.now(),
                        updated_at=updated_at or datetime.now(),
                        facts=case_file_data.get("facts", {}),
                        documents=case_file_data.get("documents", [])
                    )
            except Exception as e:
                print(f"Error loading case file {file_path}: {e}")
    
    def _save_case_file(self, case_file_id: str):
        """Save a case file to disk."""
        case_file = self.case_files.get(case_file_id)
        if not case_file:
            return
        
        # Convert to dict for saving
        case_file_dict = case_file.dict()
        
        # Convert datetime to string
        case_file_dict["created_at"] = case_file.created_at.isoformat()
        case_file_dict["updated_at"] = case_file.updated_at.isoformat()
        
        # Save as YAML
        file_path = Path("case_files") / f"{case_file_id}.yaml"
        with open(file_path, 'w') as f:
            yaml.dump(case_file_dict, f, default_flow_style=False, sort_keys=False)
        
        # Also save as JSON for easier programmatic access
        json_path = Path("case_files") / f"{case_file_id}.json"
        with open(json_path, 'w') as f:
            json.dump(case_file_dict, f, indent=2)
    
    def create_case_file(
        self,
        case_file_id: str,
        title: str,
        description: Optional[str] = None,
        session_id: str = ""
    ) -> CaseFile:
        """
        Create a new case file.
        
        Args:
            case_file_id: Unique ID for the case file.
            title: Title of the case file.
            description: Optional description of the case file.
            session_id: Optional session ID to associate with this case file.
            
        Returns:
            The newly created case file.
        """
        now = datetime.now()
        
        case_file = CaseFile(
            case_file_id=case_file_id,
            title=title,
            description=description,
            session_id=session_id,
            created_at=now,
            updated_at=now,
            facts={},
            documents=[]
        )
        
        self.case_files[case_file_id] = case_file
        self._save_case_file(case_file_id)
        
        return case_file
    
    def get_case_file(self, case_file_id: str) -> Optional[CaseFile]:
        """Get a case file by ID."""
        return self.case_files.get(case_file_id)
    
    def update_case_file(
        self,
        case_file_id: str,
        facts: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None
    ) -> Optional[CaseFile]:
        """
        Update a case file with new facts or documents.
        
        Args:
            case_file_id: ID of the case file to update.
            facts: New facts to add or update in the case file.
            documents: New list of document IDs.
            
        Returns:
            The updated case file, or None if not found.
        """
        case_file = self.case_files.get(case_file_id)
        if not case_file:
            return None
        
        # Update facts if provided
        if facts:
            current_facts = dict(case_file.facts)
            current_facts.update(facts)
            case_file.facts = current_facts
        
        # Update documents if provided
        if documents is not None:
            case_file.documents = documents
        
        # Update timestamp
        case_file.updated_at = datetime.now()
        
        # Save changes
        self._save_case_file(case_file_id)
        
        return case_file
    
    def list_case_files(self, session_id: Optional[str] = None) -> List[CaseFile]:
        """
        List all case files, optionally filtered by session ID.
        
        Args:
            session_id: Optional session ID to filter by.
            
        Returns:
            List of case files.
        """
        if session_id:
            return [cf for cf in self.case_files.values() if cf.session_id == session_id]
        else:
            return list(self.case_files.values())
    
    def delete_case_file(self, case_file_id: str) -> bool:
        """
        Delete a case file.
        
        Args:
            case_file_id: ID of the case file to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        if case_file_id not in self.case_files:
            return False
        
        # Remove from memory
        del self.case_files[case_file_id]
        
        # Remove from disk
        file_path = Path("case_files") / f"{case_file_id}.yaml"
        if file_path.exists():
            file_path.unlink()
        
        json_path = Path("case_files") / f"{case_file_id}.json"
        if json_path.exists():
            json_path.unlink()
        
        return True 