import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

class ConversationState:
    """Manages conversation state and case file information gathered from user interactions."""
    
    def __init__(self, session_id: str):
        """
        Initialize a new conversation state.
        
        Args:
            session_id: The ID of the conversation session
        """
        self.session_id = session_id
        self.case_file = {
            "role": None,  # tenant or landlord
            "property_location": None,  # must be in San Francisco
            "problem_category": None,  # e.g., rent increase, habitability, harassment
            "key_dates": {},  # e.g., lease start, notice served
            "lease_terms": {},  # e.g., monthly rent, lease duration
            "notices": [],  # list of notices sent or received
            "special_concerns": [],  # e.g., retaliation, discrimination
            "desired_resolution": None,  # what user wants to achieve
            "documents": [],  # list of document IDs uploaded by user
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.is_first_response = True
        self.essential_facts_complete = False
        
        # Create directory for storing conversation states
        Path("conversation_states").mkdir(exist_ok=True)
        
        # Try to load existing conversation state
        self._load_state()
    
    def _load_state(self):
        """Load conversation state from disk if it exists."""
        state_path = Path(f"conversation_states/{self.session_id}.json")
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    data = json.load(f)
                    self.case_file = data.get("case_file", self.case_file)
                    self.is_first_response = data.get("is_first_response", False)
                    self.essential_facts_complete = data.get("essential_facts_complete", False)
            except Exception as e:
                print(f"Error loading conversation state: {e}")
    
    def save(self):
        """Save conversation state to disk."""
        state_path = Path(f"conversation_states/{self.session_id}.json")
        
        # Update timestamp
        self.case_file["updated_at"] = datetime.now().isoformat()
        
        state_data = {
            "case_file": self.case_file,
            "is_first_response": self.is_first_response,
            "essential_facts_complete": self.essential_facts_complete
        }
        
        with open(state_path, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def update_case_file(self, new_facts: Dict[str, Any]):
        """
        Update the case file with new facts gathered from the conversation.
        
        Args:
            new_facts: Dictionary of new or updated facts
        """
        # Update simple fields
        for field in ["role", "property_location", "problem_category", "desired_resolution"]:
            if field in new_facts and new_facts[field]:
                self.case_file[field] = new_facts[field]
        
        # Update nested dictionaries
        for field in ["key_dates", "lease_terms"]:
            if field in new_facts and isinstance(new_facts[field], dict):
                if not self.case_file[field]:
                    self.case_file[field] = {}
                self.case_file[field].update(new_facts[field])
        
        # Update lists
        for field in ["notices", "special_concerns", "documents"]:
            if field in new_facts and isinstance(new_facts[field], list):
                # Add only new items
                for item in new_facts[field]:
                    if item not in self.case_file[field]:
                        self.case_file[field].append(item)
        
        # Check if all essential facts are gathered
        self._check_if_complete()
        
        # Save updated state
        self.save()
    
    def _check_if_complete(self):
        """Check if all essential facts are gathered."""
        required_fields = ["role", "property_location", "problem_category"]
        
        # Basic check for required fields
        if all(self.case_file.get(field) for field in required_fields):
            # Property must be in San Francisco
            if self.case_file["property_location"].lower() == "san francisco":
                # Different requirements based on problem category
                category = self.case_file["problem_category"].lower()
                
                # For rent increases, we need lease terms
                if "rent increase" in category and not self.case_file["lease_terms"]:
                    return
                
                # For habitability issues, we need specifics
                if "habitability" in category and not self.case_file["special_concerns"]:
                    return
                
                # For eviction or notices, we need the notice details
                if any(term in category for term in ["evict", "notice"]) and not self.case_file["notices"]:
                    return
                
                # If we got here, mark as complete
                self.essential_facts_complete = True
    
    def get_yaml_case_file(self) -> str:
        """
        Get the case file in YAML format.
        
        Returns:
            YAML formatted case file
        """
        # Filter out empty or None values
        filtered_case_file = {}
        for key, value in self.case_file.items():
            if value is not None:
                if isinstance(value, dict) and not value:
                    continue
                if isinstance(value, list) and not value:
                    continue
                filtered_case_file[key] = value
        
        return yaml.dump(filtered_case_file, default_flow_style=False, sort_keys=False)
    
    def mark_first_response_complete(self):
        """Mark that the first response has been sent."""
        self.is_first_response = False
        self.save()
    
    def add_document(self, document_id: str, document_name: str):
        """
        Add a document to the case file.
        
        Args:
            document_id: The ID of the uploaded document
            document_name: The name of the document
        """
        if "documents" not in self.case_file:
            self.case_file["documents"] = []
        
        document_info = {
            "id": document_id,
            "name": document_name,
            "uploaded_at": datetime.now().isoformat()
        }
        
        self.case_file["documents"].append(document_info)
        self.save() 