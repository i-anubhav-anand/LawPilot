import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from app.models.case_summary import CaseSummary, CaseSummarySection

# Configure logging
logger = logging.getLogger("case_summary_manager")

class CaseSummaryManager:
    """
    Manager for creating and updating structured case summaries.
    """
    
    def __init__(self):
        """Initialize the case summary manager."""
        self.summaries = {}
        self.summary_dir = Path("case_summaries")
        self.summary_dir.mkdir(exist_ok=True)
        self._load_summaries()
        
    def _load_summaries(self):
        """Load existing summaries from disk."""
        try:
            for file_path in self.summary_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        case_file_id = data.get("case_file_id")
                        if case_file_id:
                            self.summaries[case_file_id] = CaseSummary(**data)
                except Exception as e:
                    logger.error(f"❌ ERROR LOADING SUMMARY: {file_path} - {str(e)}")
        except Exception as e:
            logger.error(f"❌ ERROR LOADING SUMMARIES: {str(e)}")
    
    def _save_summary(self, case_file_id: str):
        """Save a summary to disk."""
        try:
            summary = self.summaries.get(case_file_id)
            if summary:
                file_path = self.summary_dir / f"{case_file_id}.json"
                with open(file_path, "w") as f:
                    f.write(summary.json())
                logger.info(f"✅ SUMMARY SAVED: {case_file_id}")
        except Exception as e:
            logger.error(f"❌ ERROR SAVING SUMMARY: {case_file_id} - {str(e)}")
    
    def create_summary(self, case_file_id: str, session_id: str) -> CaseSummary:
        """Create a new case summary."""
        summary = CaseSummary(
            case_file_id=case_file_id,
            session_id=session_id
        )
        
        # Add default sections
        default_sections = [
            CaseSummarySection(
                title="Prospective Client",
                key_details={
                    "Name/contact info": "not yet provided",
                    "Age": "not provided",
                    "Role": "not specified"
                }
            ),
            CaseSummarySection(
                title="Rental Unit",
                key_details={
                    "Location": "not provided",
                    "Unit type": "not specified",
                    "Lease type/term": "not stated"
                }
            ),
            CaseSummarySection(
                title="Primary Issue",
                key_details={
                    "Description": "not yet provided",
                    "Duration": "not specified"
                }
            ),
            CaseSummarySection(
                title="Timeline",
                key_details={
                    "Initial occurrence": "not provided",
                    "Communication with landlord": "not specified",
                    "Legal consultation": datetime.now().strftime("%b %d %Y")
                }
            ),
            CaseSummarySection(
                title="Landlord Response",
                key_details={
                    "Status": "not yet provided"
                }
            ),
            CaseSummarySection(
                title="Tenant Actions",
                key_details={
                    "Steps taken": "not yet provided"
                }
            ),
            CaseSummarySection(
                title="Potential Legal Claims",
                key_details={
                    "Claims": "to be determined based on further information"
                }
            ),
            CaseSummarySection(
                title="Evidence",
                key_details={
                    "On hand": "not yet provided",
                    "To collect": "not yet provided"
                }
            ),
            CaseSummarySection(
                title="Client Goals",
                key_details={
                    "Primary objective": "not yet provided"
                }
            ),
            CaseSummarySection(
                title="Urgency / Impact",
                key_details={
                    "Level": "not yet assessed"
                }
            ),
            CaseSummarySection(
                title="Suggested Attorney Next Steps",
                key_details={
                    "Recommendations": "to be determined after initial consultation"
                }
            )
        ]
        
        summary.sections = default_sections
        self.summaries[case_file_id] = summary
        self._save_summary(case_file_id)
        
        logger.info(f"✅ CREATED NEW SUMMARY: case_file_id={case_file_id}")
        return summary
    
    def get_summary(self, case_file_id: str) -> Optional[CaseSummary]:
        """Get a case summary by ID."""
        return self.summaries.get(case_file_id)
    
    def update_section(
        self,
        case_file_id: str,
        section_title: str,
        key_details: Dict[str, str]
    ) -> Optional[CaseSummary]:
        """Update a section in a case summary."""
        summary = self.summaries.get(case_file_id)
        if not summary:
            logger.warning(f"⚠️ SUMMARY NOT FOUND: case_file_id={case_file_id}")
            return None
        
        # Find the section to update
        for section in summary.sections:
            if section.title == section_title:
                # Update only the provided details
                for key, value in key_details.items():
                    section.key_details[key] = value
                break
        else:
            # Section not found, create it
            new_section = CaseSummarySection(
                title=section_title,
                key_details=key_details
            )
            summary.sections.append(new_section)
        
        # Update timestamp
        summary.updated_at = datetime.now()
        
        # Save updated summary
        self._save_summary(case_file_id)
        
        logger.info(f"✅ UPDATED SUMMARY SECTION: case_file_id={case_file_id}, section={section_title}")
        return summary
    
    def add_section(
        self,
        case_file_id: str,
        section_title: str,
        key_details: Dict[str, str]
    ) -> Optional[CaseSummary]:
        """Add a new section to a case summary."""
        summary = self.summaries.get(case_file_id)
        if not summary:
            logger.warning(f"⚠️ SUMMARY NOT FOUND: case_file_id={case_file_id}")
            return None
        
        # Check if section already exists
        for section in summary.sections:
            if section.title == section_title:
                logger.warning(f"⚠️ SECTION ALREADY EXISTS: case_file_id={case_file_id}, section={section_title}")
                return self.update_section(case_file_id, section_title, key_details)
        
        # Create new section
        new_section = CaseSummarySection(
            title=section_title,
            key_details=key_details
        )
        summary.sections.append(new_section)
        
        # Update timestamp
        summary.updated_at = datetime.now()
        
        # Save updated summary
        self._save_summary(case_file_id)
        
        logger.info(f"✅ ADDED SUMMARY SECTION: case_file_id={case_file_id}, section={section_title}")
        return summary
    
    def format_summary_for_display(self, case_file_id: str) -> str:
        """Format the summary as a readable string for display."""
        summary = self.summaries.get(case_file_id)
        if not summary:
            return "No summary found."
        
        output = ["# Lead-Qualification Case Summary", "(for attorney intake – San Francisco, CA)", "\n"]
        
        # Add table header
        output.append("Section\tKey Details")
        
        # Add sections
        for section in summary.sections:
            details_text = []
            for key, value in section.key_details.items():
                details_text.append(f"• {key}: {value}")
            
            # Join details with newlines
            details_str = "\n".join(details_text)
            
            # Add section with its details
            output.append(f"{section.title}\t{details_str}")
        
        return "\n".join(output)

    async def generate_summary_from_chat(self, case_file_id: str, session_id: str, chat_history: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate a structured case summary from chat history using the specialized prompt.
        
        Args:
            case_file_id: ID of the case file
            session_id: ID of the chat session
            chat_history: List of chat messages with 'role' and 'content' fields
            
        Returns:
            Formatted case summary text or None if generation failed
        """
        try:
            # Import here to avoid circular imports
            from app.core.llm_service import OpenAIService
            from app.core.system_prompt import get_system_prompt
            
            # Initialize LLM service
            llm_service = OpenAIService()
            
            # Format chat history into a string
            chat_text = []
            for msg in chat_history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                chat_text.append(f"{role}: {content}")
            
            conversation = "\n\n".join(chat_text)
            
            # Build the prompt
            user_prompt = f"""Please analyze this legal chat conversation and create a structured case summary for attorney handoff:

CONVERSATION:
{conversation}

Based on this conversation, generate a professionally formatted case summary with all relevant legal facts structured into the sections described in your instructions.
"""
            
            # Use the specialized case summary prompt
            system_prompt = get_system_prompt(prompt_type="case_summary")
            
            # Generate the summary
            logger.info(f"🔄 GENERATING CASE SUMMARY: case_file_id={case_file_id}")
            summary_text = await llm_service.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,  # Low temperature for factual output
                use_streaming=False
            )
            
            logger.info(f"✅ CASE SUMMARY GENERATED: case_file_id={case_file_id}, length={len(summary_text)}")
            
            # Parse the generated summary to update our structured summary
            self._update_summary_from_generated_text(case_file_id, summary_text)
            
            return summary_text
            
        except Exception as e:
            logger.error(f"❌ ERROR GENERATING CASE SUMMARY: case_file_id={case_file_id}, error={str(e)}")
            return None

    def _update_summary_from_generated_text(self, case_file_id: str, summary_text: str) -> None:
        """
        Parse a generated summary text and update our structured CaseSummary object.
        
        Args:
            case_file_id: ID of the case file to update
            summary_text: Generated summary text to parse
        """
        summary = self.summaries.get(case_file_id)
        if not summary:
            logger.warning(f"⚠️ NO SUMMARY FOUND FOR CASE: case_file_id={case_file_id}")
            return
        
        try:
            # Look for section headers and their content
            sections = {
                "Prospective Client": {},
                "Rental Unit": {},
                "Primary Issue": {},
                "Timeline": {},
                "Landlord Response": {},
                "Tenant Actions": {},
                "Potential Legal Claims": {},
                "Evidence": {},
                "Client Goals": {},
                "Urgency / Impact": {},
                "Suggested Attorney Next Steps": {}
            }
            
            # Simple parsing for demonstration - in production, use more robust parsing
            current_section = None
            lines = summary_text.split("\n")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a section header
                for section_title in sections.keys():
                    if section_title in line or section_title.replace(" / ", "/") in line:
                        current_section = section_title
                        break
                        
                # If we're in a section and have a key-value line
                if current_section and ":" in line and line.startswith(("-", "•", "*")):
                    # Extract key and value from bullet points
                    parts = line.lstrip("- •*").split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        sections[current_section][key] = value
            
            # Update the structured summary with the parsed information
            for section_title, details in sections.items():
                if details:  # Only update if we found details
                    self.update_section(case_file_id, section_title, details)
                    
            logger.info(f"✅ UPDATED STRUCTURED SUMMARY FROM TEXT: case_file_id={case_file_id}")
        except Exception as e:
            logger.error(f"❌ ERROR PARSING SUMMARY TEXT: case_file_id={case_file_id}, error={str(e)}") 