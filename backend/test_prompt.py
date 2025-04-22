#!/usr/bin/env python3
import os
import argparse
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

from app.core.system_prompt import get_system_prompt
from app.core.conversation_state import ConversationState

# Load environment variables
load_dotenv()

def test_prompt(query: str, session_id: str = None, show_extracted_facts: bool = False):
    """
    Test the system prompt by generating a response to a query.
    
    Args:
        query: The query to test.
        session_id: Optional session ID to use for conversation state.
        show_extracted_facts: Whether to show extracted facts.
    """
    try:
        # Initialize OpenAI client
        client = OpenAI()
        
        # Get the system prompt
        system_prompt = get_system_prompt()
        
        # Initialize or get conversation state
        conversation_state = None
        if session_id:
            conversation_state = ConversationState(session_id)
            
        # Build the user prompt
        user_prompt = f"""QUERY: {query}

RELEVANT LEGAL SOURCES (ProvidedCorpus):
[Source 1]: California Civil Code §1950.5 allows landlords to collect security deposits but limits the amount to two months' rent for unfurnished dwellings or three months' rent for furnished ones.
[Source 2]: San Francisco Rent Ordinance Chapter 37 prohibits landlords from evicting tenants without just cause, which includes non-payment of rent, breach of lease terms, or the owner moving into the unit.

The legal sources above are from the ProvidedCorpus. Answer the query based ONLY on these provided legal sources.

If you extract any new facts about the user's situation, include them in a separate YAML-formatted "EXTRACTED_FACTS" section at the end of your response.
"""

        # Display conversation state if available
        if conversation_state:
            print("\nCurrent Case File:")
            print(conversation_state.get_yaml_case_file())
            print("\n")
            
            # Add case file to prompt
            user_prompt = f"""QUERY: {query}

CURRENT CASE FILE:
{conversation_state.get_yaml_case_file()}

RELEVANT LEGAL SOURCES (ProvidedCorpus):
[Source 1]: California Civil Code §1950.5 allows landlords to collect security deposits but limits the amount to two months' rent for unfurnished dwellings or three months' rent for furnished ones.
[Source 2]: San Francisco Rent Ordinance Chapter 37 prohibits landlords from evicting tenants without just cause, which includes non-payment of rent, breach of lease terms, or the owner moving into the unit.

The legal sources above are from the ProvidedCorpus. Answer the query based ONLY on these provided legal sources.

If you extract any new facts about the user's situation, include them in a separate YAML-formatted "EXTRACTED_FACTS" section at the end of your response.
"""

        # Make the API call
        print("Generating response...\n")
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        # Extract and display the facts if needed
        extracted_facts = {}
        if "EXTRACTED_FACTS:" in answer:
            import yaml
            facts_section = answer.split("EXTRACTED_FACTS:")[1].strip()
            if "\n\n" in facts_section:
                facts_section = facts_section.split("\n\n")[0]
            
            try:
                extracted_facts = yaml.safe_load(facts_section)
                
                # Update conversation state if available
                if conversation_state and extracted_facts:
                    conversation_state.update_case_file(extracted_facts)
            except Exception as e:
                print(f"Error parsing extracted facts: {e}")
            
            # Clean up the answer
            clean_answer = answer.split("EXTRACTED_FACTS:")[0].strip()
            
            if show_extracted_facts:
                print(clean_answer)
                print("\nExtracted Facts:")
                print(facts_section)
            else:
                print(clean_answer)
        else:
            print(answer)
        
        # Save the updated conversation state
        if conversation_state and extracted_facts:
            conversation_state.save()
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test the legal assistant system prompt.")
    parser.add_argument("query", help="The query to test.")
    parser.add_argument("--session", "-s", help="Session ID for conversation state.")
    parser.add_argument("--facts", "-f", action="store_true", help="Show extracted facts.")
    
    args = parser.parse_args()
    
    test_prompt(args.query, args.session, args.facts)

if __name__ == "__main__":
    main() 