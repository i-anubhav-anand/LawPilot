# Legal AI Assistant System Prompt

This directory contains a system prompt implementation for the Legal AI Assistant, designed to provide guidance on San Francisco tenant-landlord law. The system is built around a conversational RAG (Retrieval-Augmented Generation) approach with a specialized prompt designed to:

1. Maintain a specific conversational style (empathetic, step-by-step, one question at a time)
2. Track user information in a structured case file
3. Cite only laws that appear in the provided corpus
4. Focus on San Francisco-specific tenant-landlord issues

## Key Files

- `app/core/system_prompt.py`: Contains the system prompt and configuration
- `app/core/conversation_state.py`: Manages the conversation state and case file
- `app/core/updated_rag_engine.py`: Enhanced RAG engine that uses the system prompt
- `test_prompt.py`: Command-line tool to test the system prompt

## Testing the System Prompt

You can test the system prompt directly using the included test script:

```bash
# Set up your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Test a simple query
python test_prompt.py "I'm having issues with my landlord in San Francisco"

# Use a session ID to maintain conversation state
python test_prompt.py --session test123 "I'm a tenant in San Francisco"
python test_prompt.py --session test123 "My landlord increased my rent by 10%"

# Show the extracted facts
python test_prompt.py --session test123 --facts "My building was built in 1975"
```

## System Prompt Features

The system prompt guides the assistant to:

1. Make it clear it's not providing legal advice
2. Ask exactly one question per message to gather information
3. Maintain a conversational, empathetic tone
4. Only assist with San Francisco tenant-landlord issues
5. Build a YAML-style case file with the user's information
6. Cite only laws from the provided corpus

## Conversation State

The conversation state manager tracks:

1. The user's role (tenant or landlord)
2. Property location (must be in San Francisco)
3. Problem category
4. Key dates
5. Lease terms
6. Special concerns
7. Uploaded documents
8. Desired resolution

## Integration with the Backend

The system prompt is integrated with the backend through:

1. The updated RAG engine (`app/core/updated_rag_engine.py`)
2. The chat endpoint (`app/api/endpoints/chat.py`)

## Structure of the Legal Response

Each response from the assistant will follow this pattern:

1. Acknowledgment/empathy (if the user mentions a difficult situation)
2. Concise, helpful information based only on the provided legal sources
3. Citations to specific legal sources
4. Exactly one follow-up question to gather more information

## Customizing the System Prompt

You can modify the system prompt in `app/core/system_prompt.py` to adjust the assistant's:

1. Tone and conversational style
2. Information gathering approach
3. Response formatting
4. Legal disclaimers

After making changes, update the `.env` file configuration as needed. 