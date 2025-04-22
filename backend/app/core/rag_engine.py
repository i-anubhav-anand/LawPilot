import os
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import yaml
from datetime import datetime
import asyncio
import logging
import uuid

from app.core.vector_store import VectorStore
from app.core.text_chunker import TextChunker
from app.models.chat import ChatMessage, RAGResponse, Source
from app.models.case_file import CaseFile
from app.models.documents import DocumentAnalysis
from app.core.system_prompt import get_system_prompt
from app.core.conversation_state import ConversationState

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine for generating legal assistant responses."""
        self.vector_store = VectorStore()
        self.text_chunker = TextChunker()
        self.chat_histories = {}  # Session ID -> List[ChatMessage]
        self.conversation_states = {}  # Session ID -> ConversationState
        
        # Create directories for storing data
        os.makedirs("legal_corpus", exist_ok=True)
        os.makedirs("chat_histories", exist_ok=True)
        os.makedirs("conversation_states", exist_ok=True)
        
        # Load chat histories from disk
        self._load_chat_histories()
        
        # Legal corpus paths - these will be populated by admin endpoints
        self.legal_corpus_paths = []
        for file_path in Path("legal_corpus").glob("*.txt"):
            self.legal_corpus_paths.append(str(file_path))
    
    def _load_chat_histories(self):
        """Load chat histories from disk."""
        histories_dir = Path("chat_histories")
        for file_path in histories_dir.glob("*.json"):
            try:
                session_id = file_path.stem
                with open(file_path, 'r') as f:
                    history_data = json.load(f)
                
                messages = []
                for msg_data in history_data:
                    timestamp = datetime.fromisoformat(msg_data.get("timestamp", datetime.now().isoformat()))
                    messages.append(ChatMessage(
                        role=msg_data.get("role", "user"),
                        content=msg_data.get("content", ""),
                        timestamp=timestamp
                    ))
                
                self.chat_histories[session_id] = messages
            except Exception as e:
                print(f"Error loading chat history {file_path}: {e}")
    
    def _save_chat_history(self, session_id: str):
        """Save a chat history to disk."""
        history = self.chat_histories.get(session_id, [])
        if not history:
            return
        
        history_data = []
        for msg in history:
            history_data.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            })
        
        file_path = Path("chat_histories") / f"{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session."""
        return self.chat_histories.get(session_id, [])
    
    def list_chat_sessions(self) -> List[str]:
        """
        Get a list of all available chat session IDs.
        
        Returns:
            List of session IDs.
        """
        # First load any chat histories from disk that might not be in memory
        self._load_chat_histories()
        
        # Return all session IDs
        return list(self.chat_histories.keys())
    
    def _get_conversation_state(self, session_id: str) -> ConversationState:
        """
        Get or create a conversation state for a session.
        
        Args:
            session_id: The session ID.
            
        Returns:
            Conversation state for the session.
        """
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = ConversationState(session_id)
        
        return self.conversation_states[session_id]
    
    async def process_query(
        self,
        query: str,
        session_id: str,
        case_file: Optional[CaseFile] = None,
        num_results: int = 5
    ) -> RAGResponse:
        """
        Process a user query using RAG.
        
        Args:
            query: The user's query.
            session_id: The session ID.
            case_file: Optional case file with context.
            num_results: Number of results to retrieve.
            
        Returns:
            Generated response with sources and suggested questions.
        """
        logger.info(f"🔄 PROCESSING QUERY: session_id={session_id}, query='{query}'")
        
        # Get or create conversation state
        conversation_state = self._get_conversation_state(session_id)
        
        # Add user message to history
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        
        self.chat_histories[session_id].append(ChatMessage(
            role="user",
            content=query
        ))
        logger.info(f"✅ USER MESSAGE ADDED TO HISTORY: session_id={session_id}")
        
        # Perform retrieval
        logger.info(f"🔄 RETRIEVING RELEVANT CHUNKS: session_id={session_id}")
        retrieved_chunks = await self._retrieve_relevant_chunks(
            query=query,
            session_id=session_id,
            case_file=case_file,
            num_results=num_results
        )
        logger.info(f"✅ RETRIEVED {len(retrieved_chunks)} CHUNKS")
        
        # Generate response using LLM with enhanced prompt
        logger.info(f"🔄 GENERATING RESPONSE: session_id={session_id}")
        response = await self._generate_response(
            query=query,
            retrieved_chunks=retrieved_chunks,
            chat_history=self.chat_histories[session_id],
            case_file=case_file,
            conversation_state=conversation_state
        )
        logger.info(f"✅ RESPONSE GENERATED: session_id={session_id}")
        
        # Add assistant response to history
        self.chat_histories[session_id].append(ChatMessage(
            role="assistant",
            content=response.answer
        ))
        logger.info(f"✅ ASSISTANT RESPONSE ADDED TO HISTORY: session_id={session_id}")
        
        # Save updated chat history
        self._save_chat_history(session_id)
        
        # Update conversation state with extracted facts
        if response.extracted_facts:
            logger.info(f"✅ EXTRACTED FACTS FOUND: session_id={session_id}")
            conversation_state.update_case_file(response.extracted_facts)
            
        # Mark first response as complete if needed
        if conversation_state.is_first_response:
            logger.info(f"✅ MARKING FIRST RESPONSE COMPLETE: session_id={session_id}")
            conversation_state.mark_first_response_complete()
        
        logger.info(f"✅ QUERY PROCESSING COMPLETED: session_id={session_id}")
        return response
    
    async def analyze_document(
        self,
        document_id: str,
        session_id: str,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document from the vector store.
        
        Args:
            document_id: The ID of the document to analyze.
            session_id: The session ID.
            query: Optional query to focus the analysis.
            
        Returns:
            Dictionary with analysis results.
        """
        logger.info(f"🔍 ANALYZING DOCUMENT: document_id={document_id}, session_id={session_id}, query={query}")
        
        # Get document
        document = await self.vector_store.get_document(document_id)
        if not document:
            logger.error(f"❌ DOCUMENT NOT FOUND: document_id={document_id}")
            raise ValueError(f"Document with ID {document_id} not found")
        
        # Create a CaseFile to focus retrieval for this document
        case_file = CaseFile(
            case_file_id=f"temp-{str(uuid.uuid4())}",
            title=f"Analysis for {document.get('filename', 'document')}",
            description="Temporary case file for document analysis",
            documents=[document_id]
        )
        
        # If no query is provided, generate a default one
        if not query:
            query = "Summarize the key points and information in this document"
        
        # Retrieve chunks and generate a response
        retrieved_chunks = await self._retrieve_relevant_chunks(
            query=query,
            session_id=session_id,
            case_file=case_file,
            num_results=10  # Use more chunks for document analysis
        )
        
        # Extract the content from chunks
        chunk_texts = [chunk.get("content", "") for chunk in retrieved_chunks]
        combined_context = "\n\n".join(chunk_texts)
        
        # Generate analysis
        analysis_prompt = f"""
        You are a legal expert analyzing a document. Based on the following document excerpts, provide a comprehensive analysis.
        
        Document: {document.get('filename', 'Unnamed document')}
        
        Excerpts:
        {combined_context}
        
        Provide an analysis with the following sections:
        1. Summary: A brief overview of the document's content and purpose
        2. Key Points: The most important information or requirements in bullet points
        3. Legal Implications: Any legal considerations that should be noted
        4. Recommendations: Suggested next steps or actions based on this document
        
        Focus specifically on: {query}
        """
        
        # Call the LLM for document analysis
        analysis = await self.llm_service.generate_response(
            user_prompt=analysis_prompt,
            use_streaming=False
        )
        
        # Return the analysis with metadata
        return {
            "document_id": document_id,
            "filename": document.get("filename", ""),
            "query": query,
            "chunks_used": len(retrieved_chunks),
            "analysis": analysis,
        }
    
    async def _retrieve_relevant_chunks(
        self,
        query: str,
        session_id: str,
        case_file: Optional[CaseFile] = None,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from the vector store.
        
        Args:
            query: The user's query.
            session_id: The session ID for retrieving session-specific documents.
            case_file: Optional case file with context.
            num_results: Number of results to retrieve.
            
        Returns:
            List of relevant chunks with metadata.
        """
        logger.info(f"🔄 RETRIEVING CHUNKS: query='{query}', session_id={session_id}, num_results={num_results}")
        
        # First, search global documents (allocate 40% of results)
        global_limit = max(2, int(num_results * 0.4))
        logger.info(f"🔄 SEARCHING GLOBAL DOCUMENTS: limit={global_limit}")
        
        try:
            global_results = await self.vector_store.search(
                query=query, 
                limit=global_limit,
                filter_metadata={"is_global": True}
            )
            logger.info(f"✅ FOUND {len(global_results)} RELEVANT GLOBAL CHUNKS")
        except Exception as e:
            logger.error(f"❌ ERROR SEARCHING GLOBAL DOCUMENTS: {str(e)}")
            global_results = []

        # Then, search session-specific documents (allocate 40% of results)
        session_limit = max(2, int(num_results * 0.4))
        logger.info(f"🔄 SEARCHING SESSION DOCUMENTS: session_id={session_id}, limit={session_limit}")
        
        try:
            # We filter for documents with matching session_id and is_global=False
            session_results = await self.vector_store.search(
                query=query,
                limit=session_limit,
                filter_metadata={"session_id": session_id, "is_global": False}
            )
            logger.info(f"✅ FOUND {len(session_results)} RELEVANT SESSION CHUNKS")
        except Exception as e:
            logger.error(f"❌ ERROR SEARCHING SESSION DOCUMENTS: {str(e)}")
            session_results = []

        # Combine global and session results
        results = []
        results.extend(global_results)
        results.extend(session_results)
        
        # Search case file documents if provided (allocate 20% of results)
        case_file_results = []
        if case_file and case_file.documents:
            logger.info(f"🔄 SEARCHING CASE FILE DOCUMENTS: case_file_id={case_file.case_file_id}, docs={len(case_file.documents)}")
            for doc_id in case_file.documents:
                # Get the document first
                document = await self.vector_store.get_document(doc_id)
                if not document:
                    logger.warning(f"⚠️ CASE FILE DOCUMENT NOT FOUND: doc_id={doc_id}")
                    continue
                
                # Check if document has chunks
                doc_chunks = document.get("chunks", [])
                if not doc_chunks:
                    logger.warning(f"⚠️ CASE FILE DOCUMENT HAS NO CHUNKS: doc_id={doc_id}")
                    continue
                    
                logger.info(f"🔄 SEARCHING IN DOCUMENT: doc_id={doc_id}, chunks={len(doc_chunks)}")
                
                # Get results from the existing results for this document
                existing_doc_results = [r for r in results if r.get("document_id") == doc_id]
                
                # If we don't have enough results, do a targeted search
                if len(existing_doc_results) < 2:
                    # Search again with the query
                    try:
                        additional_results = await self.vector_store.search(
                            query=query,
                            limit=3,  # Just a few more results
                            filter_metadata={"document_id": doc_id}
                        )
                        
                        # Add to case file results
                        for result in additional_results:
                            if result not in case_file_results:
                                case_file_results.append(result)
                                
                        logger.info(f"✅ FOUND {len(additional_results)} ADDITIONAL CHUNKS FROM DOCUMENT: doc_id={doc_id}")
                    except Exception as e:
                        logger.error(f"❌ ERROR SEARCHING DOCUMENT {doc_id}: {str(e)}")
            
            # Add case file results to combined results
            results.extend(case_file_results)
        
        # Remove duplicates (based on chunk_id and document_id)
        unique_results = []
        seen_chunks = set()
        for result in results:
            chunk_key = f"{result.get('document_id')}:{result.get('chunk_id')}"
            if chunk_key not in seen_chunks:
                unique_results.append(result)
                seen_chunks.add(chunk_key)
        
        # Sort by similarity and limit to num_results
        unique_results = sorted(unique_results, key=lambda x: x.get("similarity", 0), reverse=True)[:num_results]
        logger.info(f"✅ FINAL CHUNKS RETRIEVED: global={len(global_results)}, session={len(session_results)}, case_file={len(case_file_results)}, total={len(unique_results)}")
        
        return unique_results
    
    async def _generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: List[ChatMessage],
        case_file: Optional[CaseFile] = None,
        conversation_state: Optional[ConversationState] = None
    ) -> RAGResponse:
        """
        Generate a response using the language model with retrieved context.
        
        Args:
            query: The user's query.
            retrieved_chunks: The retrieved relevant chunks.
            chat_history: The chat history for this session.
            case_file: Optional case file with context.
            conversation_state: The conversation state with case file information.
            
        Returns:
            Generated response with sources and other metadata.
        """
        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Prepare context from retrieved chunks
            context = []
            sources = []
            
            for i, chunk in enumerate(retrieved_chunks):
                context.append(f"[Source {i+1}]: {chunk['text']}")
                
                # Create source object for referencing
                source_type = "law"
                if "document_id" in chunk["metadata"]:
                    source_type = "document"
                
                sources.append(Source(
                    source_type=source_type,
                    title=chunk["metadata"].get("filename", f"Source {i+1}"),
                    content=chunk["text"],
                    relevance_score=chunk["similarity"],
                    document_id=chunk["metadata"].get("document_id")
                ))
            
            context_str = "\n\n".join(context)
            
            # Prepare chat history context
            history_str = ""
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            for msg in recent_history[:-1]:  # Exclude the current query
                history_str += f"{msg.role.upper()}: {msg.content}\n"
            
            # Prepare case file context
            case_file_str = ""
            if conversation_state:
                case_file_str = f"CURRENT CASE FILE:\n{conversation_state.get_yaml_case_file()}\n"
            elif case_file:
                case_file_str = "CASE FILE INFORMATION:\n"
                case_file_str += f"Title: {case_file.title}\n"
                if case_file.description:
                    case_file_str += f"Description: {case_file.description}\n"
                
                if case_file.facts:
                    case_file_str += "Facts:\n"
                    for key, value in case_file.facts.items():
                        if isinstance(value, dict):
                            case_file_str += f"- {key}:\n"
                            for subkey, subvalue in value.items():
                                case_file_str += f"  - {subkey}: {subvalue}\n"
                        else:
                            case_file_str += f"- {key}: {value}\n"
            
            # Use the enhanced system prompt
            system_prompt = get_system_prompt()
            
            # Build the prompt for generating the response
            user_prompt = f"""QUERY: {query}

CHAT HISTORY:
{history_str}

{case_file_str}

RELEVANT LEGAL SOURCES (ProvidedCorpus):
{context_str}

The legal sources above are from the ProvidedCorpus. Answer the query based ONLY on these provided legal sources. Cite specific sources in your answer (e.g., "According to [Source 3]..."). Be helpful, accurate, and clear.

If you extract any new facts about the user's situation, include them in a separate YAML-formatted "EXTRACTED_FACTS" section at the end of your response.

Remember to ask exactly one question per message, and make it the next most relevant question based on what you've learned so far.
"""

            # Check if this is the first response
            is_first_response = conversation_state and conversation_state.is_first_response
            temperature = 0.1  # Lower temperature for more consistent responses
            
            # Make the API call
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            
            # Extract the suggested questions (we'll still extract them, but our prompt will only show one)
            suggested_questions = []
            next_question_patterns = [
                "Next question:", 
                "My next question for you is:", 
                "My question for you is:",
                "Can you tell me"
            ]
            
            for pattern in next_question_patterns:
                if pattern in answer:
                    parts = answer.split(pattern, 1)
                    if len(parts) > 1:
                        question_text = parts[1].strip().split("\n")[0].strip()
                        if question_text and question_text not in suggested_questions:
                            if question_text.endswith("?"):
                                suggested_questions.append(question_text)
            
            # If we didn't find any specific questions but there are question marks, look for those
            if not suggested_questions:
                for line in answer.split("\n"):
                    line = line.strip()
                    if "?" in line and len(line) < 200:  # Reasonable length for a question
                        question_part = line.split("?")[0] + "?"
                        if question_part not in suggested_questions:
                            suggested_questions.append(question_part)
            
            # Extract any facts if present
            extracted_facts = {}
            if "EXTRACTED_FACTS" in answer:
                facts_section = answer.split("EXTRACTED_FACTS:")[1].strip()
                # Take until next section if there is one
                if "\n\n" in facts_section:
                    facts_section = facts_section.split("\n\n")[0]
                
                try:
                    # Try to parse as YAML
                    extracted_facts = yaml.safe_load(facts_section)
                except Exception as e:
                    print(f"Error parsing extracted facts: {e}")
            
            # Clean up the answer (remove the special sections)
            if "EXTRACTED_FACTS:" in answer:
                answer = answer.split("EXTRACTED_FACTS:")[0].strip()
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                suggested_questions=suggested_questions[:3],  # Limit to top 3 suggestions
                extracted_facts=extracted_facts
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return RAGResponse(
                answer=f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or contact support if the issue persists.",
                sources=[],
                suggested_questions=[],
                extracted_facts={}
            ) 