import os
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import yaml
from datetime import datetime
import asyncio
import logging
import uuid
import re

from app.core.vector_store import VectorStore
from app.core.text_chunker import TextChunker
from app.models.chat import ChatMessage, RAGResponse, Source
from app.models.case_file import CaseFile
from app.models.documents import DocumentResponse, DocumentAnalysis
from app.core.system_prompt import get_system_prompt
from app.core.conversation_state import ConversationState
from app.core.direct_processor import DirectTextProcessor
from app.core.document_processor import DocumentProcessor
from app.core.llm_service import OpenAIService

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine for generating legal assistant responses."""
        self.vector_store = VectorStore()
        self.text_chunker = TextChunker()
        self.chat_histories = {}  # Session ID -> List[ChatMessage]
        self.conversation_states = {}  # Session ID -> ConversationState
        
        # Initialize basic LLM service for API calls
        self.llm_service = OpenAIService()
        
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
            # Validate results to make sure all needed fields are present
            global_results = [
                r for r in global_results 
                if "document_id" in r and "chunk_id" in r and "text" in r and "metadata" in r
            ]
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
            # Validate results to make sure all needed fields are present
            session_results = [
                r for r in session_results 
                if "document_id" in r and "chunk_id" in r and "text" in r and "metadata" in r
            ]
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
            
            # Only try to search through valid documents
            valid_doc_ids = []
            for doc_id in case_file.documents:
                try:
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
                        
                    valid_doc_ids.append(doc_id)
                except Exception as e:
                    logger.error(f"❌ ERROR CHECKING DOCUMENT {doc_id}: {str(e)}")
            
            # Only continue with valid documents
            for doc_id in valid_doc_ids:
                logger.info(f"🔄 SEARCHING IN DOCUMENT: doc_id={doc_id}")
                
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
                        
                        # Validate results to make sure all needed fields are present
                        additional_results = [
                            r for r in additional_results 
                            if "document_id" in r and "chunk_id" in r and "text" in r and "metadata" in r
                        ]
                        
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
            try:
                chunk_key = f"{result.get('document_id')}:{result.get('chunk_id')}"
                if chunk_key not in seen_chunks:
                    unique_results.append(result)
                    seen_chunks.add(chunk_key)
            except Exception as e:
                logger.error(f"❌ ERROR PROCESSING RESULT: {str(e)}")
                continue
        
        # Sort by similarity and limit to num_results
        try:
            # Make sure we have the similarity key
            for result in unique_results:
                if "similarity" not in result:
                    result["similarity"] = 0.0
                    
            unique_results = sorted(unique_results, key=lambda x: x.get("similarity", 0), reverse=True)[:num_results]
        except Exception as e:
            logger.error(f"❌ ERROR SORTING RESULTS: {str(e)}")
            # Fallback - just take the first few
            unique_results = unique_results[:num_results] if len(unique_results) > num_results else unique_results
            
        logger.info(f"✅ FINAL CHUNKS RETRIEVED: global={len(global_results)}, session={len(session_results)}, case_file={len(case_file_results)}, total={len(unique_results)}")
        
        return unique_results
    
    async def process_query_with_direct_text(
        self,
        query: str,
        text: str,
        document_name: str,
        session_id: str,
        case_file: Optional[CaseFile] = None,
        num_results: int = 5
    ) -> RAGResponse:
        """
        Process a user query using the direct text approach (no vector embedding).
        This allows for immediate responses without waiting for document indexing.
        
        Args:
            query: The user's query.
            text: The direct text to use for context.
            document_name: Name of the document or text source.
            session_id: The session ID.
            case_file: Optional case file with context.
            num_results: Number of results to retrieve.
            
        Returns:
            Generated response with sources and suggested questions.
        """
        logger.info(f"🔄 PROCESSING QUERY WITH DIRECT TEXT: session_id={session_id}, query='{query}'")
        
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
        
        # Process the text directly
        direct_processor = DirectTextProcessor()
        direct_result = await direct_processor.process_text(
            text=text,
            document_name=document_name,
            query=query,
            session_id=session_id,
            case_file_id=case_file.case_file_id if case_file else None
        )
        
        logger.info(f"✅ DIRECT TEXT PROCESSED: chunks={len(direct_result['chunks'])}")
        
        # Convert chunks to the format expected by generate_response
        retrieved_chunks = []
        for i, chunk in enumerate(direct_result['chunks']):
            retrieved_chunks.append({
                "content": chunk,
                "metadata": {
                    "filename": document_name,
                    "session_id": session_id,
                    "document_id": direct_result["document"].document_id,
                    "chunk_id": f"direct_{i}",
                    "is_direct": True
                },
                "score": 0.95 - (i * 0.05)  # Simulate relevance scores
            })
        
        # Also try to get relevant chunks from the vector store for broader context
        try:
            vector_chunks = await self._retrieve_relevant_chunks(
                query=query,
                session_id=session_id,
                case_file=case_file,
                num_results=max(2, num_results // 2)  # Fewer results from vector store
            )
            logger.info(f"✅ RETRIEVED {len(vector_chunks)} ADDITIONAL CHUNKS FROM VECTOR STORE")
            
            # Combine the results, prioritizing direct chunks
            retrieved_chunks.extend(vector_chunks)
        except Exception as e:
            logger.warning(f"⚠️ COULD NOT RETRIEVE FROM VECTOR STORE: {str(e)}")
        
        # Generate response using LLM with enhanced prompt
        logger.info(f"🔄 GENERATING RESPONSE WITH DIRECT TEXT: session_id={session_id}")
        response = await self._generate_response(
            query=query,
            retrieved_chunks=retrieved_chunks,
            chat_history=self.chat_histories[session_id],
            case_file=case_file,
            conversation_state=conversation_state,
            is_direct_text=True,
            direct_document=direct_result["document"]
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
        
        # Schedule background indexing of this text
        document_processor = DocumentProcessor()
        asyncio.create_task(direct_processor.schedule_background_indexing(
            direct_result["document"].document_id,
            document_processor,
            session_id,
            case_file.case_file_id if case_file else None
        ))
        
        logger.info(f"✅ QUERY PROCESSING WITH DIRECT TEXT COMPLETED: session_id={session_id}")
        return response
    
    async def _generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: List[ChatMessage],
        case_file: Optional[CaseFile] = None,
        conversation_state: Optional[ConversationState] = None,
        is_direct_text: bool = False,
        direct_document: Optional[DocumentResponse] = None
    ) -> RAGResponse:
        """
        Generate a response using retrieved chunks and the LLM.
        
        Args:
            query: The user's query.
            retrieved_chunks: List of relevant chunks with metadata.
            chat_history: Chat history for this session.
            case_file: Optional case file with context.
            conversation_state: Optional conversation state.
            is_direct_text: Whether this is a direct text processing request.
            direct_document: Optional document info for direct text processing.
            
        Returns:
            RAGResponse with answer, sources, and suggested questions.
        """
        logger.info(f"🔄 GENERATING RESPONSE: query='{query}', chunks={len(retrieved_chunks)}")
        
        # Create a context with the retrieved chunks
        context_parts = []
        sources = []
        
        # Set used to track unique sources to avoid duplication
        used_sources = set()
        
        # Add direct text notice if applicable
        if is_direct_text and direct_document:
            context_parts.append(f"NOTICE: This response includes direct analysis of the document '{direct_document.filename}' that was just uploaded and is still being processed.")
        
        # Sort chunks by score if available
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get("score", 0), reverse=True)
        
        # Process each chunk
        for i, chunk in enumerate(sorted_chunks):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            score = chunk.get("score", 0)
            
            # Skip empty content
            if not content.strip():
                continue
                
            # Create a unique identifier for this source
            source_id = f"{metadata.get('document_id', 'unknown')}_{metadata.get('chunk_id', i)}"
            
            # Skip if we've already used this exact source
            if source_id in used_sources:
                continue
                
            used_sources.add(source_id)
            
            # Add to context
            context_parts.append(f"[{i+1}] {content}")
            
            # Create source
            source_type = "document"
            if metadata.get("is_direct", False):
                source_type = "direct_document"
            elif "law" in metadata.get("corpus_type", "").lower():
                source_type = "law"
                
            source = Source(
                source_type=source_type,
                title=metadata.get("filename", f"Source {i+1}"),
                content=content[:500] + ("..." if len(content) > 500 else ""),
                citation=metadata.get("citation", None),
                relevance_score=score,
                document_id=metadata.get("document_id", None)
            )
            
            sources.append(source)
            
            # Limit the number of sources to avoid overwhelming the LLM
            if len(sources) >= 10:
                break
        
        # Combine the context parts
        if context_parts:
            context = "\n\n".join(context_parts)
        else:
            # Provide a fallback for no context
            context = "No specific information found for this query."
            
        # Prepare chat history
        history_text = []
        for msg in chat_history[-6:]:  # Include up to 6 most recent messages
            role = "User" if msg.role == "user" else "Assistant"
            history_text.append(f"{role}: {msg.content}")
        
        history_str = "\n".join(history_text)
        
        # Get case file yaml if available
        case_file_yaml = ""
        if conversation_state:
            case_file_yaml = conversation_state.get_yaml_case_file()
            
        # Prepare case file section if available
        case_file_section = ""
        if case_file_yaml:
            case_file_section = f"""CASE FILE:
{case_file_yaml}

"""

        # Build the prompt
        user_prompt = f"""QUERY: {query}

CONVERSATION HISTORY:
{history_str}

{case_file_section}
RELEVANT INFORMATION:
{context}

Based on the above information, please provide a comprehensive answer to the user's query.
Include specific references to the sources when appropriate.
If you extract any new facts about the user's situation, include them in a separate YAML-formatted "EXTRACTED_FACTS" section.

Also suggest 2-3 follow-up questions the user might want to ask next based on their current query and the available information."""
        
        # Use system prompt from the core
        system_prompt = get_system_prompt(conversation_state=conversation_state)
        
        # Call the LLM for a response
        try:
            response_text = await self.llm_service.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_streaming=False
            )
            
            logger.info(f"✅ LLM RESPONSE GENERATED: length={len(response_text)}")
        except Exception as e:
            logger.error(f"❌ ERROR GENERATING LLM RESPONSE: {str(e)}")
            response_text = f"I'm sorry, I encountered an error while processing your request. Error: {str(e)}"
        
        # Extract facts and suggested questions
        extracted_facts = {}
        suggested_questions = []
        
        # Parse the response for facts and questions
        if "EXTRACTED_FACTS:" in response_text:
            import yaml
            try:
                facts_section = response_text.split("EXTRACTED_FACTS:")[1].strip()
                if "\n\n" in facts_section:
                    facts_section = facts_section.split("\n\n")[0]
                
                # Parse the YAML facts
                extracted_facts = yaml.safe_load(facts_section)
                
                # Remove the facts section from the response
                response_text = response_text.split("EXTRACTED_FACTS:")[0].strip()
                
                logger.info(f"✅ EXTRACTED FACTS: {extracted_facts}")
            except Exception as e:
                logger.error(f"❌ ERROR PARSING EXTRACTED FACTS: {str(e)}")
        
        # Extract suggested questions
        if "follow-up questions" in response_text.lower():
            try:
                # Try to find the section with suggested questions
                sections = response_text.split("\n\n")
                for section in sections:
                    if "follow-up questions" in section.lower() or "suggested questions" in section.lower():
                        # Extract questions (look for numbered or bulleted items)
                        question_matches = re.findall(r"(?:^|\n)[*\-\d.)\s]+([^*\-\d.)\n][^\n]+\?)", section)
                        if question_matches:
                            suggested_questions = [q.strip() for q in question_matches]
                            
                            # Remove the questions section if it's at the end
                            if sections[-1] == section:
                                response_text = response_text.replace(section, "").strip()
                                
                        break
                
                logger.info(f"✅ SUGGESTED QUESTIONS: {suggested_questions}")
            except Exception as e:
                logger.error(f"❌ ERROR EXTRACTING SUGGESTED QUESTIONS: {str(e)}")
        
        # Create and return the RAG response
        return RAGResponse(
            answer=response_text,
            sources=sources,
            suggested_questions=suggested_questions[:3],  # Limit to 3 questions
            extracted_facts=extracted_facts
        ) 