import os
import numpy as np
import faiss
import json
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import logging
import asyncio
from datetime import datetime
import time
import uuid
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_store")

# Load environment variables
load_dotenv()

class VectorStore:
    def __init__(self, embedding_model_name: str = "text-embedding-3-small"):
        """
        Initialize the vector store for document retrieval.
        
        Args:
            embedding_model_name: Name of the OpenAI embedding model to use.
        """
        # Store configuration
        self.embedding_model_name = embedding_model_name
        self.data_dir = "data"
        self.index_file = os.path.join(self.data_dir, "faiss_index.bin")
        self.data_file = os.path.join(self.data_dir, "vector_store_data.json")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Document and embedding storage
        self.document_store = {}  # Document ID -> Document data (chunks, metadata)
        self.embeddings = []      # List of embeddings
        self.document_ids = []    # List of document IDs (parallel to embeddings)
        self.chunk_ids = []       # List of chunk IDs (parallel to embeddings)
        
        # Initialize OpenAI clients
        logger.info(f"🔄 INITIALIZING OPENAI CLIENT WITH MODEL: {embedding_model_name}")
        self.openai_client = OpenAI()
        self.openai_async_client = AsyncOpenAI()
        
        # Initialize FAISS index and load existing data
        self._load_data()
        
        if not hasattr(self, 'index') or self.index is None:
            self._initialize_index()
    
    def _load_data(self):
        """Load existing vector store data if available."""
        index_path = Path("vector_db/faiss_index.bin")
        metadata_path = Path("vector_db/metadata.pkl")
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load FAISS index
                logger.info("🔄 LOADING EXISTING FAISS INDEX")
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                logger.info("🔄 LOADING VECTOR STORE METADATA")
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.document_store = metadata["document_store"]
                    self.document_ids = metadata["document_ids"]
                    self.chunk_ids = metadata["chunk_ids"]
                
                logger.info(f"✅ LOADED VECTOR STORE: {len(self.document_store)} documents, {len(self.chunk_ids)} chunks, {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"❌ ERROR LOADING VECTOR STORE: {e}. Starting with empty store.", exc_info=True)
                self._initialize_index()
        else:
            logger.info("ℹ️ NO EXISTING VECTOR STORE FOUND. Initializing new index.")
            self._initialize_index()
    
    def _initialize_index(self):
        """Initialize a new FAISS index."""
        # Define embedding dimension based on OpenAI model
        # text-embedding-3-small produces 1536-dimensional embeddings
        embedding_dim = 1536  # Default for OpenAI's text-embedding-3-small
        
        logger.info(f"🔄 INITIALIZING NEW FAISS INDEX: dimension={embedding_dim}")
        
        # Create a new L2 index (Euclidean distance)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.document_ids = []
        self.chunk_ids = []
        
        logger.info(f"✅ NEW FAISS INDEX INITIALIZED: dimension={embedding_dim}")
    
    async def _save_data(self):
        """Save the vector store data to disk asynchronously."""
        index_path = Path("vector_db/faiss_index.bin")
        metadata_path = Path("vector_db/metadata.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(index_path.parent, exist_ok=True)
        
        # Save FAISS index
        logger.info(f"🔄 SAVING FAISS INDEX: {self.index.ntotal} vectors")
        
        # Run CPU-bound operations in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            lambda: faiss.write_index(self.index, str(index_path))
        )
        
        # Save metadata
        logger.info("🔄 SAVING VECTOR STORE METADATA")
        metadata = {
            "document_store": self.document_store,
            "document_ids": self.document_ids,
            "chunk_ids": self.chunk_ids
        }
        
        await loop.run_in_executor(
            None,
            lambda: pickle.dump(metadata, open(metadata_path, 'wb'))
        )
        
        logger.info(f"✅ VECTOR STORE DATA SAVED: {index_path.parent}")
    
    async def _get_embedding(self, text: str, chunk_id: Optional[str] = None) -> np.ndarray:
        """Generate an embedding for the given text using OpenAI's API"""
        # Log the start of embedding generation with text details
        text_preview = text[:50] + "..." if len(text) > 50 else text
        chunk_info = f" for chunk {chunk_id}" if chunk_id else ""
        
        logger.info(f"🔄 GENERATING OPENAI EMBEDDING{chunk_info}: {text_preview}")
        
        start_time = time.time()
        
        try:
            # Call OpenAI's embedding API
            response = await self.openai_async_client.embeddings.create(
                model=self.embedding_model_name,
                input=text,
                encoding_format="float"
            )
            
            # Get the embedding vector
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            gen_time = time.time() - start_time
            embedding_shape = embedding.shape
            embedding_size = embedding.size
            
            # Log successful embedding generation with details
            logger.info(f"✅ OPENAI EMBEDDING GENERATED{chunk_info}: shape={embedding_shape}, size={embedding_size}, time={gen_time:.2f}s")
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ OPENAI EMBEDDING GENERATION FAILED{chunk_info}: {str(e)}")
            raise
    
    async def add_document(
        self, 
        document_id: str, 
        chunks: List[str], 
        metadata: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ):
        """Add a document to the vector store
        
        Args:
            document_id: Unique ID for the document
            chunks: List of text chunks from the document
            metadata: Document metadata (title, path, etc.)
            progress_callback: Optional callback function to report progress
        """
        logger.info(f"🔄 ADDING DOCUMENT: id={document_id}, chunks={len(chunks)}")
        
        # Initialize document entry with metadata and empty chunks
        self.document_store[document_id] = {
            "metadata": metadata,
            "chunks": {},
            "added_at": datetime.now().isoformat()
        }
        
        # Process chunks in parallel for efficiency
        chunk_tasks = []
        for i, chunk_text in enumerate(chunks):
            # Generate a unique chunk ID
            chunk_id = str(uuid.uuid4())
            
            # Add chunk to document store
            self.document_store[document_id]["chunks"][chunk_id] = {
                "text": chunk_text,
                "index": i
            }
            
            # Add task to generate embedding for this chunk
            task = self._get_embedding(chunk_text, chunk_id)
            chunk_tasks.append((chunk_id, task, i))
        
        # Generate embeddings for all chunks, but in smaller batches to avoid blocking
        logger.info(f"🔄 GENERATING EMBEDDINGS FOR {len(chunk_tasks)} CHUNKS")
        start_time = time.time()
        
        new_embeddings = []
        batch_size = 5  # Process embeddings in batches of 5
        
        # Process chunks in batches
        for i in range(0, len(chunk_tasks), batch_size):
            batch = chunk_tasks[i:i+batch_size]
            batch_results = []
            
            # Process each batch concurrently
            for chunk_id, task, chunk_index in batch:
                try:
                    embedding = await task
                    batch_results.append((chunk_id, embedding, chunk_index))
                except Exception as e:
                    logger.error(f"❌ FAILED TO GENERATE EMBEDDING FOR CHUNK {chunk_id}: {str(e)}")
                    
                    # Still call progress callback on error if provided
                    if progress_callback:
                        progress_callback(chunk_index)
            
            # Process batch results
            for chunk_id, embedding, chunk_index in batch_results:
                # Store embedding and metadata
                self.embeddings.append(embedding)
                self.document_ids.append(document_id)
                self.chunk_ids.append(chunk_id)
                
                new_embeddings.append(embedding)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(chunk_index)
            
            # Yield control back to the event loop after each batch
            # This allows other requests to be processed
            await asyncio.sleep(0.01)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(chunks) if chunks else 0
        logger.info(f"✅ EMBEDDINGS COMPLETED: {len(new_embeddings)} embeddings generated in {total_time:.2f}s (avg: {avg_time:.2f}s per chunk)")
        
        # Add embeddings to FAISS index
        if new_embeddings:
            logger.info(f"🔄 UPDATING FAISS INDEX: Adding {len(new_embeddings)} embeddings")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.index.add(np.array(new_embeddings))
            )
            
            # Save updated data
            logger.info(f"🔄 SAVING VECTOR STORE DATA: {len(new_embeddings)} new embeddings")
            await self._save_data()
            
            logger.info(f"✅ DOCUMENT ADDED: id={document_id}, chunks={len(chunks)}, index_size={len(self.embeddings)}")
        else:
            logger.warning(f"⚠️ NO EMBEDDINGS GENERATED FOR DOCUMENT: id={document_id}")
    
    async def search(
        self, 
        query: str, 
        limit: int = 5, 
        similarity_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query text
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1) for results
            filter_metadata: Filter results by document metadata
            
        Returns:
            List of search results with document metadata and chunk text
        """
        logger.info(f"🔄 SEARCHING: query='{query}', limit={limit}, threshold={similarity_threshold}")
        start_time = time.time()
        
        if not hasattr(self, 'index') or self.index is None or len(self.document_ids) == 0:
            logger.warning("⚠️ EMPTY VECTOR STORE: No documents found for search")
            return []
            
        try:
            # Generate embedding for the query
            query_embedding = await self._get_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search the index - use run_in_executor for CPU-bound FAISS operations
            loop = asyncio.get_event_loop()
            logger.info(f"🔄 QUERYING FAISS INDEX: dimensions={query_vector.shape}")
            
            # FAISS search returns distances and indices
            distances, indices = await loop.run_in_executor(
                None,
                lambda: self.index.search(query_vector, min(limit * 4, len(self.document_ids)))  # Get more results for filtering
            )
            
            # Process search results
            distances = distances[0]  # First (and only) query result
            indices = indices[0]      # First (and only) query result
            
            # Calculate similarity scores (convert distance to similarity in 0-1 range)
            # Lower distance = higher similarity
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
            similarities = [1.0 - (dist / max_distance) for dist in distances]
            
            # Apply similarity threshold filter
            results = []
            filtered_count = 0
            metadata_filtered_count = 0
            missing_chunk_count = 0
            
            logger.info(f"🔄 PROCESSING {len(indices)} SEARCH RESULTS")
            
            for idx, (index, distance, similarity) in enumerate(zip(indices, distances, similarities)):
                if index < 0 or index >= len(self.document_ids):
                    continue  # Skip invalid indices
                    
                # Apply similarity threshold
                if similarity < similarity_threshold:
                    filtered_count += 1
                    continue
                    
                # Get document and chunk IDs
                doc_id = self.document_ids[index]
                chunk_id = self.chunk_ids[index]
                
                # Skip if document doesn't exist (possibly deleted)
                if doc_id not in self.document_store:
                    logger.debug(f"⚠️ DOCUMENT NOT FOUND IN STORE: id={doc_id}")
                    continue
                
                # Skip if chunk doesn't exist in the document
                if chunk_id not in self.document_store[doc_id]["chunks"]:
                    logger.debug(f"⚠️ CHUNK NOT FOUND IN DOCUMENT: doc_id={doc_id}, chunk_id={chunk_id}")
                    missing_chunk_count += 1
                    continue
                    
                # Apply metadata filter if specified
                if filter_metadata:
                    metadata_match = True
                    for key, value in filter_metadata.items():
                        if key not in self.document_store[doc_id]["metadata"] or self.document_store[doc_id]["metadata"][key] != value:
                            metadata_match = False
                            break
                            
                    if not metadata_match:
                        metadata_filtered_count += 1
                        continue
                
                try:        
                    # Get chunk text and metadata
                    chunk_text = self.document_store[doc_id]["chunks"][chunk_id]["text"]
                    doc_metadata = self.document_store[doc_id]["metadata"]
                    
                    # Add to results
                    results.append({
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": doc_metadata,
                        "similarity": similarity,
                        "distance": float(distance)  # Convert from numpy to Python float
                    })
                except KeyError as e:
                    # Handle case where chunk data is malformed
                    logger.error(f"❌ SEARCH ERROR: Invalid chunk data - doc_id={doc_id}, chunk_id={chunk_id}, error={str(e)}")
                    missing_chunk_count += 1
                    continue
                
                # Stop once we have enough results
                if len(results) >= limit:
                    break
                    
            # Sort results by similarity score (highest first)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            search_time = time.time() - start_time
            logger.info(
                f"✅ SEARCH COMPLETED: found={len(results)} results, "
                f"filtered_by_threshold={filtered_count}, "
                f"filtered_by_metadata={metadata_filtered_count}, "
                f"missing_chunks={missing_chunk_count}, "
                f"time={search_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ SEARCH ERROR: {str(e)}", exc_info=True)
            return []
            
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the vector store
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if document was deleted, False if it was not found
        """
        logger.info(f"🔄 DELETING DOCUMENT: id={document_id}")
        start_time = time.time()
        
        if document_id not in self.document_store:
            logger.warning(f"⚠️ DOCUMENT NOT FOUND FOR DELETION: id={document_id}")
            return False
            
        try:
            # Get all chunk IDs for this document
            chunk_ids_to_remove = list(self.document_store[document_id]["chunks"].keys())
            logger.info(f"🔄 REMOVING {len(chunk_ids_to_remove)} CHUNKS FOR DOCUMENT: id={document_id}")
            
            # Get indices to remove from the FAISS index
            indices_to_remove = []
            for i, (doc_id, chunk_id) in enumerate(zip(self.document_ids, self.chunk_ids)):
                if doc_id == document_id:
                    indices_to_remove.append(i)
                    
            logger.info(f"🔄 REMOVING {len(indices_to_remove)} VECTORS FROM INDEX")
            
            if hasattr(self, 'index') and self.index is not None and len(indices_to_remove) > 0:
                # Sort indices in descending order to avoid shifting problems
                indices_to_remove.sort(reverse=True)
                
                # Create a new index (FAISS doesn't support direct removal)
                temp_index = faiss.IndexFlatL2(self.index.d)
                
                # Copy vectors to the new index, skipping those to be removed
                new_document_ids = []
                new_chunk_ids = []
                new_embeddings = []
                
                for i in range(self.index.ntotal):
                    if i not in indices_to_remove:
                        # Get the embedding from the original index
                        vector = np.zeros((1, self.index.d), dtype=np.float32)
                        self.index.reconstruct(i, vector[0])
                        
                        # Add to the new index
                        temp_index.add(vector)
                        
                        # Update tracking lists
                        new_document_ids.append(self.document_ids[i])
                        new_chunk_ids.append(self.chunk_ids[i])
                        new_embeddings.append(vector[0])
                        
                # Replace old index and tracking lists with new ones
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: None)  # Yield control briefly
                
                self.index = temp_index
                self.document_ids = new_document_ids
                self.chunk_ids = new_chunk_ids
                self.embeddings = new_embeddings
                
            # Remove document from document store
            del self.document_store[document_id]
            
            # Save updated data
            await self._save_data()
            
            delete_time = time.time() - start_time
            logger.info(f"✅ DOCUMENT DELETED: id={document_id}, chunks={len(chunk_ids_to_remove)}, time={delete_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ERROR DELETING DOCUMENT: id={document_id}, error={str(e)}", exc_info=True)
            return False
            
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document data with metadata and chunks, or None if not found
        """
        logger.info(f"🔄 RETRIEVING DOCUMENT: id={document_id}")
        start_time = time.time()
        
        if document_id not in self.document_store:
            logger.warning(f"⚠️ DOCUMENT NOT FOUND: id={document_id}")
            return None
            
        # Get document from store
        document = self.document_store[document_id]
        chunks_count = len(document['chunks'])
        
        result = {
            "document_id": document_id,
            "metadata": document["metadata"],
            "chunks": [
                {"chunk_id": chunk_id, "text": chunk_data["text"]}
                for chunk_id, chunk_data in document["chunks"].items()
            ],
            "added_at": document.get("added_at", None)
        }
        
        retrieval_time = time.time() - start_time
        logger.info(f"✅ DOCUMENT RETRIEVED: id={document_id}, chunks={chunks_count}, time={retrieval_time:.2f}s")
        
        return result
        
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their metadata (without chunks)
        
        Returns:
            List of documents with their metadata
        """
        logger.info("🔄 RETRIEVING ALL DOCUMENTS")
        start_time = time.time()
        
        # Get list of document IDs and metadata
        documents = [
            {
                "document_id": doc_id,
                "metadata": self.document_store[doc_id]["metadata"],
                "chunk_count": len(self.document_store[doc_id]["chunks"]),
                "added_at": self.document_store[doc_id].get("added_at", None)
            }
            for doc_id in self.document_store
        ]
        
        retrieval_time = time.time() - start_time
        logger.info(f"✅ RETRIEVED ALL DOCUMENTS: count={len(documents)}, time={retrieval_time:.2f}s")
        
        return documents
        
    async def clear(self) -> bool:
        """Clear all data from the vector store
        
        Returns:
            bool: True if successful
        """
        logger.warning("⚠️ CLEARING ALL VECTOR STORE DATA")
        start_time = time.time()
        
        try:
            # Reset all data structures
            loop = asyncio.get_event_loop()
            
            # Initialize a new index
            if hasattr(self, 'model'):
                embedding_dim = len(await self._get_embedding("test"))
            else:
                embedding_dim = 384  # Default dimension
                
            await loop.run_in_executor(
                None,
                lambda: setattr(self, 'index', faiss.IndexFlatL2(embedding_dim))
            )
            
            # Clear all other data
            self.document_store = {}
            self.document_ids = []
            self.chunk_ids = []
            self.embeddings = []
            
            # Save empty data
            await self._save_data()
            
            clear_time = time.time() - start_time
            logger.info(f"✅ VECTOR STORE CLEARED: time={clear_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ERROR CLEARING VECTOR STORE: {str(e)}", exc_info=True)
            return False 