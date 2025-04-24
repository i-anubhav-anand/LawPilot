import os
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("llm_service")

class OpenAIService:
    """
    Service for interfacing with OpenAI models.
    Provides methods for generating responses, embeddings, and other LLM operations.
    """
    
    def __init__(self, model_name: str = "gpt-4-turbo"):
        """
        Initialize the OpenAI service.
        
        Args:
            model_name: The name of the model to use for generation
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        logger.info(f"🔄 INITIALIZED OPENAI SERVICE: model={model_name}, embedding_model={self.embedding_model}")
    
    async def generate_response(
        self, 
        user_prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        use_streaming: bool = False
    ) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: Optional system prompt for context
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            use_streaming: Whether to use streaming for the response
            
        Returns:
            Generated text response
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            messages.append({"role": "user", "content": user_prompt})
            
            logger.info(f"🔄 GENERATING RESPONSE: model={self.model_name}, temperature={temperature}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"✅ RESPONSE GENERATED: length={len(generated_text)}")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ ERROR GENERATING RESPONSE: {str(e)}")
            raise RuntimeError(f"Error generating response: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"🔄 GENERATING EMBEDDINGS: count={len(texts)}, model={self.embedding_model}")
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"✅ EMBEDDINGS GENERATED: count={len(embeddings)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ ERROR GENERATING EMBEDDINGS: {str(e)}")
            raise RuntimeError(f"Error generating embeddings: {str(e)}") 