import os
import base64
import logging
from typing import Optional, List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("vision_service")

class VisionService:
    """
    Service for analyzing images using vision-capable LLMs.
    Provides methods for generating descriptions and analyzing content of images.
    """
    
    def __init__(self, model_name: str = "gpt-4-vision-preview"):
        """
        Initialize the Vision service.
        
        Args:
            model_name: The name of the vision-capable model to use
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        logger.info(f"🔄 INITIALIZED VISION SERVICE: model={model_name}")
    
    async def analyze_image(
        self, 
        image_path: str, 
        prompt: str = "Analyze this image in detail, focusing on any legal documents, notices, or relevant information it contains.",
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Analyze an image using the vision-capable model.
        
        Args:
            image_path: Path to the image file
            prompt: Instruction for how to analyze the image
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"🔄 ANALYZING IMAGE: path={image_path}")
            
            # Convert image to base64
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"❌ IMAGE NOT FOUND: path={image_path}")
                return {"error": "Image file not found", "analysis": "Unable to analyze image - file not found"}
            
            # Determine content type based on extension
            content_type = self._get_content_type(image_path)
            if not content_type:
                logger.error(f"❌ UNSUPPORTED IMAGE FORMAT: {image_path.suffix}")
                return {"error": "Unsupported image format", "analysis": "Unable to analyze image - unsupported format"}
            
            # Encode image as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create the messages for the API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Call the vision model
            logger.info(f"🔄 CALLING VISION MODEL: {self.model_name}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            
            analysis = response.choices[0].message.content
            logger.info(f"✅ IMAGE ANALYSIS COMPLETE: length={len(analysis)}")
            
            return {
                "analysis": analysis,
                "filename": image_path.name,
                "content_type": content_type
            }
            
        except Exception as e:
            logger.error(f"❌ ERROR ANALYZING IMAGE: {str(e)}")
            return {
                "error": str(e),
                "analysis": f"An error occurred while analyzing the image: {str(e)}"
            }
    
    def _get_content_type(self, file_path: Path) -> Optional[str]:
        """
        Determine the content type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content type string or None if unsupported
        """
        extension = file_path.suffix.lower()
        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        
        return content_types.get(extension) 