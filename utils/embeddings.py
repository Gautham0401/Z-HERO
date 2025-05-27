# C:\PROJECTS\ZHERO\ZHEROBE\zhero_adk_backend\utils\embeddings.py
# utils/embeddings.py
import logging
import os 
from typing import List
from functools import lru_cache

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel # Explicit import

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1000) # Caching embeddings to prevent repeated calls for the same text
def generate_embedding(text: str) -> List[float]:
    """Generate text embedding using Vertex AI Text Embeddings API."""
    try:
        # aiplatform.init for project and location should be called once in main.py
        # You can optionally retrieve them here again using os.getenv if this utility
        # might be used independently without global aiplatform.init, but it's redundant
        # if main.py always initializes it.
        # project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        # location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        # Instantiate a text embedding model directly
        # Ensure the model name is correct and available in your project/location.
        # "text-embedding-004" is a common choice.
        model = TextEmbeddingModel.from_pretrained("text-embedding-004") 
        
        embeddings_response = model.get_embeddings([text])
        
        if embeddings_response and embeddings_response[0].values:
            logger.debug(f"Generated embedding for text: '{text[:50]}...'")
            return embeddings_response[0].values
        else:
            logger.error(f"Vertex AI returned no embedding values for text: '{text}'")
            raise ValueError("Unexpected embedding response format from Vertex AI: No values found.")

    except Exception as e:
        logger.error(f"Embedding generation error for text: '{text[:50]}...': {str(e)}", exc_info=True)
        raise # Re-raise the exception after logging for proper error propagation