# multimodal_agent.py (NEW FILE)
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
import base64
import httpx # For potentially fetching image from URL

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.clients import agent_client # For potential calls to other agents

app = FastAPI(title="Multimodal Agent")

# Initialize multimodal LLM (Gemini with Vision capabilities)
multimodal_model: Optional[Any] = None # Use Any as GenerativeModel import can differ for multimodal
try:
    from google.cloud import aiplatform
    from vertexai.preview.generative_models import GenerativeModel, Part
    aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["VERTEX_AI_LOCATION"])
    multimodal_model = GenerativeModel(os.environ["GEMINI_PRO_MODEL_ID"]) # Check if this directly supports Vision or needs specific ID
    # For actual multimodal capabilities, usually need a specific model like 'gemini-pro-vision'
    # or specific endpoint for Vision API.
    logger.info("Multimodal Agent: Initialized Gemini model (conceptual for vision).")
except Exception as e:
    logger.error(f"Multimodal Agent: Failed to initialize multimodal model: {e}", exc_info=True)
    multimodal_model = None

@app.post("/process_content", summary="Processes multimodal (text + image) content")
async def process_multimodal_content(request_data: Dict[str, Any]):
    """
    Receives a text query and an image URL, processes them using a multimodal LLM,
    and returns insights.
    Expected data: {"user_id": "...", "query_text": "...", "image_url": "..."}
    """
    user_id = request_data.get("user_id")
    query_text = request_data.get("query_text")
    image_url = request_data.get("image_url")

    if not all([user_id, query_text, image_url]):
        raise HTTPException(status_code=400, detail="user_id, query_text, and image_url are required.")
    if not multimodal_model:
        raise HTTPException(status_code=500, detail="Multimodal model not initialized.")

    logger.info(f"Multimodal Agent: Processing multimodal query for user {user_id}: '{query_text}' with image '{image_url}'")

    try:
        # 1. Fetch the image from the URL
        async with httpx.AsyncClient() as client:
            image_response = await client.get(image_url)
            image_response.raise_for_status()
            image_bytes = image_response.content
        logger.info(f"Multimodal Agent: Image fetched successfully (bytes: {len(image_bytes)})")

        # 2. Prepare content for multimodal LLM (Gemini-Pro-Vision or Vision API)
        # Assuming `multimodal_model` is `GenerativeModel` with Vision capabilities
        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg") # Adjust mime_type as needed

        contents = [image_part, query_text]

        # 3. Request insights from multimodal LLM
        response = await multimodal_model.generate_content_async(
            contents=contents,
            generation_config={"temperature": 0.2, "max_output_tokens": 500}
        )
        multimodal_insight = response.candidates[0].text
        logger.info("Multimodal Agent: Insight generated from multimodal content.")

        # 4. Potentially call other agents based on insight (e.g., KM Agent to save insight)
        # This would be a place for more complex orchestration within Multimodal Agent itself.
        # For simplicity, returning the insight directly.

        return {"status": "success", "insight": multimodal_insight, "user_id": user_id, "query": query_text, "image_url": image_url}
    except httpx.HTTPStatusError as e:
        logger.error(f"Multimodal Agent: HTTP Error fetching image: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {e.response.text}")
    except Exception as e:
        logger.error(f"Multimodal Agent: Error processing multimodal content for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Multimodal processing failed: {e}")

# To run this agent: uvicorn multimodal_agent:app --port 8009 --reload