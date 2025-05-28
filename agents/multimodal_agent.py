# agents/multimodal_agent.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import base64
import httpx # For potentially fetching image from URL
import datetime # For generic_exception_handler

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import MultimodalProcessRequest # NEW: Use Pydantic model for request
from zhero_common.clients import agent_client # For potential calls to other agents
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroInvalidInputError,
    ZHeroDependencyError, ZHeroVertexAIError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher


app = FastAPI(title="Multimodal Agent")

# --- Global Exception Handlers (REQUIRED IN ALL AGENT FILES) ---
@app.exception_handler(ZHeroException)
async def zhero_exception_handler(request: Request, exc: ZHeroException):
    logger.error(f"ZHeroException caught for request {request.url.path}: {exc.message}", exc_info=True, extra={"details": exc.details, "status_code": exc.status_code})
    return JSONResponse(status_code=exc.status_code, content={"error_type": exc.__class__.__name__,"message": exc.message,"details": exc.details})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException caught for request {request.url.path}: {exc.detail}", exc_info=True, extra={"status_code": exc.status_code, "request_body": await request.body()})
    return JSONResponse(status_code=exc.status_code, content={"error_type": "HTTPException","message": exc.detail,"details": getattr(exc, 'body', None)})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log_id = f"ERR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.critical(f"Unhandled Exception caught for request {request.url.path} (ID: {log_id}): {exc}", exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error_type": "InternalServerError","message": "An unexpected internal server error occurred. Please try again later.","error_id": log_id, "details": str(exc) if app.debug else None})
# --- END Global Exception Handlers ---


# Initialize multimodal LLM (Gemini with Vision capabilities)
multimodal_model: Optional[Any] = None

@app.on_event("startup")
async def startup_event():
    global multimodal_model
    # Initialize Pub/Sub Publisher
    await initialize_pubsub_publisher()

    try:
        from google.cloud import aiplatform
        from vertexai.preview.generative_models import GenerativeModel, Part
        aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["VERTEX_AI_LOCATION"])
        multimodal_model = GenerativeModel(os.environ.get("GEMINI_PRO_VISION_MODEL_ID", os.environ["GEMINI_PRO_MODEL_ID"]))
        logger.info("Multimodal Agent: Initialized Gemini model (conceptual for vision).")
    except Exception as e:
        logger.error(f"Multimodal Agent: Failed to initialize multimodal model: {e}", exc_info=True)
        raise ZHeroVertexAIError("MultimodalAgent", "Gemini Vision Model", "Failed to initialize multimodal model on startup.", original_error=e)


@app.post("/process_content", summary="Processes multimodal (text + image) content")
async def process_multimodal_content(request: MultimodalProcessRequest): # UPDATED to use Pydantic model
    """
    Receives a text query and an image URL, processes them using a multimodal LLM,
    and returns insights.
    """
    if not multimodal_model:
        raise ZHeroDependencyError("MultimodalAgent", "Gemini Vision Model", "Multimodal model not initialized.", 500)
    if not pubsub_publisher: # Check if Pub/Sub is ready
        raise ZHeroDependencyError("MultimodalAgent", "Pub/Sub", "Pub/Sub publisher not initialized.", 500)


    logger.info(f"Multimodal Agent: Processing multimodal query for user {request.user_id}: '{request.query_text}' with image '{request.image_url}'")
    try:
        # 1. Fetch the image from the URL
        async with httpx.AsyncClient() as client:
            try:
                image_response = await client.get(str(request.image_url)) # Convert AnyUrl to str for httpx
                image_response.raise_for_status()
                image_bytes = image_response.content
                logger.info(f"Multimodal Agent: Image fetched successfully (bytes: {len(image_bytes)})")
            except httpx.RequestError as e:
                raise ZHeroDependencyError("MultimodalAgent", "Image Fetch Service", f"Failed to connect to image URL: {request.image_url}", original_error=e, status_code=503)
            except httpx.HTTPStatusError as e:
                raise ZHeroDependencyError("MultimodalAgent", "Image Fetch Service", f"Failed to fetch image: HTTP {e.response.status_code}", original_error=e, status_code=e.response.status_code)

        # 2. Prepare content for multimodal LLM
        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg") # Adjust mime_type as needed

        contents = [image_part, request.query_text]

        # 3. Request insights from multimodal LLM
        response = await multimodal_model.generate_content_async(
            contents=contents,
            generation_config={"temperature": 0.2, "max_output_tokens": 500}
        )
        multimodal_insight = response.candidates[0].text
        logger.info("Multimodal Agent: Insight generated from multimodal content.")

        return {"status": "success", "insight": multimodal_insight, "user_id": request.user_id, "query": request.query_text, "image_url": str(request.image_url)}
    except ZHeroException: raise
    except Exception as e:
        raise ZHeroAgentError("MultimodalAgent", "Error processing multimodal content.", original_error=e)
