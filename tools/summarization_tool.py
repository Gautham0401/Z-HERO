# tools/summarization_tool.py

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from vertexai.preview.generative_models import GenerativeModel
import datetime
import asyncio # <--- NEW

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroInvalidInputError,
    ZHeroDependencyError, ZHeroVertexAIError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher
# NEW: Import metrics helper
from zhero_common.metrics import log_performance_metric, PerformanceMetricName # <--- NEW


app = FastAPI(title="Summarization Tool")

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


summarization_model: Optional[GenerativeModel] = None

@app.on_event("startup")
async def startup_event():
    global summarization_model
    # Initialize Pub/Sub Publisher (this is crucial for metric logging!)
    await initialize_pubsub_publisher()
    try:
        from google.cloud import aiplatform # Ensure this is imported for Vertex AI features
        aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["VERTEX_AI_LOCATION"])
        summarization_model = GenerativeModel(os.environ["GEMINI_PRO_MODEL_ID"])
        logger.info("Summarization Tool: Initialized Gemini model for summarization.")
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="summarization_tool",
                metric_name=PerformanceMetricName.AGENT_STARTUP_SUCCESS,
                context={"model": os.environ["GEMINI_PRO_MODEL_ID"]}
            )
        )
    except Exception as e:
        logger.error(f"Summarization Tool: Failed to initialize Gemini model: {e}", exc_info=True)
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="summarization_tool",
                metric_name=PerformanceMetricName.AGENT_STARTUP_FAILURE,
                context={"model": os.environ["GEMINI_PRO_MODEL_ID"], "error": str(e)}
            )
        )
        raise ZHeroVertexAIError("SummarizationTool", "Gemini Model", "Failed to initialize Gemini model on startup.", original_error=e)


@app.post("/summarize", response_model=Dict[str, str], summary="Summarizes provided text content")
async def summarize_text(request: Dict[str, str]): # Simple dict payload is fine for a single arg
    text_content = request.get("text_content")
    if not text_content:
        raise ZHeroInvalidInputError(message="'text_content' is required for summarization.")
    if not summarization_model: # Although initialized at startup, defensive check
        raise ZHeroDependencyError("SummarizationTool", "Gemini Model", "Summarization model not initialized.", 500)

    logger.info(f"Summarization Tool: Summarizing text of length {len(text_content)}.")
    start_time = datetime.datetime.now() # <--- NEW

    prompt = f"""
    Please summarize the following text concisely and clearly.
    Text:
    ---
    {text_content}
    ---
    Summary:
    """
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 200,
    }

    try:
        response = await summarization_model.generate_content_async(
            contents=[prompt],
            generation_config=generation_config
        )
        summary = response.candidates[0].text
        logger.info("Summarization Tool: Text summarized successfully.")

        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="summarization_tool",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log summarization success
                value=1.0, # No specific user_id in this tool
                context={"text_length": len(text_content), "summary_length": len(summary), "duration_ms": total_duration_ms}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="summarization_tool",
                metric_name=PerformanceMetricName.API_CALL_LATENCY_MS, # <--- NEW
                value=total_duration_ms,
                context={"endpoint": "/summarize", "type": "api_latency"}
            )
        )

        return {"summary": summary}
    except Exception as e:
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="summarization_tool",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log summarization failure
                value=0.0,
                context={"text_length": len(text_content), "error": str(e), "type": "unexpected"}
            )
        )
        raise ZHeroVertexAIError("SummarizationTool", "Gemini Model", f"Error summarizing text from Gemini: {e}", original_error=e)
