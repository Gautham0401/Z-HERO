# tools/sentiment_analysis_agent.py

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from google.cloud import language_v1
from typing import Dict, Any, Optional
import datetime
import asyncio # <--- NEW

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import AnalyzeSentimentRequest, SentimentResponse # AnalyzeSentimentRequest is already Pydantic
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroInvalidInputError,
    ZHeroDependencyError, ZHeroVertexAIError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher
# NEW: Import metrics helper
from zhero_common.metrics import log_performance_metric, PerformanceMetricName # <--- NEW


app = FastAPI(title="Sentiment Analysis Agent")

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
    log_id = f"ERR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}" # Simple unique ID for error
    logger.critical(f"Unhandled Exception caught for request {request.url.path} (ID: {log_id}): {exc}", exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error_type": "InternalServerError","message": "An unexpected internal server error occurred. Please try again later.","error_id": log_id, "details": str(exc) if app.debug else None})
# --- END Global Exception Handlers ---


sentiment_client: Optional[language_v1.LanguageServiceClient] = None

@app.on_event("startup")
async def startup_event():
    global sentiment_client
    # Initialize Pub/Sub Publisher (this is crucial for metric logging!)
    await initialize_pubsub_publisher()
    try:
        sentiment_client = language_v1.LanguageServiceClient()
        logger.info("Sentiment Analysis Agent: Initialized Google Cloud Natural Language client.")
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="sentiment_analysis_agent",
                metric_name=PerformanceMetricName.AGENT_STARTUP_SUCCESS,
                context={"component": "google_nl_client"}
            )
        )
    except Exception as e:
        logger.error(f"Sentiment Analysis Agent: Failed to initialize NL client: {e}", exc_info=True)
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="sentiment_analysis_agent",
                metric_name=PerformanceMetricName.AGENT_STARTUP_FAILURE,
                context={"component": "google_nl_client", "error": str(e)}
            )
        )
        raise ZHeroVertexAIError("SentimentAnalysisAgent", "Google NL Client", "Failed to initialize Natural Language client on startup.", original_error=e)


@app.post("/analyze", response_model=SentimentResponse, summary="Analyzes the sentiment of text")
async def analyze_sentiment(request: AnalyzeSentimentRequest): # AnalyzeSentimentRequest is already Pydantic
    """
    Analyzes the emotional tone (sentiment) of a given text string.
    """
    if not sentiment_client: # Although initialized at startup, defensive check
        raise ZHeroDependencyError("SentimentAnalysisAgent", "Google NL Client", "Sentiment Analysis client not initialized.", 500)

    logger.info(f"Sentiment Analysis Agent: Analyzing sentiment for user {request.user_id} (text len: {len(request.text)})")
    start_time = datetime.datetime.now() # <--- NEW
    try:
        document = language_v1.Document(
            content=request.text, type_=language_v1.Document.Type.PLAIN_TEXT
        )
        
        response = await asyncio.to_thread(sentiment_client.analyze_sentiment, # <--- UPDATED to use asyncio.to_thread
            request={"document": document, "encoding_type": language_v1.EncodingType.UTF8}
        )

        sentiment_score = response.document_sentiment.score
        sentiment_magnitude = response.document_sentiment.magnitude

        if sentiment_score > 0.2:
            sentiment_category = "positive"
        elif sentiment_score < -0.2:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"

        logger.info(f"Sentiment Analysis Agent: Analyzed sentiment for {request.user_id}: {sentiment_category} (Score: {sentiment_score})")

        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="sentiment_analysis_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log analysis success
                value=1.0, user_id=request.user_id,
                context={"text_length": len(request.text), "sentiment": sentiment_category, "score": sentiment_score, "duration_ms": total_duration_ms}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="sentiment_analysis_agent",
                metric_name=PerformanceMetricName.API_CALL_LATENCY_MS, # <--- NEW
                value=total_duration_ms, user_id=request.user_id,
                context={"endpoint": "/analyze", "type": "api_latency"}
            )
        )
        return SentimentResponse(
            sentiment=sentiment_category,
            score=sentiment_score,
            magnitude=sentiment_magnitude
        )
    except Exception as e:
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="sentiment_analysis_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log analysis failure
                value=0.0, user_id=request.user_id,
                context={"text_length": len(request.text), "error": str(e), "type": "unexpected"}
            )
        )
        raise ZHeroVertexAIError("SentimentAnalysisAgent", "Google NL API", f"Error during sentiment analysis API call: {e}", original_error=e)
# Note: The `analyze_sentiment` endpoint is the main entry point for sentiment analysis requests.