# tools/research_agent.py

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import googleapiclient.discovery
from typing import List, Optional
import datetime
import asyncio # <--- NEW

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import SearchRequest, WebSearchResult # SearchRequest is already Pydantic
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroInvalidInputError,
    ZHeroDependencyError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher
# NEW: Import metrics helper
from zhero_common.metrics import log_performance_metric, PerformanceMetricName # <--- NEW


app = FastAPI(title="Research Agent")

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


cse_service: Optional[googleapiclient.discovery.Resource] = None
@app.on_event("startup")
async def startup_event():
    global cse_service
    # Initialize Pub/Sub Publisher (this is crucial for metric logging!)
    await initialize_pubsub_publisher()
    try:
        cse_service = googleapiclient.discovery.build(
            "customsearch", "v1", developerKey=os.environ["GOOGLE_CSE_API_KEY"]
        )
        logger.info("Research Agent: Initialized Google Custom Search service.")
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="research_agent",
                metric_name=PerformanceMetricName.AGENT_STARTUP_SUCCESS,
                context={"component": "google_cse_service"}
            )
        )
    except Exception as e:
        logger.error(f"Research Agent: Failed to initialize Google Custom Search service: {e}", exc_info=True)
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="research_agent",
                metric_name=PerformanceMetricName.AGENT_STARTUP_FAILURE,
                context={"component": "google_cse_service", "error": str(e)}
            )
        )
        raise ZHeroDependencyError("ResearchAgent", "Google Custom Search", "Failed to initialize Google Custom Search service on startup.", original_error=e)


@app.post("/perform_research", response_model=List[WebSearchResult], summary="Performs web research using the Web Search Tool")
async def perform_research(request: SearchRequest): # SearchRequest is already Pydantic
    """
    Performs comprehensive web research by calling the dedicated Web Search Tool.
    This agent can also include logic for advanced parsing, filtering, and source assessment.
    """
    if not cse_service: # Although initialized at startup, defensive check
        raise ZHeroDependencyError("ResearchAgent", "Google Custom Search", "Google Custom Search service not initialized.", 500)
        
    logger.info(f"Research Agent: Received research request for '{request.query}' (User: {request.user_id})")
    start_time = datetime.datetime.now() # <--- NEW
    try:
        # Call the dedicated Web Search Tool (will use agent_client's retry/circuit breaker)
        web_search_call_start = datetime.datetime.now() # <--- NEW
        web_results_json = await tool_client.post(
            "web_search_tool",
            "/search",
            request.model_dump(exclude_unset=True)
        )
        web_results = [WebSearchResult(**r) for r in web_results_json]
        logger.info(f"Research Agent: Web Search Tool returned {len(web_results)} results.")
        web_search_call_end = datetime.datetime.now() # <--- NEW
        web_search_duration_ms = (web_search_call_end - web_search_call_start).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="research_agent",
                metric_name=PerformanceMetricName.AGENT_API_CALL_SUCCESS, # <--- NEW
                value=1.0, user_id=request.user_id,
                context={"target_tool": "web_search_tool", "endpoint": "/search", "duration_ms": web_search_duration_ms, "num_results": len(web_results)}
            )
        )

        for result in web_results:
            if "wikipedia.org" in result.link:
                result.source_reliability_score = 0.9
            elif "blog" in result.link and "personal" in result.link:
                result.source_reliability_score = 0.3
            else:
                result.source_reliability_score = 0.7

            # For demonstration, use current time as publication date if not available
            result.publication_date = datetime.datetime.now().isoformat()

        logger.info(f"Research Agent: Processed {len(web_results)} web research results.")
        
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="research_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # <--- NEW
                value=1.0, user_id=request.user_id,
                context={"query": request.query, "num_results": len(web_results), "total_duration_ms": total_duration_ms, "source": "web_search_tool"}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="research_agent",
                metric_name=PerformanceMetricName.API_CALL_LATENCY_MS, # <--- NEW
                value=total_duration_ms, user_id=request.user_id,
                context={"endpoint": "/perform_research", "type": "api_latency"}
            )
        )

        return web_results
    except ZHeroException:
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="research_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # <--- NEW
                value=0.0, user_id=request.user_id, # Log failure
                context={"query": request.query, "error": "ZHeroException"}
            )
        )
        raise
    except Exception as e:
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="research_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # <--- NEW
                value=0.0, user_id=request.user_id, # Log failure
                context={"query": request.query, "error": str(e), "type": "unexpected"}
            )
        )
        raise ZHeroAgentError("ResearchAgent", f"Error performing research for '{request.query}'.", original_error=e)
