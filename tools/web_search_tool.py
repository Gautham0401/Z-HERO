# tools/web_search_tool.py

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


app = FastAPI(title="Web Search Tool")

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
        logger.info("Web Search Tool: Initialized Google Custom Search service.")
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="web_search_tool",
                metric_name=PerformanceMetricName.AGENT_STARTUP_SUCCESS,
                context={"component": "google_cse_service"}
            )
        )
    except Exception as e:
        logger.error(f"Web Search Tool: Failed to initialize Google Custom Search service: {e}", exc_info=True)
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="web_search_tool",
                metric_name=PerformanceMetricName.AGENT_STARTUP_FAILURE,
                context={"component": "google_cse_service", "error": str(e)}
            )
        )
        raise ZHeroDependencyError("WebSearchTool", "Google Custom Search API", "Failed to initialize Google Custom Search service on startup.", original_error=e)


@app.post("/search", response_model=List[WebSearchResult], summary="Performs a web search using Google Custom Search")
async def search_web(request: SearchRequest): # SearchRequest is already Pydantic
    """
    Performs a web search given a query and returns formatted results.
    """
    if not cse_service:
        raise ZHeroDependencyError("WebSearchTool", "Google Custom Search API", "Google Custom Search service not initialized.", 500)

    logger.info(f"Web Search Tool: Received search request for '{request.query}' (User: {request.user_id})")
    start_time = datetime.datetime.now() # <--- NEW
    try:
        cse_api_call_start = datetime.datetime.now() # <--- NEW
        cse_results = await asyncio.to_thread(cse_service.cse().list( # <--- UPDATED: Use asyncio.to_thread for blocking call
            q=request.query,
            cx=os.environ["GOOGLE_CSE_CX"],
            num=request.num_results,
            lr=request.language
        ).execute)
        cse_api_call_end = datetime.datetime.now() # <--- NEW
        cse_api_duration_ms = (cse_api_call_end - cse_api_call_start).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="web_search_tool",
                metric_name=PerformanceMetricName.SEARCH_ENGINE_CALL_SUCCESS,
                value=1.0, user_id=request.user_id,
                context={"query": request.query, "duration_ms": cse_api_duration_ms}
            )
        )

        results = []
        for item in cse_results.get("items", []):
            try:
                results.append(WebSearchResult(
                    title=item.get("title", "No Title"),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", "No Snippet Available"),
                    publication_date=None, # Will be filled by Research Agent if needed
                    source_reliability_score=None # Will be filled by Research Agent if needed
                ))
            except Exception as e: # Catch errors during Pydantic validation (e.g., malformed URL)
                logger.warning(f"WebSearchTool: Skipping malformed search result item: {e} - Item: {item}")
                asyncio.create_task( # <--- NEW
                    log_performance_metric(
                        agent_name="web_search_tool",
                        metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                        value=1.0, user_id=request.user_id,
                        context={"tool_name": "web_search_result_parsing", "error": str(e), "item_link": item.get('link')}
                    )
                )

        logger.info(f"Web Search Tool: Found {len(results)} results for '{request.query}'.")
        
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="web_search_tool",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=1.0, user_id=request.user_id,
                context={"query": request.query, "num_results": len(results), "duration_ms": total_duration_ms}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="web_search_tool",
                metric_name=PerformanceMetricName.API_CALL_LATENCY_MS,
                value=total_duration_ms, user_id=request.user_id,
                context={"endpoint": "/search"}
            )
        )

        return results
    except Exception as e:
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="web_search_tool",
                metric_name=PerformanceMetricName.SEARCH_ENGINE_CALL_FAILURE,
                value=1.0, user_id=request.user_id,
                context={"query": request.query, "error": str(e), "type": "api_call_failure"}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="web_search_tool",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=0.0, user_id=request.user_id,
                context={"query": request.query, "error": str(e), "type": "unexpected"}
            )
        )
        raise ZHeroDependencyError("WebSearchTool", "Google Custom Search API", f"Error during search API call: {e}", original_error=e)
