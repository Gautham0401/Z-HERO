# web_search_tool.py
from fastapi import FastAPI, HTTPException
import googleapiclient.discovery
from typing import List

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import SearchRequest, WebSearchResult

app = FastAPI(title="Web Search Tool")

cse_service = None
@app.on_event("startup")
async def startup_event():
    global cse_service
    try:
        cse_service = googleapiclient.discovery.build(
            "customsearch", "v1", developerKey=os.environ["GOOGLE_CSE_API_KEY"]
        )
        logger.info("Web Search Tool: Initialized Google Custom Search service.")
    except Exception as e:
        logger.error(f"Web Search Tool: Failed to initialize Google Custom Search service: {e}", exc_info=True)
        cse_service = None

@app.post("/search", response_model=List[WebSearchResult], summary="Performs a web search using Google Custom Search")
async def search_web(request: SearchRequest):
    """
    Performs a web search given a query and returns formatted results.
    """
    if not cse_service:
        raise HTTPException(status_code=500, detail="Google Custom Search service not initialized.")

    logger.info(f"Web Search Tool: Received search request for '{request.query}' (User: {request.user_id})")
    try:
        cse_results = cse_service.cse().list(
            q=request.query,
            cx=os.environ["GOOGLE_CSE_CX"],
            num=request.num_results,
            lr=request.language # Language Restriction
        ).execute()

        results = []
        for item in cse_results.get("items", []):
            results.append(WebSearchResult(
                title=item.get("title"),
                link=item.get("link"),
                snippet=item.get("snippet"),
                publication_date=None, # CSE API might not directly provide this easily
                source_reliability_score=None
            ))
        logger.info(f"Web Search Tool: Found {len(results)} results for '{request.query}'.")
        return results
    except Exception as e:
        logger.error(f"Web Search Tool: Error during search for '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Web search failed: {e}")

# To run this tool: uvicorn web_search_tool:app --port 8010 --reload