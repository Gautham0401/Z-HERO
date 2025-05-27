# research_agent.py
from fastapi import FastAPI, HTTPException
import httpx # For network requests to external tools
from google.api_client import discovery # For Google Custom Search client

# Import common utilities
from zhero_common.config import logger, os, TOOL_ENDPOINTS
from zhero_common.models import SearchRequest, WebSearchResult
from zhero_common.clients import tool_client # Use the imported tool client

app = FastAPI(title="Research Agent")

# Initialize Google Custom Search client (if used directly, otherwise Orchestration handles)
# In most cases, the Web Search Tool will encapsulate this, and Research Agent just calls the tool.
# This client is here for illustration if Research Agent bypassed the dedicated tool.
cse_service = None
try:
    cse_service = discovery.build(
        "customsearch", "v1", developerKey=os.environ["GOOGLE_CSE_API_KEY"]
    )
    logger.info("Research Agent: Initialized Google Custom Search service.")
except Exception as e:
    logger.warning(f"Research Agent: Failed to initialize Google Custom Search service directly: {e}. Will rely on dedicated tool.", exc_info=True)


@app.post("/perform_research", response_model=List[WebSearchResult], summary="Performs web research using the Web Search Tool")
async def perform_research(request: SearchRequest):
    """
    Performs comprehensive web research by calling the dedicated Web Search Tool.
    This agent can also include logic for advanced parsing, filtering, and source assessment.
    """
    logger.info(f"Research Agent: Received research request for '{request.query}' (User: {request.user_id})")
    try:
        # Call the dedicated Web Search Tool
        web_results_json = await tool_client.post(
            "web_search_tool",
            "/search",
            request.model_dump(exclude_unset=True) # Pass search request data
        )
        web_results = [WebSearchResult(**r) for r in web_results_json]
        logger.info(f"Research Agent: Web Search Tool returned {len(web_results)} results.")

        # (Conceptual) Add logic for source assessment, filtering, and content extraction here
        # For a truly robust Research Agent, you'd implement:
        # - More advanced web scraping (e.g., with Playwright/Selenium for dynamic content)
        # - Content cleaning (remove ads, navigations)
        # - Source reliability scoring (e.g., based on domain, publication history)
        # - Identifying key entities from search results
        for result in web_results:
            # Mock source reliability scoring
            if "wikipedia.org" in result.link:
                result.source_reliability_score = 0.9
            elif "blog" in result.link and "personal" in result.link:
                result.source_reliability_score = 0.3
            else:
                result.source_reliability_score = 0.7 # Default

            # (Conceptual) Extract publication date from URL or content
            result.publication_date = "2024-05-24" # Mock current date

        logger.info(f"Research Agent: Processed {len(web_results)} web research results.")
        return web_results
    except Exception as e:
        logger.error(f"Research Agent: Error performing research: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Research failed: {e}")

# To run this agent: uvicorn research_agent:app --port 8003 --reload