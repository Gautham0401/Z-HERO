# main.py (Primary Entry Point for Z-HERO - the Orchestration Agent)

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import uvicorn
import httpx
import os
import asyncio
import datetime
import json
from typing import Dict, Any, Optional

# Ensure zhero_common is in your Python path or placed correctly
from zhero_common.config import logger, AGENT_ENDPOINTS
from zhero_common.models import UserQuery, AIResponse # UserQuery is now a Pydantic model
from zhero_common.exceptions import ( # Import all necessary ZHeroExceptions
    ZHeroException, ZHeroAgentError, ZHeroInvalidInputError, ZHeroDependencyError, ZHeroVertexAIError
)
from zhero_common.pubsub_client import initialize_pubsub_publisher # Import Pub/Sub initializer

# Import the actual FastAPI app instance from the orchestration_agent.py file
try:
    from agents.orchestration_agent import app as orchestration_agent_app
    logger.info("Successfully imported orchestration_agent.py as the main application for the Orchestration Agent.")
except ImportError as e:
    logger.error(f"Failed to import orchestration_agent.py: {e}", exc_info=True)
    raise SystemExit("Fatal error: Orchestration Agent module not found. "
                     "Ensure 'agents/orchestration_agent.py' exists and is accessible in PYTHONPATH.")

# Alias the imported FastAPI app instance
app = orchestration_agent_app

# --- Global Exception Handlers (REQUIRED IN ALL AGENT FILES AND MAIN ENTRY) ---
# These handlers ensure consistent error responses and logging.
@app.exception_handler(ZHeroException)
async def zhero_exception_handler(request: Request, exc: ZHeroException):
    logger.error(f"ZHeroException caught for request {request.url.path}: {exc.message}", exc_info=True, extra={"details": exc.details, "status_code": exc.status_code})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_type": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException caught for request {request.url.path}: {exc.detail}", exc_info=True, extra={"status_code": exc.status_code, "request_body": await request.body()})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_type": "HTTPException",
            "message": exc.detail,
            "details": getattr(exc, 'body', None) # For ValidationError details
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log_id = f"ERR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.critical(f"Unhandled Exception caught for request {request.url.path} (ID: {log_id}): {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": "InternalServerError",
            "message": "An unexpected internal server error occurred. Please try again later.",
            "error_id": log_id,
            "details": str(exc) if app.debug else None
        }
    )
# --- END Global Exception Handlers ---


# --- Ensure Pub/Sub Publisher is initialized on global app startup ---
# This ensures that any part of the main.py or the imported orchestration_agent_app
# that relies on pubsub_publisher has it ready.
@app.on_event("startup")
async def main_startup_event():
    await initialize_pubsub_publisher() # Initialize the global Pub/Sub publisher instance


# --- Example Client Interaction (for demonstration/testing) ---
# This simulates how a frontend (e.g., Flutter app) would send a request
# to the Orchestration Agent.

async def simulate_user_request(user_id: str, query_text: str, image_url: Optional[str] = None) -> AIResponse:
    """
    Simulates sending a user query to the Orchestration Agent's endpoint.
    This function acts as a conceptual client.
    """
    orchestration_agent_url = AGENT_ENDPOINTS["orchestration_agent"]
    client_endpoint = f"{orchestration_agent_url}/orchestrate_query"

    logger.info(f"Simulating user request to Orchestration Agent for user {user_id}: '{query_text}' (Image URL: {image_url})")

    # Prepare the UserQuery payload using the Pydantic model
    user_query_payload = UserQuery(
        user_id=user_id,
        query_text=query_text,
        conversation_history=[], # Keeping it simple for demo, can be expanded
        user_profile_data={"interests": "AI, Quantum Physics", "style": "technical"},
        image_url=image_url # Pass image_url if provided
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(client_endpoint, json=user_query_payload.model_dump(exclude_unset=True))
            response.raise_for_status() # Raise an exception for 4xx or 5xx status codes
            ai_response = AIResponse(**response.json())
            logger.info(f"Received AI Response for user {user_id}: {ai_response.response_text[:100]}...")
            return ai_response
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error from Orchestration Agent: {e.response.status_code} - {e.response.text}", exc_info=True)
        # Attempt to parse as ZHeroException from response body if possible
        try:
            error_details = e.response.json()
            if "error_type" in error_details:
                raise ZHeroException(
                    message=f"Orchestration Agent responded with: {error_details.get('message', 'Unknown error')}",
                    status_code=e.response.status_code,
                    details=error_details
                )
            else:
                raise # Re-raise standard HTTPException if not a ZHeroException
        except json.JSONDecodeError:
            raise # Re-raise if response is not JSON
    except httpx.RequestError as e:
        logger.error(f"Network Error connecting to Orchestration Agent: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during simulation: {e}", exc_info=True)
        raise

async def run_simulation():
    """Runs a series of simulated user requests."""
    logger.info("\n--- Starting Z-HERO Orchestration Agent Simulation ---")
    logger.info("!!! ENSURE ALL REQUIRED AGENTS/TOOLS ARE RUNNING INDEPENDENTLY !!!")
    logger.info("   Refer to instructions for running each agent in its respective file.")
    logger.info("----------------------------------------------------\n")

    test_user_id = "demo_user_001"

    # Test Case 1: Complex knowledge query (should trigger web search, ingestion, sentiment, profile lookup)
    logger.info("\n--- Simulation Test Case 1: Complex Research & Learning ---")
    try:
        response_1 = await simulate_user_request(
            test_user_id,
            "I'm feeling a bit overwhelmed by the news lately. Can you help me understand the core economic impact of AI automation on global job markets? I'd particularly like to know about recent findings (past 6 months). And also make sure this information is saved for me, and remind me what my preferred learning style is."
        )
        logger.info(f"\nSimulation 1 Result: {response_1.response_text}\nCitations: {response_1.source_citations}\n")
    except Exception as e:
        logger.error(f"Simulation Test Case 1 Failed: {e}\n")

    await asyncio.sleep(5) # Give some time for background tasks (Pub/Sub)

    # Test Case 2: Simple query (should use existing knowledge or simple LLM response)
    logger.info("\n--- Simulation Test Case 2: Simple Knowledge Retrieval ---")
    try:
        response_2 = await simulate_user_request(
            test_user_id,
            "Tell me about the general principles of quantum entanglement. I hope you saved my previous request correctly."
        )
        logger.info(f"\nSimulation 2 Result: {response_2.response_text}\nCitations: {response_2.source_citations}\n")
    except Exception as e:
        logger.error(f"Simulation Test Case 2 Failed: {e}\n")

    await asyncio.sleep(5) # Give some time for background tasks

    # Test Case 3: Multimodal Query (conceptual to be added when multimodal_agent is fully implemented)
    logger.info("\n--- Simulation Test Case 3: Multimodal Interaction (Conceptual) ---")
    try:
        response_3 = await simulate_user_request(
            test_user_id,
            "What do you observe in this image, and how does it relate to renewable energy?",
            image_url="https://example.com/solar_panel_array.jpg" # Replace with an actual test image URL or mock
        )
        logger.info(f"\nSimulation 3 Result: {response_3.response_text}\nCitations: {response_3.source_citations}\n")
    except Exception as e:
        logger.error(f"Simulation Test Case 3 Failed: {e}\n")
        logger.info("Note: Multimodal functionality relies on the multimodal_agent.py being properly set up and pointing to a vision-capable LLM.")


    logger.info("\n--- Z-HERO Orchestration Agent Simulation Ended ---")


# --- Main execution block ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Before running this `main.py`, you MUST HAVE the following agents/tools running
    # in separate terminal windows, on the ports specified in zhero_common/config.py.
    # Start them all FIRST.
    # For example (in separate terminals or a setup script):
    # uvicorn agents.meta_agent:app --port 8008 --reload
    # uvicorn tools.user_profile_agent:app --port 8001 --reload
    # uvicorn tools.knowledge_management_agent:app --port 8002 --reload
    # uvicorn tools.research_agent:app --port 8003 --reload
    # uvicorn tools.conversational_agent:app --port 8004 --reload
    # uvicorn tools.voice_interface_agent:app --port 8005 --reload
    # uvicorn tools.sentiment_analysis_agent:app --port 8006 --reload
    # uvicorn tools.learning_agent:app --port 8007 --reload
    # uvicorn agents.multimodal_agent:app --port 8009 --reload
    # uvicorn tools.web_search_tool:app --port 8010 --reload
    # uvicorn tools.summarization_tool:app --port 8013 --reload

    # To run the Orchestration Agent's API service:
    # Use: uvicorn main:app --port 8000 --reload
    logger.info("Starting Orchestration Agent (Primary Z-HERO Entry Point API Service).")
    logger.info("Access its API at http://localhost:8000/docs for Swagger UI.")

    # To run the simulation functions (these will call the API service, which should be running separately):
    # Use: python main.py
    # If running the simulation:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_simulation())
    loop.close()