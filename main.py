# main.py (Primary Entry Point for Z-HERO - the Orchestration Agent)

from fastapi import FastAPI, HTTPException
import uvicorn
import httpx
import os
import asyncio
from typing import Dict, Any

# Ensure zhero_common is in your Python path or placed correctly
# (e.g., in the same directory as main.py)
from zhero_common.config import logger, AGENT_ENDPOINTS
from zhero_common.models import UserQuery, AIResponse

# Import the FastAPI app instance from the orchestration_agent.py file
# Assuming orchestration_agent.py is in the same directory or accessible via PYTHONPATH
try:
    from orchestration_agent import app as orchestration_agent_app
    logger.info("Successfully imported orchestration_agent.py as the main application.")
except ImportError as e:
    logger.error(f"Failed to import orchestration_agent.py: {e}", exc_info=True)
    logger.error("Please ensure 'orchestration_agent.py' is in the same directory or accessible in PYTHONPATH.")
    # Exit or throw an error if the core component is missing
    raise SystemExit("Fatal error: Orchestration Agent module not found.")

# This `app` instance is now the FastAPI application for the Orchestration Agent.
# We are simply aliasing it to `app` for potential direct `uvicorn main:app` usage
# and for including example client interaction.
app = orchestration_agent_app

# --- Example Client Interaction (for demonstration/testing) ---
# This simulates how a frontend (e.g., Flutter app) would send a request
# to the Orchestration Agent.

async def simulate_user_request(user_id: str, query_text: str) -> AIResponse:
    """
    Simulates sending a user query to the Orchestration Agent's endpoint.
    This function acts as a conceptual client.
    """
    orchestration_agent_url = AGENT_ENDPOINTS["orchestration_agent"]
    client_endpoint = f"{orchestration_agent_url}/orchestrate_query"

    logger.info(f"Simulating user request to Orchestration Agent: {query_text}")

    # Prepare the UserQuery payload
    user_query_payload = UserQuery(
        user_id=user_id,
        query_text=query_text,
        conversation_history=[], # Can be expanded for multi-turn
        user_profile_data={"interests": "AI, Quantum Physics", "style": "technical"} # Example profile
    ).model_dump(exclude_unset=True) # Exclude unset fields for cleaner payload

    try:
        async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout for full orchestration
            response = await client.post(client_endpoint, json=user_query_payload)
            response.raise_for_status() # Raise an exception for 4xx or 5xx status codes
            ai_response = AIResponse(**response.json())
            logger.info(f"Received AI Response for user {user_id}: {ai_response.response_text[:100]}...")
            return ai_response
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from Orchestration Agent: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network Error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not connect to Orchestration Agent: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error during simulation: {e}")

async def run_simulation():
    """Runs a series of simulated user requests."""
    logger.info("\n--- Starting Z-HERO Orchestration Agent Simulation ---")
    logger.info("!!! ENSURE ALL REQUIRED AGENTS/TOOLS ARE RUNNING INDEPENDENTLY !!!")
    logger.info("   e.g., user_profile_agent, knowledge_management_agent, research_agent, conversational_agent, etc.")
    logger.info("   Refer to instructions for running each agent.")
    logger.info("----------------------------------------------------\n")

    test_user_id = "demo_user_001"

    # Test 1: General knowledge query (might trigger web search if no specific notes)
    try:
        response_1 = await simulate_user_request(test_user_id, "What is the recent news about AI in healthcare?")
        logger.info(f"Simulation 1 Result: {response_1.response_text}\nCitations: {response_1.source_citations}\n")
    except HTTPException as e:
        logger.error(f"Simulation 1 Failed: {e.detail}\n")

    await asyncio.sleep(2) # brief pause

    # Test 2: Query for potentially personalized knowledge (might trigger semantic search)
    try:
        response_2 = await simulate_user_request(test_user_id, "Can you tell me about the concept of quantum entanglement from my notes?")
        logger.info(f"Simulation 2 Result: {response_2.response_text}\nCitations: {response_2.source_citations}\n")
    except HTTPException as e:
        logger.error(f"Simulation 2 Failed: {e.detail}\n")

    await asyncio.sleep(2) # brief pause

    # Test 3: Query involving a sentiment aspect (for Sentiment Analysis Agent)
    try:
        response_3 = await simulate_user_request(test_user_id, "I'm really frustrated with my quantum physics assignment. Can you simplify wave-particle duality for me?")
        logger.info(f"Simulation 3 Result: {response_3.response_text}\nCitations: {response_3.source_citations}\n")
    except HTTPException as e:
        logger.error(f"Simulation 3 Failed: {e.detail}\n")

    logger.info("\n--- Z-HERO Orchestration Agent Simulation Ended ---")


# --- Main execution block ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Before running this `main.py`, you MUST HAVE the following agents/tools running
    # in separate terminal windows, on the ports specified in zhero_common/config.py:
    # 1. user_profile_agent: uvicorn user_profile_agent:app --port 8001 --reload
    # 2. knowledge_management_agent: uvicorn knowledge_management_agent:app --port 8002 --reload
    # 3. research_agent: uvicorn research_agent:app --port 8003 --reload
    # 4. conversational_agent: uvicorn conversational_agent:app --port 8004 --reload
    # 5. voice_interface_agent: uvicorn voice_interface_agent:app --port 8005 --reload (Optional if no voice requests)
    # 6. sentiment_analysis_agent: uvicorn sentiment_analysis_agent:app --port 8006 --reload
    # 7. learning_agent: uvicorn learning_agent:app --port 8007 --reload (Can run in background)
    # 8. meta_agent: uvicorn meta_agent:app --port 8008 --reload (Can run in background)
    # 9. web_search_tool: uvicorn web_search_tool:app --port 8010 --reload
    # 10. summarization_tool: uvicorn summarization_tool:app --port 8013 --reload

    # To run the Orchestration Agent (this main.py file):
    logger.info("Starting Orchestration Agent (Primary Z-HERO Entry Point).")
    logger.info("This agent will listen on port 8000 (as per zhero_common/config.py).")
    logger.info("You can access its API at http://localhost:8000/docs for Swagger UI.")
    logger.info("To run the simulation, ensure all dependent services are up and running.")

    # Option 1: Run the FastAPI app directly (for local development/testing)
    # This uses uvicorn's run function.
    # Note: If you want to use the `run_simulation()` function,
    # you'll need to run this script as a regular Python script,
    # and then the `run_simulation()` can make requests to
    # a separate process running uvicorn.
    # Example:
    # In Terminal 1: uvicorn main:app --port 8000 --reload
    # In Terminal 2: python main.py (to run the simulation part)

    # Simplified way to run the simulation and then maybe keep the server running if wanted
    # This specific block will run the simulation once, then exit if not setup correctly.
    # For a real dev loop, run uvicorn in one terminal and test separately (as above).
    try:
        # Running the simulation as an asyncio task
        asyncio.run(run_simulation())
        logger.info("Simulation completed.")
    except Exception as e:
        logger.critical(f"Error during simulation setup or execution: {e}", exc_info=True)


    # If you wanted to run the API directly from this script,
    # you would typically remove the `asyncio.run(run_simulation())` and
    # use `uvicorn.run(app, host="0.0.0.0", port=8000)`.
    # However, running the simulation in the same process requires more advanced
    # patterns (e.g., using a test client if within tests) or separate processes.
    # The current setup assumes you'll run `uvicorn main:app` in one terminal,
    # and `python main.py` in another to trigger the simulation.