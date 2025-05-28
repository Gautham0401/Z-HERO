# tools/conversational_agent.py
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from vertexai.preview.generative_models import GenerativeModel
from typing import List, Dict, Any, Optional
import datetime # For generic_exception_handler

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import AIResponse, UserQuery # UserQuery is now Pydantic
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError,
    ZHeroInvalidInputError, ZHeroDependencyError, ZHeroVertexAIError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher


app = FastAPI(title="Conversational Agent")

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


conversational_model: Optional[GenerativeModel] = None

@app.on_event("startup")
async def startup_event():
    global conversational_model
    # Initialize Pub/Sub Publisher
    await initialize_pubsub_publisher()
    try:
        from google.cloud import aiplatform # Ensure this is imported for Vertex AI features
        aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["VERTEX_AI_LOCATION"])
        conversational_model = GenerativeModel(os.environ["GEMINI_PRO_MODEL_ID"])
        logger.info("Conversational Agent: Initialized Gemini model for response generation.")
    except Exception as e:
        logger.error(f"Conversational Agent: Failed to initialize Gemini model: {e}", exc_info=True)
        raise ZHeroVertexAIError("ConversationalAgent", "Gemini Model", "Failed to initialize Gemini model on startup.", original_error=e)


@app.post("/generate_response", response_model=AIResponse, summary="Generates a natural language response")
async def generate_response(request: UserQuery): # UPDATED to use Pydantic UserQuery validation
    """
    Generates a natural language response based on the user's query,
    provided context, and conversation history.
    """
    if not request.user_id or not request.query_text:
        raise ZHeroInvalidInputError(message="User ID and query text are required.")
    if not conversational_model:
        raise ZHeroDependencyError("ConversationalAgent", "Gemini Model", "Conversational model not initialized.", 500)
    if pubsub_publisher is None: # Check if Pub/Sub is ready
        logger.warning("Conversational Agent: Pub/Sub publisher not initialized. Performance metrics won't be logged.")


    logger.info(f"Conversational Agent: Generating response for user {request.user_id} based on query: '{request.query_text}'")

    retrieved_data = request.user_profile_data # context_info.get("retrieved_data", "")
    user_profile = request.user_profile_data # context_info.get("user_profile", {})
    sentiment = request.sentiment # context_info.get("sentiment", "neutral")

    tone_instruction = ""
    if sentiment == "negative" or sentiment == "frustrated":
        tone_instruction = "Be empathetic and helpful. Acknowledge any difficulty or frustration."
    elif sentiment == "positive":
        tone_instruction = "Maintain a positive and encouraging tone."
    elif sentiment == "curious":
        tone_instruction = "Provide detailed and engaging explanations, fostering further curiosity."

    style_instruction = ""
    if user_profile.get("style") == "formal":
        style_instruction = "Use formal language."
    elif user_profile.get("style") == "technical":
        style_instruction = "Use precise technical terms where appropriate."
    elif user_profile.get("style") == "informal":
        style_instruction = "Use a friendly and informal tone."

    system_prompt = f"""
    You are Z-HERO, a personalized AI companion. Your goal is to provide clear, concise,
    and helpful responses to the user.
    Contextual Information:
    - User Profile: {user_profile}
    - User Sentiment: {sentiment} ({tone_instruction})
    - Retrieved Data: {retrieved_data if retrieved_data else "No specific relevant data was found from searches. Rely on general knowledge if necessary."}

    Instructions for response generation:
    - Address the user's query directly.
    - If retrieved data is available, integrate it naturally and concisely.
    - Ensure your response is coherent and easy to understand.
    - {style_instruction}
    - If the retrieved data explicitly comes from a source, mention it if appropriate or indicate it will be cited separately.
    - If you are unsure or the context is insufficient, politely state so or ask for clarification.
    """

    gemini_history = []
    if request.conversation_history:
        for turn in request.conversation_history:
            role = "user" if turn["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [turn["parts"]]})

    messages = [{"role": "user", "parts": [system_prompt, request.query_text]}]

    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 500,
    }

    try:
        response_stream = await conversational_model.generate_content_async(
            contents=messages,
            generation_config=generation_config
        )
        final_response_text = response_stream.candidates[0].text
        logger.info(f"Conversational Agent: Generated response for {request.user_id}.")

        citations = request.user_profile_data.get("source_citations", []) # Assuming citations are passed via user_profile_data or context_info

        # Optional: Log performance metrics (fire-and-forget)
        asyncio.create_task(
            _log_performance_metric_via_pubsub(
                request.user_id, request.query_text, len(final_response_text), "conversational_agent"
            )
        )

        return AIResponse(
            user_id=request.user_id,
            response_text=final_response_text,
            source_citations=citations
        )
    except Exception as e:
        raise ZHeroVertexAIError("ConversationalAgent", "Gemini Model", f"Error generating response from Gemini: {e}", original_error=e)

# Helper function for logging performance metrics via Pub/Sub (replicated for modularity)
async def _log_performance_metric_via_pubsub(user_id: str, query_text: str, response_length: int, agent_name: str):
    if pubsub_publisher:
        try:
            await pubsub_publisher.publish_message(
                "performance_metrics",
                AgentPerformanceMetric(
                    agent_name=agent_name,
                    user_id=user_id,
                    metric_name="query_processed",
                    value=1.0,
                    context={"query": query_text, "response_length": response_length}
                ).model_dump(exclude_unset=True)
            )
            logger.info(f"{agent_name} Performance metric logged via Pub/Sub for user {user_id}.")
        except ZHeroException as e:
            logger.error(f"{agent_name}: Failed to log performance metric (ZHeroException): {e.message}", exc_info=True)
        except Exception as e:
            logger.error(f"{agent_name}: Failed to log performance metric (unexpected): {e}", exc_info=True)
    else:
        logger.warning(f"{agent_name}: Pub/Sub publisher not initialized. Cannot log performance metric.")