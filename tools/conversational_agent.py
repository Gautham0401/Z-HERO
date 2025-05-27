# conversational_agent.py
from fastapi import FastAPI, HTTPException
from vertexai.preview.generative_models import GenerativeModel
from typing import List, Dict, Any, Optional

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import AIResponse, UserQuery

app = FastAPI(title="Conversational Agent")

conversational_model: Optional[GenerativeModel] = None

@app.on_event("startup")
async def startup_event():
    """Initializes Vertex AI Gemini model on application startup."""
    global conversational_model
    try:
        from google.cloud import aiplatform # Ensure this is imported for Vertex AI features
        aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["VERTEX_AI_LOCATION"])
        conversational_model = GenerativeModel(os.environ["GEMINI_PRO_MODEL_ID"])
        logger.info("Conversational Agent: Initialized Gemini model for response generation.")
    except Exception as e:
        logger.error(f"Conversational Agent: Failed to initialize Gemini model: {e}", exc_info=True)
        conversational_model = None

@app.post("/generate_response", response_model=AIResponse, summary="Generates a natural language response")
async def generate_response(
    user_id: str,
    query_text: str,
    context_info: Dict[str, Any], # e.g., {"retrieved_data": "...", "user_profile": "...", "sentiment": "..."}
    conversation_history: Optional[List[Dict[str, str]]] = None # {"role": "user/model", "parts": "...}
):
    """
    Generates a natural language response based on the user's query,
    provided context, and conversation history.
    """
    if not conversational_model:
        raise HTTPException(status_code=500, detail="Conversational model (LLM) not initialized.")

    logger.info(f"Conversational Agent: Generating response for user {user_id} based on query: '{query_text}'")

    # Combine context and history for the prompt
    retrieved_data = context_info.get("retrieved_data", "")
    user_profile = context_info.get("user_profile", {})
    sentiment = context_info.get("sentiment", "neutral")

    # Adapt tone based on sentiment
    tone_instruction = ""
    if sentiment == "negative" or sentiment == "frustrated":
        tone_instruction = "Be empathetic and helpful. Acknowledge any difficulty or frustration."
    elif sentiment == "positive":
        tone_instruction = "Maintain a positive and encouraging tone."
    elif sentiment == "curious":
        tone_instruction = "Provide detailed and engaging explanations, fostering further curiosity."

    # Adapt style based on user profile (mock)
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

    # Format history for Gemini API (if applicable)
    gemini_history = []
    if conversation_history:
        for turn in conversation_history:
            role = "user" if turn["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [turn["parts"]]})

    # For simplicity, we just send a single message here.
    # A real chat session would maintain the history to maintain state.
    # We send system prompt as part of the content for initial turn.
    messages = [
        {"role": "user", "parts": [system_prompt, query_text]}
    ]

    generation_config = {
        "temperature": 0.7, # Higher temperature for more creative/varied responses
        "max_output_tokens": 500,
    }

    try:
        response_stream = await conversational_model.generate_content_async(
            contents=messages,
            generation_config=generation_config
            # stream=True # For streaming responses
        )
        # Assuming non-streaming here for simplicity.
        final_response_text = response_stream.candidates[0].text
        logger.info(f"Conversational Agent: Generated response for {user_id}.")

        # In a real setup, source_citations would come from the Orchestration Agent.
        # This is just a placeholder example.
        citations = context_info.get("source_citations", [])

        return AIResponse(
            user_id=user_id,
            response_text=final_response_text,
            source_citations=citations
        )
    except Exception as e:
        logger.error(f"Conversational Agent: Error generating response for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")

# To run this agent: uvicorn conversational_agent:app --port 8004 --reload