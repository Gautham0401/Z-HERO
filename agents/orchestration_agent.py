# orchestration_agent.py
from fastapi import FastAPI, HTTPException
from vertexai.preview.generative_models import GenerativeModel, Part, Tool as GeminiTool
from vertexai.language_models import ChatSession
import google.api_core.exceptions
import json
import asyncio # For potential future concurrency
from typing import Optional

# Import common utilities
from zhero_common.config import logger, AGENT_ENDPOINTS, TOOL_ENDPOINTS, os
from zhero_common.models import (
    UserQuery, AIResponse, SearchResult, WebSearchResult, KnowledgeItem,
    KnowledgeSearchQuery, SearchRequest, SentimentResponse, AnalyzeSentimentRequest,
    AgentPerformanceMetric, LearningTrigger # Add these models for new tool handling/logging
)
from zhero_common.clients import agent_client, tool_client, AgentClient # Import AgentClient for local mock client init if needed

app = FastAPI(title="Orchestration Agent")

# Initialize LLM for the Orchestration Agent's reasoning
orchestration_model: Optional[GenerativeModel] = None
try:
    from google.cloud import aiplatform # Ensure this is imported for Vertex AI features
    aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["VERTEX_AI_LOCATION"])
    orchestration_model = GenerativeModel(os.environ["GEMINI_PRO_MODEL_ID"])
    logger.info("Orchestration Agent: Initialized Gemini model for reasoning.")
except Exception as e:
    logger.error(f"Orchestration Agent: Failed to initialize Gemini model: {e}", exc_info=True)
    # Set to None to handle cases where model init fails, preventing immediate crash
    orchestration_model = None

# --- Pre-defined tools for Gemini's tool-calling capability ---
# These definitions need to match the actual APIs of the microservices.
# The 'func' here is a placeholder that Gemini calls; the actual HTTP call is via httpx.
# In a real LangChain setup, these would be LangChain tools.
GEMINI_TOOLS = [
    # Web Search Tool
    # --- UPDATED GEMINI_TOOLS List ---
GEMINI_TOOLS = [
    # Existing Tool: Semantic Knowledge Search (via Knowledge Management Agent)
    GeminiTool.from_function(
        func=lambda query_text, user_id, top_k=5: agent_client.post( # Direct call to KM Agent
            "knowledge_management_agent",
            "/search_knowledge_semantic",
            {"user_id": user_id, "query_text": query_text, "top_k": top_k}
        ),
        name="semantic_knowledge_search",
        description="Searches the user's personalized knowledge base semantically for relevant information (Z-HERO's 'Mind Palace'). Prioritize this for known topics or user-specific information.",
        parameters={
            "query_text": {"type": "string", "description": "The semantic query text to search for."},
            "user_id": {"type": "string", "description": "The ID of the user whose knowledge base is to be searched. ALWAYS provide this."},
            "top_k": {"type": "integer", "description": "Optional: The number of top relevant results to return (default is 5)."}
        }
    ),
    # Semantic Knowledge Search Tool (Part of Knowledge Management Agent)
    GeminiTool.from_function(
        func=lambda query_text, user_id, top_k=5: agent_client.post(
            "knowledge_management_agent",
            "/search_knowledge_semantic",
            {"user_id": user_id, "query_text": query_text, "top_k": top_k}
        ),
        name="semantic_knowledge_search",
        description="Searches the user's personalized knowledge base semantically for relevant information.",
        parameters={
            "query_text": {"type": "string", "description": "The semantic query text to search for."},
            "user_id": {"type": "string", "description": "The ID of the user whose knowledge base is to be searched."},
            "top_k": {"type": "integer", "description": "Optional: The number of top relevant results to return (default is 5)."}
        }
    ),
    # Existing Tool: Web Search (via Web Search Tool)
    GeminiTool.from_function(
        func=lambda query, user_id, num_results=3: tool_client.post( # Direct call to Web Search Tool
            "web_search_tool",
            "/search",
            {"query": query, "user_id": user_id, "num_results": num_results}
        ),
        name="web_search",
        description="Performs a general web search to find external, public, or very up-to-date information. Use this if internal knowledge is insufficient.",
        parameters={
            "query": {"type": "string", "description": "The search query to perform."},
            "user_id": {"type": "string", "description": "The ID of the user initiating the query. ALWAYS provide this."},
            "num_results": {"type": "integer", "description": "Optional: The number of search results to return (default is 3)."}
        }
    ),
    # Summarization Tool (would be part of a dedicated Summarization Agent/Tool)
    GeminiTool.from_function(
        func=lambda text_content: tool_client.post(
            "summarization_tool",
            "/summarize",
            {"text_content": text_content}
        ),
        name="summarize_text_content",
        description="Summarizes provided long text into a concise format.",
        parameters={
            "text_content": {"type": "string", "description": "The text content to be summarized."}
        }
    ),
    # Add other tools here as their APIs are defined (e.g., structured_database_tool, external APIs)
]


@app.post("/orchestrate_query", response_model=AIResponse, summary="Orchestrates a user query across Z-HERO agents and tools")
async def orchestrate_query(user_query: UserQuery):
    """
    Receives a user query and orchestrates the necessary agents and tools
    to generate a comprehensive AI response.

    The Orchestration Agent leverages Gemini's tool-calling capabilities to dynamically
    decide whether to perform a semantic knowledge search, a web search, or other actions.
    It then synthesizes the information and potentially calls the Conversational Agent
    for final response generation.
    """
    if not orchestration_model:
        raise HTTPException(status_code=500, detail="Orchestration model (LLM) not initialized.")

    logger.info(f"Orchestration Agent: Received query from user {user_query.user_id}: '{user_query.query_text}'")

    # 1. Pre-fetch User Profile & Sentiment (Orchestration Agent's responsibility)
    # In a real setup, these would be API calls to respective agents.
    # For this demo, user_query *might* contain pre-fetched data, or we mock it.
    try:
        user_profile_data_res = await agent_client.post("user_profile_agent", "/get_profile", {"user_id": user_query.user_id})
        user_context = user_profile_data_res.get("profile", {})
        # Note: If user_profile_data was already in user_query, we use that.
        if user_query.user_profile_data:
            user_context.update(user_query.user_profile_data)
        logger.info(f"Orchestration Agent: Fetched user profile for {user_query.user_id}.")
    except HTTPException as e:
        logger.warning(f"Orchestration Agent: Could not fetch user profile for {user_query.user_id}: {e.detail}. Using fallback.")
        user_context = {"interests": "general knowledge", "style": "informal"}

    try:
        sentiment_analysis_res = await agent_client.post("sentiment_analysis_agent", "/analyze", {"text": user_query.query_text, "user_id": user_query.user_id})
        current_sentiment = sentiment_analysis_res.get("sentiment", "neutral")
        if user_query.sentiment: # If sentiment was already passed to orchestrator
            current_sentiment = user_query.sentiment
        logger.info(f"Orchestration Agent: Analyzed sentiment as: {current_sentiment}.")
    except HTTPException as e:
        logger.warning(f"Orchestration Agent: Could not analyze sentiment: {e.detail}. Using fallback.")
        current_sentiment = "neutral"

    # 2. LLM-driven Decision Making and Tool Execution
    chat_session = orchestration_model.start_chat()

    # Initial prompt for Gemini's reasoning
    # This guides the LLM on its behavior and tool selection logic.
    initial_prompt = f"""
    You are the Z-HERO Orchestration Agent, aiding the user's personal AI companion.
    Your goal is to formulate a comprehensive and personalized response to the user's query.
    Follow this thought process:
    1.  **Understand Intent:** Analyze the user's query, considering their profile and conversation history.
    2.  **Internal Knowledge First:** Prefer using 'semantic_knowledge_search' to retrieve information from the user's personalized knowledge base. This is the fastest and most relevant source.
    3.  **External Knowledge if Needed:** If internal knowledge is insufficient, outdated, or the query explicitly asks for recent/general web information, use 'web_search'.
    4.  **Synthesize & Respond:** Once information is retrieved (from internal or external search), synthesize it clearly. If necessary, use 'summarize_text_content' for long retrieved content before passing it back. Then, formulate the final user-facing response. If no specific tool is required, you can directly suggest a response or ask for clarification.

    User ID: {user_query.user_id}
    User Query: "{user_query.query_text}"
    Conversation History: {json.dumps(user_query.conversation_history)}
    User Profile (inferred/explicit): {json.dumps(user_context)}
    Current Sentiment of User's message: {current_sentiment}

    Think step-by-step. Prioritize using the user's personal knowledge if relevant.
    """

    final_response_text = "I'm processing your request..."
    source_citations = []
    retrieved_content: List[str] = [] # To store snippets fetched by tools

    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 800,
    }

    try:
        # Loop for potential multi-turn tool use by Gemini
        for _ in range(3): # Max 3 turns of tool use for simplicity/safety
            response = await chat_session.send_message_async(
                initial_prompt if _ == 0 else "Continue based on previous results. If no more tools needed, just formulate the final response.",
                tools=GEMINI_TOOLS,
                generation_config=generation_config
            )

            if response.candidates and response.candidates[0].tool_calls:
                # Execute tool calls identified by Gemini
                for tool_call in response.candidates[0].tool_calls:
                    logger.info(f"Orchestration Agent: Gemini requested tool: {tool_call.function.name}")
                    tool_output = None
                    try:
                        if tool_call.function.name == "semantic_knowledge_search":
                            # Ensure user_id is always passed to semantic_knowledge_search
                            tool_call.function.args["user_id"] = user_query.user_id
                            search_results_json = await tool_client.post(
                                "knowledge_management_agent", "/search_knowledge_semantic", tool_call.function.args
                            )
                            # Convert JSON dicts back to Pydantic models for type safety if desired
                            search_results = [SearchResult(**r) for r in search_results_json]
                            knowledge_snippets = []
                            for res in search_results:
                                snippet = f"Title: {res.knowledge_item.title or 'No Title'}\nContent: {res.knowledge_item.content}"
                                knowledge_snippets.append(snippet)
                                retrieved_content.append(snippet) # Accumulate content for final response
                                if res.knowledge_item.source_url:
                                    source_citations.append({"url": res.knowledge_item.source_url, "title": res.knowledge_item.title or res.knowledge_item.content[:50]})
                            tool_output = {"semantic_knowledge_search_result": knowledge_snippets}
                            logger.info(f"Orchestration Agent: Semantic search returned {len(knowledge_snippets)} snippets.")

                        elif tool_call.function.name == "web_search":
                            # Ensure user_id is always passed to web_search
                            tool_call.function.args["user_id"] = user_query.user_id
                            web_results_json = await tool_client.post(
                                "web_search_tool", "/search", tool_call.function.args
                            )
                            web_results = [WebSearchResult(**r) for r in web_results_json]
                            web_snippets = []
                            for res in web_results:
                                snippet = f"Title: {res.title}\nLink: {res.link}\nSnippet: {res.snippet}"
                                web_snippets.append(snippet)
                                retrieved_content.append(snippet) # Accumulate content for final response
                                source_citations.append({"url": res.link, "title": res.title})

                                # Asynchronously queue web search results for ingestion by Knowledge Management Agent
                                if res.link and res.title and res.snippet:
                                    # This should be a fire-and-forget background task
                                    try:
                                        await tool_client.post(
                                            "knowledge_management_agent",
                                            "/ingest_knowledge",
                                            KnowledgeItem(
                                                user_id=user_query.user_id,
                                                content=res.snippet,
                                                source_url=res.link,
                                                title=res.title,
                                                rack="Discovered Web Data", # Default rack for web
                                                book=res.title, # Default book
                                            ).model_dump(exclude_unset=True)
                                        )
                                        logger.info(f"Orchestration Agent: Ingestion request sent for '{res.title}'")
                                    except Exception as ingest_e:
                                        logger.warning(f"Orchestration Agent: Failed to queue ingestion for '{res.title}': {ingest_e}")

                            tool_output = {"web_search_result": web_snippets}
                            logger.info(f"Orchestration Agent: Web search returned {len(web_snippets)} snippets.")

                        elif tool_call.function.name == "summarize_text_content":
                            summarization_res = await tool_client.post(
                                "summarization_tool", "/summarize", tool_call.function.args
                            )
                            tool_output = {"summarization_result": summarization_res.get("summary")}

                        else:
                            tool_output = {"error": f"Unknown tool: {tool_call.function.name}"}

                    except Exception as e:
                        logger.error(f"Orchestration Agent: Error during tool call {tool_call.function.name}: {e}", exc_info=True)
                        tool_output = {"error": f"Failed to execute tool {tool_call.function.name}: {e}"}

                    # Send tool output back to Gemini
                    response = await chat_session.send_message_async(
                        Part.from_function_response(name=tool_call.function.name, response=tool_output),
                        generation_config=generation_config
                    )
                    logger.info(f"Orchestration Agent: Sent tool output back to Gemini. Next response candidate exists: {response.candidates[0].text is not None or response.candidates[0].tool_calls}")
            elif response.candidates and response.candidates[0].text:
                final_response_text = response.candidates[0].text
                break # Gemini generated a final text response, so we stop.
            else:
                logger.warning("Orchestration Agent: Gemini returned no tool calls and no text in a loop iteration.")
                final_response_text = "I'm sorry, I could not complete that request."
                break

    except google.api_core.exceptions.InvalidArgument as e:
        logger.error(f"Orchestration Agent: Invalid argument to Gemini: {e}", exc_info=True)
        final_response_text = "I'm sorry, I encountered an internal error trying to process your request."
    except Exception as e:
        logger.error(f"Orchestration Agent: Unexpected error during orchestration process: {e}", exc_info=True)
        final_response_text = "I apologize, I'm having trouble processing that request right now."

    # If the LLM didn't produce a final text response after tool calls,
    # or if there was an error, try to pass the retrieved content to the Conversational Agent
    # For a robust system, you would *always* pass through a Conversational Agent
    # for consistent tone and formatting.
    if not final_response_text or final_response_text == "I'm sorry, I could not complete that request.":
        context_for_conversational_agent = "\n".join(retrieved_content)
        if context_for_conversational_agent:
            logger.info("Orchestration Agent: Falling back to Conversational Agent with retrieved content.")
            try:
                # Call Conversational Agent
                conv_agent_response = await agent_client.post(
                    "conversational_agent",
                    "/generate_response",
                    {
                        "user_id": user_query.user_id,
                        "query_text": user_query.query_text,
                        "context_info": {"retrieved_data": context_for_conversational_agent, "user_profile": user_context, "sentiment": current_sentiment},
                        "conversation_history": user_query.conversation_history
                    }
                )
                final_response_text = conv_agent_response.get("response_text", "I'm sorry, I couldn't generate a full response.")
            except Exception as e:
                logger.error(f"Orchestration Agent: Fallback to Conversational Agent failed: {e}", exc_info=True)
                final_response_text = "I found some information, but I'm having trouble formulating a complete answer."
        else:
            logger.info("Orchestration Agent: No retrieved content and no direct LLM response. Providing generic apology.")
            final_response_text = "I'm sorry, I couldn't find relevant information or generate a response for that."


    # Log performance metrics (for Meta-Agent to consume)
    await agent_client.post(
        "meta_agent",
        "/log_performance",
        AgentPerformanceMetric(
            agent_name="orchestration_agent",
            user_id=user_query.user_id,
            metric_name="query_processed",
            value=1.0,
            context={"query": user_query.query_text, "response_length": len(final_response_text)}
        ).model_dump(exclude_unset=True)
    )

    return AIResponse(
        user_id=user_query.user_id,
        response_text=final_response_text,
        source_citations=source_citations
    )

# To run this agent: uvicorn orchestration_agent:app --port 8000 --reload