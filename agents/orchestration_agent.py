# agents/orchestration_agent.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from vertexai.preview.generative_models import GenerativeModel, Part, Tool as GeminiTool
from vertexai.language_models import ChatSession
import google.api_core.exceptions
import json
import asyncio
from typing import Optional, Dict, Any, List
import datetime
import uuid # NEW: To generate unique request IDs

# Import common utilities
from zhero_common.config import logger, AGENT_ENDPOINTS, TOOL_ENDPOINTS, os
from zhero_common.models import (
    UserQuery, AIResponse, SearchResult, WebSearchResult, KnowledgeItem,
    KnowledgeSearchQuery, SearchRequest, SentimentResponse, AnalyzeSentimentRequest,
    AgentPerformanceMetric, LearningTrigger,
    UserPreferenceUpdateRequest, MultimodalProcessRequest
)
from zhero_common.clients import agent_client, tool_client
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroInvalidInputError, ZHeroDependencyError, ZHeroVertexAIError
)
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher
from zhero_common.metrics import metrics_collector # NEW: Import metrics collector

# Import the instruction from its location
from zhero_adk_backend.config.agent_instructions import DEFAULT_ZHERO_CORE_AGENT_INSTRUCTION


app = FastAPI(title="Orchestration Agent")

# --- Global Exception Handlers (REQUIRED IN ALL AGENT FILES) ---
@app.exception_handler(ZHeroException)
async def zhero_exception_handler(request: Request, exc: ZHeroException):
    logger.error(f"ZHeroException caught for request {request.url.path}: {exc.message}", exc_info=True, extra={"details": exc.details, "status_code": exc.status_code, "request_id": request.headers.get("X-Request-ID")}) # NEW: Log request_id
    return JSONResponse(status_code=exc.status_code, content={"error_type": exc.__class__.__name__,"message": exc.message,"details": exc.details})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException caught for request {request.url.path}: {exc.detail}", exc_info=True, extra={"status_code": exc.status_code, "request_body": await request.body(), "request_id": request.headers.get("X-Request-ID")}) # NEW: Log request_id
    return JSONResponse(status_code=exc.status_code, content={"error_type": "HTTPException","message": exc.detail,"details": getattr(exc, 'body', None)})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log_id = f"ERR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}" # Simple unique ID for error
    logger.critical(f"Unhandled Exception caught for request {request.url.path} (ID: {log_id}) [Request-ID: {request.headers.get('X-Request-ID')}]: {exc}", exc_info=True) # NEW: Log request_id
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error_type": "InternalServerError","message": "An unexpected internal server error occurred. Please try again later.","error_id": log_id, "details": str(exc) if app.debug else None})
# --- END Global Exception Handlers ---


# Initialize LLM for the Orchestration Agent's reasoning
orchestration_model: Optional[GenerativeModel] = None
@app.on_event("startup")
async def startup_event():
    global orchestration_model
    # Initialize Pub/Sub Publisher
    await initialize_pubsub_publisher()

    try:
        from google.cloud import aiplatform # Ensure this is imported for Vertex AI features
        aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["VERTEX_AI_LOCATION"])
        orchestration_model = GenerativeModel(os.environ["GEMINI_PRO_MODEL_ID"])
        logger.info("Orchestration Agent: Initialized Gemini model for reasoning.")
    except Exception as e:
        logger.error(f"Orchestration Agent: Failed to initialize Gemini model: {e}", exc_info=True)
        raise ZHeroVertexAIError("OrchestrationAgent", "Gemini Model", "Failed to initialize Gemini model on startup.", original_error=e)


# --- GEMINI_TOOLS List ---
GEMINI_TOOLS = [
    GeminiTool.from_function(
        func=lambda query_text, user_id, top_k=5, request_id=None: agent_client.post( # NEW: request_id
            "knowledge_management_agent", "/search_knowledge_semantic", KnowledgeSearchQuery(user_id=user_id, query_text=query_text, top_k=top_k, request_id=request_id).model_dump(exclude_unset=True), request_id=request_id), # NEW: pass request_id
        name="semantic_knowledge_search",
        description="Searches the user's personalized knowledge base semantically for relevant information (Z-HERO's 'Mind Palace'). Prioritize this for known topics or user-specific information.",
        parameters={
            "query_text": {"type": "string", "description": "The semantic query text to search for."},
            "user_id": {"type": "string", "description": "The ID of the user whose knowledge base is to be searched. ALWAYS provide this."},
            "top_k": {"type": "integer", "description": "Optional: The number of top relevant results to return (default is 5)."},
            "request_id": {"type": "string", "description": "Optional: Unique ID for tracing the request."} # NEW
        }
    ),
    GeminiTool.from_function(
        func=lambda query, user_id, num_results=3, request_id=None: tool_client.post( # NEW: request_id
            "web_search_tool", "/search", SearchRequest(user_id=user_id, query=query, num_results=num_results, request_id=request_id).model_dump(exclude_unset=True), request_id=request_id), # NEW: pass request_id
        name="web_search",
        description="Performs a general web search to find external, public, or very up-to-date information. Use this if internal knowledge is insufficient.",
        parameters={
            "query": {"type": "string", "description": "The search query to perform."},
            "user_id": {"type": "string", "description": "The ID of the user initiating the query. ALWAYS provide this."},
            "num_results": {"type": "integer", "description": "Optional: The number of search results to return (default is 3)."},
            "request_id": {"type": "string", "description": "Optional: Unique ID for tracing the request."} # NEW
        }
    ),
    GeminiTool.from_function(
        func=lambda user_id, content, source_url=None, title=None, rack=None, book=None, request_id=None: agent_client.post( # NEW: request_id
            "knowledge_management_agent", "/ingest_knowledge",
            KnowledgeItem(user_id=user_id, content=content, source_url=source_url, title=title, rack=rack, book=book).model_dump(exclude_unset=True), request_id=request_id), # NEW: pass request_id
        name="ingest_knowledge_item",
        description="Stores new or updated information into the user's personalized internal knowledge base. Crucial for Z-HERO's continuous learning. Provide as much metadata (title, rack, book) as possible.",
        parameters={
            "user_id": {"type": "string", "description": "The ID of the user for whom to store the knowledge. Use 'system' for general knowledge. ALWAYS provide this."},
            "content": {"type": "string", "description": "The actual text content of the knowledge item."},
            "source_url": {"type": "string", "description": "Optional: The URL from which the information was retrieved."},
            "title": {"type": "string", "description": "Optional: A brief, descriptive title for the knowledge item (e.g., 'Quantum Physics Basics')."},
            "rack": {"type": "string", "description": "Optional: The main category/rack where this knowledge belongs (e.g., 'Science', 'Technology')."},
            "book": {"type": "string", "description": "Optional: The specific sub-topic/book title within the rack (e.g., 'Wave-Particle Duality')."},
            "request_id": {"type": "string", "description": "Optional: Unique ID for tracing the request."} # NEW
        }
    ),
    GeminiTool.from_function(
        func=lambda text_content, request_id=None: tool_client.post( # NEW: request_id
            "summarization_tool", "/summarize", {"text_content": {"text_content": text_content}, "request_id": request_id}), # NEW: pass request_id
        name="summarize_tool",
        description="Condenses long blocks of text into concise summaries. Use for distilling search results.",
        parameters={
            "text_content": {"type": "string", "description": "The text content to be summarized."},
            "request_id": {"type": "string", "description": "Optional: Unique ID for tracing the request."} # NEW
        }
    ),
    GeminiTool.from_function(
        func=lambda user_id, preference_key, preference_value, request_id=None: agent_client.post( # NEW: request_id
            "user_profile_agent", "/update_preference",
            UserPreferenceUpdateRequest(user_id=user_id, preference_key=preference_key, preference_value=preference_value, request_id=request_id).model_dump(exclude_unset=True)), # NEW: pass request_id
        name="update_user_preference",
        description="Records or updates a specific user preference (e.g., learning style, favorite topics).",
        parameters={
            "user_id": {"type": "string", "description": "The ID of the user whose preference to update. ALWAYS provide this."},
            "preference_key": {"type": "string", "description": "The key of the preference to update (e.g., 'learning_style', 'tone_preference')."},
            "preference_value": {"type": "any", "description": "The value of the preference (e.g., 'technical', 'friendly', 'visual')."},
            "request_id": {"type": "string", "description": "Optional: Unique ID for tracing the request."} # NEW
        }
    ),
    GeminiTool.from_function(
        func=lambda user_id, query_text, reason, request_id=None: pubsub_publisher.publish_message( # Publishes to Pub/Sub
            "learning_triggers",
            LearningTrigger(user_id=user_id, trigger_type="knowledge_gap_event", details={"query_text": query_text, "reason": reason}, request_id=request_id).model_dump(exclude_unset=True)), # NEW: pass request_id
        name="log_internal_knowledge_gap",
        description="Informs the Meta-Agent about instances where Z-HERO could not find relevant information or confidently answer a query. This signals a learning opportunity for the system.",
        parameters={
            "user_id": {"type": "string", "description": "The ID of the user for whom the gap was detected. ALWAYS provide this."},
            "query_text": {"type": "string", "description": "The user's query that led to the knowledge gap."},
            "reason": {"type": "string", "description": "A concise reason for the gap (e.g., 'no_relevant_internal_data', 'outdated_knowledge', 'complex_reasoning_required')."},
            "request_id": {"type": "string", "description": "Optional: Unique ID for tracing the request."} # NEW
        }
    ),
    GeminiTool.from_function(
        func=lambda user_id, query_text, image_url, request_id=None: agent_client.post( # NEW: request_id
            "multimodal_agent", "/process_content",
            MultimodalProcessRequest(user_id=user_id, query_text=query_text, image_url=image_url, request_id=request_id).model_dump(exclude_unset=True)), # NEW: pass request_id
        name="process_multimodal_content",
        description="Forwards multimodal queries (text + image) to a specialized agent for deeper analysis and interpretation of visual content combined with textual context.",
        parameters={
            "user_id": {"type": "string", "description": "The ID of the user initiating the query. ALWAYS provide this."},
            "query_text": {"type": "string", "description": "The textual part of the user's query."},
            "image_url": {"type": "string", "description": "The URL of the image associated with the query. ALWAYS provide this if available."},
            "request_id": {"type": "string", "description": "Optional: Unique ID for tracing the request."} # NEW
        }
    )
]


# Helper function for background ingestion via Pub/Sub
async def _ingest_web_result_via_pubsub(user_id: str, content: str, source_url: str, title: str, request_id: Optional[str] = None): # NEW: request_id
    if pubsub_publisher:
        try:
            await pubsub_publisher.publish_message(
                "knowledge_ingestion",
                KnowledgeItem(
                    user_id=user_id, content=content, source_url=source_url,
                    title=title, rack="Discovered Web Data", book=title, request_id=request_id # NEW: pass request_id
                ).model_dump(exclude_unset=True)
            )
            logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Web result queued for ingestion via Pub/Sub: '{title}'") # NEW: log request_id
        except ZHeroException as e:
            logger.error(f"[Request-ID: {request_id}] Orchestration Agent: Background ingestion failed (ZHeroException): {title} - {e.message}", exc_info=True) # NEW: log request_id
        except Exception as e:
            logger.error(f"[Request-ID: {request_id}] Orchestration Agent: Background ingestion failed (unexpected): {title} - {e}", exc_info=True) # NEW: log request_id
    else:
        logger.warning(f"[Request-ID: {request_id}] Orchestration Agent: Pub/Sub publisher not initialized. Cannot queue web result for ingestion: '{title}'.") # NEW: log request_id


# Helper function for logging performance metrics via Pub/Sub
async def _log_performance_metric_via_pubsub(user_id: str, query_text: str, response_length: int, agent_name: str = "orchestration_agent", request_id: Optional[str] = None): # NEW: request_id
    if pubsub_publisher:
        try:
            await pubsub_publisher.publish_message(
                "performance_metrics",
                AgentPerformanceMetric(
                    agent_name=agent_name,
                    user_id=user_id,
                    metric_name="query_processed",
                    value=1.0,
                    context={"query": query_text, "response_length": response_length},
                    request_id=request_id # NEW: pass request_id
                ).model_dump(exclude_unset=True)
            )
            logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Performance metric logged via Pub/Sub for user {user_id}.") # NEW: log request_id
        except ZHeroException as e:
            logger.error(f"[Request-ID: {request_id}] Orchestration Agent: Failed to log performance metric (ZHeroException): {e.message}", exc_info=True) # NEW: log request_id
        except Exception as e:
            logger.error(f"[Request-ID: {request_id}] Orchestration Agent: Failed to log performance metric (unexpected): {e}", exc_info=True) # NEW: log request_id
    else:
        logger.warning(f"[Request-ID: {request_id}] Orchestration Agent: Pub/Sub publisher not initialized. Cannot log performance metric via Pub/Sub for user {user_id}.") # NEW: log request_id


@app.post("/orchestrate_query", response_model=AIResponse, summary="Orchestrates a user query across Z-HERO agents and tools")
async def orchestrate_query(user_query: UserQuery, request: Request): # NEW: Request object to get headers
    # Extract/Generate Request ID
    request_id = user_query.request_id or request.headers.get("X-Request-ID") or str(uuid.uuid4()) # NEW
    logger.info(f"Incoming request {request_id} for user {user_query.user_id}: '{user_query.query_text}'") # NEW: Log request_id at start

    if not orchestration_model:
        raise ZHeroDependencyError("OrchestrationAgent", "Gemini Model", "Orchestration model (LLM) not initialized.", 500, details={"request_id": request_id})
    if pubsub_publisher is None: # Check if Pub/Sub is ready before major operations
        raise ZHeroDependencyError("OrchestrationAgent", "Pub/Sub", "Pub/Sub publisher not initialized.", 500, details={"request_id": request_id})

    # NEW: Context manager for timing the entire orchestration
    async with metrics_collector.time_async_operation("orchestration_query_duration", labels={"user_id": user_query.user_id, "request_id": request_id}):
        logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Received query for user {user_query.user_id}: '{user_query.query_text}'")

        # 1. Pre-fetch User Profile & Sentiment
        user_context = {}
        current_sentiment = "neutral"

        try:
            user_profile_data_res = await agent_client.post("user_profile_agent", "/get_profile", {"user_id": user_query.user_id}, request_id=request_id) # NEW: pass request_id
            user_context = user_profile_data_res.get("profile", user_context)
            if user_query.user_profile_data:
                user_context.update(user_query.user_profile_data)
            logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Fetched user profile for {user_query.user_id}.") # NEW: log request_id
        except ZHeroException as e:
            logger.warning(f"[Request-ID: {request_id}] Orchestration Agent: Could not fetch user profile for {user_query.user_id} due to ZHeroException: {e.message}. Using fallback.", exc_info=True)
            user_context = {"interests": "general knowledge", "style": "informal"}
        except Exception as e:
            logger.warning(f"[Request-ID: {request_id}] Orchestration Agent: Unexpected error fetching user profile: {e}. Using fallback.", exc_info=True)
            user_context = {"interests": "general knowledge", "style": "informal"}


        try:
            sentiment_analysis_res = await agent_client.post("sentiment_analysis_agent", "/analyze", {"text": user_query.query_text, "user_id": user_query.user_id}, request_id=request_id) # NEW: pass request_id
            current_sentiment = sentiment_analysis_res.get("sentiment", current_sentiment)
            if user_query.sentiment:
                current_sentiment = user_query.sentiment
            logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Analyzed sentiment as: {current_sentiment}.") # NEW: log request_id
        except ZHeroException as e:
            logger.warning(f"[Request-ID: {request_id}] Orchestration Agent: Could not analyze sentiment due to ZHeroException: {e.message}. Using fallback.", exc_info=True)
            current_sentiment = "neutral"
        except Exception as e:
            logger.warning(f"[Request-ID: {request_id}] Orchestration Agent: Unexpected error analyzing sentiment: {e}. Using fallback.", exc_info=True)
            current_sentiment = "neutral"


        # 2. LLM-driven Decision Making and Tool Execution
        chat_session = orchestration_model.start_chat()

        orchestration_prompt_context = {
            "user_id": user_query.user_id,
            "query_text": user_query.query_text,
            "conversation_history": json.dumps(user_query.conversation_history),
            "user_profile_data": json.dumps(user_context),
            "current_sentiment": current_sentiment,
            "image_url": str(user_query.image_url) if user_query.image_url else "None",
            "request_id": request_id # NEW: Pass request_id to LLM prompt for context/debugging
        }

        # The instruction must be formatted with the current context values
        from zhero_adk_backend.config.agent_instructions import DEFAULT_ZHERO_CORE_AGENT_INSTRUCTION
        formatted_initial_prompt = DEFAULT_ZHERO_CORE_AGENT_INSTRUCTION.format(**orchestration_prompt_context)

        final_response_text = "I'm processing your request..."
        source_citations = []
        retrieved_content: List[str] = []

        generation_config = {
            "temperature": 0.1,
            "max_output_tokens": 800,
        }

        try:
            for _ in range(3):
                with metrics_collector.time_async_operation("gemini_tool_call_decision_duration", labels={"request_id": request_id, "iteration": str(_)}): # NEW: Metrics
                    response = await chat_session.send_message_async(
                        formatted_initial_prompt if _ == 0 else "Continue based on previous results. If no more tools needed, just formulate the final response.",
                        tools=GEMINI_TOOLS,
                        generation_config=generation_config
                    )

                if response.candidates and response.candidates[0].tool_calls:
                    for tool_call in response.candidates[0].tool_calls:
                        tool_output = None
                        tool_name = tool_call.function.name # NEW
                        tool_args = tool_call.function.args # NEW

                        # Ensure request_id is present in tool arguments
                        tool_args["request_id"] = request_id

                        with metrics_collector.time_async_operation(f"tool_execution_duration.{tool_name}", labels={"request_id": request_id}): # NEW: Metrics
                            try:
                                if tool_name == "semantic_knowledge_search":
                                    search_results_json = await agent_client.post("knowledge_management_agent", "/search_knowledge_semantic", tool_args, request_id=request_id)
                                    search_results = [SearchResult(**r) for r in search_results_json]
                                    knowledge_snippets = []
                                    for res in search_results:
                                        snippet = f"Title: {res.knowledge_item.title or 'No Title'}\nContent: {res.knowledge_item.content}"
                                        knowledge_snippets.append(snippet)
                                        retrieved_content.append(snippet)
                                        if res.knowledge_item.source_url:
                                            source_citations.append({"url": str(res.knowledge_item.source_url), "title": res.knowledge_item.title or res.knowledge_item.content[:50]})
                                    tool_output = {"semantic_knowledge_search_result": knowledge_snippets}
                                    logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Semantic search returned {len(knowledge_snippets)} snippets.")

                                elif tool_name == "web_search":
                                    web_results_json = await tool_client.post("web_search_tool", "/search", tool_args, request_id=request_id)
                                    web_results = [WebSearchResult(**r) for r in web_results_json]
                                    web_snippets = []
                                    for res in web_results:
                                        snippet = f"Title: {res.title}\nLink: {res.link}\nSnippet: {res.snippet}"
                                        web_snippets.append(snippet)
                                        retrieved_content.append(snippet)
                                        source_citations.append({"url": str(res.link), "title": res.title})

                                        if pubsub_publisher and res.link and res.title and res.snippet:
                                            asyncio.create_task(
                                                _ingest_web_result_via_pubsub(
                                                    user_query.user_id, res.snippet, str(res.link), res.title, request_id=request_id
                                                )
                                            )
                                            logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Web search result queued for ingestion via Pub/Sub: '{res.title}'")

                                    tool_output = {"web_search_result": web_snippets}
                                    logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Web search returned {len(web_snippets)} snippets.")

                                elif tool_name == "ingest_knowledge_item":
                                    ingest_res = await agent_client.post("knowledge_management_agent", "/ingest_knowledge", tool_args, request_id=request_id)
                                    tool_output = {"ingest_knowledge_item_result": ingest_res}

                                elif tool_name == "summarize_tool":
                                    summarization_res = await tool_client.post("summarization_tool", "/summarize", tool_args, request_id=request_id)
                                    tool_output = {"summarization_result": summarization_res.get("summary")}

                                elif tool_name == "update_user_preference":
                                    update_res = await agent_client.post("user_profile_agent", "/update_preference", tool_args, request_id=request_id)
                                    tool_output = {"update_user_preference_result": update_res}

                                elif tool_name == "log_internal_knowledge_gap":
                                    # This tool call already publishes to Pub/Sub via pubsub_publisher.publish_message in GEMINI_TOOLS
                                    pass

                                elif tool_name == "process_multimodal_content":
                                    process_res = await agent_client.post("multimodal_agent", "/process_content", tool_args, request_id=request_id)
                                    tool_output = {"process_multimodal_content_result": process_res}

                                else:
                                    tool_output = {"error": f"Unknown tool: {tool_name}"}

                            except ZHeroException as e:
                                logger.error(f"[Request-ID: {request_id}] Orchestration Agent: Tool call failed with ZHeroException: {tool_name} - {e.message}", exc_info=True)
                                tool_output = {"error": f"Tool '{tool_name}' failed: {e.message}"}
                            except Exception as e:
                                logger.error(f"[Request-ID: {request_id}] Orchestration Agent: Unexpected error during tool call {tool_name}: {e}", exc_info=True)
                                tool_output = {"error": f"Tool '{tool_name}' failed due to unexpected error: {e}"}

                        response = await chat_session.send_message_async(
                            Part.from_function_response(name=tool_name, response=tool_output),
                            generation_config=generation_config
                        )
                        logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Sent tool output back to Gemini. Next response candidate exists: {bool(response.candidates)}")
                elif response.candidates and response.candidates[0].text:
                    final_response_text = response.candidates[0].text
                    break
                else:
                    logger.warning(f"[Request-ID: {request_id}] Orchestration Agent: Gemini returned no tool calls and no text in a loop iteration. Check prompt or LLM behavior.")
                    final_response_text = "I'm sorry, I encountered an issue completing your request."
                    break

        except google.api_core.exceptions.InvalidArgument as e:
            raise ZHeroVertexAIError("OrchestrationAgent", "Gemini Model", "Invalid argument to Gemini API.", original_error=e, status_code=400, details={"request_id": request_id})
        except ZHeroException:
            raise
        except Exception as e:
            raise ZHeroAgentError("OrchestrationAgent", "An unexpected error occurred during orchestration process.", original_error=e, details={"request_id": request_id})

        if not final_response_text or final_response_text == "I'm sorry, I encountered an issue completing your request.":
            context_for_conversational_agent = "\n".join(retrieved_content)
            if context_for_conversational_agent:
                logger.info(f"[Request-ID: {request_id}] Orchestration Agent: Falling back to Conversational Agent with retrieved content.")
                try:
                    conv_agent_response = await agent_client.post(
                        "conversational_agent",
                        "/generate_response",
                        UserQuery(
                            user_id=user_query.user_id,
                            query_text=user_query.query_text,
                            conversation_history=user_query.conversation_history,
                            user_profile_data=user_context,
                            sentiment=current_sentiment,
                            request_id=request_id # NEW: pass request_id
                        ).model_dump(exclude_unset=True)
                    )
                    final_response_text = conv_agent_response.get("response_text", "I'm sorry, I couldn't generate a full response.")
                except ZHeroException as e:
                    logger.error(f"[Request-ID: {request_id}] Orchestration Agent: Fallback to Conversational Agent failed with ZHeroException: {e.message}", exc_info=True)
                    final_response_text = "I found some information, but I'm having trouble formulating a complete answer."
                except Exception as e:
                    logger.error(f"[Request-ID: {request_id}] Orchestration Agent: Fallback to Conversational Agent failed with unexpected error: {e}", exc_info=True)
                    final_response_text = "I found some information, but I'm having trouble formulating a complete answer."
            else:
                logger.info(f"[Request-ID: {request_id}] Orchestration Agent: No retrieved content and no direct LLM response. Providing generic apology.")
                final_response_text = "I'm sorry, I couldn't find relevant information or generate a response for that."


        # Log performance metrics to Pub/Sub (fire-and-forget)
        if pubsub_publisher:
            asyncio.create_task(
                _log_performance_metric_via_pubsub(
                    user_query.user_id,
                    user_query.query_text,
                    len(final_response_text),
                    "orchestration_agent",
                    request_id # NEW: Pass request_id
                )
            )
        else:
            logger.warning(f"[Request-ID: {request_id}] Orchestration Agent: Pub/Sub publisher not initialized. Cannot log performance metric.")


        return AIResponse(
            user_id=user_query.user_id,
            response_text=final_response_text,
            source_citations=source_citations,
            request_id=request_id # NEW: Include request_id in final response
        )