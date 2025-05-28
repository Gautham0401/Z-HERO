# tools/learning_agent.py

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import datetime
import asyncio
import re

# NEW CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_google_vertexai import ChatVertexAI

# Import zhero_common tools and models
from zhero_common.config import logger, os
from zhero_common.models import ( # UPDATED IMPORTS
    LearningTrigger, KnowledgeGap, SearchRequest, KnowledgeItem, WebSearchResult,
    AgentPerformanceMetric, MarkGapAsAddressedRequest, GetKnowledgeGapsRequest # For updating gap status
)
from zhero_common.clients import agent_client, supabase
from zhero_common.crew_tools import ( # KEPT, as these are CrewAI Tool wrappers
    WebSearchTool, SummarizationTool, IngestKnowledgeItemTool,
    SemanticKnowledgeSearchTool, LogInternalKnowledgeGapTool
)
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError,
    ZHeroInvalidInputError, ZHeroDependencyError, ZHeroSupabaseError,
    ZHeroVertexAIError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher
# NEW: Import metrics helper
from zhero_common.metrics import log_performance_metric, PerformanceMetricName # <--- NEW


app = FastAPI(title="Learning Agent")

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


# Initialize CrewAI LLM
crew_llm_instance: Optional[ChatVertexAI] = None

@app.on_event("startup")
async def startup_event():
    global crew_llm_instance
    # Initialize Pub/Sub Publisher (this is crucial for metric logging!)
    await initialize_pubsub_publisher()

    try:
        crew_llm_instance = ChatVertexAI(
            model_name=os.environ["GEMINI_PRO_MODEL_ID"],
            project=os.environ["GCP_PROJECT_ID"],
            location=os.environ["VERTEX_AI_LOCATION"]
        )
        logger.info("Learning Agent: Initialized CrewAI LLM (ChatVertexAI).")
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.AGENT_STARTUP_SUCCESS,
                context={"model": os.environ["GEMINI_PRO_MODEL_ID"], "component": "crewai_llm"}
            )
        )
    except Exception as e:
        logger.error(f"Learning Agent: Failed to initialize CrewAI LLM (ChatVertexAI): {e}", exc_info=True)
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.AGENT_STARTUP_FAILURE,
                context={"model": os.environ["GEMINI_PRO_MODEL_ID"], "component": "crewai_llm", "error": str(e)}
            )
        )
        raise ZHeroVertexAIError("LearningAgent", "CrewAI LLM", "Failed to initialize CrewAI LLM on startup.", original_error=e)

    # Optionally, schedule a periodic knowledge review task (can trigger CrewAI)
    # asyncio.create_task(periodic_knowledge_review_task())


@app.post("/trigger_learning", summary="Triggers a specific learning process for the agent")
async def trigger_learning(trigger: LearningTrigger): # Uses Pydantic model for validation
    """
    Receives a trigger to initiate a learning process,
    e.g., from Meta-Agent based on a detected knowledge gap.
    This endpoint can also decide to delegate to a CrewAI workflow if the task is complex.
    """
    if not trigger.user_id or not trigger.trigger_type:
        raise ZHeroInvalidInputError(message="User ID and trigger type are required for learning trigger.")
    if trigger.trigger_type == "knowledge_gap" and not trigger.details:
        raise ZHeroInvalidInputError(message="Details are required for 'knowledge_gap' trigger type.")
    # pubsub_publisher check is now handled gracefully within log_performance_metric
    # if pubsub_publisher is None:
    #     logger.warning("Learning Agent: Pub/Sub publisher not initialized. Performance metrics won't be logged or complex gaps published.")

    logger.info(f"Learning Agent: Received learning trigger: '{trigger.trigger_type}' for user {trigger.user_id}")
    start_time = datetime.datetime.now() # <--- NEW
    try:
        crew_result_str: Optional[str] = None
        if trigger.trigger_type == "knowledge_gap":
            gap_details = trigger.details
            query = gap_details.get("query_text")
            reason = gap_details.get("reason")
            user_id = trigger.user_id

            if not query or not reason:
                raise ZHeroInvalidInputError(message="Query text and reason are required in details for 'knowledge_gap' trigger.")

            logger.info(f"Learning Agent: Evaluating knowledge gap '{query}' for CrewAI delegation.")

            if "deep research" in query.lower() or "comprehensive overview" in query.lower() or "meta_analysis" in reason:
                logger.info(f"Learning Agent: Delegating complex knowledge gap '{query}' to CrewAI workflow.")
                crew_run_result = await run_crewai_knowledge_acquisition_crew(user_id, query, f"Initial reason for research: {reason}")
                crew_result_str = crew_run_result.get("crew_result")
                return {"status": "delegated_to_crewai", "crew_result": crew_result_str}
            else:
                logger.info(f"Learning Agent: Performing simple research for gap '{query}'.")
                
                research_call_start = datetime.datetime.now() # <--- NEW
                web_results_list_json = await agent_client.post(
                    "research_agent",
                    "/perform_research",
                    SearchRequest(user_id=user_id, query=query, num_results=5).model_dump(exclude_unset=True) # Pass Pydantic
                )
                web_results_list = [WebSearchResult(**r) for r in web_results_list_json]
                research_call_end = datetime.datetime.now() # <--- NEW
                research_duration_ms = (research_call_end - research_call_start).total_seconds() * 1000 # <--- NEW
                asyncio.create_task( # <--- NEW
                    log_performance_metric(
                        agent_name="learning_agent",
                        metric_name=PerformanceMetricName.AGENT_API_CALL_SUCCESS, # <--- NEW
                        value=1.0, user_id=user_id,
                        context={"target_agent": "research_agent", "endpoint": "/perform_research", "duration_ms": research_duration_ms}
                    )
                )

                if web_results_list:
                    logger.info(f"Learning Agent: Simple research found {len(web_results_list)} results. Initiating direct ingestion.")
                    ingested_count = 0
                    for result in web_results_list:
                        # This should be via Pub/Sub for background ingestion
                        asyncio.create_task(
                            _background_ingest_knowledge_via_pubsub( # Use Pub/Sub for ingestion
                                user_id, result.snippet, result.link, result.title
                            )
                        )
                        ingested_count += 1
                    logger.info(f"Learning Agent: Initiated {ingested_count} ingestion tasks via Pub/Sub for gap '{query}'.")
                else:
                    logger.warning(f"Learning Agent: Simple research found no new information for gap '{query}'.")
                    asyncio.create_task( # <--- NEW
                        log_performance_metric(
                            agent_name="learning_agent",
                            metric_name=PerformanceMetricName.KNOWLEDGE_GAP_HANDLED, # <--- NEW
                            value=0.0, user_id=user_id, # Value 0.0 for "failed" to address meaningfully
                            context={"query": query, "reason": "no_new_info", "delegated_to_crewai": False}
                        )
                    )

                asyncio.create_task( # Log performance via Pub/Sub
                    _log_gap_handling_performance( # <--- UPDATED to use new helper
                        user_id, query, True, ingested_count, (crew_result_str is not None), crew_result_str
                    )
                )
                return {"status": "processing_completed", "crew_result": crew_result_str} # crew_result_str will be None here

        elif trigger.trigger_type == "user_feedback":
            feedback_details = trigger.details
            if not feedback_details:
                raise ZHeroInvalidInputError(message="Details are required for 'user_feedback' trigger type.")
            logger.info(f"Learning Agent: Analyzing user feedback: {feedback_details}")
            
            asyncio.create_task( # Log user feedback via Pub/Sub
                _log_user_feedback_performance( # <--- UPDATED to use new helper
                    trigger.user_id, feedback_details.get("rating"), feedback_details
                )
            )
            return {"status": "feedback_analyzed"}

        else:
            raise ZHeroInvalidInputError(message=f"Unknown learning trigger type: {trigger.trigger_type}")

    except ZHeroException:
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log overall failure
                value=0.0,
                user_id=trigger.user_id,
                context={"trigger_type": trigger.trigger_type, "error": "ZHeroException", "total_duration_ms": total_duration_ms}
            )
        )
        raise
    except Exception as e:
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log overall failure
                value=0.0,
                user_id=trigger.user_id,
                context={"trigger_type": trigger.trigger_type, "error": str(e), "type": "unexpected", "total_duration_ms": total_duration_ms}
            )
        )
        raise ZHeroAgentError("LearningAgent", f"Error processing learning trigger: {e}", original_error=e)


# Helper functions for background tasks with error handling - MODIFIED to use log_performance_metric
async def _background_ingest_knowledge_via_pubsub(user_id: str, content: str, source_url: Optional[str], title: Optional[str]):
    if pubsub_publisher:
        try:
            await pubsub_publisher.publish_message(
                "knowledge_ingestion",
                KnowledgeItem(
                    user_id=user_id, content=content, source_url=source_url,
                    title=title, rack="Learned Gap Knowledge", book=title
                ).model_dump(exclude_unset=True)
            )
            logger.info(f"Learning Agent: Background ingestion for '{title}' queued via Pub/Sub.")
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="learning_agent",
                    metric_name=PerformanceMetricName.KNOWLEDGE_INGESTED,
                    value=1.0, user_id=user_id,
                    context={"source": "learning_agent_gap_fill", "title": title}
                )
            )
        except ZHeroException as e:
            logger.error(f"Learning Agent: Background ingestion failed (ZHeroException): {title} - {e.message}", exc_info=True)
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="learning_agent",
                    metric_name=PerformanceMetricName.KNOWLEDGE_INGESTED,
                    value=0.0, user_id=user_id,
                    context={"source": "learning_agent_gap_fill", "title": title, "error": e.message}
                )
            )
        except Exception as e:
            logger.error(f"Learning Agent: Background ingestion failed (unexpected): {title} - {e}", exc_info=True)
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="learning_agent",
                    metric_name=PerformanceMetricName.KNOWLEDGE_INGESTED,
                    value=0.0, user_id=user_id,
                    context={"source": "learning_agent_gap_fill", "title": title, "error": str(e), "type": "unexpected"}
                )
            )
    else:
        logger.warning(f"Learning Agent: Pub/Sub publisher not initialized. Cannot queue background knowledge ingestion.")

async def _log_gap_handling_performance(user_id: str, query: str, success: bool, ingested_count: int, delegated_to_crewai: bool, crew_result_str: Optional[str] = None):
    # This helper function is now updated to use log_performance_metric internally
    status_value = 1.0 if success else 0.0
    metric_context = {
        "gap_query": query,
        "ingested_count": ingested_count,
        "delegated_to_crewai": delegated_to_crewai,
        "crew_result_summary": crew_result_str
    }
    asyncio.create_task( # <--- UPDATED
        log_performance_metric(
            agent_name="learning_agent",
            metric_name=PerformanceMetricName.KNOWLEDGE_GAP_HANDLED,
            value=status_value,
            user_id=user_id,
            context=metric_context
        )
    )

async def _log_user_feedback_performance(user_id: str, rating: Optional[str], details: Dict[str, Any]):
    # This helper function is now updated to use log_performance_metric internally
    metric_context = {"feedback_type": rating, "details": details}
    asyncio.create_task( # <--- UPDATED
        log_performance_metric(
            agent_name="learning_agent",
            metric_name=PerformanceMetricName.USER_FEEDBACK_PROCESSED,
            value=1.0, # Assumed success if it reaches here
            user_id=user_id,
            context=metric_context
        )
    )


# --- CrewAI Knowledge Acquisition Functions ---
@app.post("/trigger_crewai_knowledge_acquisition", summary="Triggers a CrewAI process for knowledge acquisition")
async def trigger_crewai_knowledge_acquisition_endpoint(request_data: Dict[str, Any]) -> Dict[str, Any]: # This can be a Pydantic model for request_data if the client sends structured data.
    user_id = request_data.get("user_id")
    topic = request_data.get("topic")
    context_notes = request_data.get("context_notes", "")

    if not user_id or not topic:
        raise ZHeroInvalidInputError(message="User ID and topic are required for CrewAI knowledge acquisition.")
        
    # pubsub_publisher check is handled by log_performance_metric internally
    # if pubsub_publisher is None:
    #    raise ZHeroDependencyError("LearningAgent", "Pub/Sub", "Pub/Sub publisher not initialized. Cannot run CrewAI tasks that publish.", 500)
        
    if not crew_llm_instance:
        raise ZHeroDependencyError("LearningAgent", "CrewAI LLM", "CrewAI LLM not initialized. Cannot run crew.", 500)

    start_time = datetime.datetime.now() # <--- NEW
    try:
        result = await run_crewai_knowledge_acquisition_crew(user_id, topic, context_notes)
        end_time = datetime.datetime.now() # <--- NEW
        duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # <--- NEW
                value=1.0, user_id=user_id,
                context={"endpoint": "/trigger_crewai_knowledge_acquisition", "topic": topic, "duration_ms": duration_ms}
            )
        )
        return {"message": "CrewAI knowledge acquisition initiated", "result": result}
    except ZHeroException:
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # <--- NEW
                value=0.0, user_id=user_id, # Log failure
                context={"endpoint": "/trigger_crewai_knowledge_acquisition", "topic": topic, "error": "ZHeroException"}
            )
        )
        raise
    except Exception as e:
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # <--- NEW
                value=0.0, user_id=user_id, # Log failure
                context={"endpoint": "/trigger_crewai_knowledge_acquisition", "topic": topic, "error": str(e), "type": "unexpected"}
            )
        )
        raise ZHeroAgentError("LearningAgent", "Failed to trigger CrewAI knowledge acquisition endpoint.", original_error=e)


async def run_crewai_knowledge_acquisition_crew(user_id: str, topic: str, context_notes: str) -> Dict[str, Any]:
    logger.info(f"Learning Agent: Initiating CrewAI Knowledge Acquisition Crew for topic: '{topic}' (User: {user_id})")
    crew_start_time = datetime.datetime.now() # <--- NEW

    if not crew_llm_instance: # Defensive check, should be caught by endpoint
        raise ZHeroDependencyError("LearningAgent", "CrewAI LLM", "CrewAI LLM not initialized. Cannot run knowledge acquisition crew.", 500)

    researcher = Agent(
        role='Knowledge Researcher',
        goal=f'Find comprehensive and accurate information about "{topic}" from external web sources. Focus on facts, definitions, history, and applications.',
        backstory='An expert information retriever, meticulous in finding relevant and up-to-date data.',
        verbose=True, allow_delegation=False, llm=crew_llm_instance,
        tools=[WebSearchTool, SemanticKnowledgeSearchTool]
    )

    summarizer = Agent(
        role='Content Summarizer',
        goal=f'Condense raw research data on "{topic}" into concise, coherent, and highly informative summaries suitable for Z-HERO’s knowledge base. Highlight key takeaways.',
        backstory='A master of distillation, able to extract the essence from vast amounts of information.',
        verbose=True, allow_delegation=False, llm=crew_llm_instance,
        tools=[SummarizationTool]
    )

    ingestor = Agent(
        role='Knowledge Ingestor',
        goal=f'Properly structure and store the summarized knowledge about "{topic}" into Z-HERO’s internal knowledge base, assigning appropriate racks and books, and ensuring it’s linked to user {user_id}.',
        backstory='The meticulous librarian of Z-HERO, ensuring all new knowledge is perfectly organized and accessible. Must ensure correct use of `IngestKnowledgeItemTool` parameters.',
        verbose=True, allow_delegation=False, llm=crew_llm_instance,
        tools=[IngestKnowledgeItemTool]
    )

    fact_checker = Agent(
        role='Fact Checker',
        goal=f'Verify the accuracy and reliability of key information points about "{topic}" before final ingestion. Cross-check against multiple reputable sources. If discrepancies found, provide detailed evidence and suggest a course of action (e.g., mark for human review or re-research).',
        backstory='A meticulous verifier, ensuring every piece of information is accurate and trustworthy. Does not compromise on truth.',
        verbose=True, allow_delegation=False, llm=crew_llm_instance,
        tools=[WebSearchTool]
    )

    task_research = Task(
        description=f'Conduct in-depth web research on "{topic}". Identify at least 3-5 high-quality, reputable sources. Initial context: {context_notes} ',
        agent=researcher,
        expected_output=f'A list of URLs and snippets from reputable sources about "{topic}". The response should be a JSON string representing a list of dicts, each with "url", "title", "snippet".',
        output_file=f'research_{user_id}_{topic.replace(" ", "_")}.json'
    )

    task_summarize = Task(
        description=f'Analyze the research results provided. Condense and summarize the key information about "{topic}" into clear, distinct paragraphs. Ensure all critical facts are included. Your output must be the FINAL summary text.',
        agent=summarizer,
        context=[task_research],
        expected_output=f'A comprehensive, concise text summary of "{topic}" ready for storage, highlighting main points and key facts. Output only the summarized text.',
        output_file=f'summary_{user_id}_{topic.replace(" ", "_")}.md'
    )

    task_fact_check = Task(
        description=f'Review the summary generated by the Content Summarizer. For critical claims or facts, perform quick web searches to cross-verify their accuracy and source reliability. Report any discrepancies found. If possible, provide evidence for both confirmed facts and discrepancies. YOUR OUTPUT MUST STATE "VERIFIED" if all checks pass, or detailed "DISCREPANCIES FOUND" with specifics.',
        agent=fact_checker,
        context=[task_summarize],
        expected_output=f'A verification report: either "VERIFIED" if the summary is accurate and reliable, or "DISCREPANCIES FOUND: [details]" listing specific issues with evidence.'
    )

    task_ingest = Task(
        description=f'Based on the final summary (from Content Summarizer) and the fact-checking result (from Fact Checker), use the IngestKnowledgeItemTool to store the knowledge. Properly associate it with user {user_id}. You MUST infer appropriate "rack" and "book" names based on the topic and content. Title the knowledge item as "{topic} for {user_id} - Z-HERO Derived". If fact-checking found issues that cannot be resolved, state why ingestion will not proceed and log a knowledge gap with the Meta-Agent (LogInternalKnowledgeGapTool). YOUR OUTPUT MUST BE THE KNOWLEDGE ITEM ID IF SUCCESSFUL, OR A REASON FOR FAILURE.',
        agent=ingestor,
        context=[task_summarize, task_fact_check],
        expected_output=f'Confirmation of successful knowledge ingestion with the new knowledge item ID (e.g., "KNOWLEDGE_ITEM_ID: [id]"), or a reason for failure to ingest (e.g., "FAILED_INGESTION: [reason]").'
    )

    
    knowledge_acquisition_crew = Crew(
        agents=[researcher, summarizer, ingestor, fact_checker],
        tasks=[task_research, task_summarize, task_fact_check, task_ingest],
        verbose=2,
        process=Process.sequential
    )

    crew_result_str: str = ""
    try:
        crew_result_str = await asyncio.to_thread(knowledge_acquisition_crew.kickoff)
        logger.info(f"Learning Agent: CrewAI Knowledge Acquisition Crew finished. Raw result: {crew_result_str[:200]}...")
        crew_end_time = datetime.datetime.now() # <--- NEW
        crew_duration_ms = (crew_end_time - crew_start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS, # Generic tool/crew success
                value=1.0, user_id=user_id,
                context={"tool_name": "crewai_knowledge_acquisition", "topic": topic, "duration_ms": crew_duration_ms}
            )
        )
    except Exception as e: # Catch any lower level CrewAI/Langchain exception
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                value=1.0, user_id=user_id,
                context={"tool_name": "crewai_knowledge_acquisition", "topic": topic, "error": str(e)}
            )
        )
        raise ZHeroAgentError("LearningAgent", "CrewAI Knowledge Acquisition Crew kickoff failed.", original_error=e)

    ingestion_id_match = re.search(r"KNOWLEDGE_ITEM_ID:\s*\[(.*?)\]", crew_result_str)
    if ingestion_id_match:
        ingestion_id = ingestion_id_match.group(1)
        status_result = "completed"
        message_result = f"Knowledge successfully acquired and ingested with ID: {ingestion_id}"
    else:
        status_result = "completed_with_issues"
        message_result = f"Crew completed but ingestion might have failed. Review crew output: {crew_result_str}"

    return {"status": status_result, "message": message_result, "crew_result": crew_result_str}


async def periodic_knowledge_review_task():
    """
    Simulates a periodic task where Learning Agent asks Meta-Agent for unaddressed knowledge gaps.
    In production, this would be triggered by Google Cloud Tasks.
    """
    logger.info("Learning Agent (Periodic): Delaying initial knowledge review for startup.") # <--- NEW
    await asyncio.sleep(10) # Initial delay for startup
    while True:
        logger.info("Learning Agent (Periodic): Checking for outstanding knowledge gaps...")
        task_start_time = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="learning_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log task start using a generic metric
                value=1.0,
                context={"task_type": "periodic_knowledge_review", "status": "started"}
            )
        )

        if not crew_llm_instance:
            logger.warning("Learning Agent (Periodic): CrewAI LLM not initialized. Skipping advanced gap review.")
            await asyncio.sleep(60 * 60)
            continue
        # Check for pubsub_publisher no longer needed explicitly as log_performance_metric handles it.
        # This check is more for publishing the final plan.
        if pubsub_publisher is None: # Check if Pub/Sub is ready
            logger.warning("Learning Agent (Periodic): Pub/Sub publisher not initialized. Skipping performance logging.")
            await asyncio.sleep(60 * 60)
            continue

        try:
            gap_fetch_start = datetime.datetime.now() # <--- NEW
            gaps_response = await agent_client.post(
                "meta_agent",
                "/get_knowledge_gaps",
                GetKnowledgeGapsRequest(status="unaddressed", limit=5).model_dump(exclude_unset=True) # Pass Pydantic
            )
            gaps: List[KnowledgeGap] = [KnowledgeGap(**g) for g in gaps_response.get("gaps", [])]
            gap_fetch_end = datetime.datetime.now() # <--- NEW
            gap_fetch_duration_ms = (gap_fetch_end - gap_fetch_start).total_seconds() * 1000 # <--- NEW
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="learning_agent",
                    metric_name=PerformanceMetricName.AGENT_API_CALL_SUCCESS, # <--- NEW
                    value=1.0,
                    context={"target_agent": "meta_agent", "endpoint": "/get_knowledge_gaps", "count": len(gaps), "duration_ms": gap_fetch_duration_ms}
                )
            )

            if gaps:
                logger.info(f"Learning Agent (Periodic): Found {len(gaps)} unaddressed gaps. Evaluating for handling.")
                for gap in gaps:
                    if "complex" in gap.reason.lower() or "comprehensive" in gap.reason.lower() or "meta_analysis" in gap.reason.lower():
                        logger.info(f"Learning Agent (Periodic): Delegating complex gap '{gap.query_text}' to CrewAI.")
                        crew_run_result = await run_crewai_knowledge_acquisition_crew(gap.user_id, gap.query_text, gap.reason)
                        # Log gap handling success/failure
                        asyncio.create_task(
                            _log_gap_handling_performance(
                                gap.user_id, gap.query_text, True, 0, True, crew_run_result.get("crew_result")
                            )
                        )
                        # Mark gap as in progress via API call
                        mark_gap_start = datetime.datetime.now() # <--- NEW
                        await agent_client.post(
                            "meta_agent",
                            "/mark_gap_as_addressed",
                            MarkGapAsAddressedRequest(gap_id=gap.id, status="in_progress").model_dump(exclude_unset=True)
                        )
                        mark_gap_end = datetime.datetime.now() # <--- NEW
                        mark_gap_duration_ms = (mark_gap_end - mark_gap_start).total_seconds() * 1000 # <--- NEW
                        asyncio.create_task( # <--- NEW
                            log_performance_metric(
                                agent_name="learning_agent",
                                metric_name=PerformanceMetricName.AGENT_API_CALL_SUCCESS, # <--- NEW
                                value=1.0, user_id=gap.user_id,
                                context={"target_agent": "meta_agent", "endpoint": "/mark_gap_as_addressed", "gap_id": gap.id, "new_status": "in_progress", "duration_ms": mark_gap_duration_ms}
                            )
                        )
                    else:
                        logger.info(f"Learning Agent (Periodic): Handling simple gap '{gap.query_text}' directly.")
                        research_start = datetime.datetime.now() # <--- NEW
                        web_results_list_json = await agent_client.post(
                            "research_agent",
                            "/perform_research",
                            SearchRequest(user_id=gap.user_id, query=gap.query_text, num_results=3).model_dump(exclude_unset=True)
                        )
                        web_results_list = [WebSearchResult(**r) for r in web_results_list_json]
                        research_end = datetime.datetime.now() # <--- NEW
                        research_duration_ms = (research_end - research_start).total_seconds() * 1000 # <--- NEW
                        asyncio.create_task( # <--- NEW
                            log_performance_metric(
                                agent_name="learning_agent",
                                metric_name=PerformanceMetricName.AGENT_API_CALL_SUCCESS, # <--- NEW
                                value=1.0, user_id=gap.user_id,
                                context={"target_agent": "research_agent", "endpoint": "/perform_research", "duration_ms": research_duration_ms}
                            )
                        )
                        ingested_count = 0
                        if web_results_list:
                            for result in web_results_list:
                                asyncio.create_task(
                                    _background_ingest_knowledge_via_pubsub(gap.user_id, result.snippet, result.link, result.title)
                                )
                                ingested_count += 1
                        # Mark gap as addressed via API call
                        mark_gap_start = datetime.datetime.now() # <--- NEW
                        await agent_client.post(
                            "meta_agent",
                            "/mark_gap_as_addressed",
                            MarkGapAsAddressedRequest(gap_id=gap.id, status="addressed").model_dump(exclude_unset=True)
                        )
                        mark_gap_end = datetime.datetime.now() # <--- NEW
                        mark_gap_duration_ms = (mark_gap_end - mark_gap_start).total_seconds() * 1000 # <--- NEW
                        asyncio.create_task( # <--- NEW
                            log_performance_metric(
                                agent_name="learning_agent",
                                metric_name=PerformanceMetricName.AGENT_API_CALL_SUCCESS, # <--- NEW
                                value=1.0, user_id=gap.user_id,
                                context={"target_agent": "meta_agent", "endpoint": "/mark_gap_as_addressed", "gap_id": gap.id, "new_status": "addressed", "duration_ms": mark_gap_duration_ms}
                            )
                        )
                        # Log gap handling success/failure
                        asyncio.create_task(
                            _log_gap_handling_performance(
                                gap.user_id, gap.query_text, True, ingested_count, False
                            )
                        )

            else:
                logger.info(f"Learning Agent (Periodic): No unaddressed knowledge gaps found.")

            task_end_time = datetime.datetime.now() # <--- NEW
            task_duration_ms = (task_end_time - task_start_time).total_seconds() * 1000 # <--- NEW
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="learning_agent",
                    metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log task end
                    value=1.0,
                    context={"task_type": "periodic_knowledge_review", "status": "completed", "duration_ms": task_duration_ms}
                )
            )

        except ZHeroException as e:
            logger.error(f"Learning Agent (Periodic): Error during periodic knowledge review (ZHeroException): {e.message}", exc_info=True)
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="learning_agent",
                    metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log task failure
                    value=0.0, user_id="system", # user_id could be "system" for periodic tasks
                    context={"task_type": "periodic_knowledge_review", "status": "failed", "error": e.message}
                )
            )
        except Exception as e:
            logger.error(f"Learning Agent (Periodic): Error during periodic knowledge review (unexpected error): {e}", exc_info=True)
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="learning_agent",
                    metric_name=PerformanceMetricName.QUERY_PROCESSED, # Log task failure
                    value=0.0, user_id="system",
                    context={"task_type": "periodic_knowledge_review", "status": "failed", "error": str(e), "type": "unexpected"}
                )
            )

        await asyncio.sleep(60 * 60)
