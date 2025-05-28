# agents/meta_agent.py
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import datetime
import asyncio
import json
import re

# NEW CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_google_vertexai import ChatVertexAI # For CrewAI agent LLM

# Import zhero_common tools and models
from zhero_common.config import logger, os
from zhero_common.models import (
    AgentPerformanceMetric, KnowledgeGap, LearningTrigger,
    GetKnowledgeGapsRequest, MarkGapAsAddressedRequest # NEW request models for validation
)
from zhero_common.clients import supabase, agent_client # Use imported clients
# Import the specific CrewAI tools
from zhero_common.crew_tools import ReadSystemLogsTool # Only need ReadSystemLogsTool class here
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroNotFoundError,
    ZHeroInvalidInputError, ZHeroDependencyError, ZHeroSupabaseError,
    ZHeroVertexAIError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher

# Instantiate the ReadSystemLogsTool here, as it's a class and is used in the CrewAI Agent/Task definitions
read_system_logs_tool_instance = ReadSystemLogsTool()

app = FastAPI(title="Meta-Agent")

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


# Initialize CrewAI LLM
crew_llm_instance: Optional[ChatVertexAI] = None

@app.on_event("startup")
async def startup_event():
    global crew_llm_instance
    # Initialize Pub/Sub Publisher (crucial before any Pub/Sub publish operations)
    await initialize_pubsub_publisher()

    try:
        crew_llm_instance = ChatVertexAI(
            model_name=os.environ["GEMINI_PRO_MODEL_ID"],
            project=os.environ["GCP_PROJECT_ID"],
            location=os.environ["VERTEX_AI_LOCATION"]
        )
        logger.info("Meta-Agent: Initialized CrewAI LLM (ChatVertexAI).")
    except Exception as e:
        logger.error(f"Meta-Agent: Failed to initialize CrewAI LLM (ChatVertexAI): {e}", exc_info=True)
        raise ZHeroVertexAIError("MetaAgent", "CrewAI LLM", "Failed to initialize CrewAI LLM on startup.", original_error=e)

    # Schedule the periodic self-reflection task to run in the background
    asyncio.create_task(periodic_self_reflection_task())
    logger.info("Meta-Agent: Scheduled periodic self-reflection task.")


@app.post("/log_performance", summary="Logs performance metrics from other agents (via Pub/Sub Push/direct call)")
async def log_performance(metric: AgentPerformanceMetric):
    """
    Receives and logs performance metrics from any agent in the system.
    This endpoint can be called by a Pub/Sub push subscription or directly (e.g., for testing).
    """
    logger.info(f"Meta-Agent: Logging performance for {metric.agent_name}: {metric.metric_name}={metric.value}")
    try:
        metric_data = metric.model_dump(exclude_unset=True)
        metric_data["timestamp"] = metric_data["timestamp"].isoformat()
        response = await supabase.from_("agent_performance_logs").insert(metric_data).execute()
        if response["error"]:
            raise ZHeroSupabaseError(agent_name="MetaAgent", message=response["error"]["message"], original_error=response["error"])
        return {"status": "logged", "metric_id": response["data"][0].get('id')}
    except ZHeroException:
        raise
    except Exception as e:
        raise ZHeroAgentError("MetaAgent", "Error logging performance metric.", original_error=e)

@app.post("/log_knowledge_gap_event", summary="Logs an internal knowledge gap event for review by the Learning Agent (via Pub/Sub Push/direct call)")
async def log_knowledge_gap_event(trigger: LearningTrigger):
    """
    Receives a LearningTrigger indicating a knowledge gap from the Orchestration Agent.
    Logs this as an unaddressed knowledge gap for the Learning Agent to process.
    This endpoint can be called by a Pub/Sub push subscription or directly (e.g., for testing).
    """
    if trigger.trigger_type != "knowledge_gap_event":
        raise ZHeroInvalidInputError(message="Invalid trigger type for this endpoint. Expected 'knowledge_gap_event'.")

    user_id = trigger.user_id
    query_text = trigger.details.get("query_text")
    reason = trigger.details.get("reason")

    if not all([user_id, query_text, reason]):
        raise ZHeroInvalidInputError(message="user_id, query_text, and reason are required in details for knowledge gap event.")

    logger.info(f"Meta-Agent: Received knowledge gap event for user {user_id}: '{query_text}' (Reason: {reason})")

    try:
        gap_to_log = KnowledgeGap(
            user_id=user_id,
            query_text=query_text,
            reason=reason,
            timestamp=datetime.datetime.utcnow()
        )
        gap_data = gap_to_log.model_dump(exclude_unset=True)
        gap_data["status"] = "unaddressed"
        gap_data["timestamp"] = gap_data["timestamp"].isoformat()

        response = await supabase.from_("knowledge_gaps").insert(gap_data).execute()
        if response["error"]:
            raise ZHeroSupabaseError(agent_name="MetaAgent", message=response["error"]["message"], original_error=response["error"])
        return {"status": "logged", "gap_id": response["data"][0].get('id')}
    except ZHeroException:
        raise
    except Exception as e:
        raise ZHeroAgentError("MetaAgent", "Error logging knowledge gap event.", original_error=e)

@app.post("/get_knowledge_gaps", response_model=Dict[str, List[KnowledgeGap]], summary="Retrieves unaddressed knowledge gaps")
async def get_knowledge_gaps(request: GetKnowledgeGapsRequest):
    """
    Retrieves a list of knowledge gaps, optionally filtered by status.
    Called by the Learning Agent.
    """
    logger.info(f"Meta-Agent: Retrieving {request.limit} knowledge gaps with status '{request.status}'")
    try:
        response = await supabase.from_("knowledge_gaps").select("*").eq("status", request.status).limit(request.limit).execute()
        if response["error"]:
            raise ZHeroSupabaseError(agent_name="MetaAgent", message=response["error"]["message"], original_error=response["error"])

        gaps_list = [KnowledgeGap(**item) for item in response["data"]]
        logger.info(f"Meta-Agent: Retrieved {len(gaps_list)} gaps.")
        return {"gaps": gaps_list}
    except ZHeroException:
        raise
    except Exception as e:
        raise ZHeroAgentError("MetaAgent", "Error retrieving knowledge gaps.", original_error=e)

@app.post("/mark_gap_as_addressed", summary="Marks a knowledge gap as addressed")
async def mark_gap_as_addressed(request: MarkGapAsAddressedRequest):
    """
    Marks a specific knowledge gap as addressed (or in progress).
    """
    logger.info(f"Meta-Agent: Marking gap {request.gap_id} as '{request.status}'")
    try:
        response = await supabase.from_("knowledge_gaps").update({"status": request.status}).eq("id", request.gap_id).execute()
        if response["error"]:
            raise ZHeroSupabaseError(agent_name="MetaAgent", message=response["error"]["message"], original_error=response["error"])
        
        if not response["data"] and response["count"] == 0:
            raise ZHeroNotFoundError(resource_name="Knowledge Gap", identifier=request.gap_id)

        return {"status": "success", "gap_id": request.gap_id, "updated_status": request.status}
    except ZHeroException:
        raise
    except Exception as e:
        raise ZHeroAgentError("MetaAgent", "Error marking gap as addressed.", original_error=e)


# --- CrewAI Self-Optimization Functions ---

async def run_crewai_self_optimization_crew() -> Dict[str, Any]:
    """
    Orchestrates a CrewAI crew to analyze system performance and propose optimizations.
    This function gathers data and then passes it to the CrewAI process.
    """
    if not crew_llm_instance:
        raise ZHeroVertexAIError("MetaAgent", "CrewAI LLM", "CrewAI LLM not initialized. Cannot run self-optimization crew.")
    if pubsub_publisher is None: # Check if Pub/Sub is ready
        raise ZHeroDependencyError("MetaAgent", "Pub/Sub", "Pub/Sub publisher not initialized.", 500)


    logger.info("Meta-Agent: Initiating CrewAI Self-Optimization Crew.")

    # 1. Gather raw data before passing to CrewAI agents
    try:
        performance_logs = await read_system_logs_tool_instance._read_logs("performance_logs", limit=50) # Use the tool's internal func
        knowledge_gaps = await read_system_logs_tool_instance._read_logs("knowledge_gaps", limit=50, status="unaddressed")
        analysis_context = {
            "performance_logs": performance_logs,
            "knowledge_gaps": knowledge_gaps
        }
        logger.info(f"Meta-Agent: Gathered data for CrewAI analysis. Performance logs: {len(performance_logs)}, Gaps: {len(knowledge_gaps)}")
    except ZHeroException:
        raise # Re-raise ZHeroExceptions from tool
    except Exception as e:
        raise ZHeroAgentError("MetaAgent", "Failed to gather initial data for CrewAI analysis.", original_error=e)

    analyst = Agent(
        role='System Performance Analyst',
        goal='Identify bottlenecks, inefficiencies, and knowledge gaps from raw performance metrics and logs. Pinpoint specific agents or processes causing issues.',
        backstory='A rigorous data scientist and diagnostician specializing in distributed AI systems. Uncovers hidden patterns and root causes in complex system data.',
        verbose=True, allow_delegation=False, llm=crew_llm_instance,
        tools=[read_system_logs_tool_instance] # Agent now has the tool to gather data itself.
    )

    strategist = Agent(
        role='Optimization Strategist',
        goal='Formulate clear, actionable, and prioritized recommendations to improve Z-HERO\'s performance, efficiency, and knowledge base based on the analyst\'s findings. Specify responsible Z-HERO microservices (e.g., Learning Agent, Orchestration Agent) and proposed actions.',
        backstory='A visionary architect and strategic planner, translating complex problems into practical, impactful solutions for AI system improvement. Always thinks about long-term system health.',
        verbose=True, allow_delegation=False, llm=crew_llm_instance,
    )

    task_analyze_performance = Task(
        description=f'Thoroughly analyze the collected system data provided. Identify top 3 critical issues affecting Z-HERO\'s operation or knowledge base. Provide concrete evidence for each issue found in the logs.\n\nRaw Data:\nPerformance Logs: {json.dumps(analysis_context["performance_logs"], indent=2, default=str)}\nKnowledge Gaps: {json.dumps(analysis_context["knowledge_gaps"], indent=2, default=str)}',
        agent=analyst,
        expected_output='A clear, concise report detailing the top 3 issues impacting Z-HERO, with supporting evidence from log data. Format as markdown list with sub-bullets.'
    )

    task_formulate_strategy = Task(
        description=f'Based on the performance analysis, formulate a strategic plan with 2-3 distinct, prioritized, and actionable recommendations for Z-HERO\'s improvement. For each recommendation, specify which EXISTING Z-HERO microservice agent(s) (e.g., Learning Agent, Orchestration Agent, User Profile Agent, etc.) should implement the change, and the expected outcome. Output should be a clear, executable plan.',
        agent=strategist,
        context=[task_analyze_performance],
        expected_output='A structured optimization plan including: 1. Problem Statement(s) from analysis, 2. Prioritized Proposed Solution(s), 3. Responsible Z-HERO Agent(s) for implementation, 4. Expected Outcome(s). Format as markdown list.'
    )

    self_optimization_crew = Crew(
        agents=[analyst, strategist],
        tasks=[task_analyze_performance, task_formulate_strategy],
        verbose=2,
        process=Process.sequential
    )

    crew_result_str: str = ""
    try:
        crew_result_str = await asyncio.to_thread(self_optimization_crew.kickoff)
        logger.info(f"Meta-Agent: CrewAI Self-Optimization Crew finished. Raw result: {crew_result_str[:200]}...")
    except Exception as e:
        raise ZHeroAgentError("MetaAgent", "CrewAI Self-Optimization Crew kickoff failed.", original_error=e)

    # 3. Parse CrewAI output and trigger actions (crucial for self-evolution)
    logger.info("Meta-Agent: Parsing CrewAI output for actionable recommendations.")

    try:
        pubsub_message_data = {
            "type": "self_optimization_plan",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "plan_summary": crew_result_str,
            "parsed_recommendations": []
        }

        # Example of parsing a specific recommendation (as before)
        if "recommendation: Learning Agent should prioritize proactive research" in crew_result_str:
            topic_match = re.search(r"recommendation: Learning Agent should prioritize proactive research in '(.*?)'", crew_result_str)
            topic = topic_match.group(1) if topic_match else "unspecified_topic"
            pubsub_message_data["parsed_recommendations"].append({
                "action": "proactive_research",
                "target_agent": "learning_agent",
                "topic": topic,
                "reason": "Identified by Meta-Agent Crew for proactive research."
            })

        await pubsub_publisher.publish_message("self_optimization_plans", pubsub_message_data)
        logger.info("Meta-Agent: Published self-optimization plan to Pub/Sub.")
    except ZHeroException as e:
        logger.error(f"Meta-Agent: Failed to publish self-optimization plan to Pub/Sub (ZHeroException): {e.message}", exc_info=True)
    except Exception as e:
        logger.error(f"Meta-Agent: Failed to publish self-optimization plan to Pub/Sub (unexpected error): {e}", exc_info=True)


    return {"status": "completed", "crew_result": crew_result_str}

@app.post("/trigger_crewai_self_optimization", summary="Triggers a CrewAI process for self-optimization")
async def trigger_crewai_self_optimization_endpoint() -> Dict[str, Any]:
    try:
        result = await run_crewai_self_optimization_crew()
        return {"message": "CrewAI self-optimization initiated", "result": result}
    except ZHeroException:
        raise
    except Exception as e:
        raise ZHeroAgentError("MetaAgent", "Failed to trigger CrewAI self-optimization endpoint.", original_error=e)


# --- Periodic self-reflection task ---
async def periodic_self_reflection_task():
    await asyncio.sleep(10) # Initial delay for startup
    while True:
        logger.info("Meta-Agent (Periodic): Initiating scheduled self-reflection cycle...")
        if not crew_llm_instance:
            logger.warning("Meta-Agent (Periodic): CrewAI LLM not initialized. Skipping advanced self-reflection.")
            await asyncio.sleep(60 * 60) # Try again in an hour if LLM is not ready
            continue
        if pubsub_publisher is None: # Check if Pub/Sub is ready
            logger.warning("Meta-Agent (Periodic): Pub/Sub publisher not initialized. Cannot publish recommendations.")
            await asyncio.sleep(60 * 60)
            continue

        try:
            await run_crewai_self_optimization_crew()
            logger.info("Meta-Agent (Periodic): Scheduled CrewAI Self-Optimization Crew completed.")
        except ZHeroException as e:
            logger.error(f"Meta-Agent (Periodic): Error during scheduled self-reflection (ZHeroException): {e.message}", exc_info=True)
        except Exception as e:
            logger.error(f"Meta-Agent (Periodic): Error during scheduled self-reflection (unexpected error): {e}", exc_info=True)

        await asyncio.sleep(60 * 60 * 24)