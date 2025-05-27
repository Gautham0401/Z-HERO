# meta_agent.py (UPDATED & REFLECTED)
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Optional
import datetime
import asyncio
import json

# NEW CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_google_vertexai import ChatVertexAI # For CrewAI agent LLM

# Import zhero_common tools and models
from zhero_common.config import logger, os
from zhero_common.models import AgentPerformanceMetric, KnowledgeGap, LearningTrigger
from zhero_common.clients import supabase, agent_client # Use imported clients
# Import the specific CrewAI tools
from zhero_common.crew_tools import ReadSystemLogsTool, read_system_logs_tool_instance # Ensure the instantiated tool is imported


app = FastAPI(title="Meta-Agent")

# Initialize CrewAI LLM
crew_llm_instance: Optional[ChatVertexAI] = None

@app.on_event("startup")
async def startup_event():
    global crew_llm_instance
    try:
        crew_llm_instance = ChatVertexAI(
            model_name=os.environ["GEMINI_PRO_MODEL_ID"],
            project=os.environ["GCP_PROJECT_ID"],
            location=os.environ["VERTEX_AI_LOCATION"]
        )
        logger.info("Meta-Agent: Initialized CrewAI LLM (ChatVertexAI).")
    except Exception as e:
        logger.error(f"Meta-Agent: Failed to initialize CrewAI LLM (ChatVertexAI): {e}", exc_info=True)
        crew_llm_instance = None # Ensure it's None if init fails

    # Schedule the periodic self-reflection task to run in the background
    asyncio.create_task(periodic_self_reflection_task())
    logger.info("Meta-Agent: Scheduled periodic self-reflection task.")


@app.post("/log_performance", summary="Logs performance metrics from other agents")
async def log_performance(metric: AgentPerformanceMetric):
    """
    Receives and logs performance metrics from any agent in the system.
    This data forms the basis for self-reflection.
    """
    logger.info(f"Meta-Agent: Logging performance for {metric.agent_name}: {metric.metric_name}={metric.value}")
    try:
        # Store in Supabase table for performance metrics
        # Example: `agent_performance_logs` table
        metric_data = metric.model_dump(exclude_unset=True)
        metric_data["timestamp"] = metric_data["timestamp"].isoformat()
        response = await supabase.from_("agent_performance_logs").insert(metric_data).execute()
        if response["error"]:
            raise Exception(response["error"])
        # Use .get('id') with default to avoid KeyError if 'id' is not present in data
        return {"status": "logged", "metric_id": response["data"][0].get('id')}
    except Exception as e:
        logger.error(f"Meta-Agent: Error logging performance for {metric.agent_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to log performance: {e}")

@app.post("/log_knowledge_gap_event", summary="Logs an internal knowledge gap event for review by the Learning Agent")
async def log_knowledge_gap_event(trigger: LearningTrigger):
    """
    Receives a LearningTrigger indicating a knowledge gap from the Orchestration Agent.
    Logs this as an unaddressed knowledge gap for the Learning Agent to process.
    """
    if trigger.trigger_type != "knowledge_gap_event":
        raise HTTPException(status_code=400, detail="Invalid trigger type for this endpoint.")

    user_id = trigger.user_id
    query_text = trigger.details.get("query_text")
    reason = trigger.details.get("reason")

    if not all([user_id, query_text, reason]):
        raise HTTPException(status_code=400, detail="user_id, query_text, and reason are required in details for knowledge gap event.")

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
            raise Exception(response["error"])
        return {"status": "logged", "gap_id": response["data"][0].get('id')}
    except Exception as e:
        logger.error(f"Meta-Agent: Error logging knowledge gap event for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to log knowledge gap event: {e}")

@app.post("/get_knowledge_gaps", response_model=Dict[str, List[KnowledgeGap]], summary="Retrieves unaddressed knowledge gaps")
async def get_knowledge_gaps(request_data: Dict[str, Any]):
    """
    Retrieves a list of knowledge gaps, optionally filtered by status.
    Called by the Learning Agent.
    """
    status = request_data.get("status", "unaddressed")
    limit = request_data.get("limit", 10)
    logger.info(f"Meta-Agent: Retrieving {limit} knowledge gaps with status '{status}'")
    try:
        response = await supabase.from_("knowledge_gaps").select("*").eq("status", status).limit(limit).execute()
        if response["error"]:
            raise Exception(response["error"])

        gaps_list = [KnowledgeGap(**item) for item in response["data"]]
        logger.info(f"Meta-Agent: Retrieved {len(gaps_list)} gaps.")
        return {"gaps": gaps_list}
    except Exception as e:
        logger.error(f"Meta-Agent: Error retrieving knowledge gaps: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge gaps: {e}")

@app.post("/mark_gap_as_addressed", summary="Marks a knowledge gap as addressed")
async def mark_gap_as_addressed(gap_id: str, status: str = "addressed"):
    """
    Marks a specific knowledge gap as addressed (or in progress).
    """
    logger.info(f"Meta-Agent: Marking gap {gap_id} as '{status}'")
    try:
        response = await supabase.from_("knowledge_gaps").update({"status": status}).eq("id", gap_id).execute()
        if response["error"]:
            raise Exception(response["error"])
        return {"status": "success", "gap_id": response["data"][0].get('id')} # Assuming response['data'] has content
    except Exception as e:
        logger.error(f"Meta-Agent: Error marking gap {gap_id} as addressed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to mark gap as addressed: {e}")


# --- CrewAI Self-Optimization Functions ---

async def run_crewai_self_optimization_crew() -> Dict[str, Any]:
    """
    Orchestrates a CrewAI crew to analyze system performance and propose optimizations.
    This function gathers data and then passes it to the CrewAI process.
    """
    if not crew_llm_instance:
        logger.error("CrewAI LLM not initialized. Cannot run self-optimization crew.")
        return {"status": "failed", "reason": "CrewAI LLM not available."}

    logger.info("Meta-Agent: Initiating CrewAI Self-Optimization Crew.")

    # 1. Gather raw data before passing to CrewAI agents
    # Use the ReadSystemLogsTool directly or similar internal database queries.
    try:
        performance_logs = await read_system_logs_tool_instance.func("performance_logs", limit=50) # Use the tool's func directly
        knowledge_gaps = await read_system_logs_tool_instance.func("knowledge_gaps", limit=50, status="unaddressed")
        analysis_context = {
            "performance_logs": performance_logs,
            "knowledge_gaps": knowledge_gaps
        }
        logger.info(f"Meta-Agent: Gathered data for CrewAI analysis. Performance logs: {len(performance_logs)}, Gaps: {len(knowledge_gaps)}")
    except Exception as e:
        logger.error(f"Meta-Agent: Failed to gather initial data for CrewAI: {e}", exc_info=True)
        return {"status": "failed", "reason": f"Failed to gather initial data: {e}"}


    # 2. Define CrewAI Agents (internal roles)
    analyst = Agent(
        role='System Performance Analyst',
        goal='Identify bottlenecks, inefficiencies, and knowledge gaps from raw performance metrics and logs. Pinpoint specific agents or processes causing issues.',
        backstory='A rigorous data scientist and diagnostician specializing in distributed AI systems. Uncovers hidden patterns and root causes in complex system data.',
        verbose=True,
        allow_delegation=False,
        llm=crew_llm_instance,
        tools=[read_system_logs_tool_instance] # Agent now has the tool to gather data itself potentially
    )

    strategist = Agent(
        role='Optimization Strategist',
        goal='Formulate clear, actionable, and prioritized recommendations to improve Z-HERO\'s performance, efficiency, and knowledge base based on the analyst\'s findings. Specify responsible Z-HERO microservices (e.g., Learning Agent, Orchestration Agent) and proposed actions.',
        backstory='A visionary architect and strategic planner, translating complex problems into practical, impactful solutions for AI system improvement. Always thinks about long-term system health.',
        verbose=True,
        allow_delegation=False,
        llm=crew_llm_instance,
        # Strategist primarily uses reasoning, but could have tools to check configs, etc.
    )

    # 3. Define Tasks for the Crew
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
        process=Process.sequential # Execute tasks in order
    )

    # Run CrewAI in a separate thread to not block async FastAPI's event loop
    crew_result_str: str = ""
    try:
        crew_result_str = await asyncio.to_thread(self_optimization_crew.kickoff)
        logger.info(f"Meta-Agent: CrewAI Self-Optimization Crew finished. Raw result: {crew_result_str[:200]}...")
    except Exception as e:
        logger.error(f"Meta-Agent: Error during CrewAI kickoff: {e}", exc_info=True)
        return {"status": "failed", "reason": f"CrewAI Execution Error: {e}"}

    # 3. Parse CrewAI output and trigger actions (crucial for self-evolution)
    # This is a conceptual parsing. A real system would use more robust techniques
    # (e.g., LLM to parse structured output, or specific output formatting instructions in tasks)
    logger.info("Meta-Agent: Parsing CrewAI output for actionable recommendations.")

    # Extremely simplified parsing - for robust systems, use AIs to parse other AIs' outputs!
    if "Recommendation: Learning Agent should prioritize proactive research" in crew_result_str:
        topic_match = re.search(r"recommendation: Learning Agent should prioritize proactive research in '(.*?)'", crew_result_str)
        topic = topic_match.group(1) if topic_match else "unspecified_topic"
        logger.info(f"Meta-Agent: Detected recommendation for Learning Agent to research: '{topic}'")
        # Trigger the Learning Agent - but via its CrewAI endpoint for complex topics
        try:
            await agent_client.post(
                "learning_agent",
                "/trigger_crewai_knowledge_acquisition",
                {"user_id": "system", "topic": topic, "context_notes": "Identified by Meta-Agent for proactive research."}
            )
            logger.info("Meta-Agent: Triggered CrewAI knowledge acquisition in Learning Agent.")
        except Exception as e:
            logger.error(f"Meta-Agent: Failed to trigger Learning Agent for proactive research: {e}")

    # Add more parsing logic for other recommendations found in `crew_result_str`
    # E.g., for "Orchestration Agent should optimize Y routing rule"
    # This would involve calling an endpoint (e.g., /update_config on Orchestration Agent itself, or a Configuration Agent)

    return {"status": "completed", "crew_result": crew_result_str}

@app.post("/trigger_crewai_self_optimization", summary="Triggers a CrewAI process for self-optimization")
async def trigger_crewai_self_optimization_endpoint() -> Dict[str, Any]:
    """
    Endpoint to manually trigger the CrewAI self-optimization process.
    """
    result = await run_crewai_self_optimization_crew()
    return {"message": "CrewAI self-optimization initiated", "result": result}


# --- Periodic self-reflection task ---
async def periodic_self_reflection_task():
    """
    Periodically initiates the Meta-Agent's self-reflection cycle using CrewAI.
    """
    await asyncio.sleep(10) # Initial delay for startup
    while True:
        logger.info("Meta-Agent (Periodic): Initiating scheduled self-reflection cycle...")
        if not crew_llm_instance:
            logger.warning("Meta-Agent (Periodic): CrewAI LLM not initialized. Skipping advanced self-reflection.")
            await asyncio.sleep(60 * 60) # Try again in an hour if LLM is not ready
            continue

        try:
            await run_crewai_self_optimization_crew()
            logger.info("Meta-Agent (Periodic): Scheduled CrewAI Self-Optimization Crew completed.")
        except Exception as e:
            logger.error(f"Meta-Agent (Periodic): Error during scheduled self-reflection: {e}", exc_info=True)

        await asyncio.sleep(60 * 60 * 24) # Run once a day (for demo)