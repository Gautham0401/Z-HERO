# learning_agent.py (UPDATED & REFLECTED)
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Optional
import datetime
import asyncio
import re # For parsing CrewAI output (e.g., suggested rack/book)

# NEW CrewAI imports
from crewai import Agent, Task, Crew, Process
# For CrewAI agent LLM - Choose one based on your setup:
from langchain_google_vertexai import ChatVertexAI # Option 1: For Vertex AI (Gemini)

# Import zhero_common tools and models
from zhero_common.config import logger, os
from zhero_common.models import LearningTrigger, KnowledgeGap, SearchRequest, KnowledgeItem
from zhero_common.clients import agent_client, supabase # Use imported clients
# Import the specific CrewAI tools
from zhero_common.crew_tools import (
    WebSearchTool, SummarizationTool, IngestKnowledgeItemTool,
    SemanticKnowledgeSearchTool, LogInternalKnowledgeGapTool # FactChecker might use LogInternalKnowledgeGap
)


app = FastAPI(title="Learning Agent")

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
        logger.info("Learning Agent: Initialized CrewAI LLM (ChatVertexAI).")
    except Exception as e:
        logger.error(f"Learning Agent: Failed to initialize CrewAI LLM (ChatVertexAI): {e}", exc_info=True)
        crew_llm_instance = None # Ensure it's None if init fails

    # Optionally, schedule a periodic knowledge review task (can trigger CrewAI)
    # asyncio.create_task(periodic_knowledge_review_task())


# --- Existing /trigger_learning endpoint (for simpler triggers or fallback) ---
@app.post("/trigger_learning", summary="Triggers a specific learning process for the agent")
async def trigger_learning(trigger: LearningTrigger):
    """
    Receives a trigger to initiate a learning process,
    e.g., from Meta-Agent based on a detected knowledge gap.
    This endpoint can also decide to delegate to a CrewAI workflow if the task is complex.
    """
    logger.info(f"Learning Agent: Received learning trigger: '{trigger.trigger_type}' for user {trigger.user_id}")
    try:
        if trigger.trigger_type == "knowledge_gap":
            gap_details = trigger.details
            query = gap_details.get("query_text")
            reason = gap_details.get("reason")
            user_id = trigger.user_id

            logger.info(f"Learning Agent: Evaluating knowledge gap '{query}' for CrewAI delegation.")

            # Decision Logic: When to use CrewAI vs. simple research
            # For complex or deep research, delegate to CrewAI
            if "deep research" in query.lower() or "comprehensive overview" in query.lower() or "meta_analysis" in reason:
                logger.info(f"Learning Agent: Delegating complex knowledge gap '{query}' to CrewAI workflow.")
                crew_result = await run_crewai_knowledge_acquisition_crew(user_id, query, f"Initial reason for research: {reason}")
                return {"status": "delegated_to_crewai", "crew_result": crew_result.get("crew_result")}
            else:
                # Fallback to simpler direct research if not complex
                logger.info(f"Learning Agent: Performing simple research for gap '{query}'.")
                web_results_list = await agent_client.post(
                    "research_agent",
                    "/perform_research",
                    SearchRequest(query=query, user_id=user_id, num_results=5).model_dump(exclude_unset=True)
                )

                if web_results_list:
                    logger.info(f"Learning Agent: Simple research found {len(web_results_list)} results. Initiating direct ingestion.")
                    ingested_count = 0
                    for result in web_results_list:
                        try:
                            # Basic ingestion of snippet
                            await agent_client.post(
                                "knowledge_management_agent",
                                "/ingest_knowledge",
                                KnowledgeItem(
                                    user_id=user_id,
                                    content=result["snippet"] if isinstance(result, dict) else result.snippet, # Handle dict or Pydantic
                                    source_url=result["link"] if isinstance(result, dict) else result.link,
                                    title=result["title"] if isinstance(result, dict) else result.title,
                                    rack="Learned Gap Knowledge", # Specific rack
                                    book=result["title"] if isinstance(result, dict) else result.title,
                                ).model_dump(exclude_unset=True)
                            )
                            ingested_count += 1
                        except HTTPException as e:
                            logger.warning(f"Learning Agent: Failed to ingest knowledge from '{result.get('title', '')}': {e.detail}")

                    logger.info(f"Learning Agent: Ingested {ingested_count} new knowledge items for gap '{query}'.")
                else:
                    logger.warning(f"Learning Agent: Simple research found no new information for gap '{query}'.")

                await agent_client.post(
                    "meta_agent",
                    "/log_performance",
                    AgentPerformanceMetric(
                        agent_name="learning_agent",
                        user_id=user_id,
                        metric_name="knowledge_gap_filled",
                        value=1.0 if (ingested_count > 0 or (isinstance(crew_result_str, str) and "status': 'completed" in crew_result_str)) else 0.0,
                        context={"gap_query": query, "ingested_count": ingested_count if isinstance(crew_result_str, dict) else None, "delegated_to_crewai": "Yes" if isinstance(crew_result_str, dict) else "No"}
                    ).model_dump(exclude_unset=True)
                )
                return {"status": "processing_completed"}

        elif trigger.trigger_type == "user_feedback":
            feedback_details = trigger.details
            logger.info(f"Learning Agent: Analyzing user feedback: {feedback_details}")
            # Log performance of feedback processing
            await agent_client.post(
                "meta_agent",
                "/log_performance",
                AgentPerformanceMetric(
                    agent_name="learning_agent",
                    user_id=trigger.user_id,
                    metric_name="user_feedback_processed",
                    value=1.0,
                    context={"feedback_type": feedback_details.get("rating"), "details": feedback_details}
                ).model_dump(exclude_unset=True)
            )
            return {"status": "feedback_analyzed"}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown learning trigger type: {trigger.trigger_type}")

    except Exception as e:
        logger.error(f"Learning Agent: Error processing learning trigger: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Learning process failed: {e}")


# --- CrewAI Knowledge Acquisition Functions ---
@app.post("/trigger_crewai_knowledge_acquisition", summary="Triggers a CrewAI process for knowledge acquisition")
async def trigger_crewai_knowledge_acquisition_endpoint(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Endpoint to manually trigger the CrewAI knowledge acquisition process.
    """
    user_id = request_data.get("user_id")
    topic = request_data.get("topic")
    context_notes = request_data.get("context_notes", "")

    if not user_id or not topic:
        raise HTTPException(status_code=400, detail="Missing user_id or topic.")
    if not crew_llm_instance:
        raise HTTPException(status_code=500, detail="CrewAI LLM not initialized. Cannot run crew.")

    result = await run_crewai_knowledge_acquisition_crew(user_id, topic, context_notes)
    return {"message": "CrewAI knowledge acquisition initiated", "result": result}


async def run_crewai_knowledge_acquisition_crew(user_id: str, topic: str, context_notes: str) -> Dict[str, Any]:
    """
    Orchestrates a CrewAI crew to acquire, summarize, and ingest knowledge on a given topic for a user.
    """
    logger.info(f"Learning Agent: Initiating CrewAI Knowledge Acquisition Crew for topic: '{topic}' (User: {user_id})")

    if not crew_llm_instance:
        logger.error("CrewAI LLM not initialized. Cannot run knowledge acquisition crew.")
        return {"status": "failed", "reason": "CrewAI LLM not available."}

    # 1. Define the CrewAI Agents (internal roles)
    researcher = Agent(
        role='Knowledge Researcher',
        goal=f'Find comprehensive and accurate information about "{topic}" from external web sources. Focus on facts, definitions, history, and applications.',
        backstory='An expert information retriever, meticulous in finding relevant and up-to-date data.',
        verbose=True,
        allow_delegation=False,
        llm=crew_llm_instance,
        tools=[WebSearchTool, SemanticKnowledgeSearchTool] # Can use both web and internal search
    )

    summarizer = Agent(
        role='Content Summarizer',
        goal=f'Condense raw research data on "{topic}" into concise, coherent, and highly informative summaries suitable for Z-HERO’s knowledge base. Highlight key takeaways.',
        backstory='A master of distillation, able to extract the essence from vast amounts of information.',
        verbose=True,
        allow_delegation=False,
        llm=crew_llm_instance,
        tools=[SummarizationTool]
    )

    ingestor = Agent(
        role='Knowledge Ingestor',
        goal=f'Properly structure and store the summarized knowledge about "{topic}" into Z-HERO’s internal knowledge base, assigning appropriate racks and books, and ensuring it’s linked to user {user_id}.',
        backstory='The meticulous librarian of Z-HERO, ensuring all new knowledge is perfectly organized and accessible. Must ensure correct use of `IngestKnowledgeItemTool` parameters.',
        verbose=True,
        allow_delegation=False,
        llm=crew_llm_instance,
        tools=[IngestKnowledgeItemTool]
    )

    fact_checker = Agent(
        role='Fact Checker',
        goal=f'Verify the accuracy and reliability of key information points about "{topic}" before final ingestion. Cross-check against multiple reputable sources. If discrepancies found, provide detailed evidence and suggest a course of action (e.g., mark for human review or re-research).',
        backstory='A meticulous verifier, ensuring every piece of information is accurate and trustworthy. Does not compromise on truth.',
        verbose=True,
        allow_delegation=False,
        llm=crew_llm_instance,
        tools=[WebSearchTool] # Uses web search for cross-referencing
    )

    # 2. Define the Tasks for the Crew
    task_research = Task(
        description=f'Conduct in-depth web research on "{topic}". Identify at least 3-5 high-quality, reputable sources. Initial context: {context_notes} ',
        agent=researcher,
        expected_output=f'A list of URLs and snippets from reputable sources about "{topic}". The response should be a JSON string representing a list of dicts, each with "url", "title", "snippet".',
        output_file=f'research_{user_id}_{topic.replace(" ", "_")}.json' # Optional: save output to file
    )

    task_summarize = Task(
        description=f'Analyze the research results provided. Condense and summarize the key information about "{topic}" into clear, distinct paragraphs. Ensure all critical facts are included. Your output must be the FINAL summary text.',
        agent=summarizer,
        context=[task_research], # This task depends on the output of task_research
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
        context=[task_summarize, task_fact_check], # Depends on both summary and fact-check
        expected_output=f'Confirmation of successful knowledge ingestion with the new knowledge item ID (e.g., "KNOWLEDGE_ITEM_ID: [id]"), or a reason for failure to ingest (e.g., "FAILED_INGESTION: [reason]").'
    )

    # 3. Form the Crew and kick it off
    knowledge_acquisition_crew = Crew(
        agents=[researcher, summarizer, ingestor, fact_checker],
        tasks=[task_research, task_summarize, task_fact_check, task_ingest],
        verbose=2, # You can set it to 1 or 2 for different verbosity levels
        process=Process.sequential # Execute tasks in order
    )

    crew_result_str: str = ""
    try:
        crew_result_str = await asyncio.to_thread(knowledge_acquisition_crew.kickoff)
        logger.info(f"Learning Agent: CrewAI Knowledge Acquisition Crew finished. Raw result: {crew_result_str[:200]}...")
    except Exception as e:
        logger.error(f"Learning Agent: Error during CrewAI kickoff: {e}", exc_info=True)
        return {"status": "failed", "reason": f"CrewAI Execution Error: {e}"}

    # Post-processing the crew's final result
    ingestion_id_match = re.search(r"KNOWLEDGE_ITEM_ID:\s*\[(.*?)\]", crew_result_str)
    if ingestion_id_match:
        ingestion_id = ingestion_id_match.group(1)
        status = "completed"
        message = f"Knowledge successfully acquired and ingested with ID: {ingestion_id}"
    else:
        status = "completed_with_issues"
        message = f"Crew completed but ingestion might have failed. Review crew output: {crew_result_str}"

    return {"status": status, "message": message, "crew_result": crew_result_str}


# (Conceptual) Periodic Task to review knowledge gaps
# This task could now trigger the CrewAI knowledge acquisition for complex gaps
# Or simply trigger the basic gap-filling logic.
async def periodic_knowledge_review_task():
    """
    Simulates a periodic task where Learning Agent asks Meta-Agent for unaddressed knowledge gaps.
    In production, this would be triggered by Google Cloud Tasks.
    """
    while True:
        logger.info("Learning Agent (Periodic): Checking for outstanding knowledge gaps...")
        try:
            gaps_response = await agent_client.post(
                "meta_agent",
                "/get_knowledge_gaps",
                {"status": "unaddressed", "limit": 5} # Get top 5 unaddressed gaps
            )
            gaps: List[KnowledgeGap] = [KnowledgeGap(**g) for g in gaps_response.get("gaps", [])]

            if gaps:
                logger.info(f"Learning Agent (Periodic): Found {len(gaps)} unaddressed gaps. Evaluating for handling.")
                for gap in gaps:
                    if "complex" in gap.reason.lower() or "requires deep" in gap.reason.lower():
                        logger.info(f"Learning Agent (Periodic): Delegating complex gap '{gap.query_text}' to CrewAI.")
                        await run_crewai_knowledge_acquisition_crew(gap.user_id, gap.query_text, gap.reason)
                        await agent_client.post(
                            "meta_agent",
                            "/mark_gap_as_addressed",
                            {"gap_id": gap.id, "status": "in_progress"} # Mark as in progress while CrewAI runs
                        )
                    else:
                        logger.info(f"Learning Agent (Periodic): Handling simple gap '{gap.query_text}' directly.")
                        # Re-use existing simple research logic if no CrewAI is needed
                        # For direct implementation, you might need to copy/paste or refactor `trigger_learning` internal logic
                        # For now, this is a conceptual placeholder.
                        pass # Implement simple research here

            else:
                logger.info(f"Learning Agent (Periodic): No unaddressed knowledge gaps found.")

        except Exception as e:
            logger.error(f"Learning Agent (Periodic): Error during periodic knowledge review: {e}", exc_info=True)

        await asyncio.sleep(60 * 60) # Run every hour (for demo)