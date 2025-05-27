# zhero_common/crew_tools.py (As provided in previous step, ensuring consistency)

from crewai_tools import Tool
from typing import Dict, Any, List, Optional
import json
import datetime # Import for datetime objects in Supabase mock

# Import your existing common client for inter-service communication
from zhero_common.clients import agent_client, tool_client, supabase # Ensure 'supabase' is imported
from zhero_common.models import KnowledgeItem, AgentPerformanceMetric, KnowledgeGap # Import necessary models for type hinting
from zhero_common.config import logger # Import logger

# Define CrewAI-compatible tools that wrap your existing Z-HERO API calls

SemanticKnowledgeSearchTool = Tool(
    name="SemanticKnowledgeSearch",
    description="Searches the user's personalized internal knowledge base (Mind Palace) for semantically relevant information. Prioritize this for known topics or user-specific information. Arguments: query_text (str), user_id (str), top_k (int).",
    func=lambda query_text, user_id, top_k=5: agent_client.post(
        "knowledge_management_agent",
        "/search_knowledge_semantic",
        {"query_text": query_text, "user_id": user_id, "top_k": top_k}
    )
)

WebSearchTool = Tool(
    name="WebSearch",
    description="Performs a general web search to find external, public, or very up-to-date information. Arguments: query (str), user_id (str), num_results (int).",
    func=lambda query, user_id, num_results=3: tool_client.post(
        "web_search_tool",
        "/search",
        {"query": query, "user_id": user_id, "num_results": num_results}
    )
)

IngestKnowledgeItemTool = Tool(
    name="IngestKnowledgeItem",
    description="Stores new or updated information into the user's personalized internal knowledge base. Provide user_id, content, and optionally source_url, title, rack, book.",
    func=lambda user_id, content, source_url=None, title=None, rack=None, book=None: agent_client.post(
        "knowledge_management_agent",
        "/ingest_knowledge",
        KnowledgeItem(user_id=user_id, content=content, source_url=source_url, title=title, rack=rack, book=book).model_dump(exclude_unset=True)
    )
)

SummarizationTool = Tool(
    name="Summarization",
    description="Condenses long blocks of text into concise summaries. Arguments: text_content (str).",
    func=lambda text_content: tool_client.post(
        "summarization_tool",
        "/summarize",
        {"text_content": text_content}
    )
)

LogInternalKnowledgeGapTool = Tool(
    name="LogInternalKnowledgeGap",
    description="Informs the Meta-Agent about instances where Z-HERO could not find relevant information or confidently answer a query. Arguments: user_id (str), query_text (str), reason (str).",
    func=lambda user_id, query_text, reason: agent_client.post(
        "meta_agent",
        "/log_knowledge_gap_event",
        {"user_id": user_id, "query_text": query_text, "reason": reason, "trigger_type": "knowledge_gap_event"}
    )
)

UpdateUserPreferenceTool = Tool(
    name="UpdateUserPreference",
    description="Records or updates a specific user preference (e.g., learning style, favorite topics). Arguments: user_id (str), preference_key (str), preference_value (Any).",
    func=lambda user_id, preference_key, preference_value: agent_client.post(
        "user_profile_agent",
        "/update_preference",
        {"user_id": user_id, "preference_key": preference_key, "preference_value": preference_value}
    )
)

ProcessMultimodalContentTool = Tool(
    name="ProcessMultimodalContent",
    description="Forwards multimodal queries (text + image) to a specialized agent for deeper analysis and interpretation. Arguments: user_id (str), query_text (str), image_url (str).",
    func=lambda user_id, query_text, image_url: agent_client.post(
        "multimodal_agent",
        "/process_content",
        {"user_id": user_id, "query_text": query_text, "image_url": image_url}
    )
)

# NEW TOOL: ReadSystemLogsTool for Meta-Agent's Crew
# This tool directly queries Supabase (mock client for demo) to get system logs.
# In a real environment, you'd ensure proper authentication/authorization for this access.
class ReadSystemLogsTool(Tool):
    def __init__(self):
        super().__init__(
            name="ReadSystemLogs",
            description="Accesses Z-HERO's internal system logs and knowledge gap records from the Meta-Agent's data store (Supabase). This tool provides raw data for analysis. Arguments: log_type (str, e.g., 'performance_logs', 'knowledge_gaps'), limit (int, max 100), status (str, for knowledge_gaps, e.g., 'unaddressed').",
            func=self._read_logs # func should refer to an async method
        )

    # CrewAI tools require sync functions. To use an async method, we need a wrapper.
    # CrewAI allows specifying a sync func and running async code inside.
    # However, direct async method as func= is better.
    # For simplicity in Tool definition, we pass a callable that manages the async nature.
    # CrewAI's `Tool` can accept an async function directly if it's new enough (>= v0.2.0 for `crewai-tools`).
    # If not, you'd wrap it like this: `async def _read_logs_sync_wrapper(self, *args, **kwargs): return await self._read_logs(*args, **kwargs); return asyncio.run(self._read_logs_sync_wrapper(*args, **kwargs))`
    # For now, assuming CrewAI can handle awaitable objects for the `func` parameter.
    async def _read_logs(self, log_type: str, limit: int = 10, status: Optional[str] = None) -> List[Dict]:
        """
        Reads system logs or knowledge gaps from Supabase.
        """
        logger.info(f"ReadSystemLogsTool: Attempting to read {limit} {log_type} (status: {status}).")
        try:
            if log_type == "performance_logs":
                response = await supabase.from_("agent_performance_logs").select("*").limit(limit).execute()
            elif log_type == "knowledge_gaps":
                query_builder = supabase.from_("knowledge_gaps").select("*")
                if status:
                    query_builder = query_builder.eq("status", status)
                response = await query_builder.limit(limit).execute()
            else:
                return {"error": f"Invalid log_type: {log_type}. Choose 'performance_logs' or 'knowledge_gaps'."}

            if response["error"]:
                raise Exception(response["error"])

            logger.info(f"ReadSystemLogsTool: Retrieved {len(response['data'])} records for {log_type}.")
            cleaned_data = []
            for item in response['data']:
                cleaned_item = item.copy()
                for key, value in cleaned_item.items():
                    if isinstance(value, datetime.datetime):
                        cleaned_item[key] = value.isoformat()
                cleaned_data.append(cleaned_item)

            return cleaned_data # Return raw list of dicts for LLM to parse
        except Exception as e:
            logger.error(f"ReadSystemLogsTool: Error reading logs of type {log_type}: {e}", exc_info=True)
            return {"error": f"Failed to read logs: {str(e)}"}

# Instantiate the tool once
read_system_logs_tool_instance = ReadSystemLogsTool() # Global instance

ALL_ZHERO_CREW_TOOLS = [
    SemanticKnowledgeSearchTool, WebSearchTool, IngestKnowledgeItemTool,
    SummarizationTool, LogInternalKnowledgeGapTool, UpdateUserPreferenceTool,
    ProcessMultimodalContentTool, read_system_logs_tool_instance # Use the instantiated object
]