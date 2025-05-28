# zhero_common/metrics.py

import asyncio
from enum import Enum
from typing import Dict, Any, Optional

from zhero_common.config import logger
from zhero_common.models import AgentPerformanceMetric # Assuming this model is defined in zhero_common/models.py
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher # Re-using existing Pub/Sub client
from zhero_common.exceptions import ZHeroException, ZHeroDependencyError # For error handling


class PerformanceMetricName(str, Enum):
    """Standardized names for performance metrics."""
    QUERY_PROCESSED = "query_processed"
    # General API/Tool interaction metrics (example of specific events)
    TOOL_CALL_SUCCESS = "tool_call_success"
    TOOL_CALL_FAILURE = "tool_call_failure"
    AGENT_API_CALL_SUCCESS = "agent_api_call_success"
    AGENT_API_CALL_FAILURE = "agent_api_call_failure"
    # Agent lifecycle metrics
    AGENT_STARTUP_SUCCESS = "agent_startup_success"
    AGENT_STARTUP_FAILURE = "agent_startup_failure"
    # User-related metrics
    CONVERSATION_TURN = "conversation_turn"
    USER_FEEDBACK_PROCESSED = "user_feedback_processed"
    # Knowledge management metrics
    KNOWLEDGE_INGESTED = "knowledge_ingested"
    KNOWLEDGE_SEARCH_PERFORMED = "knowledge_search_performed"
    KNOWLEDGE_GAP_CREATED = "knowledge_gap_created"
    KNOWLEDGE_GAP_HANDLED = "knowledge_gap_handled"
    # Specific agent metrics (general LLM calls)
    GEMINI_GENERATION_SUCCESS = "gemini_generation_success"
    GEMINI_GENERATION_FAILURE = "gemini_generation_failure"
    SEARCH_ENGINE_CALL_SUCCESS = "search_engine_call_success"
    SEARCH_ENGINE_CALL_FAILURE = "search_engine_call_failure"
    # Add more as needed for specific events or aggregated data
    # For example, to track latency (duration in milliseconds)
    API_CALL_LATENCY_MS = "api_call_latency_ms"


async def log_performance_metric(
    agent_name: str,
    metric_name: PerformanceMetricName,
    value: float = 1.0, # Default value for event-based metrics (e.g., 1 for success)
    user_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    fire_and_forget: bool = True # If True, publishes asynchronously without awaiting its result
) -> Optional[str]:
    """
    Logs a performance metric to the Pub/Sub 'performance_metrics' topic.

    This function abstracts away the details of the Pub/Sub publisher and
    the AgentPerformanceMetric model, making it easier for individual agents
    to emit standardized metrics.

    Args:
        agent_name (str): The name of the agent or service emitting the metric
                          (e.g., "orchestration_agent", "knowledge_management_agent").
        metric_name (PerformanceMetricName): The standardized name of the metric
                                             (e.g., PerformanceMetricName.QUERY_PROCESSED).
        value (float): The value of the metric (e.g., 1.0 for an event, a duration for latency).
        user_id (Optional[str]): The ID of the user related to the metric, if applicable.
        context (Optional[Dict[str, Any]]): Additional contextual information for the metric (e.g., {"query": "...", "response_length": 123}).
        fire_and_forget (bool): If True, publishes asynchronously without awaiting its result.
                                This is generally preferred for metrics to avoid blocking main logic.
                                If False, awaits the Pub/Sub publish completion and returns message ID.

    Returns:
        Optional[str]: The message ID if published successfully and not fire_and_forget, else None.

    Raises:
        ZHeroDependencyError: If Pub/Sub publisher is not initialized or publishing fails
                              (only if `fire_and_forget` is False). This means the caller needs to handle it.
    """
    # Defensive check: ensure Pub/Sub publisher is initialized.
    # This check/initialization is defensive; ideally, it should happen on agent startup
    # and the pubsub_publisher should be globally available.
    if pubsub_publisher is None:
        logger.warning(
            f"Metrics: Pub/Sub publisher not yet initialized. Attempting to initialize for metric '{metric_name.value}'."
        )
        try:
            # Attempt to initialize. If it fails, initialize_pubsub_publisher will log and raise.
            await initialize_pubsub_publisher()
            # If after initialization, it's still None, something went wrong.
            if pubsub_publisher is None:
                error_msg = "Metrics: Pub/Sub publisher could not be initialized after attempt. Cannot log metric."
                logger.error(error_msg, exc_info=True)
                if not fire_and_forget: # Only raise if not "fire-and-forget"
                    raise ZHeroDependencyError(
                        agent_name="Metrics",
                        dependency="Pub/Sub",
                        message=error_msg,
                        status_code=500
                    )
                return None # Return None if fire_and_forget or error not re-raised
        except ZHeroDependencyError as e:
            logger.error(
                f"Metrics: Failed to initialize Pub/Sub for metric logging due to ZHeroDependencyError: {e.message}",
                exc_info=True
            )
            if not fire_and_forget: # Only raise if not "fire-and-forget"
                raise # Re-raise the original ZHeroDependencyError
            return None # Return None if fire_and_forget
        except Exception as e:
            logger.error(
                f"Metrics: Failed to initialize Pub/Sub for metric logging due to unexpected error: {e}",
                exc_info=True
            )
            if not fire_and_forget: # Only raise if not "fire-and-forget"
                raise ZHeroDependencyError( # Wrap unexpected errors in ZHeroDependencyError
                    agent_name="Metrics",
                    dependency="Pub/Sub",
                    message=f"Failed to initialize Pub/Sub for metric logging: {e}",
                    status_code=500,
                    original_error=str(e)
                )
            return None # Return None if fire_and_forget

    # Construct the AgentPerformanceMetric Pydantic model
    metric_data = AgentPerformanceMetric(
        agent_name=agent_name,
        metric_name=metric_name.value,  # Use .value for the string representation from Enum
        value=value,
        user_id=user_id,
        context=context or {}  # Ensure context is always a dictionary
    ).model_dump(exclude_unset=True) # exclude_unset=True removes fields not explicitly set

    try:
        if fire_and_forget:
            # Publish asynchronously without waiting for the result. Use asyncio.create_task for true fire-and-forget.
            asyncio.create_task(pubsub_publisher.publish_message(
                "performance_metrics", # This is the topic key defined in zhero_common/config.py
                metric_data
            ))
            logger.debug(f"Metrics: Queued performance metric '{metric_name.value}' for '{agent_name}' (fire-and-forget).")
            return None
        else:
            # Publish and wait for the result - typically used only when the caller absolutely needs the message_id or needs to be sure of delivery
            message_id = await pubsub_publisher.publish_message(
                "performance_metrics",
                metric_data
            )
            logger.info(f"Metrics: Published performance metric '{metric_name.value}' for '{agent_name}' with ID: {message_id}.")
            return message_id
    except ZHeroException as e:
        logger.error(
            f"Metrics: Failed to publish metric '{metric_name.value}' for '{agent_name}' (ZHeroException): {e.message}",
            exc_info=True
        )
        if not fire_and_forget:
            raise # Re-raise the caught ZHeroException
    except Exception as e:
        logger.error(
            f"Metrics: Failed to publish metric '{metric_name.value}' for '{agent_name}' (unexpected error): {e}",
            exc_info=True
        )
        if not fire_and_forget:
            raise ZHeroDependencyError( # Wrap unexpected errors in ZHeroDependencyError
                agent_name="Metrics",
                dependency="Pub/Sub",
                message=f"Failed to publish metric: {e}",
                status_code=500,
                original_error=str(e)
            )
    return None # Catches any other exceptions or situations where fire_and_forget is True but publish_message still completes "successfully" (e.g. without being awaited)
