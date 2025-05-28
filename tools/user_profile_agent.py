# tools/user_profile_agent.py

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import datetime
import json
import asyncio # <--- NEW - Needed for async metric logging

# Import common utilities
from zhero_common.config import logger
from zhero_common.models import UserProfile, UserProfileUpdateRequest, UserPreferenceUpdateRequest # Updated Pydantic models
from zhero_common.clients import supabase
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroNotFoundError,
    ZHeroInvalidInputError, ZHeroSupabaseError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher
# NEW: Import metrics helper
from zhero_common.metrics import log_performance_metric, PerformanceMetricName # <--- NEW


app = FastAPI(title="User Profile Agent")

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


# Initial Default Racks for new users (as described in the report)
DEFAULT_RACKS = [
    {"name": "Technology", "description": "Cutting-edge advancements and digital trends."},
    {"name": "Health & Wellness", "description": "Physical, mental, and emotional well-being."},
    {"name": "Hobbies", "description": "Personal interests, crafts, and leisure activities."},
    {"name": "Professional Development", "description": "Skills, career growth, and industry insights."},
    {"name": "General Knowledge", "description": "Facts, historical events, and broad concepts."}
]

@app.on_event("startup")
async def startup_event():
    # Initialize Pub/Sub Publisher (this is crucial for metric logging!)
    await initialize_pubsub_publisher()
    try:
        # Simulate a quick check for Supabase connectivity
        # In a real app, you might try a simple query like fetching a dummy row.
        # Here, we'll just log startup success.
        logger.info("User Profile Agent: Startup checks complete.")
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.AGENT_STARTUP_SUCCESS
            )
        )
    except Exception as e:
        logger.error(f"User Profile Agent: Startup failed: {e}", exc_info=True)
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.AGENT_STARTUP_FAILURE,
                context={"error": str(e)}
            )
        )
        # Re-raise to prevent app startup if critical dependencies fail
        raise ZHeroDependencyError("UserProfileAgent", "Startup", f"Startup failed: {e}", original_error=e)


@app.post("/initialize_profile", response_model=UserProfile, summary="Initializes a new user profile with default racks")
async def initialize_profile(user_id: str): # Could use Pydantic model
    if not user_id:
        raise ZHeroInvalidInputError(message="User ID is required for initialization.", details={"field": "user_id"})

    logger.info(f"User Profile Agent: Initializing profile for user: {user_id}")
    start_time = datetime.datetime.now() # <--- NEW
    try:
        new_profile_data = {
            "user_id": user_id,
            "email": None, # Explicitly set Optional fields to None
            "explicit_preferences": {},
            "inferred_interests": [],
            "last_active": datetime.datetime.now(datetime.timezone.utc).isoformat(), # <--- UPDATED utcnow to now(utc)
            "initialized_racks": False
        }
        
        # Insert user profile
        supabase_call_start_1 = datetime.datetime.now() # <--- NEW
        response = await supabase.from_("user_profiles").insert(new_profile_data).execute()
        if response["error"]:
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="user_profile_agent",
                    metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                    value=1.0, user_id=user_id,
                    context={"tool_name": "supabase_insert_profile", "error": response["error"].get("message")}
                )
            )
            raise ZHeroSupabaseError(agent_name="UserProfileAgent", message=response["error"]["message"], original_error=response["error"]["details"])
        supabase_call_end_1 = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS,
                value=1.0, user_id=user_id,
                context={"tool_name": "supabase_insert_profile", "duration_ms": (supabase_call_end_1 - supabase_call_start_1).total_seconds() * 1000}
            )
        )


        user_racks_to_insert = []
        for rack_data in DEFAULT_RACKS:
            user_racks_to_insert.append({"user_id": user_id, "rack_name": rack_data["name"], "description": rack_data["description"]})

        # Insert default racks
        supabase_call_start_2 = datetime.datetime.now() # <--- NEW
        rack_insert_response = await supabase.from_("user_racks").insert(user_racks_to_insert).execute()
        if rack_insert_response["error"]:
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="user_profile_agent",
                    metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                    value=1.0, user_id=user_id,
                    context={"tool_name": "supabase_insert_racks", "error": rack_insert_response["error"].get("message")}
                )
            )
            raise ZHeroSupabaseError(agent_name="UserProfileAgent", message=rack_insert_response["error"]["message"], original_error=rack_insert_response["error"]["details"])
        supabase_call_end_2 = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS,
                value=1.0, user_id=user_id,
                context={"tool_name": "supabase_insert_racks", "duration_ms": (supabase_call_end_2 - supabase_call_start_2).total_seconds() * 1000}
            )
        )

        # Update profile to mark as initialized
        supabase_call_start_3 = datetime.datetime.now() # <--- NEW
        update_response = await supabase.from_("user_profiles").update({"initialized_racks": True}).eq("user_id", user_id).execute()
        if update_response["error"]:
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="user_profile_agent",
                    metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                    value=1.0, user_id=user_id,
                    context={"tool_name": "supabase_update_profile_init", "error": update_response["error"].get("message")}
                )
            )
            raise ZHeroSupabaseError(agent_name="UserProfileAgent", message=update_response["error"]["message"], original_error=update_response["error"]["details"])
        supabase_call_end_3 = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS,
                value=1.0, user_id=user_id,
                context={"tool_name": "supabase_update_profile_init", "duration_ms": (supabase_call_end_3 - supabase_call_start_3).total_seconds() * 1000}
            )
        )

        logger.info(f"User Profile Agent: Profile and default racks initialized for {user_id}.")
        
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=1.0, user_id=user_id,
                context={"endpoint": "/initialize_profile", "duration_ms": total_duration_ms}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.API_CALL_LATENCY_MS,
                value=total_duration_ms, user_id=user_id,
                context={"endpoint": "/initialize_profile"}
            )
        )

        return UserProfile(**new_profile_data, initialized_racks=True)
    except ZHeroException: raise
    except Exception as e:
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=0.0, user_id=user_id,
                context={"endpoint": "/initialize_profile", "error": str(e), "duration_ms": total_duration_ms}
            )
        )
        raise ZHeroAgentError("UserProfileAgent", f"Failed to initialize user profile for {user_id}", original_error=e)


@app.post("/get_profile", response_model=Dict[str, Any], summary="Retrieves a user's profile")
async def get_profile(request_data: Dict[str, str]): # Consider a Pydantic model for request_data if more complex
    user_id = request_data.get("user_id")
    if not user_id:
        raise ZHeroInvalidInputError(message="User ID is required.", details={"field": "user_id"})

    logger.info(f"User Profile Agent: Fetching profile for user: {user_id}")
    start_time = datetime.datetime.now() # <--- NEW
    try:
        supabase_call_start = datetime.datetime.now() # <--- NEW
        response = await supabase.from_("user_profiles").select("*").eq("user_id", user_id).execute()
        if response["error"]:
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="user_profile_agent",
                    metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                    value=1.0, user_id=user_id,
                    context={"tool_name": "supabase_select_profile", "error": response["error"].get("message")}
                )
            )
            raise ZHeroSupabaseError(agent_name="UserProfileAgent", message=response["error"]["message"], original_error=response["error"]["details"])
        
        supabase_call_end = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS,
                value=1.0, user_id=user_id,
                context={"tool_name": "supabase_select_profile", "duration_ms": (supabase_call_end - supabase_call_start).total_seconds() * 1000}
            )
        )

        profile_data = response["data"]
        if not profile_data:
            raise ZHeroNotFoundError(resource_name="User profile", identifier=user_id)

        logger.info(f"User Profile Agent: Profile retrieved for {user_id}.")
        
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=1.0, user_id=user_id,
                context={"endpoint": "/get_profile", "duration_ms": total_duration_ms}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.API_CALL_LATENCY_MS,
                value=total_duration_ms, user_id=user_id,
                context={"endpoint": "/get_profile"}
            )
        )
        return {"profile": profile_data[0]}
    except ZHeroException: raise
    except Exception as e:
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=0.0, user_id=user_id,
                context={"endpoint": "/get_profile", "error": str(e), "duration_ms": total_duration_ms}
            )
        )
        raise ZHeroAgentError("UserProfileAgent", f"Failed to retrieve profile for {user_id}", original_error=e)


@app.post("/update_profile", response_model=Dict[str, Any], summary="Updates parts of a user's profile")
async def update_profile(request: UserProfileUpdateRequest): # UPDATED to use Pydantic model
    logger.info(f"User Profile Agent: Updating profile for user {request.user_id} with: {request.updates}")
    start_time = datetime.datetime.now() # <--- NEW
    try:
        # Check if profile exists before updating
        supabase_call_start_1 = datetime.datetime.now() # <--- NEW
        fetch_res = await supabase.from_("user_profiles").select("user_id").eq("user_id", request.user_id).execute()
        if fetch_res["error"]:
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="user_profile_agent",
                    metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                    value=1.0, user_id=request.user_id,
                    context={"tool_name": "supabase_select_profile_exist", "error": fetch_res["error"].get("message")}
                )
            )
            raise ZHeroSupabaseError(agent_name="UserProfileAgent", message=fetch_res["error"]["message"], original_error=fetch_res["error"]["details"])
        supabase_call_end_1 = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS,
                value=1.0, user_id=request.user_id,
                context={"tool_name": "supabase_select_profile_exist", "duration_ms": (supabase_call_end_1 - supabase_call_start_1).total_seconds() * 1000}
            )
        )
        if not fetch_res["data"]:
            raise ZHeroNotFoundError(resource_name="User profile", identifier=request.user_id)

        # Perform the update
        supabase_call_start_2 = datetime.datetime.now() # <--- NEW
        response = await supabase.from_("user_profiles").update(request.updates.model_dump(exclude_unset=True)).eq("user_id", request.user_id).execute() # <--- UPDATED: Ensure pydantic model is dumped
        if response["error"]:
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="user_profile_agent",
                    metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                    value=1.0, user_id=request.user_id,
                    context={"tool_name": "supabase_update_profile", "error": response["error"].get("message")}
                )
            )
            raise ZHeroSupabaseError(agent_name="UserProfileAgent", message=response["error"]["message"], original_error=response["error"]["details"])
        supabase_call_end_2 = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS,
                value=1.0, user_id=request.user_id,
                context={"tool_name": "supabase_update_profile", "duration_ms": (supabase_call_end_2 - supabase_call_start_2).total_seconds() * 1000}
            )
        )

        logger.info(f"User Profile Agent: Profile updated for {request.user_id}.")
        
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=1.0, user_id=request.user_id,
                context={"endpoint": "/update_profile", "duration_ms": total_duration_ms}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.API_CALL_LATENCY_MS,
                value=total_duration_ms, user_id=request.user_id,
                context={"endpoint": "/update_profile"}
            )
        )

        return {"status": "success", "user_id": request.user_id, "updates": request.updates.model_dump(exclude_unset=True)}
    except ZHeroException: raise
    except Exception as e:
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=0.0, user_id=request.user_id,
                context={"endpoint": "/update_profile", "error": str(e), "duration_ms": total_duration_ms}
            )
        )
        raise ZHeroAgentError("UserProfileAgent", f"Failed to update profile for {request.user_id}", original_error=e)


@app.post("/update_preference", summary="Updates a specific user preference")
async def update_user_preference(request: UserPreferenceUpdateRequest): # UPDATED to use Pydantic model
    logger.info(f"User Profile Agent: Updating preference '{request.preference_key}' for user {request.user_id} to '{request.preference_value}'")
    start_time = datetime.datetime.now() # <--- NEW

    try:
        # Fetch current preferences
        supabase_call_start_1 = datetime.datetime.now() # <--- NEW
        fetch_res = await supabase.from_("user_profiles").select("explicit_preferences").eq("user_id", request.user_id).execute()
        if fetch_res["error"]:
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="user_profile_agent",
                    metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                    value=1.0, user_id=request.user_id,
                    context={"tool_name": "supabase_select_preferences_fetch", "error": fetch_res["error"].get("message")}
                )
            )
            raise ZHeroSupabaseError(agent_name="UserProfileAgent", message=fetch_res["error"]["message"], original_error=fetch_res["error"]["details"])
        supabase_call_end_1 = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS,
                value=1.0, user_id=request.user_id,
                context={"tool_name": "supabase_select_preferences_fetch", "duration_ms": (supabase_call_end_1 - supabase_call_start_1).total_seconds() * 1000}
            )
        )

        if not fetch_res["data"]:
            raise ZHeroNotFoundError(resource_name="User profile", identifier=request.user_id)

        current_preferences = fetch_res["data"][0].get("explicit_preferences", {})
        if isinstance(current_preferences, str):
            current_preferences = json.loads(current_preferences) # Ensure it's a dict if previously stored as string

        current_preferences[request.preference_key] = request.preference_value

        update_data = {"explicit_preferences": json.dumps(current_preferences)}

        # Update preferences
        supabase_call_start_2 = datetime.datetime.now() # <--- NEW
        response = await supabase.from_("user_profiles").update(update_data).eq("user_id", request.user_id).execute()
        if response["error"]:
            asyncio.create_task( # <--- NEW
                log_performance_metric(
                    agent_name="user_profile_agent",
                    metric_name=PerformanceMetricName.TOOL_CALL_FAILURE,
                    value=1.0, user_id=request.user_id,
                    context={"tool_name": "supabase_update_preferences", "error": response["error"].get("message")}
                )
            )
            raise ZHeroSupabaseError(agent_name="UserProfileAgent", message=response["error"]["message"], original_error=response["error"]["details"])
        
        supabase_call_end_2 = datetime.datetime.now() # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.TOOL_CALL_SUCCESS,
                value=1.0, user_id=request.user_id,
                context={"tool_name": "supabase_update_preferences", "duration_ms": (supabase_call_end_2 - supabase_call_start_2).total_seconds() * 1000}
            )
        )

        logger.info(f"User Profile Agent: Preference '{request.preference_key}' updated successfully for {request.user_id}.")
        
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=1.0, user_id=request.user_id,
                context={"endpoint": "/update_preference", "duration_ms": total_duration_ms}
            )
        )
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.API_CALL_LATENCY_MS,
                value=total_duration_ms, user_id=request.user_id,
                context={"endpoint": "/update_preference"}
            )
        )
        return {"status": "success", "user_id": request.user_id, "preference_key": request.preference_key,"preference_value": request.preference_value}
    except ZHeroException: raise
    except Exception as e:
        end_time = datetime.datetime.now() # <--- NEW
        total_duration_ms = (end_time - start_time).total_seconds() * 1000 # <--- NEW
        asyncio.create_task( # <--- NEW
            log_performance_metric(
                agent_name="user_profile_agent",
                metric_name=PerformanceMetricName.QUERY_PROCESSED,
                value=0.0, user_id=request.user_id,
                context={"endpoint": "/update_preference", "error": str(e), "duration_ms": total_duration_ms}
            )
        )
        raise ZHeroAgentError("UserProfileAgent", f"Failed to update preference for {request.user_id}", original_error=e)