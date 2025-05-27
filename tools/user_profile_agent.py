# user_profile_agent.py
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import datetime
from zhero_common.config import logger # Assuming logger is defined in config
from zhero_common.models import UserProfile
from zhero_common.clients import supabase # Use the imported Supabase client

app = FastAPI(title="User Profile Agent")

# Initial Default Racks for new users (as described in the report)
DEFAULT_RACKS = [
    {"name": "Technology", "description": "Cutting-edge advancements and digital trends."},
    {"name": "Health & Wellness", "description": "Physical, mental, and emotional well-being."},
    {"name": "Hobbies", "description": "Personal interests, crafts, and leisure activities."},
    {"name": "Professional Development", "description": "Skills, career growth, and industry insights."},
    {"name": "General Knowledge", "description": "Facts, historical events, and broad concepts."}
]

@app.post("/initialize_profile", response_model=UserProfile, summary="Initializes a new user profile with default racks")
async def initialize_profile(user_id: str):
    """
    Initializes a new user profile in Supabase upon registration.
    This would typically be triggered by Supabase Edge Function on new user signup.
    """
    logger.info(f"User Profile Agent: Initializing profile for user: {user_id}")
    try:
        # Create a basic profile entry
        new_profile_data = {
            "user_id": user_id,
            "explicit_preferences": {},
            "inferred_interests": [],
            "initialized_racks": False,
            "last_active": datetime.datetime.utcnow().isoformat() # ISO format for DB
        }
        response = await supabase.from_("user_profiles").insert(new_profile_data).execute()
        if response["error"]:
            raise Exception(response["error"])

        # Create default racks for the user (in a 'user_racks' table)
        user_racks_to_insert = []
        for rack_data in DEFAULT_RACKS:
            user_racks_to_insert.append({
                "user_id": user_id,
                "rack_name": rack_data["name"],
                "description": rack_data["description"]
            })

        rack_insert_response = await supabase.from_("user_racks").insert(user_racks_to_insert).execute()
        if rack_insert_response["error"]:
            raise Exception(rack_insert_response["error"])

        # Update profile to mark racks as initialized
        update_response = await supabase.from_("user_profiles").update({"initialized_racks": True}).eq("user_id", user_id).execute()
        if update_response["error"]:
            raise Exception(update_response["error"])


        logger.info(f"User Profile Agent: Profile and default racks initialized for {user_id}.")
        return UserProfile(**new_profile_data, initialized_racks=True) # Return the created profile data
    except Exception as e:
        logger.error(f"User Profile Agent: Error initializing profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize user profile: {e}")

@app.post("/get_profile", response_model=Dict[str, Any], summary="Retrieves a user's profile")
async def get_profile(request_data: Dict[str, str]):
    """
    Retrieves the comprehensive profile for a given user.
    """
    user_id = request_data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required.")

    logger.info(f"User Profile Agent: Fetching profile for user: {user_id}")
    try:
        response = await supabase.from_("user_profiles").select("*").eq("user_id", user_id).execute()
        if response["error"]:
            raise Exception(response["error"])

        profile_data = response["data"]
        if not profile_data:
            logger.warning(f"User Profile Agent: Profile not found for {user_id}.")
            raise HTTPException(status_code=404, detail="User profile not found.")

        # In a real app, also fetch user_racks and user_books metadata here
        # For this demo, just return the base profile.
        logger.info(f"User Profile Agent: Profile retrieved for {user_id}.")
        return {"profile": profile_data[0]}
    except Exception as e:
        logger.error(f"User Profile Agent: Error retrieving profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user profile: {e}")

@app.post("/update_profile", response_model=Dict[str, Any], summary="Updates parts of a user's profile")
async def update_profile(user_id: str, updates: Dict[str, Any]):
    """
    Updates specific fields in a user's profile.
    """
    logger.info(f"User Profile Agent: Updating profile for user {user_id} with: {updates}")
    try:
        await supabase.from_("user_profiles").update(updates).eq("user_id", user_id).execute()
        logger.info(f"User Profile Agent: Profile updated for {user_id}.")
        return {"status": "success", "user_id": user_id, "updates": updates}
    except Exception as e:
        logger.error(f"User Profile Agent: Error updating profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update user profile: {e}")

@app.post("/update_preference", summary="Updates a specific user preference")
async def update_user_preference(request_data: Dict[str, Any]):
    """
    Receives and updates a single user preference in the user's profile.
    Expected data: {"user_id": "...", "preference_key": "...", "preference_value": "..."}
    """
    user_id = request_data.get("user_id")
    preference_key = request_data.get("preference_key")
    preference_value = request_data.get("preference_value")

    if not all([user_id, preference_key is not None, preference_value is not None]):
        raise HTTPException(status_code=400, detail="user_id, preference_key, and preference_value are all required.")

    logger.info(f"User Profile Agent: Updating preference '{preference_key}' for user {user_id} to '{preference_value}'")

    # In a real setup, you might want to validate preference_key and value allowed types
    updates_payload = {f"explicit_preferences->>{preference_key}": json.dumps(preference_value)} # Store as JSONB
    # Note: Supabase's PG client might need specific syntax for JSONB update
    # For now, we'll use a simpler representation or update the entire `explicit_preferences` dict

    try:
        # First, fetch current preferences to merge
        fetch_res = await supabase.from_("user_profiles").select("explicit_preferences").eq("user_id", user_id).execute()
        if fetch_res["error"] or not fetch_res["data"]:
            raise HTTPException(status_code=404, detail="User profile not found or error fetching existing preferences.")

        current_preferences = fetch_res["data"][0].get("explicit_preferences", {})
        if isinstance(current_preferences, str): # If it's stored as plain stringified JSON
             current_preferences = json.loads(current_preferences)

        current_preferences[preference_key] = preference_value

        update_data = {"explicit_preferences": json.dumps(current_preferences)} # Store back as stringified JSON

        response = await supabase.from_("user_profiles").update(update_data).eq("user_id", user_id).execute()
        if response["error"]:
            raise Exception(response["error"])

        logger.info(f"User Profile Agent: Preference '{preference_key}' updated successfully for {user_id}.")
        return {"status": "success", "user_id": user_id, "preference_key": preference_key, "preference_value": preference_value}
    except Exception as e:
        logger.error(f"User Profile Agent: Error updating preference for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update preference: {e}")
    
# To run this agent: uvicorn user_profile_agent:app --port 8001 --reload