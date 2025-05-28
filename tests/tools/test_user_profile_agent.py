# tests/tools/test_user_profile_agent.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json # For JSON parsing errors
import datetime

# Import the FastAPI app instance from your user_profile_agent.py
# Make sure your project structure or PYTHONPATH allows this import
from tools.user_profile_agent import app

# Import models and common components
from zhero_common.models import UserProfile, UserProfileUpdateRequest, UserPreferenceUpdateRequest
from zhero_common.clients import supabase # The Supabase client mock is used here
from zhero_common.exceptions import ZHeroNotFoundError, ZHeroInvalidInputError, ZHeroSupabaseError

# Create a TestClient instance for your FastAPI app
client = TestClient(app=app)

# Fixture to clear Supabase mock data before each test
@pytest.fixture(autouse=True)
def clear_supabase_mock():
    supabase.data_store = {} # Clear the in-memory data store for each test
    yield

### Test Cases for initialize_profile ###
def test_initialize_profile_success(mocker):
    user_id = "test_user_init_1"
    # Mock Supabase responses for insert operations
    mocker.patch.object(supabase, 'from_', return_value=AsyncMock(
        insert=AsyncMock(return_value=AsyncMock(
            execute=AsyncMock(return_value={"data": [{}], "error": None})
        ))
    ))

    response = client.post(f"/initialize_profile?user_id={user_id}")
    assert response.status_code == 200
    assert response.json()["user_id"] == user_id
    assert response.json()["initialized_racks"] is True

    # Verify Supabase calls (simplified for mock, real check would be more granular)
    supabase.from_.assert_any_call("user_profiles")
    supabase.from_.assert_any_call("user_racks")

def test_initialize_profile_missing_user_id():
    response = client.post("/initialize_profile") # Missing user_id as query param
    assert response.status_code == 400
    assert response.json()["error_type"] == "ZHeroInvalidInputError"
    assert "User ID is required" in response.json()["message"]

def test_initialize_profile_supabase_error(mocker):
    user_id = "test_user_supabase_fail"
    mocker.patch.object(supabase, 'from_', return_value=AsyncMock(
        insert=AsyncMock(return_value=AsyncMock(
            execute=AsyncMock(return_value={"data": None, "error": {"message": "DB connection failed", "details": "timeout"}})
        ))
    ))

    response = client.post(f"/initialize_profile?user_id={user_id}")
    assert response.status_code == 500 # ZHeroSupabaseError maps to 500
    assert response.json()["error_type"] == "ZHeroSupabaseError"
    assert "Supabase operation failed" in response.json()["message"]

### Test Cases for get_profile ###
def test_get_profile_success(mocker):
    user_id = "existing_user"
    mock_profile_data = {
        "user_id": user_id, "email": "test@example.com", "explicit_preferences": {},
        "inferred_interests": [], "initialized_racks": True, "last_active": "2023-01-01T12:00:00"
    }
    mocker.patch.object(supabase, 'from_', return_value=AsyncMock(
        select=AsyncMock(return_value=AsyncMock(
            eq=AsyncMock(return_value=AsyncMock(
                execute=AsyncMock(return_value={"data": [mock_profile_data], "error": None})
            ))
        ))
    ))
    response = client.post("/get_profile", json={"user_id": user_id})
    assert response.status_code == 200
    assert response.json()["profile"]["user_id"] == user_id

def test_get_profile_not_found(mocker):
    user_id = "non_existent_user"
    mocker.patch.object(supabase, 'from_', return_value=AsyncMock(
        select=AsyncMock(return_value=AsyncMock(
            eq=AsyncMock(return_value=AsyncMock(
                execute=AsyncMock(return_value={"data": [], "error": None}) # Simulate not found
            ))
        ))
    ))
    response = client.post("/get_profile", json={"user_id": user_id})
    assert response.status_code == 404
    assert response.json()["error_type"] == "ZHeroNotFoundError"
    assert "User profile not found" in response.json()["message"]

def test_get_profile_missing_user_id_in_request_data():
    response = client.post("/get_profile", json={}) # Missing user_id
    assert response.status_code == 400
    assert response.json()["error_type"] == "ZHeroInvalidInputError"
    assert "User ID is required" in response.json()["message"]

### Test Cases for update_profile (using Pydantic model) ###
def test_update_profile_success(mocker):
    user_id = "update_user_1"
    updates = {"email": "new@example.com"}
    # Mock initial fetch for user existence:
    mocker.patch.object(supabase, 'from_', side_effect=[
        AsyncMock(select=AsyncMock(return_value=AsyncMock(eq=AsyncMock(
            execute=AsyncMock(return_value={"data": [{"user_id": user_id}], "error": None})
        )))),
        # Mock the update operation itself:
        AsyncMock(update=AsyncMock(return_value=AsyncMock(eq=AsyncMock(
            execute=AsyncMock(return_value={"data": [{"user_id": user_id, **updates}], "error": None})
        ))))
    ])
    response = client.post("/update_profile", json={"user_id": user_id, "updates": updates})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["user_id"] == user_id

def test_update_profile_user_not_found(mocker):
    user_id = "non_existent"
    updates = {"email": "new@example.com"}
    mocker.patch.object(supabase, 'from_', return_value=AsyncMock(
        select=AsyncMock(return_value=AsyncMock(eq=AsyncMock(
            execute=AsyncMock(return_value={"data": [], "error": None}) # User not found on fetch
        )))
    ))
    response = client.post("/update_profile", json={"user_id": user_id, "updates": updates})
    assert response.status_code == 404
    assert response.json()["error_type"] == "ZHeroNotFoundError"

def test_update_profile_missing_updates_payload():
    user_id = "some_user"
    response = client.post("/update_profile", json={"user_id": user_id, "updates": {}}) # Empty updates dict
    assert response.status_code == 400
    assert response.json()["error_type"] == "HTTPException" # Pydantic validation error for min_items=1

### Test Cases for update_preference (using Pydantic model) ###
def test_update_preference_success(mocker):
    user_id = "pref_user"
    pref_key = "learning_style"
    pref_value = "visual"
    initial_prefs = {"tone": "friendly"}
    
    # Mock initial fetch for user existence & current prefs
    mocker.patch.object(supabase, 'from_', side_effect=[
        AsyncMock(select=AsyncMock(return_value=AsyncMock(eq=AsyncMock(
            execute=AsyncMock(return_value={"data": [{"user_id": user_id, "explicit_preferences": json.dumps(initial_prefs)}], "error": None})
        )))),
        # Mock the update operation itself
        AsyncMock(update=AsyncMock(return_value=AsyncMock(eq=AsyncMock(
            execute=AsyncMock(return_value={"data": [{"user_id": user_id}], "error": None})
        ))))
    ])
    
    response = client.post("/update_preference", json={"user_id": user_id, "preference_key": pref_key, "preference_value": pref_value})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["preference_key"] == pref_key
    assert response.json()["preference_value"] == pref_value

def test_update_preference_user_not_found(mocker):
    user_id = "non_exist_pref"
    mocker.patch.object(supabase, 'from_', return_value=AsyncMock(
        select=AsyncMock(return_value=AsyncMock(eq=AsyncMock(
            execute=AsyncMock(return_value={"data": [], "error": None}) # User not found
        )))
    ))
    response = client.post("/update_preference", json={"user_id": user_id, "preference_key": "x", "preference_value": "y"})
    assert response.status_code == 404
    assert response.json()["error_type"] == "ZHeroNotFoundError"

def test_update_preference_missing_key_in_request_data():
    response = client.post("/update_preference", json={"user_id": "some_user", "preference_value": "y"})
    assert response.status_code == 400
    assert response.json()["error_type"] == "HTTPException" # Pydantic validation error for missing field