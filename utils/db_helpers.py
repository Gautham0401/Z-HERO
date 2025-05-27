# utils/db_helpers.py
import logging
import asyncio
from typing import Dict, Any, Optional
from supabase import Client # type: ignore
from postgrest.exceptions import APIError as PostgrestAPIError

logger = logging.getLogger(__name__)

async def get_supabase_client(external_supabase_client: Optional[Client] = None) -> Client:
    """
    Returns a Supabase client instance. If an external client is provided, use it.
    Otherwise, re-initialize using environment variables. This ensures tools can work
    independently if needed, or use a shared client from main.py.
    """
    if external_supabase_client:
        return external_supabase_client
    else:
        # Avoid circular import by importing here if needed for independent use
        from supabase import create_client
        import os
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            logger.critical("Supabase URL or Key not set in environment variables during independent client creation.")
            raise ValueError("Supabase URL or Key not set in environment variables.")
        return create_client(supabase_url, supabase_key)

async def check_and_create_user(supabase_client: Client, user_id: str) -> Dict[str, Any]:
    """
    Checks if a user exists in the 'users' table and creates them if not.
    Returns the user data or an error dict.
    NOTE: While this ensures a user record, rely on Supabase Row Level Security (RLS)
    to enforce data isolation for a specific user_id in all database operations.
    """
    try:
        user_check = await asyncio.to_thread(supabase_client.table("users").select("user_id").eq("user_id", user_id).limit(1).execute)
        if not user_check.data:
            # Only create an entry in the 'users' table if it doesn't exist
            # This is separate from Supabase Auth where the user is primarily registered.
            await asyncio.to_thread(supabase_client.table("users").insert({"user_id": user_id, "preferences": {}}).execute) # Initialize preferences as empty dict
            logger.info(f"Created new user entry for {user_id} in 'users' table.")
            return {"status": "created", "user_id": user_id}
        logger.debug(f"User {user_id} already exists in 'users' table.")
        return {"status": "exists", "user_id": user_id}
    except PostgrestAPIError as e:
        logger.error(f"Supabase error during user check/create for {user_id}: {str(e)}", exc_info=True)
        return {"error": f"Supabase user check/create failed: {str(e)}"}
    except Exception as e:
        logger.error(f"General error during user check/create for {user_id}: {str(e)}", exc_info=True)
        return {"error": f"Failed during user check/create: {str(e)}"}

async def insert_record(supabase_client: Client, table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Inserts a single record into a Supabase table."""
    try:
        response = await asyncio.to_thread(supabase_client.table(table_name).insert(data).execute)
        if response.data:
            logger.info(f"Successfully inserted record into {table_name}: Id={response.data[0].get('id')}")
            return {"status": "success", "data": response.data[0]}
        else:
            logger.error(f"Supabase insert returned no data for {table_name}, data: {data}")
            return {"error": "No data returned on insert."}
    except PostgrestAPIError as e:
        logger.error(f"Supabase error inserting into {table_name}, data: {data}: {str(e)}", exc_info=True)
        return {"error": f"Supabase insert failed: {str(e)}"}
    except Exception as e:
        logger.error(f"General error inserting into {table_name}, data: {data}: {str(e)}", exc_info=True)
        return {"error": f"Failed to insert record: {str(e)}"}

async def update_record(supabase_client: Client, table_name: str, filter_column: str, filter_value: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """Updates records in a Supabase table based on a filter."""
    try:
        response = await asyncio.to_thread(supabase_client.table(table_name).update(data).eq(filter_column, filter_value).execute)
        if response.data:
            logger.info(f"Successfully updated record(s) in {table_name} where {filter_column}={filter_value}")
            return {"status": "success", "data": response.data}
        else:
            logger.warning(f"No records updated in {table_name} where {filter_column}={filter_value}")
            return {"status": "not_found", "message": "No matching record to update."}
    except PostgrestAPIError as e:
        logger.error(f"Supabase error updating {table_name}, filter: {filter_column}={filter_value}, data: {data}: {str(e)}", exc_info=True)
        return {"error": f"Supabase update failed: {str(e)}"}
    except Exception as e:
        logger.error(f"General error updating {table_name}, filter: {filter_column}={filter_value}, data: {data}: {str(e)}", exc_info=True)
        return {"error": f"Failed to update record: {str(e)}"}

async def select_records(supabase_client: Client, table_name: str, filters: Dict[str, Any], columns: str = "*", limit: int = 1) -> Dict[str, Any]:
    """Selects records from a Supabase table based on filters."""
    try:
        query = supabase_client.table(table_name).select(columns)
        for col, val in filters.items():
            query = query.eq(col, val)
        
        response = await asyncio.to_thread(query.limit(limit).execute)
        if response.data:
            logger.debug(f"Found {len(response.data)} record(s) in {table_name} with filters {filters}")
            return {"status": "success", "data": response.data}
        else:
            logger.debug(f"No records found in {table_name} with filters {filters}")
            return {"status": "not_found", "message": "No matching records found."}
    except PostgrestAPIError as e:
        logger.error(f"Supabase error selecting from {table_name}, filters: {filters}: {str(e)}", exc_info=True)
        return {"error": f"Supabase select failed: {str(e)}"}
    except Exception as e:
        logger.error(f"General error selecting from {table_name}, filters: {filters}: {str(e)}", exc_info=True)
        return {"error": f"Failed to select records: {str(e)}"}