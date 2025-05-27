# zhero_common/clients.py
import httpx # For making HTTP requests between microservices
from zhero_common.config import AGENT_ENDPOINTS, TOOL_ENDPOINTS, logger
from fastapi import HTTPException
from typing import Dict, Any

class AgentClient:
    """
    A unified client for making asynchronous HTTP calls to other agents/tools.
    """
    def __init__(self, endpoint_map: Dict[str, str]):
        self.endpoint_map = endpoint_map

    async def post(self, service_name: str, path: str, json_data: Dict[str, Any]):
        """
        Sends a POST request to a specific agent's or tool's endpoint.
        """
        base_url = self.endpoint_map.get(service_name)
        if not base_url:
            logger.error(f"AgentClient: Unknown service name: {service_name}")
            raise HTTPException(status_code=500, detail=f"Internal error: Unknown service '{service_name}'")

        full_url = f"{base_url}{path}"
        try:
            logger.info(f"AgentClient: Calling {service_name} at {full_url}")
            async with httpx.AsyncClient(timeout=30.0) as client: # Increased timeout for potential LLM calls
                response = await client.post(full_url, json=json_data)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                logger.info(f"AgentClient: Received success response from {service_name}")
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"AgentClient: HTTP error calling {service_name} ({full_url}): {e.response.status_code} - {e.response.text}", exc_info=True)
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from {service_name}: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"AgentClient: Network error calling {service_name} ({full_url}): {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"Network error calling {service_name}: {e}")
        except Exception as e:
            logger.error(f"AgentClient: Unexpected error calling {service_name} ({full_url}): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected error calling {service_name}: {e}")

# Global clients for inter-agent communication
agent_client = AgentClient(AGENT_ENDPOINTS)
tool_client = AgentClient(TOOL_ENDPOINTS)

# --- Supabase Mock (for demonstration where true Supabase client is not initialized) ---
# In a real environment, you'd use the official Supabase Python client (supabase-py).
# This mock prevents errors in demonstration code where we don't set up the full client.
class MockSupabaseClient:
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.data_store = {} # Simulating database tables in memory (resets on restart)
        logger.info(f"Initialized MockSupabaseClient for {url}. DATA WILL NOT PERSIST.")

    def from_(self, table_name: str):
        self._table_name = table_name
        self.data_store.setdefault(table_name, [])
        return self # Return self for chaining methods

    def select(self, columns: str = "*"):
        self._last_op = "select"
        self._columns = columns
        self._filters = []
        return self

    def insert(self, data: Dict):
        self._last_op = "insert"
        self._insert_data = data
        return self

    def update(self, data: Dict):
        self._last_op = "update"
        self._update_data = data
        self._filters = []
        return self

    def eq(self, column: str, value: Any):
        self._filters.append(("eq", column, value))
        return self

    def filter_data(self, data):
        filtered = list(data)
        for op, col, val in self._filters:
            if op == "eq":
                filtered = [row for row in filtered if row.get(col) == val]
        return filtered

    async def execute(self):
        # Simulate async DB operation
        table_data = self.data_store[self._table_name]
        if self._last_op == "insert":
            new_item = self._insert_data.copy()
            if 'id' not in new_item or new_item['id'] is None:
                # Generate a simple mock ID for tracking
                new_item['id'] = str(len(table_data) + 1) # Simple ID generator
            if 'timestamp' not in new_item:
                new_item['timestamp'] = datetime.datetime.utcnow().isoformat()
            table_data.append(new_item)
            logger.info(f"Mock Supabase: Inserted into '{self._table_name}': {new_item.get('id')}")
            return {"data": [new_item], "count": 1, "error": None}
        elif self._last_op == "select":
            results = self.filter_data(table_data)
            logger.info(f"Mock Supabase: Selected from '{self._table_name}', {len(results)} results.")
            return {"data": results, "count": len(results), "error": None}
        elif self._last_op == "update":
            updated_count = 0
            for i, row in enumerate(table_data):
                if all(row.get(col) == val for op, col, val in self._filters if op == 'eq'):
                    table_data[i].update(self._update_data)
                    updated_count += 1
            logger.info(f"Mock Supabase: Updated {updated_count} items in '{self._table_name}'.")
            return {"data": [], "count": updated_count, "error": None}
        return {"data": [], "error": "Invalid mock operation"}

# You would typically instantiate the real Supabase client like this:
# from supabase import create_client, Client
# supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
# For this demo, we'll use the mock:
supabase = MockSupabaseClient(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])