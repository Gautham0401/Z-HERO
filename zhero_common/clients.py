# zhero_common/clients.py
import httpx
from zhero_common.config import AGENT_ENDPOINTS, TOOL_ENDPOINTS, logger
from fastapi import HTTPException
from typing import Dict, Any, Optional # Added Optional
from tenacity import (
    retry, wait_exponential, stop_after_attempt, retry_if_exception_type,
    CircuitBreakerError, before_sleep_log
)
import logging
import datetime # For MockSupabaseClient timestamp
import os # For MockSupabaseClient init

from zhero_common.exceptions import (
    ZHeroException, ZHeroDependencyError, ZHeroAgentError
)

tenacity_logger = logging.getLogger('tenacity')
tenacity_logger.setLevel(logging.INFO)

class AgentClient:
    """
    A unified client for making asynchronous HTTP calls to other agents/tools.
    """
    def __init__(self, endpoint_map: Dict[str, str]):
        self.endpoint_map = endpoint_map
        self._circuit_breakers: Dict[str, bool] = {}

    def _is_5xx_error(response: httpx.Response) -> bool:
        return 500 <= response.status_code < 600

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=(
            retry_if_exception_type(httpx.RequestError) |
            retry_if_exception_type(httpx.HTTPStatusError)
        ),
        before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
        reraise=True
    )
    async def _post_with_retry_logic(self, service_name: str, full_url: str, json_data: Dict[str, Any], request_id: Optional[str]):
        headers = {"X-Request-ID": request_id} if request_id else None # NEW: Add X-Request-ID header
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(full_url, json=json_data, headers=headers) # NEW: Pass headers
                if 400 <= response.status_code < 500:
                    tenacity_logger.error(f"[Request-ID: {request_id}] AgentClient: Non-retryable 4xx error from {service_name} ({full_url}): {response.status_code} - {response.text}")
                    raise httpx.HTTPStatusError(
                        f"Non-retryable client error: {response.text}",
                        request=response.request,
                        response=response
                    )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            tenacity_logger.error(f"[Request-ID: {request_id}] AgentClient: HTTPStatusError from {service_name} ({full_url}): {e.response.status_code} - {e.response.text}", exc_info=True)
            raise ZHeroDependencyError(
                agent_name="AgentClient",
                dependency=service_name,
                message=f"HTTP Error {e.response.status_code}",
                status_code=e.response.status_code,
                original_error=e.response.text,
                details={"request_id": request_id} # NEW: Add request_id to error details
            )
        except httpx.RequestError as e:
            tenacity_logger.error(f"[Request-ID: {request_id}] AgentClient: RequestError (network/timeout) from {service_name} ({full_url}): {e}", exc_info=True)
            raise ZHeroDependencyError(
                agent_name="AgentClient",
                dependency=service_name,
                message=f"Network/Connection error: {e}",
                status_code=503,
                original_error=str(e),
                details={"request_id": request_id} # NEW: Add request_id to error details
            )

    async def post(self, service_name: str, path: str, json_data: Dict[str, Any], request_id: Optional[str] = None): # NEW param: request_id
        base_url = self.endpoint_map.get(service_name)
        if not base_url:
            logger.error(f"[Request-ID: {request_id}] AgentClient: Unknown service name: {service_name}")
            raise ZHeroAgentError("AgentClient", f"Unknown service '{service_name}' in endpoint map.", 500, details={"request_id": request_id}) # NEW: Add request_id

        full_url = f"{base_url}{path}"

        if self._circuit_breakers.get(service_name):
            logger.warning(f"[Request-ID: {request_id}] AgentClient: Circuit breaker is OPEN for {service_name}. Failing fast.")
            raise ZHeroDependencyError(
                agent_name="AgentClient",
                dependency=service_name,
                message=f"Circuit breaker is open for {service_name}. Try again later.",
                status_code=503,
                details={"request_id": request_id} # NEW: Add request_id
            )

        try:
            result = await self._post_with_retry_logic(service_name, full_url, json_data, request_id) # NEW: Pass request_id
            if self._circuit_breakers.get(service_name):
                logger.info(f"[Request-ID: {request_id}] AgentClient: Circuit breaker for {service_name} closed due to success.")
                self._circuit_breakers[service_name] = False
            return result
        except ZHeroDependencyError as e:
            if e.status_code >= 500:
                self._circuit_breakers[service_name] = True
                logger.error(f"[Request-ID: {request_id}] AgentClient: Circuit breaker OPENED for {service_name} due to repeated failures.")
            raise
        except Exception as e:
            logger.error(f"[Request-ID: {request_id}] AgentClient: Unexpected non-ZHero exception calling {service_name} ({full_url}): {e}", exc_info=True)
            raise ZHeroException(f"An unexpected error occurred during call to {service_name}", 500, str(e), details={"request_id": request_id}) # NEW: Add request_id

agent_client = AgentClient(AGENT_ENDPOINTS)
tool_client = AgentClient(TOOL_ENDPOINTS)

class MockSupabaseClient:
    def __init__(self, url, key):
        logger.info(f"Initialized MockSupabaseClient: {url}")
        self.url = url
        self.key = key
        self.data_store = {}

    def from_(self, table_name: str):
        self._table_name = table_name
        self.data_store.setdefault(table_name, [])
        return self

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
        table_data = self.data_store[self._table_name]
        if self._last_op == "insert":
            new_item = self._insert_data.copy()
            if 'id' not in new_item or new_item['id'] is None:
                new_item['id'] = str(len(table_data) + 1)
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

supabase = MockSupabaseClient(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])