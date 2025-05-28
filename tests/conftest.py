# tests/conftest.py
import sys
import os
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure agent_client and supabase_mock are initialized only once for tests
# This prevents re-initialization warnings or issues if multiple tests import agents.
from zhero_common.clients import agent_client, supabase
# You might also want to mock other startup tasks like pubsub_publisher for tests
# from zhero_common.pubsub_client import pubsub_publisher


# You can also use fixtures here if needed across multiple test files
@pytest.fixture(autouse=True)
def clean_supabase_mock():
    """Fixture to clear Supabase mock data before each test."""
    # Ensure the mock is the one used by zhero_common.clients
    supabase.data_store = {}
    yield