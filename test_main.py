import pytest
from fastapi.testclient import TestClient
from main import app, EchoAgent, runner, supabase
from unittest.mock import Mock, patch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = TestClient(app)

async def mock_run_async(user_id, session_id, new_message):
    mock_event = Mock()
    mock_event.is_final_response.return_value = True
    mock_event.content = Mock()
    mock_part = Mock()
    mock_part.text = f"EchoAgent acknowledges: {new_message.parts[0].text}"
    mock_event.content.parts = [mock_part]
    logger.debug(f"Mock event yielded: is_final={mock_event.is_final_response()}, parts={[p.text for p in mock_event.content.parts]}")
    yield mock_event

@pytest.mark.asyncio
async def test_echo_endpoint(mocker):
    # Mock Gemini API response
    mocker.patch.object(runner, 'run_async', side_effect=mock_run_async)

    # Mock Supabase insert (synchronous)
    mock_builder = Mock()
    mock_builder.execute = Mock(return_value=Mock(data=[{}]))
    mock_table = Mock()
    mock_table.insert = Mock(return_value=mock_builder)
    mocker.patch('main.supabase.table', return_value=mock_table)

    # Mock Supabase auth
    mock_user = Mock()
    mock_user.user.id = "test_user"
    mocker.patch('main.supabase.auth.get_user', return_value=mock_user)

    response = client.post("/echo", json={"message": "Hello World", "user_id": "test_user", "session_id": "test_session"}, headers={"Authorization": "Bearer test_token"})
    logger.debug(f"test_echo_endpoint response: {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"response": "EchoAgent acknowledges: Hello World"}
    mock_table.insert.assert_called_once_with({
        "user_id": "test_user",
        "session_id": "test_session",
        "message": "Hello World",
        "response": "EchoAgent acknowledges: Hello World"
    })
    mock_builder.execute.assert_called_once()

@pytest.mark.asyncio
async def test_echo_invalid_input():
    response = client.post("/echo", json={"wrong_field": "Hello"}, headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 422
    assert "detail" in response.json()

@pytest.mark.asyncio
async def test_echo_empty_message(mocker):
    # Mock Gemini API response
    mocker.patch.object(runner, 'run_async', side_effect=mock_run_async)

    # Mock Supabase insert (synchronous)
    mock_builder = Mock()
    mock_builder.execute = Mock(return_value=Mock(data=[{}]))
    mock_table = Mock()
    mock_table.insert = Mock(return_value=mock_builder)
    mocker.patch('main.supabase.table', return_value=mock_table)

    # Mock Supabase auth
    mock_user = Mock()
    mock_user.user.id = "test_user"
    mocker.patch('main.supabase.auth.get_user', return_value=mock_user)

    response = client.post("/echo", json={"message": "", "user_id": "test_user", "session_id": "test_session"}, headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 200
    assert "EchoAgent acknowledges:" in response.json()["response"]
    mock_table.insert.assert_called_once()
    mock_builder.execute.assert_called_once()

@pytest.mark.asyncio
async def test_debug_endpoint():
    response = client.post("/echo/debug", json={"message": "Debug Test"})
    assert response.status_code == 200
    assert response.json() == {"received": {"message": "Debug Test"}}

@pytest.mark.asyncio
async def test_supabase_storage(mocker):
    # Mock Gemini API response
    mocker.patch.object(runner, 'run_async', side_effect=mock_run_async)

    # Mock Supabase insert (synchronous)
    mock_builder = Mock()
    mock_builder.execute = Mock(return_value=Mock(data=[{}]))
    mock_table = Mock()
    mock_table.insert = Mock(return_value=mock_builder)
    mocker.patch('main.supabase.table', return_value=mock_table)

    # Mock Supabase auth
    mock_user = Mock()
    mock_user.user.id = "test_user"
    mocker.patch('main.supabase.auth.get_user', return_value=mock_user)

    response = client.post("/echo", json={"message": "Storage Test", "user_id": "test_user", "session_id": "test_session"}, headers={"Authorization": "Bearer test_token"})
    logger.debug(f"test_supabase_storage response: {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"response": "EchoAgent acknowledges: Storage Test"}
    mock_table.insert.assert_called_once_with({
        "user_id": "test_user",
        "session_id": "test_session",
        "message": "Storage Test",
        "response": "EchoAgent acknowledges: Storage Test"
    })
    mock_builder.execute.assert_called_once()

@pytest.mark.asyncio
async def test_search_endpoint(mocker):
    # Mock aiohttp for Bing API
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = Mock(return_value={
        "webPages": {
            "value": [{"snippet": "This is a test result"}]
        }
    })
    mock_session = Mock()
    mock_session.get = Mock(return_value=mock_response)
    mocker.patch('aiohttp.ClientSession', return_value=mock_session)

    # Mock Supabase insert (synchronous)
    mock_builder = Mock()
    mock_builder.execute = Mock(return_value=Mock(data=[{}]))
    mock_table = Mock()
    mock_table.insert = Mock(return_value=mock_builder)
    mocker.patch('main.supabase.table', return_value=mock_table)

    # Mock Supabase auth
    mock_user = Mock()
    mock_user.user.id = "test_user"
    mocker.patch('main.supabase.auth.get_user', return_value=mock_user)

    response = client.post("/search", json={"message": "Test Search", "user_id": "test_user", "session_id": "test_session"}, headers={"Authorization": "Bearer test_token"})
    logger.debug(f"test_search_endpoint response: {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"response": "This is a test result"}
    mock_table.insert.assert_called_once_with({
        "user_id": "test_user",
        "session_id": "test_session",
        "message": "Test Search",
        "response": "This is a test result"
    })
    mock_builder.execute.assert_called_once()