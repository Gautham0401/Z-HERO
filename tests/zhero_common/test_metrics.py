# tests/zhero_common/test_metrics.py
import pytest
import time
import asyncio
from unittest.mock import patch, AsyncMock # AsyncMock for httpx client

# Import the metrics_collector instance from your zhero_common module
# Ensure your PYTHONPATH is set up correctly, or run pytest from project root
from zhero_common.metrics import metrics_collector

@pytest.fixture(autouse=True)
def clear_metrics_before_each_test():
    """Fixture to clear the metrics_collector for each test."""
    metrics_collector._metrics = {} # Directly access internal state for testing
    yield

def test_metrics_collector_increment_counter():
    """Test that increment_counter correctly increments a counter."""
    metrics_collector.increment_counter("test_counter")
    metrics_collector.increment_counter("test_counter", value=2.0)
    metrics_collector.increment_counter("labeled_counter", labels={"agent": "orch"})
    metrics_collector.increment_counter("labeled_counter", labels={"agent": "orch"})

    metrics = metrics_collector.get_all_metrics()

    assert "test_counter" in metrics
    assert metrics["test_counter"]["type"] == "counter"
    assert metrics["test_counter"]["value"] == 3.0

    assert "labeled_counter_agent:orch" in metrics
    assert metrics["labeled_counter_agent:orch"]["type"] == "counter"
    assert metrics["labeled_counter_agent:orch"]["value"] == 2.0
    assert metrics["labeled_counter_agent:orch"]["labels"] == {"agent": "orch"}

def test_metrics_collector_observe_histogram():
    """Test that observe_histogram records values."""
    metrics_collector.observe_histogram("test_duration", 1.5)
    metrics_collector.observe_histogram("test_duration", 2.5)
    metrics_collector.observe_histogram("labeled_duration", 3.0, labels={"agent": "km"})

    metrics = metrics_collector.get_all_metrics()

    assert "test_duration" in metrics
    assert metrics["test_duration"]["type"] == "histogram"
    assert metrics["test_duration"]["values"] == [1.5, 2.5]

    assert "labeled_duration_agent:km" in metrics
    assert metrics["labeled_duration_agent:km"]["type"] == "histogram"
    assert metrics["labeled_duration_agent:km"]["values"] == [3.0]

@pytest.mark.asyncio # Marks the test as an async function for pytest-asyncio
async def test_metrics_collector_time_async_operation(mocker):
    """Test that time_async_operation correctly measures duration."""
    # Mock time.perf_counter for deterministic timing
    mock_perf_counter = AsyncMock(side_effect=[0.0, 1.0]) # First call returns 0.0, second 1.0
    mocker.patch('time.perf_counter', new=mock_perf_counter)

    async def dummy_async_op():
        await asyncio.sleep(0.01) # Simulate some async work
        return "done"

    async with metrics_collector.time_async_operation("dummy_op_time", labels={"type": "test_fn"}):
        result = await dummy_async_op()

    metrics = metrics_collector.get_all_metrics()

    assert "dummy_op_time_type:test_fn" in metrics
    assert metrics["dummy_op_time_type:test_fn"]["type"] == "histogram"
    assert metrics["dummy_op_time_type:test_fn"]["values"] == [1.0] # Should be end_time - start_time
    assert result == "done" # Ensure the wrapped operation still returns its result