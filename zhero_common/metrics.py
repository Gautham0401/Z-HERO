# zhero_common/metrics.py (NEW FILE)
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import asyncio

from zhero_common.config import logger # Import shared logger

class MetricsCollector:
    """
    A conceptual class for collecting application metrics.
    In a real system, this would push to Prometheus, Opentelemetry, GCP Cloud Monitoring, etc.
    """
    def __init__(self):
        # In-memory storage for demonstration. Not production-ready.
        self._metrics: Dict[str, Dict[str, Any]] = {}
        logger.info("MetricsCollector: Initialized conceptual metrics collector.")

    def _get_metric_key(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Generates a unique key for storing metrics with labels."""
        key = metric_name
        if labels:
            sorted_labels = sorted(labels.items())
            key += "_" + "_".join([f"{k}:{v}" for k, v in sorted_labels])
        return key

    def increment_counter(self, metric_name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        key = self._get_metric_key(metric_name, labels)
        self._metrics.setdefault(key, {"type": "counter", "value": 0.0, "labels": labels})
        self._metrics[key]["value"] += value
        logger.debug(f"MetricsCollector: Counter '{key}' incremented by {value} to {self._metrics[key]['value']}")

    def observe_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        key = self._get_metric_key(metric_name, labels)
        self._metrics.setdefault(key, {"type": "histogram", "values": [], "labels": labels})
        self._metrics[key]["values"].append(value)
        logger.debug(f"MetricsCollector: Histogram '{key}' observed value {value}")

    @asynccontextmanager
    async def time_async_operation(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to measure asynchronous operation duration."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.observe_histogram(metric_name, duration, labels)
            logger.debug(f"MetricsCollector: Operation '{metric_name}' took {duration:.4f}s")

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Returns the current state of all collected metrics."""
        return self._metrics.copy()

# Global instance of the metrics collector
metrics_collector = MetricsCollector()