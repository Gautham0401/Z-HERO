# zhero_common/pubsub_client.py
from google.cloud import pubsub_v1
import json
import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError # For Pub/Sub Await
from typing import Dict, Any, Optional

from zhero_common.config import PUBSUB_TOPICS, logger
from zhero_common.exceptions import ZHeroDependencyError
import os

class PubSubPublisher:
    def __init__(self, project_id: Optional[str] = None): # Corrected 'GCP_PROJECT_ID' to 'project_id' when calling super()
        self._project_id = project_id if project_id else os.environ.get("GCP_PROJECT_ID")
        if not self._project_id:
            raise ZHeroDependencyError(
                agent_name="PubSubPublisher",
                dependency="PubSub",
                message="GCP Project ID must be provided or set in environment variables to initialize Pub/Sub publisher.",
                status_code=500
            )
        self._publisher = pubsub_v1.PublisherClient()
        self._topic_paths: Dict[str, str] = {}
        logger.info("PubSubPublisher: Initialized PublisherClient.")

    def _get_topic_path(self, topic_name: str) -> str:
        if topic_name not in self._topic_paths:
            self._topic_paths[topic_name] = self._publisher.topic_path(self._project_id, topic_name)
        return self._topic_paths[topic_name]

    async def publish_message(self, topic_key: str, data: Dict[str, Any], timeout: float = 5.0) -> str:
        """
        Publish_message a dictionary message to a specified Pub/Sub topic.
        The topic_key corresponds to keys in PUBSUB_TOPICS.
        Returns the message ID on success.
        """
        if topic_key not in PUBSUB_TOPICS:
            raise ZHeroDependencyError(
                agent_name="PubSubPublisher",
                dependency="PubSub",
                message=f"Undefined Pub/Sub topic key: {topic_key}",
                status_code=500
            )

        topic_name = PUBSUB_TOPICS[topic_key]
        topic_path = self._get_topic_path(topic_name)
        
        # Data must be a bytestring
        data_json = json.dumps(data)
        data_bytes = data_json.encode("utf-8")

        future = self._publisher.publish(topic_path, data_bytes)

        try:
            message_id = await asyncio.to_thread(future.result, timeout=timeout)
            logger.info(f"PubSubPublisher: Published message to {topic_name} with ID: {message_id}")
            return message_id
        except FuturesTimeoutError:
            logger.error(f"PubSubPublisher: Publishing to {topic_name} timed out after {timeout} seconds.")
            raise ZHeroDependencyError(
                agent_name="PubSubPublisher",
                dependency="PubSub",
                message=f"Publishing to topic '{topic_name}' timed out.",
                status_code=503
            )
        except Exception as e:
            logger.error(f"PubSubPublisher: Failed to publish message to {topic_name}: {e}", exc_info=True)
            raise ZHeroDependencyError(
                agent_name="PubSubPublisher",
                dependency="PubSub",
                message=f"Failed to publish message: {e}",
                status_code=500,
                original_error=str(e)
            )

pubsub_publisher: Optional[PubSubPublisher] = None

async def initialize_pubsub_publisher():
    global pubsub_publisher
    if pubsub_publisher is None: # Only initialize once
        try:
            pubsub_publisher = PubSubPublisher()
            logger.info("PubSubPublisher: Global instance initialized.")
        except ZHeroDependencyError as e: # Catch our specific error here
            logger.critical(f"PubSubPublisher: Failed to initialize global instance due to dependency error: {e.message}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(f"PubSubPublisher: Failed to initialize global instance due to unexpected error: {e}", exc_info=True)
            raise