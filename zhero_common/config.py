# zhero_common/config.py
import os
import logging
from typing import Dict

# Configure basic logging for all agents
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variables (Replace with your actual values) ---
# For local development, consider using python-dotenv to load these from a .env file.
# In production, use Google Secret Manager or similar secure credential management.

# GCP Project and AI-related
os.environ["GCP_PROJECT_ID"] = os.getenv("GCP_PROJECT_ID", "your-zhero-gcp-project-id")
os.environ["VERTEX_AI_LOCATION"] = os.getenv("VERTEX_AI_LOCATION", "us-central1")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "path/to/your/gcp-credentials.json")

# Vertex AI Model IDs
os.environ["GEMINI_PRO_MODEL_ID"] = os.getenv("GEMINI_PRO_MODEL_ID", "gemini-pro")
os.environ["GEMINI_PRO_VISION_MODEL_ID"] = os.getenv("GEMINI_PRO_VISION_MODEL_ID", "gemini-pro-vision") # Added for multimodal agent
os.environ["VERTEX_AI_EMBEDDING_MODEL_ID"] = os.getenv("VERTEX_AI_EMBEDDING_MODEL_ID", "text-embedding-004")
os.environ["VERTEX_AI_NLP_MODEL_ID"] = os.getenv("VERTEX_AI_NLP_MODEL_ID", "builtin/legacy")

# Vertex AI Search (Vector Database)
os.environ["VERTEX_AI_SEARCH_ENDPOINT_ID"] = os.getenv("VERTEX_AI_SEARCH_ENDPOINT_ID", "your-matching-engine-endpoint-id")
os.environ["VERTEX_AI_SEARCH_INDEX_ID"] = os.getenv("VERTEX_AI_SEARCH_INDEX_ID", "your-matching-engine-index-id")

# Supabase
os.environ["SUPABASE_URL"] = os.getenv("SUPABASE_URL", "https://your-supabase-url.supabase.co")
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "your-supabase-service-role-key")

# Google Custom Search API (for Research Agent)
os.environ["GOOGLE_CSE_API_KEY"] = os.getenv("GOOGLE_CSE_API_KEY", "your-google-cse-api-key")
os.environ["GOOGLE_CSE_CX"] = os.getenv("GOOGLE_CSE_CX", "your-google-cse-cx")

# --- Google Cloud Pub/Sub Topic Names (NEW) ---
# Ensure these topics (and subscriptions) are created in your GCP project.
PUBSUB_TOPICS: Dict[str, str] = {
    "performance_metrics": os.getenv("PUBSUB_TOPIC_PERFORMANCE_METRICS", "zhero-performance-metrics"),
    "knowledge_ingestion": os.getenv("PUBSUB_TOPIC_KNOWLEDGE_INGESTION", "zhero-knowledge-ingestion"),
    "learning_triggers": os.getenv("PUBSUB_TOPIC_LEARNING_TRIGGERS", "zhero-learning-triggers"), # For Meta-Agent to Learning Agent
    "self_optimization_plans": os.getenv("PUBSUB_TOPIC_SELF_OPTIMIZATION_PLANS", "zhero-self-optimization-plans"), # For Meta-Agent to internal config/human
}

# --- Agent and Tool Endpoint URLs (for inter-service communication) ---
AGENT_ENDPOINTS: Dict[str, str] = {
    "orchestration_agent": os.getenv("ORCHESTRATION_AGENT_URL", "http://localhost:8000"),
    "user_profile_agent": os.getenv("USER_PROFILE_AGENT_URL", "http://localhost:8001"),
    "knowledge_management_agent": os.getenv("KNOWLEDGE_MANAGEMENT_AGENT_URL", "http://localhost:8002"),
    "research_agent": os.getenv("RESEARCH_AGENT_URL", "http://localhost:8003"),
    "conversational_agent": os.getenv("CONVERSATIONAL_AGENT_URL", "http://localhost:8004"),
    "voice_interface_agent": os.getenv("VOICE_INTERFACE_AGENT_URL", "http://localhost:8005"),
    "sentiment_analysis_agent": os.getenv("SENTIMENT_ANALYSIS_AGENT_URL", "http://localhost:8006"),
    "learning_agent": os.getenv("LEARNING_AGENT_URL", "http://localhost:8007"),
    "meta_agent": os.getenv("META_AGENT_URL", "http://localhost:8008"),
    "multimodal_agent": os.getenv("MULTIMODAL_AGENT_URL", "http://localhost:8009"),
}

TOOL_ENDPOINTS: Dict[str, str] = {
    "web_search_tool": os.getenv("WEB_SEARCH_TOOL_URL", "http://localhost:8010"),
    "vector_search_tool": os.getenv("VECTOR_SEARCH_TOOL_URL", "http://localhost:8011"),
    "structured_database_tool": os.getenv("STRUCTURE_DATABASE_TOOL_URL", "http://localhost:8012"),
    "summarization_tool": os.getenv("SUMMARIZATION_TOOL_URL", "http://localhost:8013"),
}