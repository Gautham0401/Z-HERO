# zhero_common/models.py (UPDATED)
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import datetime

# --- General Communication Models ---
class AgentMessage(BaseModel):
    sender: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None # For agents requesting tool usage

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class ToolResponse(BaseModel):
    tool_name: str
    output: Any
    success: bool
    error: Optional[str] = None

class UserQuery(BaseModel):
    user_id: str
    query_text: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    # user_profile_data and sentiment would typically be fetched by Orchestration Agent
    # For simplicity, if pre-computed for demo, can be passed.
    user_profile_data: Optional[Dict[str, Any]] = None
    sentiment: Optional[str] = None # e.g., 'positive', 'neutral', 'negative'
    image_url: Optional[str] = None # <--- ADDED THIS FIELD HERE

class AIResponse(BaseModel):
    user_id: str
    response_text: str
    source_citations: List[Dict[str, str]] = Field(default_factory=list) # e.g., {"url": "...", "title": "..."}
    internal_notes: Optional[Dict[str, Any]] = None # For debugging/logging

# --- Agent-Specific Models ---

# User Profile Agent
class UserProfile(BaseModel):
    user_id: str
    email: Optional[str] = None
    explicit_preferences: Dict[str, Any] = Field(default_factory=dict) # e.g., {"favorite_topics": ["AI", "space"], "learning_style": "visual"}
    inferred_interests: List[str] = Field(default_factory=list)
    last_active: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    # Add fields for racks/books metadata linking
    initialized_racks: bool = False # Flag for initial setup

# Knowledge Management Agent
class KnowledgeItem(BaseModel):
    id: Optional[str] = None # Unique ID for the knowledge chunk
    user_id: str # Or 'system' for general knowledge
    content: str # The actual text content
    source_url: Optional[str] = None
    title: Optional[str] = None
    rack: Optional[str] = None # Z-HERO specific metadata (e.g., "Technology")
    book: Optional[str] = None # Z-HERO specific metadata (e.g., "Quantum Computing Basics")
    embeddings: Optional[List[float]] = None # Stored after generation
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    score: Optional[float] = None # For retrieval scores

class KnowledgeSearchQuery(BaseModel):
    user_id: str
    query_text: str
    top_k: int = 5
    filter_by_racks: Optional[List[str]] = None
    filter_by_books: Optional[List[str]] = None

class SearchResult(BaseModel):
    knowledge_item: KnowledgeItem
    score: float # Relevance score

# Research Agent / Web Search Tool
class SearchRequest(BaseModel):
    query: str
    user_id: str
    num_results: int = 3
    language: str = "en"

class WebSearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    publication_date: Optional[str] = None
    source_reliability_score: Optional[float] = None # From Research Agent assessment

# Voice Interface Agent
class SpeechToTextRequest(BaseModel):
    audio_content_base64: str # Base64 encoded audio bytes
    encoding: str # e.g., "LINEAR16", "FLAC"
    sample_rate_hertz: int
    language_code: str = "en-US"
    user_id: str

class SpeechToTextResponse(BaseModel):
    transcription: str
    confidence: float

class TextToSpeechRequest(BaseModel):
    text: str
    user_id: str
    voice_name: Optional[str] = "en-US-Standard-A"
    speaking_rate: float = 1.0
    pitch: float = 0.0

class TextToSpeechResponse(BaseModel):
    audio_content_base64: str # Base64 encoded audio bytes

# Sentiment Analysis Agent
class AnalyzeSentimentRequest(BaseModel):
    text: str
    user_id: str

class SentimentResponse(BaseModel):
    sentiment: str # e.g., "positive", "neutral", "negative", "mixed"
    score: float # From -1.0 (negative) to 1.0 (positive)
    magnitude: float # Strength of emotion (0 to infinity)

# Learning Agent
class LearningTrigger(BaseModel):
    user_id: str
    trigger_type: str # e.g., "knowledge_gap", "low_confidence_response", "user_feedback"
    details: Dict[str, Any]

class KnowledgeGap(BaseModel):
    query_text: str
    user_id: str
    reason: str # e.g., "no_relevant_book_found", "outdated_info", "low_confidence"
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

# Meta-Agent
class AgentPerformanceMetric(BaseModel):
    agent_name: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    user_id: Optional[str] = None
    metric_name: str # e.g., "latency_ms", "success_rate", "tokens_consumed"
    value: float
    context: Dict[str, Any] = Field(default_factory=dict)