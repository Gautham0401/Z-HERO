# zhero_common/models.py
from pydantic import BaseModel, Field, EmailStr, AnyUrl
from typing import List, Dict, Optional, Any, Literal
import datetime

# --- General Communication Models ---
class AgentMessage(BaseModel):
    sender: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class ToolResponse(BaseModel):
    tool_name: str
    output: Any
    success: bool
    error: Optional[str] = None

class BaseRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the user.")

class UserQuery(BaseRequest):
    query_text: str = Field(..., min_length=1, description="The user's input query text.")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Previous turns in the conversation.")
    user_profile_data: Optional[Dict[str, Any]] = Field(None, description="Pre-fetched user profile data.")
    sentiment: Optional[str] = Field(None, description="Inferred sentiment of the user's query.")
    image_url: Optional[AnyUrl] = Field(None, description="URL of an image associated with the query, if any.")

class AIResponse(BaseModel):
    user_id: str
    response_text: str
    source_citations: List[Dict[str, str]] = Field(default_factory=list)
    internal_notes: Optional[Dict[str, Any]] = None

# --- Agent-Specific Models ---

# User Profile Agent
class UserProfile(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    explicit_preferences: Dict[str, Any] = Field(default_factory=dict)
    inferred_interests: List[str] = Field(default_factory=list)
    last_active: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    initialized_racks: bool = False

class UserProfileUpdateRequest(BaseRequest):
    updates: Dict[str, Any] = Field(..., min_items=1)

class UserPreferenceUpdateRequest(BaseRequest):
    user_id: str = Field(..., min_length=1)
    preference_key: str = Field(..., min_length=1)
    preference_value: Any

# Knowledge Management Agent
class KnowledgeItem(BaseModel):
    id: Optional[str] = None
    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_url: Optional[AnyUrl] = None
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    rack: Optional[str] = Field(None, min_length=1, max_length=100)
    book: Optional[str] = Field(None, min_length=1, max_length=100)
    embeddings: Optional[List[float]] = None
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    score: Optional[float] = None

class KnowledgeSearchQuery(BaseRequest):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(5, gt=0, le=100)
    filter_by_racks: Optional[List[str]] = None
    filter_by_books: Optional[List[str]] = None

class SearchResult(BaseModel):
    knowledge_item: KnowledgeItem
    score: float

# Research Agent / Web Search Tool
class SearchRequest(BaseRequest):
    query: str = Field(..., min_length=1)
    num_results: int = Field(3, gt=0, le=10)
    language: str = Field("en", min_length=2, max_length=5)

class WebSearchResult(BaseModel):
    title: str = Field(..., min_length=1)
    link: AnyUrl = Field(...)
    snippet: str = Field(..., min_length=1)
    publication_date: Optional[str] = None
    source_reliability_score: Optional[float] = None

# Voice Interface Agent
class SpeechToTextRequest(BaseRequest):
    audio_content_base64: str = Field(..., min_length=1)
    encoding: str = Field(..., min_length=1)
    sample_rate_hertz: int = Field(..., gt=0)
    language_code: str = Field("en-US", min_length=2, max_length=10)

class SpeechToTextResponse(BaseModel):
    transcription: str
    confidence: float

class TextToSpeechRequest(BaseRequest):
    text: str = Field(..., min_length=1)
    voice_name: Optional[str] = Field("en-US-Standard-A", min_length=1)
    speaking_rate: float = Field(1.0, gt=0.0, le=4.0)
    pitch: float = Field(0.0, ge=-20.0, le=20.0)

class TextToSpeechResponse(BaseModel):
    audio_content_base64: str

# Sentiment Analysis Agent
class AnalyzeSentimentRequest(BaseRequest):
    text: str = Field(..., min_length=1)

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    score: float = Field(..., ge=-1.0, le=1.0)
    magnitude: float = Field(..., ge=0.0)

# Learning Agent
class LearningTrigger(BaseRequest):
    trigger_type: str = Field(..., min_length=1)
    details: Dict[str, Any] = Field(default_factory=dict)

# Sub-model for knowledge gap details within LearningTrigger
class KnowledgeGapDetails(BaseModel):
    query_text: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)

class KnowledgeGap(BaseRequest):
    query_text: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    status: Literal["unaddressed", "in_progress", "addressed", "rejected"] = "unaddressed"
    timestamp: datetime.datetime = Field(default_factory=datetime.utcnow)

# Meta-Agent
class AgentPerformanceMetric(BaseRequest):
    agent_name: str = Field(..., min_length=1)
    metric_name: str = Field(..., min_length=1)
    value: float
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime.datetime = Field(default_factory=datetime.utcnow)

class GetKnowledgeGapsRequest(BaseModel):
    status: Optional[Literal["unaddressed", "in_progress", "addressed", "rejected"]] = "unaddressed"
    limit: int = Field(10, gt=0, le=100)

class MarkGapAsAddressedRequest(BaseModel):
    gap_id: str = Field(..., min_length=1)
    status: Literal["addressed", "in_progress", "rejected"] = "addressed"

# Multimodal Agent.process_content
class MultimodalProcessRequest(BaseRequest):
    query_text: str = Field(..., min_length=1)
    image_url: AnyUrl = Field(...)
