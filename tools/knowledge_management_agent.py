# tools/knowledge_management_agent.py
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchingEngineIndexEndpoint
from typing import List, Optional, Dict, Any
import datetime
import asyncio
import logging

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import KnowledgeItem, KnowledgeSearchQuery, SearchResult # Pydantic models
from zhero_common.clients import supabase
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroNotFoundError,
    ZHeroInvalidInputError, ZHeroDependencyError, ZHeroSupabaseError,
    ZHeroVertexAIError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher


app = FastAPI(title="Knowledge Management Agent")

# --- Global Exception Handlers (REQUIRED IN ALL AGENT FILES) ---
@app.exception_handler(ZHeroException)
async def zhero_exception_handler(request: Request, exc: ZHeroException):
    logger.error(f"ZHeroException caught for request {request.url.path}: {exc.message}", exc_info=True, extra={"details": exc.details, "status_code": exc.status_code})
    return JSONResponse(status_code=exc.status_code, content={"error_type": exc.__class__.__name__,"message": exc.message,"details": exc.details})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException caught for request {request.url.path}: {exc.detail}", exc_info=True, extra={"status_code": exc.status_code, "request_body": await request.body()})
    return JSONResponse(status_code=exc.status_code, content={"error_type": "HTTPException","message": exc.detail,"details": getattr(exc, 'body', None)})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log_id = f"ERR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.critical(f"Unhandled Exception caught for request {request.url.path} (ID: {log_id}): {exc}", exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error_type": "InternalServerError","message": "An unexpected internal server error occurred. Please try again later.","error_id": log_id, "details": str(exc) if app.debug else None})
# --- END Global Exception Handlers ---


embedding_model: Optional[aiplatform.language_models.TextEmbeddingModel] = None
matching_engine_index_endpoint: Optional[MatchingEngineIndexEndpoint] = None

@app.on_event("startup")
async def startup_event():
    global embedding_model, matching_engine_index_endpoint
    # Initialize Pub/Sub Publisher
    await initialize_pubsub_publisher()

    try:
        aiplatform.init(
            project=os.environ["GCP_PROJECT_ID"],
            location=os.environ["VERTEX_AI_LOCATION"]
        )
        embedding_model = aiplatform.language_models.TextEmbeddingModel.from_pretrained(
            os.environ["VERTEX_AI_EMBEDDING_MODEL_ID"]
        )
        logger.info("Knowledge Management Agent: Initialized Vertex AI Embeddings model.")
    except Exception as e:
        logger.error(f"Knowledge Management Agent: Failed to initialize embedding model: {e}", exc_info=True)
        raise ZHeroVertexAIError("KnowledgeManagementAgent", "Embeddings Model", "Failed to initialize embeddings model on startup.", original_error=e)

    try:
        matching_engine_index_endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=os.environ["VERTEX_AI_SEARCH_ENDPOINT_ID"]
        )
        logger.info("Knowledge Management Agent: Initialized Vertex AI Search Index Endpoint.")
    except Exception as e:
        logger.error(f"Knowledge Management Agent: Failed to initialize Vertex AI Search Index Endpoint: {e}", exc_info=True)
        raise ZHeroVertexAIError("KnowledgeManagementAgent", "Vertex AI Search", "Failed to initialize Matching Engine endpoint on startup.", original_error=e)


@app.post("/ingest_knowledge", summary="Ingests a knowledge item, generates embeddings, and stores it")
async def ingest_knowledge(item: KnowledgeItem): # Uses Pydantic model for validation
    """
    Ingests a new knowledge item, generates embeddings, and stores it in the vector database
    (Vertex AI Search/Matching Engine) and its metadata in Supabase.
    This endpoint can be called directly or via Pub/Sub push subscription.
    """
    if not item.content:
        raise ZHeroInvalidInputError(message="Knowledge item content cannot be empty for ingestion.")
    if not item.user_id:
        raise ZHeroInvalidInputError(message="Knowledge item must have a user_id for ingestion.")
    if not embedding_model:
        raise ZHeroDependencyError("KnowledgeManagementAgent", "Embeddings Model", "Embedding model not initialized.", 500)


    logger.info(f"Knowledge Management Agent: Received ingestion for '{item.title or item.content[:50]}...' (User: {item.user_id})")

    try:
        embeddings_response = embedding_model.predict([item.content])
        item.embeddings = embeddings_response.embeddings[0].values
        logger.info("Knowledge Management Agent: Embeddings generated.")

        if matching_engine_index_endpoint:
            try:
                logger.info(f"Knowledge Management Agent: Simulating storage of embedding in Vertex AI Search for {item.user_id}.")
            except Exception as e:
                raise ZHeroVertexAIError("KnowledgeManagementAgent", "Vertex AI Search", "Failed to upsert data to Matching Engine.", original_error=e)
        else:
            logger.warning("Knowledge Management Agent: Matching Engine not initialized. Only metadata will be stored.")

        item.id = item.id if item.id else str(datetime.datetime.utcnow().timestamp())
        item_data_for_supabase = item.model_dump(exclude={"embeddings"}, exclude_unset=True)
        item_data_for_supabase["timestamp"] = item_data_for_supabase["timestamp"].isoformat()

        response = await supabase.from_("knowledge_items").insert(item_data_for_supabase).execute()
        if response["error"]:
            raise ZHeroSupabaseError(agent_name="KnowledgeManagementAgent", message=response["error"]["message"], original_error=response["error"])

        logger.info(f"Knowledge Management Agent: Knowledge item '{item.title}' ingested and metadata stored successfully.")
        return {"status": "success", "id": item.id}
    except ZHeroException:
        raise
    except Exception as e:
        raise ZHeroAgentError("KnowledgeManagementAgent", "Error during knowledge ingestion.", original_error=e)


@app.post("/search_knowledge_semantic", response_model=List[SearchResult], summary="Performs a semantic search against the knowledge base")
async def search_knowledge_semantic(request: KnowledgeSearchQuery): # Uses Pydantic model for validation
    """
    Performs a semantic search against the user's personalized knowledge base
    (Vertex AI Search/Matching Engine) and retrieves relevant items.
    """
    if not embedding_model:
        raise ZHeroDependencyError("KnowledgeManagementAgent", "Embeddings Model", "Embedding model not initialized for search.", 500)
    if not matching_engine_index_endpoint:
        raise ZHeroDependencyError("KnowledgeManagementAgent", "Vertex AI Search", "Matching Engine endpoint not initialized for search.", 500)
    if not request.query_text:
        raise ZHeroInvalidInputError(message="Search query text cannot be empty.")
    if not request.user_id:
        raise ZHeroInvalidInputError(message="User ID is required for semantic search.")

    logger.info(f"Knowledge Management Agent: Received semantic search request for '{request.query_text}' (User: {request.user_id})")

    try:
        embeddings_response = embedding_model.predict([request.query_text])
        query_embedding = embeddings_response.embeddings[0].values
        logger.info("Knowledge Management Agent: Query embedding generated.")

        restricts = [{"namespace": "user_id", "allow_list": [request.user_id]}]
        if request.filter_by_racks:
            restricts.append({"namespace": "rack", "allow_list": request.filter_by_racks})
        if request.filter_by_books:
            restricts.append({"namespace": "book", "allow_list": request.filter_by_books})

        results = matching_engine_index_endpoint.find_neighbors(
            queries=[query_embedding],
            num_neighbors=request.top_k,
            restricts=restricts
        )
        logger.info(f"Knowledge Management Agent: Vector search completed, found {len(results[0])} raw neighbors.")

        retrieved_items_ids = [neighbor.id for neighbor in results[0]]
        mock_db_results = []
        for item_id in retrieved_items_ids:
            db_result = await supabase.from_("knowledge_items").select("*").eq("id", item_id).execute()
            if db_result["error"]:
                raise ZHeroSupabaseError(agent_name="KnowledgeManagementAgent", message=db_result["error"]["message"], original_error=db_result["error"])
            if db_result["data"]:
                mock_db_results.append(db_result["data"][0])


        final_results: List[SearchResult] = []
        for neighbor in results[0]:
            item_data = next((item for item in mock_db_results if item['id'] == neighbor.id), None)
            if item_data:
                knowledge_item = KnowledgeItem(
                    id=item_data.get("id"),
                    user_id=item_data.get("user_id", ""),
                    content=item_data.get("content", ""),
                    source_url=item_data.get("source_url"),
                    title=item_data.get("title"),
                    rack=item_data.get("rack"),
                    book=item_data.get("book"),
                    timestamp=datetime.datetime.fromisoformat(item_data.get("timestamp")) if isinstance(item_data.get("timestamp"), str) else item_data.get("timestamp")
                )
                final_results.append(SearchResult(knowledge_item=knowledge_item, score=neighbor.distance))

        final_results.sort(key=lambda x: x.score)

        logger.info(f"Knowledge Management Agent: Retrieved and formatted {len(final_results)} detailed knowledge items.")
        return final_results
    except ZHeroException:
        raise
    except Exception as e:
        raise ZHeroAgentError("KnowledgeManagementAgent", "Error during semantic search.", original_error=e)