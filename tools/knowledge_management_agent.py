# knowledge_management_agent.py
from fastapi import FastAPI, HTTPException
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchingEngineIndexEndpoint
from typing import List, Optional
import datetime
import asyncio # For illustrative purpose for async operations

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import KnowledgeItem, KnowledgeSearchQuery, SearchResult
from zhero_common.clients import supabase # Use the imported Supabase client

app = FastAPI(title="Knowledge Management Agent")

embedding_model: Optional[aiplatform.language_models.TextEmbeddingModel] = None
matching_engine_index_endpoint: Optional[MatchingEngineIndexEndpoint] = None

@app.on_event("startup")
async def startup_event():
    """Initializes Vertex AI clients on application startup."""
    global embedding_model, matching_engine_index_endpoint
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
        # Handle cases where model init fails, preventing immediate crash
        embedding_model = None

    try:
        # Note: Ensure the endpoint ID is correct for your deployed index.
        # It typically looks like projects/PROJECT_NUMBER/locations/REGION/indexEndpoints/ENDPOINT_ID
        matching_engine_index_endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=os.environ["VERTEX_AI_SEARCH_ENDPOINT_ID"]
        )
        logger.info("Knowledge Management Agent: Initialized Vertex AI Search Index Endpoint.")
    except Exception as e:
        logger.error(f"Knowledge Management Agent: Failed to initialize Vertex AI Search Index Endpoint: {e}", exc_info=True)
        matching_engine_index_endpoint = None

@app.post("/ingest_knowledge", summary="Ingests a knowledge item, generates embeddings, and stores it")
async def ingest_knowledge(item: KnowledgeItem):
    """
    Ingests a new knowledge item, generates embeddings, and stores it in the vector database
    (Vertex AI Search/Matching Engine) and its metadata in Supabase.
    """
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model not initialized.")

    logger.info(f"Knowledge Management Agent: Received ingestion for '{item.title or item.content[:50]}...' (User: {item.user_id})")

    try:
        # 1. Generate Embeddings
        # Vertex AI client librarypredict is synchronous, but we can await it within FastAPI
        embeddings_response = embedding_model.predict([item.content])
        item.embeddings = embeddings_response.embeddings[0].values
        logger.info("Knowledge Management Agent: Embeddings generated.")

        # 2. Store in Vertex AI Search (Matching Engine) - Conceptual/Simplified
        # The true upsert operation to Matching Engine is out of scope for a direct FastAPI call.
        # In a real system, you'd use Vertex AI SDK's `upsert_datapoints` via a client
        # or stream data to the index endpoint, potentially after writing to GCS for batching.
        # This part simulates the data reaching the vector database.
        if matching_engine_index_endpoint:
            try:
                # This is a conceptual call. Direct upsert_datapoints requires proper setup
                # of datapoint IDs to match your index, and ensuring the dimensions match.
                # Example for upsert_datapoints (requires a pre-existing index ID/name):
                # from google.cloud.aiplatform_v1beta1 import MatchingEngineIndexServiceClient
                # client = MatchingEngineIndexServiceClient(client_options=...)
                # client.upsert_datapoints(
                #     index=os.environ["VERTEX_AI_SEARCH_INDEX_ID"], # This is the full resource name
                #     datapoints=[
                #         {
                #             "id": item.id if item.id else str(datetime.datetime.utcnow().timestamp()),
                #             "embedding": item.embeddings,
                #             "numeric_restricts": [], # Add numeric filters if needed
                #             "categorical_restricts": [ # Crucial for user-specific filtering
                #                 {"namespace": "user_id", "value": item.user_id},
                #                 {"namespace": "rack", "value": item.rack} if item.rack else {}
                #             ]
                #         }
                #     ]
                # )
                # For this demo, we just log that it would be stored.
                logger.info(f"Knowledge Management Agent: Simulating storage of embedding in Vertex AI Search for {item.user_id}.")
            except Exception as e:
                logger.warning(f"Knowledge Management Agent: Failed to simulate Matching Engine upsert: {e}", exc_info=True)
                # Continue process even if vector store fails, metadata might still be useful
        else:
            logger.warning("Knowledge Management Agent: Matching Engine not initialized. Will only store metadata.")


        # 3. Store metadata in Supabase
        item.id = item.id if item.id else str(datetime.datetime.utcnow().timestamp()) # Generate ID if not provided
        item_data_for_supabase = item.model_dump(exclude={"embeddings"}, exclude_unset=True) # Don't store embeddings in PostgreSQL
        item_data_for_supabase["timestamp"] = item_data_for_supabase["timestamp"].isoformat() # Convert datetime to string

        response = await supabase.from_("knowledge_items").insert(item_data_for_supabase).execute()
        if response["error"]:
            raise Exception(response["error"])

        logger.info(f"Knowledge Management Agent: Knowledge item '{item.title}' ingested and metadata stored successfully.")
        return {"status": "success", "id": item.id}
    except Exception as e:
        logger.error(f"Knowledge Management Agent: Error ingesting knowledge: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Knowledge ingestion failed: {str(e)}")


@app.post("/search_knowledge_semantic", response_model=List[SearchResult], summary="Performs a semantic search against the knowledge base")
async def search_knowledge_semantic(request: KnowledgeSearchQuery):
    """
    Performs a semantic search against the user's personalized knowledge base
    (Vertex AI Search/Matching Engine) and retrieves relevant items.
    """
    if not embedding_model or not matching_engine_index_endpoint:
        raise HTTPException(status_code=500, detail="Knowledge management services not initialized.")

    logger.info(f"Knowledge Management Agent: Received semantic search request for '{request.query_text}' (User: {request.user_id})")

    try:
        # 1. Generate embedding for the query
        query_embedding_response = embedding_model.predict([request.query_text])
        query_embedding = query_embedding_response.embeddings[0].values
        logger.info("Knowledge Management Agent: Query embedding generated.")

        # 2. Prepare filters for Matching Engine
        restricts = [{"namespace": "user_id", "allow_list": [request.user_id]}]
        if request.filter_by_racks:
            restricts.append({"namespace": "rack", "allow_list": request.filter_by_racks})
        if request.filter_by_books:
            restricts.append({"namespace": "book", "allow_list": request.filter_by_books})

        # 3. Perform vector search in Vertex AI Search (Matching Engine)
        results = matching_engine_index_endpoint.find_neighbors(
            queries=[query_embedding],
            num_neighbors=request.top_k,
            restricts=restricts
        )
        logger.info(f"Knowledge Management Agent: Vector search completed, found {len(results[0])} raw neighbors.")

        # 4. Retrieve full content/metadata for the found IDs from Supabase
        retrieved_items_ids = [neighbor.id for neighbor in results[0]]
        # In a real app, this would be `supabase.from_("knowledge_items").select("*").in_("id", retrieved_items_ids).execute()`
        # For mock, we'll just simulate by searching in the mock data store.
        mock_db_results = []
        for item_id in retrieved_items_ids:
            db_result = await supabase.from_("knowledge_items").select("*").eq("id", item_id).execute()
            if db_result["data"]:
                mock_db_results.append(db_result["data"][0])


        # 5. Combine results with scores and format
        final_results: List[SearchResult] = []
        for neighbor in results[0]: # Sort by distance (score) from Matching Engine
            item_data = next((item for item in mock_db_results if item['id'] == neighbor.id), None)
            if item_data:
                # Reconstruct KnowledgeItem from DB data
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

        logger.info(f"Knowledge Management Agent: Retrieved and formatted {len(final_results)} detailed knowledge items.")

        # Sort again by score (distance, lower is better)
        final_results.sort(key=lambda x: x.score)

        return final_results
    except Exception as e:
        logger.error(f"Knowledge Management Agent: Error during semantic search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

# To run this agent: uvicorn knowledge_management_agent:app --port 8002 --reload