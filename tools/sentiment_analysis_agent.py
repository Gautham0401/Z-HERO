# sentiment_analysis_agent.py
from fastapi import FastAPI, HTTPException
from google.cloud import language_v1 # Google Cloud Natural Language API
from typing import Dict, Any, Optional

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import AnalyzeSentimentRequest, SentimentResponse

app = FastAPI(title="Sentiment Analysis Agent")

sentiment_client: Optional[language_v1.LanguageServiceClient] = None

@app.on_event("startup")
async def startup_event():
    """Initializes Google Cloud Natural Language client on application startup."""
    global sentiment_client
    try:
        sentiment_client = language_v1.LanguageServiceClient()
        logger.info("Sentiment Analysis Agent: Initialized Google Cloud Natural Language client.")
    except Exception as e:
        logger.error(f"Sentiment Analysis Agent: Failed to initialize NL client: {e}", exc_info=True)
        sentiment_client = None

@app.post("/analyze", response_model=SentimentResponse, summary="Analyzes the sentiment of text")
async def analyze_sentiment(request: AnalyzeSentimentRequest):
    """
    Analyzes the emotional tone (sentiment) of a given text string.
    """
    if not sentiment_client:
        raise HTTPException(status_code=500, detail="Sentiment Analysis client not initialized.")

    logger.info(f"Sentiment Analysis Agent: Analyzing sentiment for user {request.user_id} (text len: {len(request.text)})")
    try:
        document = language_v1.Document(
            content=request.text, type_=language_v1.Document.Type.PLAIN_TEXT
        )
        # Perform sentiment analysis
        response = sentiment_client.analyze_sentiment(
            request={"document": document, "encoding_type": language_v1.EncodingType.UTF8}
        )

        sentiment_score = response.document_sentiment.score
        sentiment_magnitude = response.document_sentiment.magnitude

        # Map score to a categorical sentiment
        if sentiment_score > 0.2:
            sentiment_category = "positive"
        elif sentiment_score < -0.2:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"

        logger.info(f"Sentiment Analysis Agent: Analyzed sentiment for {request.user_id}: {sentiment_category} (Score: {sentiment_score})")

        return SentimentResponse(
            sentiment=sentiment_category,
            score=sentiment_score,
            magnitude=sentiment_magnitude
        )
    except Exception as e:
        logger.error(f"Sentiment Analysis Agent: Error analyzing sentiment for {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

# To run this agent: uvicorn sentiment_analysis_agent:app --port 8006 --reload