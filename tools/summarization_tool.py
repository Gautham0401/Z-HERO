# summarization_tool.py
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
from vertexai.preview.generative_models import GenerativeModel

# Import common utilities
from zhero_common.config import logger, os

app = FastAPI(title="Summarization Tool")

summarization_model: Optional[GenerativeModel] = None

@app.on_event("startup")
async def startup_event():
    global summarization_model
    try:
        from google.cloud import aiplatform # Ensure this is imported for Vertex AI features
        aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["VERTEX_AI_LOCATION"])
        summarization_model = GenerativeModel(os.environ["GEMINI_PRO_MODEL_ID"])
        logger.info("Summarization Tool: Initialized Gemini model for summarization.")
    except Exception as e:
        logger.error(f"Summarization Tool: Failed to initialize Gemini model: {e}", exc_info=True)
        summarization_model = None

@app.post("/summarize", response_model=Dict[str, str], summary="Summarizes provided text content")
async def summarize_text(request: Dict[str, str]):
    """
    Summarizes a given large block of text into a concise summary.
    """
    text_content = request.get("text_content")
    if not text_content:
        raise HTTPException(status_code=400, detail="'text_content' is required for summarization.")
    if not summarization_model:
        raise HTTPException(status_code=500, detail="Summarization model not initialized.")

    logger.info(f"Summarization Tool: Summarizing text of length {len(text_content)}.")

    prompt = f"""
    Please summarize the following text concisely and clearly.
    Text:
    ---
    {text_content}
    ---
    Summary:
    """
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 200, # Adjust for desired summary length
    }

    try:
        response = await summarization_model.generate_content_async(
            contents=[prompt],
            generation_config=generation_config
        )
        summary = response.candidates[0].text
        logger.info("Summarization Tool: Text summarized successfully.")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization Tool: Error summarizing text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text summarization failed: {e}")

# To run this tool: uvicorn summarization_tool:app --port 8013 --reload