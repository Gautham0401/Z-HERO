# tools/voice_interface_agent.py
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1beta1 as tts
import base64
import asyncio
import datetime

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import SpeechToTextRequest, SpeechToTextResponse, TextToSpeechRequest, TextToSpeechResponse # Already Pydantic
from zhero_common.exceptions import (
    ZHeroException, ZHeroAgentError, ZHeroInvalidInputError,
    ZHeroDependencyError, ZHeroVertexAIError
)
# NEW: Pub/Sub Publisher instance
from zhero_common.pubsub_client import pubsub_publisher, initialize_pubsub_publisher


app = FastAPI(title="Voice Interface Agent")

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


stt_client: Optional[speech.SpeechClient] = None
tts_client: Optional[tts.TextToSpeechClient] = None

@app.on_event("startup")
async def startup_event():
    global stt_client, tts_client
    # Initialize Pub/Sub Publisher
    await initialize_pubsub_publisher()
    try:
        stt_client = speech.SpeechClient()
        logger.info("Voice Interface Agent: Initialized Google Cloud Speech-to-Text client.")
    except Exception as e:
        logger.error(f"Voice Interface Agent: Failed to initialize STT client: {e}", exc_info=True)
        raise ZHeroVertexAIError("VoiceInterfaceAgent", "Google STT Client", "Failed to initialize Speech-to-Text client on startup.", original_error=e)

    try:
        tts_client = tts.TextToSpeechClient()
        logger.info("Voice Interface Agent: Initialized Google Cloud Text-to-Speech client.")
    except Exception as e:
        logger.error(f"Voice Interface Agent: Failed to initialize TTS client: {e}", exc_info=True)
        raise ZHeroVertexAIError("VoiceInterfaceAgent", "Google TTS Client", "Failed to initialize Text-to-Speech client on startup.", original_error=e)


@app.post("/speech_to_text", response_model=SpeechToTextResponse, summary="Transcribes audio to text")
async def speech_to_text(request: SpeechToTextRequest): # SpeechToTextRequest is already Pydantic
    """
    Transcribes base64-encoded audio content into text using Google Cloud Speech-to-Text.
    """
    if not stt_client:
        raise ZHeroDependencyError("VoiceInterfaceAgent", "Google STT Client", "Speech-to-Text client not initialized.", 500)

    logger.info(f"Voice Interface Agent: Received STT request for user {request.user_id} ({request.language_code})")
    try:
        audio_content = base64.b64decode(request.audio_content_base64)

        audio = speech.RecognitionAudio(content=audio_content)
        try:
            encoding_enum = getattr(speech.RecognitionConfig.AudioEncoding, request.encoding.upper())
        except AttributeError:
             raise ZHeroInvalidInputError(message=f"Invalid audio encoding: {request.encoding}. Must be a valid SpeechConfig.AudioEncoding enum member.")

        config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=request.sample_rate_hertz,
            language_code=request.language_code,
            enable_automatic_punctuation=True,
        )

        response = await asyncio.to_thread(stt_client.recognize, config=config, audio=audio)

        if not response.results:
            logger.warning(f"Voice Interface Agent: No speech detected for user {request.user_id}.")
            return SpeechToTextResponse(transcription="[No speech detected]", confidence=0.0)

        top_result = response.results[0].alternatives[0]
        transcription = top_result.transcript
        confidence = top_result.confidence

        logger.info(f"Voice Interface Agent: STT transcription for {request.user_id}: '{transcription[:50]}...'")
        return SpeechToTextResponse(transcription=transcription, confidence=confidence)
    except ZHeroException: raise
    except Exception as e:
        raise ZHeroVertexAIError("VoiceInterfaceAgent", "Google STT API", f"Error during STT API call: {e}", original_error=e)


@app.post("/text_to_speech", response_model=TextToSpeechResponse, summary="Synthesizes text to audio")
async def text_to_speech(request: TextToSpeechRequest): # TextToSpeechRequest is already Pydantic
    if not tts_client:
        raise ZHeroDependencyError("VoiceInterfaceAgent", "Google TTS Client", "Text-to-Speech client not initialized.", 500)

    logger.info(f"Voice Interface Agent: Received TTS request for user {request.user_id} (text len: {len(request.text)})")
    try:
        synthesis_input = tts.SynthesisInput(text=request.text)

        voice = tts.VoiceSelectionParams(
            language_code=request.voice_name.split('-')[0] + '-' + request.voice_name.split('-')[1],
            name=request.voice_name,
            ssml_gender=tts.SsmlVoiceGender.NEUTRAL
        )

        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            speaking_rate=request.speaking_rate,
            pitch=request.pitch
        )

        response = await asyncio.to_thread(tts_client.synthesize_speech, input=synthesis_input, voice=voice, audio_config=audio_config)

        audio_content_base64 = base64.b64encode(response.audio_content).decode("utf-8")

        logger.info(f"Voice Interface Agent: TTS audio generated for {request.user_id} (bytes: {len(response.audio_content)})")
        return TextToSpeechResponse(audio_content_base64=audio_content_base64)
    except ZHeroException: raise
    except Exception as e:
        raise ZHeroVertexAIError("VoiceInterfaceAgent", "Google TTS API", f"Error during TTS API call: {e}", original_error=e)
