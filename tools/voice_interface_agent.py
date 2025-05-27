# voice_interface_agent.py
from fastapi import FastAPI, HTTPException
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1beta1 as tts
import base64
import asyncio # Illustrative

# Import common utilities
from zhero_common.config import logger, os
from zhero_common.models import SpeechToTextRequest, SpeechToTextResponse, TextToSpeechRequest, TextToSpeechResponse

app = FastAPI(title="Voice Interface Agent")

stt_client: Optional[speech.SpeechClient] = None
tts_client: Optional[tts.TextToSpeechClient] = None

@app.on_event("startup")
async def startup_event():
    """Initializes Google Cloud STT/TTS clients on application startup."""
    global stt_client, tts_client
    try:
        stt_client = speech.SpeechClient()
        logger.info("Voice Interface Agent: Initialized Google Cloud Speech-to-Text client.")
    except Exception as e:
        logger.error(f"Voice Interface Agent: Failed to initialize STT client: {e}", exc_info=True)
        stt_client = None

    try:
        tts_client = tts.TextToSpeechClient()
        logger.info("Voice Interface Agent: Initialized Google Cloud Text-to-Speech client.")
    except Exception as e:
        logger.error(f"Voice Interface Agent: Failed to initialize TTS client: {e}", exc_info=True)
        tts_client = None


@app.post("/speech_to_text", response_model=SpeechToTextResponse, summary="Transcribes audio to text")
async def speech_to_text(request: SpeechToTextRequest):
    """
    Transcribes base64-encoded audio content into text using Google Cloud Speech-to-Text.
    """
    if not stt_client:
        raise HTTPException(status_code=500, detail="Speech-to-Text client not initialized.")

    logger.info(f"Voice Interface Agent: Received STT request for user {request.user_id} ({request.language_code})")
    try:
        audio_content = base64.b64decode(request.audio_content_base64)

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding(getattr(speech.RecognitionConfig.AudioEncoding, request.encoding)),
            sample_rate_hertz=request.sample_rate_hertz,
            language_code=request.language_code,
            enable_automatic_punctuation=True,
            # Add other features like diarization, word-level confidence etc.
        )

        response = await asyncio.to_thread(stt_client.recognize, config=config, audio=audio) # Make sync call async

        if not response.results:
            logger.warning(f"Voice Interface Agent: No speech detected for user {request.user_id}.")
            return SpeechToTextResponse(transcription="", confidence=0.0)

        # Get the first alternative of the first result
        top_result = response.results[0].alternatives[0]
        transcription = top_result.transcript
        confidence = top_result.confidence

        logger.info(f"Voice Interface Agent: STT transcription for {request.user_id}: '{transcription[:50]}...'")
        return SpeechToTextResponse(transcription=transcription, confidence=confidence)
    except Exception as e:
        logger.error(f"Voice Interface Agent: Error during STT for {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech-to-Text failed: {e}")


@app.post("/text_to_speech", response_model=TextToSpeechResponse, summary="Synthesizes text to audio")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Synthesizes text into base64-encoded audio content using Google Cloud Text-to-Speech.
    """
    if not tts_client:
        raise HTTPException(status_code=500, detail="Text-to-Speech client not initialized.")

    logger.info(f"Voice Interface Agent: Received TTS request for user {request.user_id} (text len: {len(request.text)})")
    try:
        synthesis_input = tts.SynthesisInput(text=request.text)

        # Build the voice request, select the language code ("en-US") and the voice
        # Type ("Standard" or "Neural")
        voice = tts.VoiceSelectionParams(
            language_code=request.voice_name.split('-')[0] + '-' + request.voice_name.split('-')[1], # e.g., "en-US"
            name=request.voice_name,
            ssml_gender=tts.SsmlVoiceGender.NEUTRAL # Can be configurable
        )

        # Select the type of audio file you want returned
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16, # Raw PCM for Flutter (or MP3/OGG for web/other)
            speaking_rate=request.speaking_rate,
            pitch=request.pitch
        )

        # Perform the text-to-speech request on the text input with the selected voice parameters and audio file type
        response = await asyncio.to_thread(tts_client.synthesize_speech, input=synthesis_input, voice=voice, audio_config=audio_config) # Make sync call async

        audio_content_base64 = base64.b64encode(response.audio_content).decode("utf-8")

        logger.info(f"Voice Interface Agent: TTS audio generated for {request.user_id} (bytes: {len(response.audio_content)})")
        return TextToSpeechResponse(audio_content_base64=audio_content_base64)
    except Exception as e:
        logger.error(f"Voice Interface Agent: Error during TTS for {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text-to-Speech failed: {e}")

# To run this agent: uvicorn voice_interface_agent:app --port 8005 --reload 