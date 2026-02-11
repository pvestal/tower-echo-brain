"""
Echo Brain Voice API Router
============================
REST endpoints for one-shot STT/TTS and a WebSocket endpoint
for real-time streaming voice conversation.

Endpoints:
  POST /api/echo/voice/transcribe   — Upload audio, get text back
  POST /api/echo/voice/synthesize   — Send text, get audio back
  POST /api/echo/voice/chat         — Full loop: audio in → text → response → audio out
  GET  /api/echo/voice/status       — Voice service health
  GET  /api/echo/voice/voices       — List available TTS voices
  WS   /api/echo/voice/ws           — Real-time bidirectional voice streaming
"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger("echo.voice.api")

router = APIRouter(prefix="/api/echo/voice", tags=["voice"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    """For JSON-based transcription (base64-encoded audio)."""
    audio_base64: str
    sample_rate: int = 16000
    language: str = "en"


class TranscribeResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration_s: float
    transcription_time_s: float
    segments: list


class SynthesizeRequest(BaseModel):
    text: str
    speaker_id: Optional[int] = None
    length_scale: float = 1.0


class VoiceChatResponse(BaseModel):
    transcript: str
    response_text: str
    audio_base64: str
    timings: dict
    query_type: Optional[str] = None
    confidence: Optional[float] = None
    sources: Optional[list] = None
    actions_taken: Optional[list] = None
    execution_time_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    sample_rate: int = Form(16000),
    language: str = Form("en"),
):
    """
    Transcribe audio to text.

    Accepts either:
    - Multipart file upload (audio/wav, audio/webm, etc.)
    - Base64-encoded audio in form field
    """
    from src.services.voice_service import voice_service

    if file:
        audio_bytes = await file.read()
    elif audio_base64:
        audio_bytes = base64.b64decode(audio_base64)
    else:
        raise HTTPException(400, "Provide either 'file' upload or 'audio_base64'")

    if len(audio_bytes) < 100:
        raise HTTPException(400, "Audio data too short")

    result = await voice_service.transcribe(
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
        language=language,
    )

    return TranscribeResponse(**result)


@router.post("/synthesize")
async def synthesize_speech(request: SynthesizeRequest):
    """
    Convert text to speech. Returns WAV audio.

    Response Content-Type: audio/wav
    """
    from src.services.voice_service import voice_service

    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    if len(request.text) > 5000:
        raise HTTPException(400, "Text too long (max 5000 chars)")

    audio_bytes = await voice_service.synthesize(
        text=request.text,
        speaker_id=request.speaker_id,
        length_scale=request.length_scale,
    )

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=echo_response.wav",
            "X-Echo-Text-Length": str(len(request.text)),
            "X-Echo-Audio-Length": str(len(audio_bytes)),
        },
    )


@router.post("/synthesize/base64")
async def synthesize_speech_base64(request: SynthesizeRequest):
    """
    Convert text to speech. Returns base64-encoded WAV.
    Useful for JSON-based frontends.
    """
    from src.services.voice_service import voice_service

    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    audio_bytes = await voice_service.synthesize(
        text=request.text,
        speaker_id=request.speaker_id,
        length_scale=request.length_scale,
    )

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode(),
        "format": "wav",
        "text_length": len(request.text),
        "audio_bytes": len(audio_bytes),
    }


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    sample_rate: int = Form(16000),
):
    """
    Full voice chat loop: audio → transcribe → Echo Brain → TTS → audio response.

    This is the "talk to Echo Brain" endpoint.
    """
    from src.services.voice_service import voice_service
    from src.intelligence.reasoner import get_reasoning_engine

    if file:
        audio_bytes = await file.read()
    elif audio_base64:
        audio_bytes = base64.b64decode(audio_base64)
    else:
        raise HTTPException(400, "Provide either 'file' upload or 'audio_base64'")

    # Stash reasoning metadata from the chat handler
    reasoning_meta = {}

    async def chat_handler(text: str) -> dict:
        """Bridge to Echo Brain's reasoning pipeline — captures full metadata."""
        try:
            reasoner = get_reasoning_engine()
            result = await reasoner.process(text)
            reasoning_meta.update({
                "query_type": result.query_type.value,
                "confidence": result.confidence,
                "sources": result.sources,
                "actions_taken": result.actions_taken,
                "execution_time_ms": result.execution_time_ms,
            })
            return {"response": result.response}
        except Exception as e:
            logger.error(f"Chat handler error: {e}")
            return {"response": "I encountered an error processing your request."}

    result = await voice_service.voice_chat(
        audio_bytes=audio_bytes,
        chat_handler=chat_handler,
        sample_rate=sample_rate,
    )

    if "error" in result:
        return VoiceChatResponse(
            transcript="",
            response_text=result.get("error", "No speech detected"),
            audio_base64="",
            timings=result.get("timings", {}),
        )

    # Persist to Echo Brain memory (fire-and-forget)
    from src.services.voice_memory import store_voice_turn
    asyncio.create_task(store_voice_turn(
        session_id=f"rest-{uuid4().hex[:12]}",
        user_text=result["transcript"],
        response_text=result["response_text"],
        metadata={
            "query_type": reasoning_meta.get("query_type"),
            "confidence": reasoning_meta.get("confidence"),
            "sources": reasoning_meta.get("sources"),
            "stt_time_ms": result["timings"].get("stt_ms"),
            "chat_time_ms": result["timings"].get("chat_ms"),
            "tts_time_ms": result["timings"].get("tts_ms"),
        },
    ))

    return VoiceChatResponse(
        transcript=result["transcript"],
        response_text=result["response_text"],
        audio_base64=base64.b64encode(result["audio_bytes"]).decode(),
        timings=result["timings"],
        **reasoning_meta,
    )


@router.get("/status")
async def voice_status():
    """Get voice service status and health."""
    from src.services.voice_service import voice_service
    return voice_service.get_status()


@router.get("/voices")
async def list_voices():
    """List available Piper TTS voice models."""
    voices_dir = Path("/opt/tower-echo-brain/models/voice/piper")
    available = []

    if voices_dir.exists():
        for onnx_file in voices_dir.glob("*.onnx"):
            if not onnx_file.name.endswith(".onnx.json"):
                available.append({
                    "name": onnx_file.stem,
                    "path": str(onnx_file),
                    "size_mb": round(onnx_file.stat().st_size / 1024 / 1024, 1),
                    "config_exists": onnx_file.with_suffix(".onnx.json").exists(),
                })

    # Suggested voices to download
    suggested = [
        {"name": "en_US-lessac-medium", "quality": "medium", "description": "Natural US English (default)"},
        {"name": "en_US-lessac-high", "quality": "high", "description": "High quality US English"},
        {"name": "en_US-amy-medium", "quality": "medium", "description": "Female US English"},
        {"name": "en_US-ryan-medium", "quality": "medium", "description": "Male US English"},
        {"name": "en_GB-alan-medium", "quality": "medium", "description": "British English"},
    ]

    return {
        "installed": available,
        "suggested": suggested,
        "models_dir": str(voices_dir),
    }


# ---------------------------------------------------------------------------
# WebSocket — Real-time Voice Streaming
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def voice_websocket(websocket: WebSocket):
    """
    Real-time bidirectional voice streaming.

    Protocol:
    ─────────────────────────────────────────────
    Client → Server messages:
      { "type": "audio_chunk", "data": "<base64 PCM>", "sample_rate": 16000 }
      { "type": "audio_end" }               — Signals end of utterance
      { "type": "config", "language": "en" } — Update settings
      { "type": "ping" }

    Server → Client messages:
      { "type": "transcript", "text": "...", "final": true }
      { "type": "response", "text": "...", "audio": "<base64 WAV>" }
      { "type": "status", "state": "listening|processing|speaking" }
      { "type": "error", "message": "..." }
      { "type": "pong" }
    ─────────────────────────────────────────────
    """
    from src.services.voice_service import voice_service
    from src.intelligence.reasoner import get_reasoning_engine
    from src.services.voice_memory import store_voice_turn

    await websocket.accept()
    session_id = f"ws-{uuid4().hex[:12]}"
    logger.info(f"Voice WebSocket connected [session={session_id}]")

    # Session state
    audio_buffer = bytearray()
    language = "en"
    sample_rate = 16000

    try:
        # Ensure voice service is initialized
        if not voice_service._initialized:
            await websocket.send_json({"type": "status", "state": "initializing"})
            await voice_service.initialize()

        await websocket.send_json({"type": "status", "state": "listening"})

        while True:
            # Receive message
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── Ping/Pong ──
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            # ── Config update ──
            elif msg_type == "config":
                language = msg.get("language", language)
                sample_rate = msg.get("sample_rate", sample_rate)
                await websocket.send_json({
                    "type": "config_ack",
                    "language": language,
                    "sample_rate": sample_rate,
                })

            # ── Audio chunk received ──
            elif msg_type == "audio_chunk":
                chunk = base64.b64decode(msg["data"])
                audio_buffer.extend(chunk)

            # ── End of utterance — process full pipeline ──
            elif msg_type == "audio_end":
                if len(audio_buffer) < 100:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Audio too short",
                    })
                    audio_buffer.clear()
                    await websocket.send_json({"type": "status", "state": "listening"})
                    continue

                # Signal processing state
                await websocket.send_json({"type": "status", "state": "processing"})

                try:
                    # 1. STT
                    transcript = await voice_service.transcribe(
                        bytes(audio_buffer), sample_rate, language
                    )

                    user_text = transcript["text"]
                    await websocket.send_json({
                        "type": "transcript",
                        "text": user_text,
                        "final": True,
                        "language": transcript["language"],
                        "confidence": transcript["language_probability"],
                    })

                    if not user_text.strip():
                        await websocket.send_json({
                            "type": "error",
                            "message": "No speech detected",
                        })
                        audio_buffer.clear()
                        await websocket.send_json({"type": "status", "state": "listening"})
                        continue

                    # 2. Echo Brain reasoning
                    reasoner = get_reasoning_engine()
                    chat_result = await reasoner.process(user_text)
                    response_text = chat_result.response

                    # 3. TTS
                    await websocket.send_json({"type": "status", "state": "speaking"})
                    tts_start = time.time()
                    response_audio = await voice_service.synthesize(response_text)
                    tts_ms = int((time.time() - tts_start) * 1000)

                    # 4. Send response (text + audio + reasoning metadata)
                    stt_ms = int(transcript["transcription_time_s"] * 1000)
                    chat_ms = chat_result.execution_time_ms or 0
                    total_ms = stt_ms + chat_ms + tts_ms
                    await websocket.send_json({
                        "type": "response",
                        "text": response_text,
                        "audio": base64.b64encode(response_audio).decode(),
                        "query_type": chat_result.query_type.value,
                        "confidence": chat_result.confidence,
                        "sources": chat_result.sources,
                        "actions_taken": chat_result.actions_taken,
                        "execution_time_ms": chat_result.execution_time_ms,
                        "audio_bytes_received": len(audio_buffer),
                        "timings": {
                            "stt_ms": stt_ms,
                            "chat_ms": chat_result.execution_time_ms,
                            "tts_ms": tts_ms,
                            "total_ms": total_ms,
                        },
                    })

                    # 5. Persist to Echo Brain memory (fire-and-forget)
                    asyncio.create_task(store_voice_turn(
                        session_id=session_id,
                        user_text=user_text,
                        response_text=response_text,
                        metadata={
                            "query_type": getattr(chat_result, "query_type", None)
                                and chat_result.query_type.value,
                            "confidence": getattr(chat_result, "confidence", None),
                            "sources": getattr(chat_result, "sources", None),
                            "stt_time_ms": int(transcript["transcription_time_s"] * 1000),
                        },
                    ))

                except Exception as e:
                    logger.error(f"Voice pipeline error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })

                # Reset buffer for next utterance
                audio_buffer.clear()
                await websocket.send_json({"type": "status", "state": "listening"})

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass