"""
Echo Brain Voice Service
========================
Handles Speech-to-Text (STT) via faster-whisper and Text-to-Speech (TTS) via Piper.
Both run locally on Tower's GPU — no cloud APIs, no data leaving the network.

Architecture:
  Browser Mic → WebSocket → faster-whisper (STT) → Echo Brain Chat → Piper (TTS) → WebSocket → Browser Speaker

Dependencies (install in /opt/tower-echo-brain/venv):
  pip install faster-whisper piper-tts webrtcvad numpy

Voice models stored at: /opt/tower-echo-brain/models/voice/
"""

import asyncio
import io
import logging
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("echo.voice")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOICE_MODELS_DIR = Path("/opt/tower-echo-brain/models/voice")
VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# STT config — faster-whisper
STT_MODEL_SIZE = "large-v3"        # Best accuracy - we have the GPU power
STT_DEVICE = "cuda"                # Primary: NVIDIA RTX 3060 (12GB VRAM)
STT_COMPUTE_TYPE = "float16"       # FP16 for GPU efficiency
STT_DEVICE_INDEX = 0               # RTX 3060 is usually device 0
STT_FALLBACK_DEVICE = "auto"       # Will try AMD if NVIDIA fails
STT_BEAM_SIZE = 5                  # Default beam search width
STT_LANGUAGE = "en"                # Primary language

# TTS config — Piper
TTS_MODEL_NAME = "en_US-lessac-medium"   # Natural-sounding US English voice
TTS_SAMPLE_RATE = 22050                   # Piper default sample rate
TTS_SPEAKER_ID = None                     # None = default speaker
TTS_LENGTH_SCALE = 1.0                    # Speech speed (1.0 = normal)

# Audio processing
SILENCE_THRESHOLD_DB = -40         # dB threshold for voice activity
CHUNK_DURATION_MS = 30             # VAD frame duration (10, 20, or 30 ms)
SAMPLE_RATE = 16000                # Input audio sample rate for STT


class VoiceService:
    """
    Manages STT and TTS models with lazy initialization.
    Models are loaded once on first use and kept in memory.

    Think of this like a translator who sits in the room —
    they don't need to re-learn the language each time you speak.
    """

    def __init__(self):
        self._stt_model = None
        self._tts_voice = None
        self._vad = None
        self._lock = asyncio.Lock()
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self):
        """Load models into GPU memory. Call once at startup."""
        async with self._lock:
            if self._initialized:
                return

            logger.info("Initializing Voice Service...")
            start = time.time()

            # Load STT model
            await self._load_stt_model()

            # Load TTS model
            await self._load_tts_model()

            # Initialize VAD (Voice Activity Detection)
            self._init_vad()

            elapsed = time.time() - start
            logger.info(f"Voice Service initialized in {elapsed:.1f}s")
            self._initialized = True

    async def _load_stt_model(self):
        """Load faster-whisper model onto GPU with fallback."""
        from faster_whisper import WhisperModel

        # Try NVIDIA first with int8_float16
        try:
            logger.info(f"Loading STT model: {STT_MODEL_SIZE} on NVIDIA GPU (cuda)")
            loop = asyncio.get_event_loop()
            self._stt_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    STT_MODEL_SIZE,
                    device="cuda",
                    device_index=0,  # RTX 3060
                    compute_type="int8_float16",  # Best for RTX 3060
                    download_root=str(VOICE_MODELS_DIR / "whisper"),
                )
            )
            logger.info("✅ STT model loaded on NVIDIA RTX 3060")
            return
        except Exception as e:
            logger.warning(f"NVIDIA GPU with int8_float16 failed: {e}, trying int8...")

        # Try NVIDIA with int8
        try:
            logger.info(f"Loading STT model: {STT_MODEL_SIZE} on NVIDIA GPU with int8")
            loop = asyncio.get_event_loop()
            self._stt_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    STT_MODEL_SIZE,
                    device="cuda",
                    device_index=0,  # RTX 3060
                    compute_type="int8",
                    download_root=str(VOICE_MODELS_DIR / "whisper"),
                )
            )
            logger.info("✅ STT model loaded on NVIDIA RTX 3060 (int8)")
            return
        except Exception as e:
            logger.warning(f"NVIDIA GPU failed completely: {e}, trying AMD...")

        # Try AMD ROCm
        try:
            logger.info(f"Loading STT model: {STT_MODEL_SIZE} on AMD GPU")
            loop = asyncio.get_event_loop()
            self._stt_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    STT_MODEL_SIZE,
                    device="auto",  # Will detect available device
                    compute_type="default",  # Let it choose
                    download_root=str(VOICE_MODELS_DIR / "whisper"),
                )
            )
            logger.info("✅ STT model loaded on AMD RX 9070 XT")
            return
        except Exception as e:
            logger.error(f"Both GPUs failed: {e}")
            raise RuntimeError("No GPU available for STT model. CPU mode is disabled.")

    async def _load_tts_model(self):
        """Load Piper TTS voice model."""
        try:
            from piper.voice import PiperVoice

            model_path = VOICE_MODELS_DIR / "piper" / f"{TTS_MODEL_NAME}.onnx"

            if not model_path.exists():
                logger.info(f"TTS model not found at {model_path}, downloading...")
                await self._download_piper_model(TTS_MODEL_NAME)

            logger.info(f"Loading TTS model: {TTS_MODEL_NAME}")
            loop = asyncio.get_event_loop()
            self._tts_voice = await loop.run_in_executor(
                None,
                lambda: PiperVoice.load(str(model_path))
            )
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    async def _download_piper_model(self, model_name: str):
        """Download a Piper voice model from HuggingFace."""

        piper_dir = VOICE_MODELS_DIR / "piper"
        piper_dir.mkdir(parents=True, exist_ok=True)

        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
        # Parse model name: en_US-lessac-medium → en/en_US/lessac/medium/
        parts = model_name.split("-")
        lang = parts[0][:2]          # en
        locale = parts[0]            # en_US
        name = parts[1]              # lessac
        quality = parts[2]           # medium

        model_url = f"{base_url}/{lang}/{locale}/{name}/{quality}/{model_name}.onnx"
        config_url = f"{model_url}.json"

        for url, filename in [(model_url, f"{model_name}.onnx"),
                              (config_url, f"{model_name}.onnx.json")]:
            target = piper_dir / filename
            if not target.exists():
                logger.info(f"Downloading {filename}...")
                proc = await asyncio.create_subprocess_exec(
                    "wget", "-q", "-O", str(target), url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Failed to download {url}")

    def _init_vad(self):
        """Initialize Voice Activity Detection for smart silence trimming."""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(2)  # Aggressiveness 0-3 (2 = balanced)
            logger.info("VAD initialized")
        except ImportError:
            logger.warning("webrtcvad not installed — VAD disabled")
            self._vad = None

    # ------------------------------------------------------------------
    # Speech-to-Text
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = SAMPLE_RATE,
        language: str = STT_LANGUAGE,
    ) -> dict:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw PCM audio (16-bit, mono) or WAV file bytes
            sample_rate: Audio sample rate (default 16kHz)
            language: Language code for transcription

        Returns:
            dict with keys: text, language, confidence, duration_s, segments
        """
        if not self._stt_model:
            await self.initialize()

        start = time.time()

        # Convert raw PCM to WAV if needed
        audio_file = self._prepare_audio(audio_bytes, sample_rate)

        # Run transcription in executor (CPU-bound despite GPU — GIL)
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._stt_model.transcribe(
                audio_file,
                beam_size=STT_BEAM_SIZE,
                language=language,
                vad_filter=True,           # Filter out non-speech
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )
        )

        # Collect all segments
        result_segments = []
        full_text_parts = []

        for segment in segments:
            result_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            })
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)
        elapsed = time.time() - start

        result = {
            "text": full_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_s": round(info.duration, 2),
            "transcription_time_s": round(elapsed, 3),
            "segments": result_segments,
        }

        logger.info(
            f"Transcribed {info.duration:.1f}s audio → "
            f"{len(full_text)} chars in {elapsed:.2f}s"
        )
        return result

    def _prepare_audio(self, audio_bytes: bytes, sample_rate: int) -> str:
        """
        Convert raw audio bytes to a temporary WAV file for faster-whisper.
        If already WAV format (starts with RIFF header), save directly.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        if audio_bytes[:4] == b"RIFF":
            # Already WAV format
            tmp.write(audio_bytes)
        else:
            # Raw PCM — wrap in WAV container
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

        tmp.close()
        return tmp.name

    # ------------------------------------------------------------------
    # Text-to-Speech
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = TTS_SPEAKER_ID,
        length_scale: float = TTS_LENGTH_SCALE,
        output_format: str = "wav",
    ) -> bytes:
        """
        Convert text to speech audio.

        Args:
            text: Text to speak
            speaker_id: Voice speaker ID (for multi-speaker models)
            length_scale: Speed multiplier (< 1.0 = faster, > 1.0 = slower)
            output_format: Output format ("wav" or "raw")

        Returns:
            Audio bytes (WAV format by default)
        """
        if not self._tts_voice:
            await self.initialize()

        start = time.time()

        # Synthesize in executor
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            lambda: self._synthesize_sync(text, speaker_id, length_scale)
        )

        elapsed = time.time() - start
        logger.info(
            f"Synthesized {len(text)} chars → "
            f"{len(audio_bytes)} bytes in {elapsed:.3f}s"
        )

        return audio_bytes

    def _synthesize_sync(
        self, text: str, speaker_id: Optional[int], length_scale: float
    ) -> bytes:
        """Synchronous Piper synthesis — runs in thread executor."""
        from piper.config import SynthesisConfig

        # Create synthesis config (only supports length_scale)
        syn_config = SynthesisConfig(
            length_scale=length_scale,
        )

        # Synthesize audio chunks
        audio_chunks = []
        for chunk in self._tts_voice.synthesize(text, syn_config):
            # chunk.audio_int16_array is the int16 numpy array
            audio_chunks.append(chunk.audio_int16_array)

        # Concatenate all chunks
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
        else:
            audio_data = np.array([], dtype=np.int16)

        # Wrap in WAV format
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self._tts_voice.config.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return buf.getvalue()

    # ------------------------------------------------------------------
    # Full Voice Chat Loop
    # ------------------------------------------------------------------

    async def voice_chat(
        self,
        audio_bytes: bytes,
        chat_handler,
        sample_rate: int = SAMPLE_RATE,
    ) -> dict:
        """
        Full voice interaction loop:
        1. Transcribe incoming audio (STT)
        2. Send text to Echo Brain chat handler
        3. Synthesize response (TTS)
        4. Return everything

        Args:
            audio_bytes: Raw audio from the client
            chat_handler: Async callable that takes a string query
                          and returns a response dict with at least 'response' key
            sample_rate: Audio sample rate

        Returns:
            dict with: transcript, response_text, audio_bytes, timings
        """
        timings = {}

        # Step 1: STT
        t0 = time.time()
        transcript = await self.transcribe(audio_bytes, sample_rate)
        timings["stt_ms"] = round((time.time() - t0) * 1000)

        user_text = transcript["text"]
        if not user_text.strip():
            return {
                "transcript": "",
                "response_text": "",
                "audio_bytes": b"",
                "timings": timings,
                "error": "No speech detected",
            }

        # Step 2: Chat
        t1 = time.time()
        chat_response = await chat_handler(user_text)
        response_text = chat_response.get("response", "I didn't understand that.")
        timings["chat_ms"] = round((time.time() - t1) * 1000)

        # Step 3: TTS
        t2 = time.time()
        response_audio = await self.synthesize(response_text)
        timings["tts_ms"] = round((time.time() - t2) * 1000)

        timings["total_ms"] = round((time.time() - t0) * 1000)

        return {
            "transcript": user_text,
            "response_text": response_text,
            "audio_bytes": response_audio,
            "timings": timings,
        }

    # ------------------------------------------------------------------
    # Voice Activity Detection
    # ------------------------------------------------------------------

    def detect_speech(self, audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        """
        Check if audio contains speech using WebRTC VAD.
        Useful for filtering silence before sending to STT.
        """
        if not self._vad:
            return True  # No VAD = assume speech

        frame_size = int(sample_rate * CHUNK_DURATION_MS / 1000) * 2  # 16-bit
        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio_bytes) - frame_size, frame_size):
            frame = audio_bytes[i:i + frame_size]
            if len(frame) == frame_size:
                total_frames += 1
                try:
                    if self._vad.is_speech(frame, sample_rate):
                        speech_frames += 1
                except Exception:
                    pass

        if total_frames == 0:
            return False

        speech_ratio = speech_frames / total_frames
        return speech_ratio > 0.1  # At least 10% speech frames

    # ------------------------------------------------------------------
    # Status & Health
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current voice service status."""
        return {
            "initialized": self._initialized,
            "stt": {
                "loaded": self._stt_model is not None,
                "model": STT_MODEL_SIZE,
                "device": STT_DEVICE,
                "compute_type": STT_COMPUTE_TYPE,
            },
            "tts": {
                "loaded": self._tts_voice is not None,
                "model": TTS_MODEL_NAME,
                "sample_rate": TTS_SAMPLE_RATE,
            },
            "vad": {
                "available": self._vad is not None,
            },
        }

    async def shutdown(self):
        """Release GPU memory."""
        self._stt_model = None
        self._tts_voice = None
        self._initialized = False
        logger.info("Voice service shut down")


# Singleton instance
voice_service = VoiceService()