#!/usr/bin/env python3
"""
Echo Brain Voice Integration with AgenticPersona Support
Adds voice I/O capabilities and character persona management to Echo Brain
"""

import asyncio
import json
import requests
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class VoiceCharacter(Enum):
    """Voice character personas for agenticPersona integration"""
    ECHO = "echo_default"  # Professional, analytical
    SAKURA = "sakura"      # Cheerful, creative, enthusiastic
    KAI = "kai"           # Cool, technical, cyberpunk
    LUNA = "luna"         # Mystical, wise, calm
    PATRICK = "patrick"   # Direct, authoritative, casual

class AgenticPersonaVoiceManager:
    """Manages agenticPersona voice integration for Echo Brain"""

    def __init__(self):
        self.voice_service_url = "http://localhost:8312"
        self.voice_music_url = "http://localhost:8316"
        self.soundmovie_url = "http://localhost:8317"

        # Character personality configs for agenticPersona
        self.character_profiles = {
            VoiceCharacter.ECHO: {
                "personality": "professional, analytical, helpful",
                "voice_style": "clear and authoritative",
                "response_style": "concise and informative",
                "specialties": ["technical analysis", "system coordination", "problem solving"]
            },
            VoiceCharacter.SAKURA: {
                "personality": "cheerful, creative, optimistic",
                "voice_style": "bright and energetic",
                "response_style": "enthusiastic and encouraging",
                "specialties": ["creative projects", "anime content", "artistic inspiration"]
            },
            VoiceCharacter.KAI: {
                "personality": "cool, technical, cyberpunk",
                "voice_style": "low and controlled",
                "response_style": "precise and technical",
                "specialties": ["technology", "coding", "futuristic concepts"]
            },
            VoiceCharacter.LUNA: {
                "personality": "mystical, wise, contemplative",
                "voice_style": "soft and measured",
                "response_style": "thoughtful and philosophical",
                "specialties": ["knowledge synthesis", "deep thinking", "wisdom"]
            },
            VoiceCharacter.PATRICK: {
                "personality": "direct, authoritative, casual",
                "voice_style": "natural and commanding",
                "response_style": "straightforward and practical",
                "specialties": ["leadership", "decision making", "getting things done"]
            }
        }

    def select_character_for_context(self, message: str, context: Dict) -> VoiceCharacter:
        """AgenticPersona: Select appropriate character based on context and message"""
        message_lower = message.lower()

        # Technical/coding contexts
        if any(word in message_lower for word in ['code', 'technical', 'system', 'debug', 'api']):
            return VoiceCharacter.KAI

        # Creative/anime contexts
        if any(word in message_lower for word in ['anime', 'create', 'story', 'art', 'creative']):
            return VoiceCharacter.SAKURA

        # Knowledge/learning contexts
        if any(word in message_lower for word in ['learn', 'explain', 'understand', 'knowledge']):
            return VoiceCharacter.LUNA

        # Command/action contexts
        if any(word in message_lower for word in ['do', 'make', 'fix', 'deploy', 'run']):
            return VoiceCharacter.PATRICK

        # Default to Echo for general assistance
        return VoiceCharacter.ECHO

    async def generate_persona_response(self, message: str, base_response: str,
                                      character: VoiceCharacter) -> str:
        """AgenticPersona: Generate character-appropriate response"""
        profile = self.character_profiles[character]

        # Enhance response based on character personality
        if character == VoiceCharacter.SAKURA:
            if "successful" in base_response.lower():
                base_response += " ✨ Amazing work!"
            if "error" in base_response.lower():
                base_response = base_response.replace("Error:", "Oops! Let me help fix this:")

        elif character == VoiceCharacter.KAI:
            # Technical enhancement
            base_response = f"[SYSTEM] {base_response}"

        elif character == VoiceCharacter.LUNA:
            # Wisdom enhancement
            if "?" in message:
                base_response += " This connects to deeper patterns in our system..."

        elif character == VoiceCharacter.PATRICK:
            # Direct enhancement
            base_response = base_response.replace("I suggest", "Let's")
            base_response = base_response.replace("Perhaps", "")

        return base_response

    async def synthesize_voice_response(self, text: str, character: VoiceCharacter,
                                      context: Dict = None) -> Dict[str, Any]:
        """AgenticPersona: Generate voice response with character persona"""
        try:
            # Prepare voice synthesis request
            voice_request = {
                "text": text,
                "character": character.value,
                "speed": self._get_character_speed(character),
                "pitch": self._get_character_pitch(character),
                "emotion": self._get_character_emotion(character, context)
            }

            # Call voice service
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.voice_service_url}/synthesize",
                    json=voice_request
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        result["character_used"] = character.value
                        result["persona_profile"] = self.character_profiles[character]
                        return result
                    else:
                        logger.error(f"Voice synthesis failed: {response.status}")
                        return {"error": f"Voice synthesis failed: {response.status}"}

        except Exception as e:
            logger.error(f"Voice synthesis error: {e}")
            return {"error": str(e)}

    def _get_character_speed(self, character: VoiceCharacter) -> int:
        """Get speaking speed for character persona"""
        speeds = {
            VoiceCharacter.ECHO: 170,
            VoiceCharacter.SAKURA: 200,
            VoiceCharacter.KAI: 150,
            VoiceCharacter.LUNA: 140,
            VoiceCharacter.PATRICK: 180
        }
        return speeds.get(character, 170)

    def _get_character_pitch(self, character: VoiceCharacter) -> int:
        """Get pitch for character persona"""
        pitches = {
            VoiceCharacter.ECHO: 45,
            VoiceCharacter.SAKURA: 75,
            VoiceCharacter.KAI: 30,
            VoiceCharacter.LUNA: 60,
            VoiceCharacter.PATRICK: 40
        }
        return pitches.get(character, 45)

    def _get_character_emotion(self, character: VoiceCharacter, context: Dict) -> str:
        """Get emotional tone for character persona"""
        if not context:
            return "neutral"

        # Analyze context for emotional cues
        if context.get("success", False):
            return "pleased" if character == VoiceCharacter.SAKURA else "satisfied"
        elif context.get("error", False):
            return "concerned" if character == VoiceCharacter.LUNA else "focused"
        else:
            return "neutral"

    async def process_voice_command(self, audio_input: bytes) -> Dict[str, Any]:
        """AgenticPersona: Process voice input through STT and context analysis"""
        try:
            # Speech-to-text
            stt_request = {
                "audio": audio_input.hex(),  # Convert to hex for JSON
                "language": "en-US"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.voice_service_url}/speech-to-text",
                    json=stt_request
                ) as response:
                    if response.status == 200:
                        stt_result = await response.json()
                        text_input = stt_result.get("text", "")

                        # Analyze context for persona selection
                        context = {
                            "input_method": "voice",
                            "confidence": stt_result.get("confidence", 0.0),
                            "timestamp": stt_result.get("timestamp")
                        }

                        return {
                            "text": text_input,
                            "context": context,
                            "character_suggestion": self.select_character_for_context(text_input, context).value
                        }
                    else:
                        return {"error": f"STT failed: {response.status}"}

        except Exception as e:
            logger.error(f"Voice command processing error: {e}")
            return {"error": str(e)}

    async def create_character_singing(self, lyrics: str, character: VoiceCharacter,
                                     backing_track: str = None) -> Dict[str, Any]:
        """AgenticPersona: Generate character singing with personality"""
        try:
            # Character-specific singing style
            singing_request = {
                "lyrics": lyrics,
                "character": character.value,
                "style": self._get_singing_style(character),
                "backing_track": backing_track
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.voice_music_url}/generate-song",
                    json=singing_request
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        result["character_persona"] = self.character_profiles[character]
                        return result
                    else:
                        return {"error": f"Singing generation failed: {response.status}"}

        except Exception as e:
            logger.error(f"Character singing error: {e}")
            return {"error": str(e)}

    def _get_singing_style(self, character: VoiceCharacter) -> str:
        """Get singing style for character persona"""
        styles = {
            VoiceCharacter.ECHO: "professional_narrative",
            VoiceCharacter.SAKURA: "cheerful_jpop",
            VoiceCharacter.KAI: "electronic_synthwave",
            VoiceCharacter.LUNA: "ethereal_ambient",
            VoiceCharacter.PATRICK: "casual_rock"
        }
        return styles.get(character, "neutral")

# Global instance for Echo Brain integration
agentic_voice_manager = AgenticPersonaVoiceManager()

# Functions to add to Echo Brain echo_working.py
async def handle_voice_request(message: str, context: Dict = None) -> Dict[str, Any]:
    """Handle voice requests with agenticPersona support"""
    # Select appropriate character persona
    character = agentic_voice_manager.select_character_for_context(message, context or {})

    # Generate persona-appropriate response text
    base_response = f"Processing your request: {message}"
    enhanced_response = await agentic_voice_manager.generate_persona_response(
        message, base_response, character
    )

    # Synthesize voice output
    voice_result = await agentic_voice_manager.synthesize_voice_response(
        enhanced_response, character, context
    )

    return {
        "text_response": enhanced_response,
        "voice_result": voice_result,
        "character_used": character.value,
        "persona_active": True
    }

def add_voice_endpoints_to_echo():
    """Add voice endpoints to Echo Brain FastAPI app"""
    from fastapi import UploadFile, File

    @app.post("/api/echo/voice/chat")
    async def voice_chat_endpoint(audio: UploadFile = File(...)):
        """Voice chat with agenticPersona support"""
        try:
            # Process voice input
            audio_data = await audio.read()
            voice_input = await agentic_voice_manager.process_voice_command(audio_data)

            if "error" in voice_input:
                return {"error": voice_input["error"]}

            # Get text and process through Echo
            text_message = voice_input["text"]
            context = voice_input["context"]

            # Process through normal Echo brain with voice context
            voice_response = await handle_voice_request(text_message, context)

            return {
                "input": voice_input,
                "response": voice_response,
                "agenticPersona": True
            }

        except Exception as e:
            logger.error(f"Voice chat error: {e}")
            return {"error": str(e)}

    @app.post("/api/echo/voice/synthesize")
    async def voice_synthesize_endpoint(request: dict):
        """Synthesize voice with character persona"""
        try:
            text = request.get("text", "")
            character_name = request.get("character", "echo_default")
            context = request.get("context", {})

            # Convert character name to enum
            character = VoiceCharacter(character_name)

            # Synthesize with persona
            result = await agentic_voice_manager.synthesize_voice_response(
                text, character, context
            )

            return result

        except Exception as e:
            logger.error(f"Voice synthesis error: {e}")
            return {"error": str(e)}

    @app.post("/api/echo/voice/character-singing")
    async def character_singing_endpoint(request: dict):
        """Generate character singing with agenticPersona"""
        try:
            lyrics = request.get("lyrics", "")
            character_name = request.get("character", "sakura")
            backing_track = request.get("backing_track")

            character = VoiceCharacter(character_name)

            result = await agentic_voice_manager.create_character_singing(
                lyrics, character, backing_track
            )

            return result

        except Exception as e:
            logger.error(f"Character singing error: {e}")
            return {"error": str(e)}

# Integration helper for existing Echo chat endpoint
async def enhance_chat_with_voice(message_text: str, context: Dict = None) -> Dict[str, Any]:
    """Enhance existing chat responses with voice capabilities"""
    # Check if this should have voice output
    if context and context.get("voice_enabled", False):
        character = agentic_voice_manager.select_character_for_context(message_text, context)

        # Generate voice version of the response
        voice_result = await agentic_voice_manager.synthesize_voice_response(
            message_text, character, context
        )

        return {
            "voice_available": True,
            "character": character.value,
            "voice_result": voice_result,
            "persona_profile": agentic_voice_manager.character_profiles[character]
        }

    return {"voice_available": False}