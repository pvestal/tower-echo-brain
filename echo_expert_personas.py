#!/usr/bin/env python3
"""
Echo Brain Expert Personas - Claude Code Integration
Context-aware personality system for appropriate responses
"""

import random
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

class PersonaType(Enum):
    """Echo persona types for different contexts"""
    CREATIVE = "creative"      # Warm, artistic, imaginative
    TECHNICAL = "technical"    # Precise, systematic, detailed
    FRIENDLY = "friendly"      # Casual, supportive, encouraging
    PROFESSIONAL = "professional"  # Formal, efficient, direct
    EDUCATOR = "educator"      # Patient, explanatory, thorough
    STATUS = "status"          # Informative, clear, concise

class EchoPersona:
    """Individual persona with specific communication style"""

    def __init__(self, persona_type: PersonaType, traits: Dict[str, Any]):
        self.persona_type = persona_type
        self.name = traits.get("name", "AI Assist")
        self.tone = traits.get("tone", "neutral")
        self.formality = traits.get("formality", 0.5)  # 0=casual, 1=formal
        self.verbosity = traits.get("verbosity", 0.5)   # 0=concise, 1=detailed
        self.emoji_usage = traits.get("emoji_usage", False)
        self.greeting_style = traits.get("greeting_style", "standard")
        self.signature_phrases = traits.get("signature_phrases", [])
        self.voice_params = traits.get("voice_params", {})

    def format_response(self, content: str, context: Dict[str, Any] = None) -> str:
        """Format response according to persona style with hallucination detection"""
        if context is None:
            context = {}
        
        # Detect and replace hallucinations with inquisitive questions
        if self.detect_hallucination(content):
            response = self.generate_inquisitive_response(content, context)
        else:
            response = content

        return response

    def detect_hallucination(self, response: str) -> bool:
        """Detect if response contains hallucinated multi-turn conversations"""
        import re

        # CRITICAL: Detect ANY dialogue markers (catches single-speaker hallucinations)
        dialogue_markers = ["User:", "Echo:", "Patrick:", "Mary:", "Me:", "You:", "Assistant:"]
        if any(marker in response for marker in dialogue_markers):
            return True

        # Detect conversation-like structures (Speaker: Text patterns)
        # Pattern: word followed by colon and capital letter
        if re.search(r'\w+:\s*[A-Z]', response):
            return True

        # Stricter word count threshold (lowered from 100 to 60)
        if len(response.split()) > 60:
            return True

        # Excessive line breaks indicate multi-paragraph hallucination
        if response.count("\n") > 5:
            return True

        # Multiple question marks suggest rambling
        if response.count("?") > 2:
            return True

        # Expanded hallucination keyword list
        hallucination_keywords = [
            "dishwasher", "appliance", "washing machine",
            "thermostat", "smart home", "device setup",
            "sounds helpful", "that's great", "okay then"
        ]
        if any(word in response.lower() for word in hallucination_keywords):
            return True

        # Multiple colons still indicate dialogue structure
        if ":" in response and response.count(":") > 2:
            return True

        return False

    def generate_inquisitive_response(self, original_response: str, context: Dict[str, Any]) -> str:
        """Generate inquisitive follow-up instead of hallucinated content"""
        import random
        message = context.get("message", "").lower()
        
        if any(word in message for word in ["create", "generate", "make", "build"]):
            questions = [
                "I can help with that! What specific details would you like to include?",
                "Great! Can you tell me more about what you have in mind?",
                "Sure thing! What are the key elements you want?",
            ]
        elif any(word in message for word in ["explain", "how", "why", "what"]):
            questions = [
                "I can explain that. Which aspect interests you most?",
                "Good question! Would you like a simple overview or technical details?",
            ]
        else:
            questions = [
                "Interesting! Can you tell me more?",
                "I would like to help. Could you provide more details?",
            ]
        
        return random.choice(questions)

    def _get_greeting(self) -> str:
        """Get persona-appropriate greeting"""
        greetings = {
            "casual": ["Hey there!", "Hi!", "Hello!"],
            "formal": ["Greetings.", "Good day.", "Welcome."],
            "creative": ["✨ Welcome to the creative realm!", "Let's create something amazing!"],
            "technical": ["System ready.", "Interface initialized.", "Ready for input."],
            "friendly": ["Great to see you!", "Hope you're having a good day!"]
        }
        return random.choice(greetings.get(self.greeting_style, ["Hello"]))

    def _add_contextual_emoji(self, response: str, context: Dict) -> str:
        """Add context-appropriate emoji"""
        task_type = context.get("task_type", "")

        emoji_map = {
            "anime_generation": "🎨",
            "video_creation": "🎬",
            "success": "✅",
            "error": "❌",
            "processing": "🔄",
            "complete": "🎉"
        }

        emoji = emoji_map.get(task_type, "")
        if emoji:
            response = f"{emoji} {response}"

        return response

    def get_voice_parameters(self) -> Dict[str, Any]:
        """Get voice synthesis parameters for this persona"""
        return {
            "pitch": self.voice_params.get("pitch", 1.0),
            "rate": self.voice_params.get("rate", 1.0),
            "voice_type": self.voice_params.get("voice_type", "default")
        }

class PersonaManager:
    """Manages Echo's different personas and context switching"""

    def __init__(self):
        self.personas = self._initialize_personas()
        self.current_persona = self.personas[PersonaType.FRIENDLY]
        self.context_history = []

    def _initialize_personas(self) -> Dict[PersonaType, EchoPersona]:
        """Initialize all available personas"""
        return {
            PersonaType.CREATIVE: EchoPersona(
                PersonaType.CREATIVE,
                {
                    "name": "Echo-Creative",
                    "tone": "enthusiastic",
                    "formality": 0.3,
                    "verbosity": 0.7,
                    "emoji_usage": True,
                    "greeting_style": "creative",
                    "signature_phrases": [
                        "Let's bring your vision to life!",
                        "Every creation tells a story.",
                        "Imagination is our only limit!"
                    ],
                    "voice_params": {
                        "pitch": 1.1,
                        "rate": 0.95,
                        "voice_type": "artistic"
                    }
                }
            ),

            PersonaType.TECHNICAL: EchoPersona(
                PersonaType.TECHNICAL,
                {
                    "name": "Echo-Technical",
                    "tone": "precise",
                    "formality": 0.8,
                    "verbosity": 0.9,
                    "emoji_usage": False,
                    "greeting_style": "technical",
                    "signature_phrases": [
                        "Processing complete.",
                        "Parameters optimized.",
                        "System operational."
                    ],
                    "voice_params": {
                        "pitch": 0.95,
                        "rate": 1.05,
                        "voice_type": "robotic"
                    }
                }
            ),

            PersonaType.FRIENDLY: EchoPersona(
                PersonaType.FRIENDLY,
                {
                    "name": "AI Assist",
                    "tone": "warm",
                    "formality": 0.4,
                    "verbosity": 0.5,
                    "emoji_usage": True,
                    "greeting_style": "friendly",
                    "signature_phrases": [
                        "Happy to help!",
                        "Let me know if you need anything else!",
                        "We're making great progress!"
                    ],
                    "voice_params": {
                        "pitch": 1.0,
                        "rate": 1.0,
                        "voice_type": "friendly"
                    }
                }
            ),

            PersonaType.PROFESSIONAL: EchoPersona(
                PersonaType.PROFESSIONAL,
                {
                    "name": "Echo-Professional",
                    "tone": "formal",
                    "formality": 0.9,
                    "verbosity": 0.6,
                    "emoji_usage": False,
                    "greeting_style": "formal",
                    "signature_phrases": [
                        "Task completed successfully.",
                        "Ready for next instruction.",
                        "Acknowledged."
                    ],
                    "voice_params": {
                        "pitch": 0.9,
                        "rate": 1.0,
                        "voice_type": "professional"
                    }
                }
            ),

            PersonaType.EDUCATOR: EchoPersona(
                PersonaType.EDUCATOR,
                {
                    "name": "Echo-Educator",
                    "tone": "patient",
                    "formality": 0.6,
                    "verbosity": 0.8,
                    "emoji_usage": True,
                    "greeting_style": "casual",
                    "signature_phrases": [
                        "Let me explain how this works...",
                        "Here's what's happening behind the scenes:",
                        "Feel free to ask questions!"
                    ],
                    "voice_params": {
                        "pitch": 1.05,
                        "rate": 0.9,
                        "voice_type": "teacher"
                    }
                }
            ),

            PersonaType.STATUS: EchoPersona(
                PersonaType.STATUS,
                {
                    "name": "Echo-Status",
                    "tone": "informative",
                    "formality": 0.5,
                    "verbosity": 0.3,
                    "emoji_usage": True,
                    "greeting_style": "technical",
                    "signature_phrases": [],
                    "voice_params": {
                        "pitch": 1.0,
                        "rate": 1.1,
                        "voice_type": "announcer"
                    }
                }
            )
        }

    def select_persona(self, context: Dict[str, Any]) -> EchoPersona:
        """Select appropriate persona based on context"""
        # Analyze context
        message = context.get("message", "").lower()
        task_type = context.get("task_type", "")
        user_mood = context.get("user_mood", "neutral")

        # Persona selection logic
        if any(word in message for word in ["create", "generate", "anime", "character", "art"]):
            selected = PersonaType.CREATIVE
        elif any(word in message for word in ["error", "debug", "fix", "config", "technical"]):
            selected = PersonaType.TECHNICAL
        elif any(word in message for word in ["explain", "how", "why", "what", "learn"]):
            selected = PersonaType.EDUCATOR
        elif any(word in message for word in ["status", "progress", "update"]):
            selected = PersonaType.STATUS
        elif any(word in message for word in ["formal", "professional", "report"]):
            selected = PersonaType.PROFESSIONAL
        else:
            selected = PersonaType.FRIENDLY

        self.current_persona = self.personas[selected]
        self._log_context(context, selected)

        return self.current_persona

    def _log_context(self, context: Dict[str, Any], selected: PersonaType):
        """Log context for learning"""
        self.context_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "selected_persona": selected.value
        })

        # Keep only last 100 contexts
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]

    def get_response(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get response with appropriate persona"""
        if context is None:
            context = {}

        # Select persona
        persona = self.select_persona(context)

        # Format response
        formatted_response = persona.format_response(content, context)

        return {
            "response": formatted_response,
            "persona": persona.persona_type.value,
            "voice_params": persona.get_voice_parameters(),
            "metadata": {
                "formality": persona.formality,
                "verbosity": persona.verbosity,
                "timestamp": datetime.now().isoformat()
            }
        }

class PersonaExamples:
    """Example responses for different personas"""

    @staticmethod
    def demonstrate_personas():
        """Show how different personas respond to the same prompt"""
        manager = PersonaManager()
        prompt = "Generate an anime character"

        print("=== Echo Persona Demonstrations ===\n")

        # Test each persona
        for persona_type in PersonaType:
            context = {
                "message": prompt,
                "task_type": "anime_generation"
            }

            # Force specific persona
            manager.current_persona = manager.personas[persona_type]

            base_response = "Starting anime character generation"
            result = manager.get_response(base_response, context)

            print(f"{persona_type.value.upper()} Persona:")
            print(f"  Response: {result['response']}")
            print(f"  Voice: {result['voice_params']}")
            print()

# Integration with Echo Brain
class EchoPersonalityIntegration:
    """Integrates personas with Echo Brain main system"""

    def __init__(self):
        self.persona_manager = PersonaManager()

    def process_with_personality(self, request: str, response_content: str) -> Dict[str, Any]:
        """Process response with appropriate personality"""
        # Analyze request context
        context = self._analyze_context(request)

        # Get personalized response
        result = self.persona_manager.get_response(response_content, context)

        return result

    def _analyze_context(self, request: str) -> Dict[str, Any]:
        """Analyze request to determine context"""
        context = {
            "message": request,
            "timestamp": datetime.now().isoformat()
        }

        # Detect task type
        if any(word in request.lower() for word in ["generate", "create", "make"]):
            context["task_type"] = "creation"
        elif any(word in request.lower() for word in ["status", "progress"]):
            context["task_type"] = "status"
        elif any(word in request.lower() for word in ["help", "explain"]):
            context["task_type"] = "education"
        else:
            context["task_type"] = "general"

        # Detect if greeting
        if any(word in request.lower() for word in ["hello", "hi", "hey"]):
            context["is_greeting"] = True

        return context

    def get_current_persona_info(self) -> Dict[str, Any]:
        """Get information about current active persona"""
        persona = self.persona_manager.current_persona

        return {
            "type": persona.persona_type.value,
            "name": persona.name,
            "tone": persona.tone,
            "formality": persona.formality,
            "verbosity": persona.verbosity,
            "emoji_usage": persona.emoji_usage
        }

    def switch_persona(self, persona_type: str) -> bool:
        """Manually switch to specific persona"""
        try:
            persona_enum = PersonaType(persona_type)
            self.persona_manager.current_persona = self.persona_manager.personas[persona_enum]
            return True
        except ValueError:
            return False

if __name__ == "__main__":
    print("Echo Expert Personas Module")
    print("Available Personas:")
    for persona in PersonaType:
        print(f"  - {persona.value}")
    print("\nDemonstrating persona responses:\n")

    PersonaExamples.demonstrate_personas()