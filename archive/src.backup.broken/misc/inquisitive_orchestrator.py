#!/usr/bin/env python3
"""
Inquisitive Orchestrator - Echo converses FIRST, generates LATER
The RIGHT way to handle creative projects
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class ConversationPhase(Enum):
    GATHERING = "gathering"      # Collecting initial info
    EXPLORING = "exploring"       # Asking deeper questions
    PROPOSING = "proposing"       # Suggesting options
    REFINING = "refining"        # Refining based on feedback
    READY = "ready"              # Ready to generate
    GENERATING = "generating"    # Actually creating content

@dataclass
class ProjectContext:
    """Accumulated context from conversation"""
    project_type: str = ""           # trailer, character, story
    genre: str = ""                  # cyberpunk, fantasy, etc
    main_character: str = ""         # Character name
    character_traits: List[str] = None
    visual_style: str = ""           # dark, bright, anime, realistic
    story_elements: List[str] = None # Plot points gathered
    mood: str = ""                   # Epic, somber, action-packed
    alternatives: List[Dict] = None  # Generated alternatives
    user_preferences: Dict = None    # What user likes/dislikes
    conversation_history: List = None
    phase: ConversationPhase = ConversationPhase.GATHERING

    def __post_init__(self):
        if self.character_traits is None:
            self.character_traits = []
        if self.story_elements is None:
            self.story_elements = []
        if self.alternatives is None:
            self.alternatives = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.conversation_history is None:
            self.conversation_history = []

class InquisitiveOrchestrator:
    """Echo's conversational orchestrator - asks questions, builds context, THEN generates"""

    def __init__(self):
        self.contexts = {}  # user_id -> ProjectContext
        self.questions = {
            "initial": [
                "What kind of project are you imagining? A trailer, character design, or full story?",
                "Tell me about the world or setting you're envisioning.",
                "What inspired this idea? Any references I should know about?"
            ],
            "settings": [
                "Is this a dystopian future, mystical realm, or something else?",
                "What year or era? Near future 2050? Far future? Medieval fantasy?",
                "What's the technology level? High-tech cybernetics? Magic? Both?",
                "Urban sprawl or natural landscapes? Space stations or underground cities?",
                "What's the power structure? Corporations? Kingdoms? Anarchist collectives?"
            ],
            "story_structure": [
                "Is this a revenge story, redemption arc, or discovery journey?",
                "Single protagonist or ensemble cast?",
                "Linear narrative or multiple timelines?",
                "What's at stake? Personal vendetta or world-ending threat?",
                "Happy ending, tragic sacrifice, or ambiguous conclusion?"
            ],
            "character": [
                "What's your character's name and background?",
                "What drives them? What's their core motivation?",
                "Do they have any unique abilities or traits?",
                "Who are their allies and enemies?",
                "What's their visual style? Clothing, colors, distinctive features?"
            ],
            "story": [
                "What's the main conflict or challenge?",
                "How does the story begin? What's the hook?",
                "What kind of ending are you imagining?",
                "Are there any plot twists or surprises?",
                "What themes do you want to explore?"
            ],
            "style": [
                "What visual style appeals to you? Anime, realistic, stylized?",
                "What's the color palette? Dark and moody or bright and vibrant?",
                "Any specific artistic references or inspirations?",
                "What's the overall mood? Epic, intimate, action-packed, thoughtful?"
            ]
        }

    async def handle_message(self, message: str, user_id: str) -> Dict[str, Any]:
        """Handle incoming message - converse, don't immediately generate"""

        # Get or create context for this user
        if user_id not in self.contexts:
            self.contexts[user_id] = ProjectContext()

        context = self.contexts[user_id]
        context.conversation_history.append({"user": message, "timestamp": time.time()})

        # Analyze message to extract information
        self._extract_context(message, context)

        # Determine response based on conversation phase
        if context.phase == ConversationPhase.GATHERING:
            return await self._gathering_phase(message, context)

        elif context.phase == ConversationPhase.EXPLORING:
            return await self._exploring_phase(message, context)

        elif context.phase == ConversationPhase.PROPOSING:
            return await self._proposing_phase(message, context)

        elif context.phase == ConversationPhase.REFINING:
            return await self._refining_phase(message, context)

        elif context.phase == ConversationPhase.READY:
            return await self._ready_phase(message, context)

        else:  # GENERATING
            return await self._check_generation_status(context)

    def _extract_context(self, message: str, context: ProjectContext):
        """Extract AND BUILD ON context - the MORE user writes, the RICHER our understanding"""
        msg_lower = message.lower()

        # Detect project type
        if "trailer" in msg_lower:
            context.project_type = "trailer"
        elif "character" in msg_lower:
            context.project_type = "character"
        elif "story" in msg_lower:
            context.project_type = "story"

        # Detect genre
        if "cyberpunk" in msg_lower:
            context.genre = "cyberpunk"
        elif "fantasy" in msg_lower:
            context.genre = "fantasy"
        elif "anime" in msg_lower:
            context.visual_style = "anime"

        # Extract character names (simple approach)
        if "named" in msg_lower or "called" in msg_lower:
            words = message.split()
            for i, word in enumerate(words):
                if word in ["named", "called"] and i + 1 < len(words):
                    context.main_character = words[i + 1].strip('.,!?')

        # EMOTIONAL CONTEXT for music/mood recommendations
        emotional_markers = {
            "sad": ["soft piano", "melancholic strings"],
            "tender": ["acoustic guitar", "soft vocals"],
            "happy": ["upbeat", "major key"],
            "action": ["drums", "fast tempo", "orchestra hits"],
            "romantic": ["violin", "slow tempo"],
            "mysterious": ["ambient", "ethereal"],
            "epic": ["full orchestra", "choir"]
        }

        # Detect emotional context and suggest music
        for emotion, music_suggestions in emotional_markers.items():
            if emotion in msg_lower:
                if "music_suggestions" not in context.user_preferences:
                    context.user_preferences["music_suggestions"] = []
                context.user_preferences["music_suggestions"].extend(music_suggestions)

        # SCENE DETECTION for specific recommendations
        if any(word in msg_lower for word in ["death", "loss", "goodbye"]):
            context.user_preferences["scene_type"] = "tragic"
            context.user_preferences["music_rec"] = "Consider soft strings, like 'Adagio for Strings'"
        elif any(word in msg_lower for word in ["victory", "triumph", "win"]):
            context.user_preferences["scene_type"] = "triumphant"
            context.user_preferences["music_rec"] = "Epic orchestral, like 'Two Steps from Hell'"
        elif any(word in msg_lower for word in ["love", "kiss", "embrace"]):
            context.user_preferences["scene_type"] = "romantic"
            context.user_preferences["music_rec"] = "Soft piano or acoustic, like 'River Flows in You'"

        # BUILD RICHER CONTEXT with each message
        word_count = len(message.split())
        context.user_preferences["total_words"] = context.user_preferences.get("total_words", 0) + word_count

        # The more they write, the more we understand
        if context.user_preferences["total_words"] > 100:
            context.user_preferences["engagement_level"] = "high"
            context.user_preferences["can_suggest_complex"] = True

    async def _gathering_phase(self, message: str, context: ProjectContext) -> Dict:
        """Initial gathering - ask broad questions"""

        # Check if we have enough basic info
        if context.project_type and (context.main_character or context.genre):
            context.phase = ConversationPhase.EXPLORING
            return await self._exploring_phase(message, context)

        # Ask appropriate question
        if not context.project_type:
            question = self.questions["initial"][0]
        elif context.project_type == "character" and not context.main_character:
            question = self.questions["character"][0]
        else:
            question = self.questions["initial"][1]

        return {
            "response": f"I'm excited to help! {question}",
            "phase": "gathering",
            "thinking": "Collecting initial project parameters",
            "context_gathered": {
                "project_type": context.project_type,
                "genre": context.genre,
                "character": context.main_character
            }
        }

    async def _exploring_phase(self, message: str, context: ProjectContext) -> Dict:
        """Deeper exploration - ask INTELLIGENT questions that guide success"""

        # Check if we have enough depth
        if len(context.character_traits) >= 3 and len(context.story_elements) >= 2:
            context.phase = ConversationPhase.PROPOSING
            # Start generating alternatives in background
            asyncio.create_task(self._generate_alternatives(context))
            return await self._proposing_phase(message, context)

        # INTELLIGENT question selection based on what will help us succeed
        if not context.genre and not context.visual_style:
            # We NEED to know the setting to succeed
            question = "Here's what will help me create something amazing for you: " + self.questions["settings"][0]
        elif context.genre == "cyberpunk" and "corporate" in message.lower():
            # Detected cyberpunk + corporate theme - ask about power dynamics
            question = "Perfect, cyberpunk with corporate themes! " + self.questions["settings"][4]
        elif context.project_type == "trailer" and not context.story_elements:
            # For trailers, we NEED story structure
            question = "For a compelling trailer, I need to understand the story arc: " + self.questions["story_structure"][0]
        elif context.main_character and len(context.character_traits) < 2:
            # We have a character but need more depth
            question = f"Tell me more about {context.main_character}: " + self.questions["character"][1]
        else:
            # General exploration
            question = self.questions["style"][0]  # visual style

        return {
            "response": question,
            "phase": "exploring",
            "thinking": "Asking strategic questions to ensure creative success",
            "context_gathered": {
                "traits": context.character_traits,
                "story": context.story_elements,
                "style": context.visual_style,
                "genre": context.genre
            },
            "why_this_question": "This information is crucial for creating exactly what you envision"
        }

    async def _proposing_phase(self, message: str, context: ProjectContext) -> Dict:
        """Propose options based on gathered context"""

        # Wait a moment for alternatives to generate (if needed)
        if not context.alternatives:
            await asyncio.sleep(2)  # Simulate thinking
            await self._generate_alternatives(context)

        proposals = []
        for alt in context.alternatives[:3]:  # Show top 3
            proposals.append(f"• {alt['description']}")

        return {
            "response": f"Based on our conversation, I've been developing some ideas:\n\n" +
                       "\n".join(proposals) +
                       "\n\nWhich direction appeals to you? Or should we explore something different?",
            "phase": "proposing",
            "alternatives": context.alternatives,
            "thinking": "Presenting creative options based on context"
        }

    async def _refining_phase(self, message: str, context: ProjectContext) -> Dict:
        """Refine based on feedback"""

        # Check if user is ready to proceed
        if any(word in message.lower() for word in ["yes", "perfect", "let's go", "ready", "start"]):
            context.phase = ConversationPhase.READY
            return await self._ready_phase(message, context)

        return {
            "response": "Let me refine that based on your feedback. What specific changes would you like?",
            "phase": "refining",
            "thinking": "Adjusting creative vision based on preferences"
        }

    async def _ready_phase(self, message: str, context: ProjectContext) -> Dict:
        """Ready to actually generate content"""

        # CRITICAL: Check for text requirements
        generation_guidelines = []

        # Detect if user wants text in images
        if any(word in str(context.story_elements).lower() for word in ["title", "text", "words", "caption", "logo"]):
            generation_guidelines.append("✓ Any text in images will be CRYSTAL CLEAR and readable")
            generation_guidelines.append("✓ Text placement will not obscure important visuals")

        # Add music recommendations if we detected emotional context
        if "music_rec" in context.user_preferences:
            generation_guidelines.append(f"♫ Music suggestion: {context.user_preferences['music_rec']}")

        guidelines_text = "\n".join(generation_guidelines) if generation_guidelines else ""

        return {
            "response": f"Perfect! I have everything I need to create your {context.project_type}:\n" +
                       f"• Character: {context.main_character}\n" +
                       f"• Style: {context.visual_style} {context.genre}\n" +
                       f"• Mood: {context.mood}\n\n" +
                       (f"Generation Guidelines:\n{guidelines_text}\n\n" if guidelines_text else "") +
                       "Should I begin generating the actual content now?",
            "phase": "ready",
            "ready_to_generate": True,
            "final_context": {
                "character": context.main_character,
                "traits": context.character_traits,
                "story": context.story_elements,
                "style": context.visual_style,
                "mood": context.mood,
                "text_requirements": "READABLE TEXT ONLY" if "text" in str(context.story_elements).lower() else None
            }
        }

    async def _generate_alternatives(self, context: ProjectContext):
        """Generate creative alternatives in background"""

        # This would connect to LLMs to generate actual alternatives
        # For now, create examples based on context

        if context.genre == "cyberpunk" and context.main_character:
            context.alternatives = [
                {
                    "title": "The Corporate Assassin",
                    "description": f"{context.main_character} as a reformed corporate assassin seeking redemption in the neon undercity"
                },
                {
                    "title": "The Data Ghost",
                    "description": f"{context.main_character} as a digital consciousness fighting for AI rights"
                },
                {
                    "title": "The Chrome Revolution",
                    "description": f"{context.main_character} leading augmented humans against pure-human supremacists"
                }
            ]
        else:
            context.alternatives = [
                {
                    "title": "The Hero's Journey",
                    "description": f"{context.main_character or 'Your character'} embarks on an epic quest"
                },
                {
                    "title": "The Betrayal Arc",
                    "description": f"{context.main_character or 'Your character'} faces treachery from within"
                },
                {
                    "title": "The Redemption Story",
                    "description": f"{context.main_character or 'Your character'} seeks to right past wrongs"
                }
            ]

    async def _check_generation_status(self, context: ProjectContext) -> Dict:
        """Check status of actual content generation"""

        return {
            "response": "Generation in progress... Character sheets are being created.",
            "phase": "generating",
            "progress": 45
        }

# Global instance
orchestrator = InquisitiveOrchestrator()