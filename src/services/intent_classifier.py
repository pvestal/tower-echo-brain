#!/usr/bin/env python3
"""
FIXED Conversation management with proper intent classification
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation context with FIXED intent classification"""

    def __init__(self):
        self.conversations = {}  # In-memory cache
        self._database = None  # Will be set lazily

        # FIXED intent patterns - more general and actually works
        self.intent_patterns = {
            # Explanations and knowledge queries - CHECK FIRST
            "explanation": [
                r"what.*is|what.*are|what's",
                r"how.*does|how.*do|how.*to",
                r"explain|describe|tell.*about",
                r"why|when|where|who",
                r"understand|learn|tutorial",
                r"difference.*between",
                r"define|definition"
            ],

            # Code generation (NEW code, not modification)
            "code_generation": [
                r"write.*function|write.*code|write.*script",
                r"create.*function|create.*code|create.*script",
                r"generate.*code|implement.*function",
                r"build.*api|make.*api",
                r"example.*of|sample.*code"
            ],

            # System queries
            "system_query": [
                r"status|health|running",
                r"service.*status|check.*service",
                r"echo.*brain|tower.*service",
                r"is.*working|is.*running"
            ],

            # Analysis requests
            "analysis": [
                r"analyze|review.*code|code.*review",
                r"check.*quality|optimize",
                r"find.*issues|debug",
                r"performance|profile"
            ],

            # Code modification - VERY SPECIFIC
            "code_modification": [
                r"modify.*file|edit.*file|change.*file",
                r"update.*file|refactor.*code",
                r"fix.*bug.*in.*file",
                r"add.*to.*existing"
            ],

            # Service-specific intents (keep existing)
            "service_testing": [
                r"test.*(comfyui|anime|dashboard|auth|echo|deepseek|kb|voice|knowledge).*(service|api)"
            ],
            "service_debugging": [
                r"debug.*(comfyui|anime|dashboard|auth|echo|deepseek|kb|voice|knowledge).*(service|api)"
            ],
            "anime_generation": [
                r"(generate|create|make).*(anime|animation|animated).*(video|scene|clip)",
                r"anime.*(video|generation)"
            ],
            "image_generation": [
                r"generate.*image|create.*image|make.*image",
                r"generate.*picture|create.*picture",
                r"draw|paint|sketch|render"
            ],
            "voice_generation": [
                r"generate.*voice|create.*voice",
                r"text.*to.*speech|tts",
                r"say.*this|speak.*this"
            ],
            "music_generation": [
                r"generate.*music|create.*music",
                r"compose|soundtrack",
                r"musical.*piece"
            ]
        }

        self.clarifying_questions = {
            "code_modification": [
                "Which specific files need to be modified?",
                "What exact functionality should be added or changed?"
            ],
            "analysis": [
                "What specific aspects should I analyze?",
                "Are you looking for performance, security, or code quality issues?"
            ],
            "explanation": [
                "Would you like a simple overview or technical details?",
                "Any specific aspects you want me to focus on?"
            ]
        }

        # Thought logging for visibility
        self.thought_log = []

    def classify_intent(self, query: str, conversation_history: List[Dict] = []) -> Tuple[str, float, Dict]:
        """FIXED intent classification with proper ordering"""
        query_lower = query.lower()
        intent_params = {}

        # Clear thought log
        self.thought_log = []

        # Log classification process
        thought = f"Classifying query: {query[:50]}..."
        logger.info(f"🧠 THOUGHT: {thought}")
        self.thought_log.append(thought)

        # Check conversation context for intent continuation
        if conversation_history:
            last_intent = conversation_history[-1].get("intent")
            if last_intent and any(pattern in query_lower for pattern in ["yes", "no", "continue", "more"]):
                thought = f"Continuing previous intent: {last_intent}"
                logger.info(f"🧠 THOUGHT: {thought}")
                self.thought_log.append(thought)
                return last_intent, 0.9, {}

        # CHECK EXPLANATION FIRST (most common)
        for pattern in self.intent_patterns["explanation"]:
            if re.search(pattern, query_lower):
                thought = f"Matched explanation pattern: {pattern}"
                logger.info(f"🧠 THOUGHT: {thought}")
                self.thought_log.append(thought)
                return "explanation", 0.95, {}

        # Then check other intents in priority order
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            if intent == "explanation":
                continue  # Already checked

            score = 0
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    score += 0.8
                    thought = f"Pattern '{pattern}' matched for intent '{intent}'"
                    logger.info(f"🧠 THOUGHT: {thought}")
                    self.thought_log.append(thought)

                    # Extract parameters for service intents
                    if intent in ['service_testing', 'service_debugging']:
                        groups = match.groups()
                        if groups:
                            intent_params['target_service'] = groups[0]

            if score > 0:
                intent_scores[intent] = min(score, 1.0)

        if not intent_scores:
            thought = "No specific intent matched, defaulting to general"
            logger.info(f"🧠 THOUGHT: {thought}")
            self.thought_log.append(thought)
            return "general", 0.5, {}

        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        thought = f"Selected intent: {best_intent[0]} (confidence: {best_intent[1]:.2f})"
        logger.info(f"🧠 THOUGHT: {thought}")
        self.thought_log.append(thought)

        return best_intent[0], best_intent[1], intent_params

    def needs_clarification(self, intent: str, confidence: float, query: str) -> bool:
        """Determine if query needs clarification - BE LESS AGGRESSIVE"""
        # Only clarify if really low confidence
        if confidence < 0.3:
            return True

        # Don't clarify for clear intents
        if intent in ['explanation', 'system_query', 'general']:
            return False

        # Only clarify code modification if no file specified
        if intent == 'code_modification' and 'file' not in query.lower():
            return True

        return False

    def get_clarifying_questions(self, intent: str, query: str) -> List[str]:
        """Get relevant clarifying questions for intent"""
        questions = self.clarifying_questions.get(intent, [
            "Can you provide more details about what you're trying to accomplish?"
        ])
        return questions[:2]  # Only 2 questions max

    def update_conversation(self, conversation_id: str, user_query: str,
                          intent: str, response: str, requires_clarification: bool):
        """Update conversation history in memory"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "history": [],
                "created_at": datetime.now(),
                "last_intent": None,
                "thought_log": []
            }

        self.conversations[conversation_id]["history"].append({
            "user_query": user_query,
            "intent": intent,
            "response": response,
            "requires_clarification": requires_clarification,
            "timestamp": datetime.now(),
            "thoughts": self.thought_log.copy()
        })

        self.conversations[conversation_id]["last_intent"] = intent
        self.conversations[conversation_id]["thought_log"].extend(self.thought_log)

        # Keep last 10 interactions in memory
        if len(self.conversations[conversation_id]["history"]) > 10:
            self.conversations[conversation_id]["history"] = \
                self.conversations[conversation_id]["history"][-10:]

    async def get_conversation_context(self, conversation_id: str) -> Dict:
        """Get conversation context from memory or database"""
        # Check memory cache first
        if conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            return {
                "history": conv["history"],
                "last_intent": conv.get("last_intent"),
                "created_at": conv.get("created_at"),
                "interaction_count": len(conv["history"]),
                "thought_log": conv.get("thought_log", [])
            }

        # Return empty context if not found
        return {
            "history": [],
            "last_intent": None,
            "created_at": datetime.now(),
            "interaction_count": 0,
            "thought_log": []
        }

# Create the global instance
conversation_manager = ConversationManager()

# For backward compatibility
if __name__ == "__main__":
    # Test the fixed classification
    test_queries = [
        "Explain quantum computing",
        "What is a neural network?",
        "Write a Python function to sort a list",
        "Fix the bug in main.py",
        "What's the status of Echo Brain?",
        "Analyze this code for performance",
        "How does async/await work?"
    ]

    print("Testing FIXED intent classification:\n")
    for query in test_queries:
        intent, confidence, params = conversation_manager.classify_intent(query)
        print(f"Query: {query}")
        print(f"  Intent: {intent} (confidence: {confidence:.2f})")
        print(f"  Thoughts: {conversation_manager.thought_log[-1] if conversation_manager.thought_log else 'None'}")
        print()