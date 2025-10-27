#!/usr/bin/env python3
"""
Conversation management with lazy database connection
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation context with lazy database persistence"""

    def __init__(self):
        self.conversations = {}  # In-memory cache
        self._database = None  # Will be set lazily
        
        # Rest of the initialization (intent patterns, etc.)
        self.intent_patterns = {
            "service_testing": [
                r"test.*(comfyui|anime|dashboard|auth|echo|deepseek|kb|voice|knowledge).*(service|api)"
            ],
            "service_debugging": [
                r"debug.*(comfyui|anime|dashboard|auth|echo|deepseek|kb|voice|knowledge).*(service|api)"
            ],
            "service_monitoring": [
                r"(comfyui|anime|dashboard|auth|echo|deepseek|kb|voice|knowledge).*(status|health)"
            ],
            "agent_delegation": [
                r"delegate.*to\s+(\w+)",
                r"send.*to\s+(architect|security|guardian|weights|balances)",
                r"assign.*task.*to",
                r"route.*to.*agent",
                r"have.*(architect|security|guardian).*handle",
                r"delegate.*complex.*task"
            ],
            "inter_service_communication": [
                r"communicate.*with\s+(\w+)",
                r"send.*message.*to\s+(\w+)",
                r"connect.*to\s+(\w+).*service",
                r"talk.*to\s+(\w+)",
                r"interface.*with\s+(\w+)"
            ],
            "code_modification": [
                r"write.*script.*at",
                r"create.*file.*at",
                r"write.*file",
                r"save.*content.*to",
                r"write.*to.*file",
                r"modify|change|update|edit|fix.*code|file",
                r"add.*function|method|class",
                r"refactor|rewrite"
            ],
            "debugging": [
                r"error|bug|issue|problem|fail|crash",
                r"debug|troubleshoot|diagnose",
                r"not working|broken"
            ],
            "architecture": [
                r"design|architecture|structure|organize",
                r"system|framework|pattern",
                r"how.*build|implement.*system"
            ],
            "ci_cd": [
                r"deploy|deployment|ci/cd|pipeline",
                r"build|test.*automation",
                r"git|version control|workflow"
            ],
            "explanation": [
                r"what.*is|how.*does|explain|understand",
                r"learn|tutorial|guide",
                r"difference between"
            ],
            "image_generation": [
                r"generate.*image|create.*image|make.*image|draw.*image",
                r"generate.*picture|create.*picture|make.*picture",
                r"image.*of|picture.*of|visual.*of",
                r"paint|sketch|render|visualize",
                r"comfyui.*generate|use.*comfyui"
            ],
            "anime_generation": [
                r"(generate|create|make).*(anime|animation|animated).*(video|scene|clip)",
                r"anime.*(video|generation)"
            ],
            "voice_generation": [
                r"generate.*voice|create.*voice|make.*voice",
                r"say.*this|speak.*this|voice.*this",
                r"text.*to.*speech|tts|speech.*synthesis",
                r"generate.*audio|create.*audio|make.*audio",
                r"read.*aloud|voice.*over|narrate"
            ],
            "music_generation": [
                r"generate.*music|create.*music|make.*music",
                r"compose.*music|write.*music|music.*composition",
                r"generate.*song|create.*song|make.*song",
                r"soundtrack|background.*music|ambient.*music",
                r"musical.*piece|audio.*track"
            ]
        }

        self.clarifying_questions = {
            "code_modification": [
                "Which specific files need to be modified?",
                "What exact functionality should be added or changed?",
                "Are there any constraints or requirements I should know about?",
                "Should this integrate with existing code patterns?"
            ],
            "debugging": [
                "What specific error message are you seeing?",
                "When does this error occur (what triggers it)?",
                "What was the last working state?",
                "Have you tried any solutions already?"
            ],
            "architecture": [
                "What's the main goal of this system?",
                "Are there performance or scalability requirements?",
                "Should this integrate with your existing Tower services?",
                "What's your timeline for implementation?"
            ],
            "ci_cd": [
                "What type of project are we deploying?",
                "Do you have existing CI/CD infrastructure?",
                "What environments need to be supported?",
                "Are there specific testing requirements?"
            ],
            "explanation": [
                "What's your current level of understanding with this topic?",
                "Are you looking for a high-level overview or technical details?",
                "Is this for a specific project or general learning?",
                "Do you prefer code examples or conceptual explanations?"
            ]
        }
    
    @property
    def database(self):
        """Lazy database connection - connects on first access"""
        if self._database is None:
            try:
                from src.db.database import database
                self._database = database
                logger.info("✅ Conversation manager connected to database")
            except Exception as e:
                logger.warning(f"Could not connect to database: {e}")
                self._database = False  # Mark as attempted but failed
        return self._database if self._database is not False else None

    def classify_intent(self, query: str, conversation_history: List[Dict] = []) -> Tuple[str, float, Dict]:
        """Classify user intent with confidence score and extracted parameters"""
        query_lower = query.lower()
        intent_params = {}

        # Check conversation context for intent continuation
        if conversation_history:
            last_intent = conversation_history[-1].get("intent")
            if last_intent and any(pattern in query_lower for pattern in ["yes", "no", "continue", "more"]):
                return last_intent, 0.9, {}

        # Pattern-based intent classification with parameter extraction
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    score += 0.8
                    # Parameter extraction logic (abbreviated for brevity - keep existing)
                    if intent in ['service_testing', 'service_debugging', 'service_monitoring']:
                        groups = match.groups()
                        if groups:
                            intent_params['target_service'] = groups[0]

            if score > 0:
                intent_scores[intent] = min(score, 1.0)

        if not intent_scores:
            return "general", 0.5, {}

        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1], intent_params

    def needs_clarification(self, intent: str, confidence: float, query: str) -> bool:
        """Determine if query needs clarification"""
        if intent in ['service_testing', 'service_debugging', 'service_monitoring', 
                      'agent_delegation', 'inter_service_communication', 'code_modification']:
            return False
        if confidence < 0.1:
            return True
        return False

    def get_clarifying_questions(self, intent: str, query: str) -> List[str]:
        """Get relevant clarifying questions for intent"""
        questions = self.clarifying_questions.get(intent, [
            "Can you provide more details about what you're trying to accomplish?",
            "What specific aspects would you like me to focus on?",
            "Are there any constraints or preferences I should know about?"
        ])
        return questions[:3]

    def update_conversation(self, conversation_id: str, user_query: str,
                          intent: str, response: str, requires_clarification: bool):
        """Update conversation history in memory (DB logging handled by API endpoint)"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "history": [],
                "created_at": datetime.now(),
                "last_intent": None
            }

        self.conversations[conversation_id]["history"].append({
            "user_query": user_query,
            "intent": intent,
            "response": response,
            "requires_clarification": requires_clarification,
            "timestamp": datetime.now()
        })

        self.conversations[conversation_id]["last_intent"] = intent

        # Keep last 10 interactions in memory
        if len(self.conversations[conversation_id]["history"]) > 10:
            self.conversations[conversation_id]["history"] =                 self.conversations[conversation_id]["history"][-10:]

    def get_conversation_context(self, conversation_id: str) -> Dict:
        """Get conversation context from memory or database"""
        # Check memory cache first
        if conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            return {
                "history": conv["history"],
                "last_intent": conv.get("last_intent"),
                "created_at": conv.get("created_at"),
                "interaction_count": len(conv["history"])
            }

        # Try loading from database
        if self.database:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                history = loop.run_until_complete(
                    self.database.get_conversation_history(conversation_id)
                )
                loop.close()

                if history:
                    # Convert database records to conversation format
                    formatted_history = []
                    for record in history:
                        formatted_history.append({
                            "user_query": record.get("query", ""),
                            "intent": record.get("intent", ""),
                            "response": record.get("response", ""),
                            "requires_clarification": False,
                            "timestamp": record.get("timestamp", datetime.now())
                        })

                    # Cache in memory
                    self.conversations[conversation_id] = {
                        "history": formatted_history[-10:],
                        "created_at": history[0].get("timestamp", datetime.now()) if history else datetime.now(),
                        "last_intent": formatted_history[-1].get("intent") if formatted_history else None
                    }

                    logger.info(f"✅ Loaded {len(formatted_history)} messages from database for conversation {conversation_id}")

                    return {
                        "history": formatted_history[-10:],
                        "last_intent": formatted_history[-1].get("intent") if formatted_history else None,
                        "created_at": history[0].get("timestamp", datetime.now()) if history else datetime.now(),
                        "interaction_count": len(formatted_history)
                    }
            except Exception as e:
                logger.error(f"Failed to load conversation from database: {e}")

        # Return empty context if not found
        return {}

# Global instance - immediately available
conversation_manager = ConversationManager()
