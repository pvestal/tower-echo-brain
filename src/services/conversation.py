#!/usr/bin/env python3
"""
Conversation management and intent recognition for Echo Brain
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation context and intent recognition"""

    def __init__(self):
        self.conversations = {}  # In-memory for now, can move to Redis/DB later
        self.intent_patterns = {
            "service_testing": [
                r"test\s+(\w+)",
                r"run.*test.*on\s+(\w+)",
                r"check\s+(\w+).*working",
                r"verify\s+(\w+).*status",
                r"test.*(comfyui|anime|dashboard|auth|echo|deepseek|kb|voice)"
            ],
            "service_debugging": [
                r"debug\s+(\w+)",
                r"troubleshoot\s+(\w+)",
                r"diagnose\s+(\w+).*problem",
                r"analyze\s+(\w+).*issue",
                r"debug.*(comfyui|anime|dashboard|auth|echo|deepseek|kb|voice)"
            ],
            "service_monitoring": [
                r"status\s+(\w+)",
                r"health\s+(\w+)",
                r"check\s+(\w+).*health",
                r"monitor\s+(\w+)",
                r"get.*stats|statistics"
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
                    score += 0.8  # Higher confidence for direct matches
                    # Extract service names from patterns
                    if intent in ['service_testing', 'service_debugging', 'service_monitoring', 'inter_service_communication']:
                        # Try to extract service name from the match
                        groups = match.groups()
                        if groups:
                            intent_params['target_service'] = groups[0]
                        else:
                            # Look for known service names in the query
                            services = ['comfyui', 'anime', 'dashboard', 'auth', 'echo', 'kb', 'deepseek', 'voice']
                            for service in services:
                                if service in query_lower:
                                    intent_params['target_service'] = service
                                    break
                            # Also try to extract from the match groups for more complex patterns
                            if len(groups) > 0 and groups[0] in ['comfyui', 'anime', 'dashboard', 'auth', 'echo', 'kb', 'deepseek', 'voice']:
                                intent_params['target_service'] = groups[0]

                    # Extract agent names for delegation
                    if intent == 'agent_delegation':
                        groups = match.groups()
                        if groups:
                            intent_params['target_agent'] = groups[0]
                        else:
                            # Look for known agent names in the query
                            agents = ['architect', 'security', 'guardian', 'weights', 'balances', 'vision']
                            for agent in agents:
                                if agent in query_lower:
                                    intent_params['target_agent'] = agent
                                    break

                    # Extract multimedia generation parameters
                    if intent == 'image_generation':
                        # Extract prompt by removing generation keywords
                        prompt = query.replace('generate image', '').replace('create image', '').replace('make image', '').replace('draw image', '').strip()
                        if 'of ' in prompt:
                            prompt = prompt.split('of ', 1)[1]
                        if prompt:
                            intent_params['prompt'] = prompt
                        # Extract style if mentioned
                        if 'anime' in query_lower:
                            intent_params['style'] = 'anime'
                        elif 'realistic' in query_lower:
                            intent_params['style'] = 'realistic'
                        elif 'cartoon' in query_lower:
                            intent_params['style'] = 'cartoon'

                    elif intent == 'voice_generation':
                        # Extract text to speak
                        text = query.replace('say', '').replace('speak', '').replace('voice', '').replace('this', '').strip()
                        if 'generate voice' in query_lower:
                            text = query.replace('generate voice', '').strip()
                        if text:
                            intent_params['text'] = text
                        # Extract character if mentioned
                        if 'echo' in query_lower:
                            intent_params['character'] = 'echo_default'

                    elif intent == 'music_generation':
                        # Extract description
                        description = query.replace('generate music', '').replace('create music', '').replace('make music', '').strip()
                        if 'compose' in query_lower:
                            description = query.replace('compose', '').strip()
                        if description:
                            intent_params['description'] = description
                        # Extract duration if mentioned
                        duration_match = re.search(r'(\d+)\s*(?:second|sec|minute|min)', query_lower)
                        if duration_match:
                            duration = int(duration_match.group(1))
                            if 'minute' in duration_match.group(0) or 'min' in duration_match.group(0):
                                duration *= 60  # Convert to seconds
                            intent_params['duration'] = duration

            if score > 0:
                intent_scores[intent] = min(score, 1.0)

        if not intent_scores:
            return "general", 0.5, {}

        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1], intent_params

    def needs_clarification(self, intent: str, confidence: float, query: str) -> bool:
        """Determine if query needs clarification"""
        # Capability intents never need clarification - execute directly
        if intent in ['service_testing', 'service_debugging', 'service_monitoring', 'agent_delegation', 'inter_service_communication', 'code_modification']:
            return False

        # Low confidence always needs clarification
        if confidence < 0.1:  # Changed from 0.6 to only clarify on very low confidence
            return True

        # Check for vague queries even with good intent classification
        vague_indicators = [
            len(query.split()) < 2,  # Very short queries
            "help" in query.lower() and len(query.split()) < 3,
            query.count("?") == 0 and intent != "explanation",  # No questions but unclear intent
        ]

        return False  # Disabled vague indicators check

    def get_clarifying_questions(self, intent: str, query: str) -> List[str]:
        """Get relevant clarifying questions for intent"""
        questions = self.clarifying_questions.get(intent, [
            "Can you provide more details about what you're trying to accomplish?",
            "What specific aspects would you like me to focus on?",
            "Are there any constraints or preferences I should know about?"
        ])

        # Limit to 2-3 most relevant questions to avoid overwhelming
        return questions[:3]

    def update_conversation(self, conversation_id: str, user_query: str,
                          intent: str, response: str, requires_clarification: bool):
        """Update conversation history"""
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

        # Keep conversation history manageable (last 10 interactions)
        if len(self.conversations[conversation_id]["history"]) > 10:
            self.conversations[conversation_id]["history"] = \
                self.conversations[conversation_id]["history"][-10:]

    def get_conversation_context(self, conversation_id: str) -> Dict:
        """Get conversation context for better processing"""
        if conversation_id not in self.conversations:
            return {}

        conv = self.conversations[conversation_id]
        return {
            "history": conv["history"],
            "last_intent": conv.get("last_intent"),
            "created_at": conv.get("created_at"),
            "interaction_count": len(conv["history"])
        }

# Global conversation manager instance
conversation_manager = ConversationManager()