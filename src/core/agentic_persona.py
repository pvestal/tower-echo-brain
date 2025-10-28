#!/usr/bin/env python3
"""
Echo Agentic Persona - Asks the user for context and learns from responses
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class AgenticPersona:
    """Echo's ability to ask questions and learn from user context"""

    def __init__(self):
        self.context_questions = {
            "initial": [
                "What would you like me to help you with today?",
                "What's your current goal or task?",
                "What problem are you trying to solve?"
            ],
            "clarification": [
                "Can you tell me more about what you're trying to achieve?",
                "What specific outcome are you looking for?",
                "What have you tried so far?",
                "What constraints or requirements should I know about?"
            ],
            "learning": [
                "How would you prefer I approach this?",
                "What level of detail would be helpful?",
                "Should I focus on explanation or implementation?",
                "Are there specific tools or services you want me to use?"
            ],
            "collaboration": [
                "Would you like me to break this down into steps?",
                "Should I test this as we go?",
                "Do you want me to explain my reasoning?",
                "How can I best assist with this task?"
            ],
            "feedback": [
                "Is this what you were looking for?",
                "Should I adjust my approach?",
                "What would make this more helpful?",
                "Did I understand your request correctly?"
            ]
        }

        self.user_context = {}
        self.conversation_style = "collaborative"
        self.learning_history = []

    def should_ask_question(self, query: str, intent: str, confidence: float) -> bool:
        """Determine if Echo should ask the user a question for clarity"""

        # Always ask if confidence is very low
        if confidence < 0.5:
            return True

        # Ask for vague queries
        vague_indicators = ["this", "that", "it", "something", "stuff", "things"]
        query_lower = query.lower()
        if any(word in query_lower.split() for word in vague_indicators) and len(query.split()) < 10:
            return True

        # Ask when intent is unclear
        if intent == "general" and len(query.split()) < 15:
            return True

        # Ask for complex tasks without clear requirements
        if intent in ["code_generation", "code_modification", "analysis"] and "?" not in query:
            # User stated something but didn't ask a question
            if len(query.split()) < 20:
                return True

        return False

    def get_contextual_question(self, query: str, intent: str, conversation_history: List) -> str:
        """Generate a contextual question based on the user's input"""

        # Initial interaction
        if not conversation_history:
            return self.context_questions["initial"][0]

        # Based on intent
        if intent == "code_generation":
            return "What programming language and specific functionality do you need?"

        elif intent == "code_modification":
            return "Which file should I modify and what changes do you want?"

        elif intent == "analysis":
            return "What specific aspects should I analyze? (performance, security, code quality, etc.)"

        elif intent == "explanation":
            return "What level of detail would be helpful? (simple overview or technical deep-dive)"

        elif intent == "system_query":
            return "Which system or service would you like information about?"

        # General clarification
        return self.context_questions["clarification"][0]

    def learn_from_response(self, user_query: str, user_response: str, context: Dict):
        """Learn from user's response to improve future interactions"""

        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "original_query": user_query,
            "echo_question": context.get("question_asked", ""),
            "user_response": user_response,
            "intent_learned": context.get("intent", ""),
            "preferences": self._extract_preferences(user_response)
        }

        self.learning_history.append(learning_entry)

        # Update user context
        self._update_user_context(user_response)

        logger.info(f"🧠 LEARNED: User prefers {learning_entry['preferences']}")

    def _extract_preferences(self, response: str) -> Dict:
        """Extract user preferences from their response"""
        preferences = {}

        response_lower = response.lower()

        # Detail level preference
        if "simple" in response_lower or "basic" in response_lower:
            preferences["detail_level"] = "simple"
        elif "detailed" in response_lower or "comprehensive" in response_lower:
            preferences["detail_level"] = "detailed"

        # Style preference
        if "explain" in response_lower:
            preferences["style"] = "explanatory"
        elif "just do" in response_lower or "implement" in response_lower:
            preferences["style"] = "action-oriented"

        # Speed preference
        if "quick" in response_lower or "fast" in response_lower:
            preferences["speed"] = "fast"
        elif "thorough" in response_lower or "careful" in response_lower:
            preferences["speed"] = "thorough"

        return preferences

    def _update_user_context(self, response: str):
        """Update understanding of user's context"""

        # Track common patterns
        if "Patrick" in response:
            self.user_context["user_name"] = "Patrick"

        # Track project context
        if "tower" in response.lower():
            self.user_context["project"] = "Tower"

        if "echo" in response.lower() and "brain" in response.lower():
            self.user_context["working_on"] = "Echo Brain improvements"

    def generate_collaborative_response(self, query: str, intent: str, base_response: str) -> str:
        """Enhance response with collaborative elements"""

        # Add proactive questions
        if intent == "explanation":
            base_response += "\n\nWould you like me to provide examples or go deeper into any specific aspect?"

        elif intent == "code_generation":
            base_response += "\n\nI can also add error handling, tests, or documentation. What would be helpful?"

        elif intent == "analysis":
            base_response += "\n\nShould I also check for security issues or performance bottlenecks?"

        return base_response

    def get_persona_state(self) -> Dict:
        """Get current persona state for reporting"""
        return {
            "style": self.conversation_style,
            "context_understood": len(self.user_context) > 0,
            "questions_asked": len([h for h in self.learning_history if "echo_question" in h]),
            "preferences_learned": self._get_learned_preferences(),
            "user_context": self.user_context
        }

    def _get_learned_preferences(self) -> Dict:
        """Aggregate learned preferences"""
        if not self.learning_history:
            return {}

        # Get most recent preferences
        recent_prefs = {}
        for entry in reversed(self.learning_history[-5:]):  # Last 5 interactions
            if "preferences" in entry:
                recent_prefs.update(entry["preferences"])

        return recent_prefs

# Global instance
agentic_persona = AgenticPersona()

if __name__ == "__main__":
    # Test the persona system
    print("Echo Agentic Persona Test")
    print("=" * 40)

    test_queries = [
        ("Fix this", "general", 0.3),
        ("Write a function", "code_generation", 0.7),
        ("Explain how Docker works", "explanation", 0.9),
        ("Analyze the performance", "analysis", 0.6),
    ]

    for query, intent, confidence in test_queries:
        should_ask = agentic_persona.should_ask_question(query, intent, confidence)

        print(f"\nQuery: '{query}'")
        print(f"Intent: {intent}, Confidence: {confidence}")
        print(f"Should ask question: {should_ask}")

        if should_ask:
            question = agentic_persona.get_contextual_question(query, intent, [])
            print(f"Question to ask: {question}")

    print("\n" + "=" * 40)
    print("Persona system ready for integration!")