#!/usr/bin/env python3
"""
FIXED Model Selection with proper complexity escalation
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class FixedModelSelector:
    """Fixed model selection with intent-aware complexity escalation"""

    def __init__(self):
        # Proper model tiers (no more jumping to 70B)
        self.model_tiers = {
            "tiny": ["llama3.1:8b", "llama3.2:3b"],
            "small": ["llama3.2:3b", "mistral:7b"],
            "medium": ["codellama:7b", "mistral:7b"],
            "large": ["codellama:13b", "qwen2.5-coder:7b"],
            "expert": ["qwen2.5-coder:32b", "mixtral:8x7b"],
            "genius": ["llama3.1:70b", "codellama:70b"]  # Only for explicit requests
        }

        # Intent to tier mapping
        self.intent_tiers = {
            "explanation": ("tiny", "small", "medium"),  # Simple to complex explanations
            "code_generation": ("medium", "large", "expert"),  # Need good code models
            "code_modification": ("large", "expert", "expert"),  # Need precision
            "analysis": ("medium", "large", "expert"),  # Need analytical capability
            "system_query": ("small", "medium", "large"),  # Need reliable data retrieval
            "general": ("tiny", "small", "medium"),  # Escalate as needed
            "image_generation": ("medium", "medium", "medium"),  # ComfyUI handles this
            "anime_generation": ("medium", "medium", "medium"),  # ComfyUI handles this
            "voice_generation": ("tiny", "small", "small"),  # Simple API calls
            "music_generation": ("small", "medium", "medium")  # Moderate complexity
        }

        self.thought_log = []

    def select_model_for_intent(self, intent: str, complexity_score: float, context: Dict) -> Tuple[str, str, str]:
        """
        Select model based on intent and complexity
        Returns: (model, tier, reasoning)
        """
        # Clear thought log
        self.thought_log = []

        # Log selection process
        thought = f"Selecting model for intent '{intent}' with complexity {complexity_score:.2f}"
        logger.info(f"🧠 MODEL SELECTION: {thought}")
        self.thought_log.append(thought)

        # Get tier range for this intent
        tier_options = self.intent_tiers.get(intent, ("tiny", "small", "medium"))

        # Select tier based on complexity
        if complexity_score < 30:
            tier = tier_options[0]  # Low complexity
            reasoning = f"Low complexity ({complexity_score:.0f}) → {tier} tier"
        elif complexity_score < 70:
            tier = tier_options[1]  # Medium complexity
            reasoning = f"Medium complexity ({complexity_score:.0f}) → {tier} tier"
        else:
            tier = tier_options[2]  # High complexity
            reasoning = f"High complexity ({complexity_score:.0f}) → {tier} tier"

        # Check for timeout constraints
        max_timeout = context.get("max_timeout", 30)
        if tier == "genius" and max_timeout < 60:
            thought = f"Genius tier would timeout ({max_timeout}s limit), using expert instead"
            logger.info(f"🧠 CONSTRAINT: {thought}")
            self.thought_log.append(thought)
            tier = "expert"
            reasoning += f" (timeout constrained to {max_timeout}s)"

        # Select specific model from tier
        models = self.model_tiers.get(tier, self.model_tiers["small"])
        model = models[0]  # Use primary model for tier

        # Check if model is available
        available_models = context.get("available_models", [])
        if available_models and model not in available_models:
            # Fall back to available model in same tier
            for alt_model in models:
                if alt_model in available_models:
                    model = alt_model
                    reasoning += f" (using alternate: {alt_model})"
                    break
            else:
                # No model in tier available, fall back
                model = "llama3.2:3b"
                reasoning += " (fallback to llama3.2:3b)"

        thought = f"Selected: {model} ({tier} tier) - {reasoning}"
        logger.info(f"🧠 DECISION: {thought}")
        self.thought_log.append(thought)

        return model, tier, reasoning

    def should_escalate(self, previous_response: str, intent: str) -> bool:
        """Determine if we should escalate to a higher tier"""
        # Check for failure indicators
        failure_indicators = [
            "I don't understand",
            "I'm not sure",
            "Could you clarify",
            "error",
            "failed",
            "unable to"
        ]

        response_lower = previous_response.lower()
        for indicator in failure_indicators:
            if indicator in response_lower:
                thought = f"Found failure indicator '{indicator}', should escalate"
                logger.info(f"🧠 ESCALATION: {thought}")
                self.thought_log.append(thought)
                return True

        # Check response quality
        if len(previous_response) < 50 and intent != "system_query":
            thought = "Response too short, should escalate"
            logger.info(f"🧠 ESCALATION: {thought}")
            self.thought_log.append(thought)
            return True

        return False

    def get_next_tier(self, current_tier: str) -> Optional[str]:
        """Get next tier for escalation"""
        tier_order = ["tiny", "small", "medium", "large", "expert"]

        try:
            current_index = tier_order.index(current_tier)
            if current_index < len(tier_order) - 1:
                return tier_order[current_index + 1]
        except ValueError:
            pass

        return None

# Global instance
model_selector = FixedModelSelector()

if __name__ == "__main__":
    # Test the fixed model selection
    test_cases = [
        ("explanation", 20, {}),
        ("explanation", 50, {}),
        ("explanation", 80, {}),
        ("code_generation", 30, {}),
        ("code_generation", 70, {}),
        ("code_modification", 50, {}),
        ("system_query", 10, {}),
        ("general", 40, {}),
    ]

    print("Testing FIXED model selection:\n")
    for intent, complexity, context in test_cases:
        model, tier, reasoning = model_selector.select_model_for_intent(intent, complexity, context)
        print(f"Intent: {intent}, Complexity: {complexity}")
        print(f"  → Model: {model} ({tier} tier)")
        print(f"  → Reasoning: {reasoning}")
        print()