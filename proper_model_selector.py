"""Proper model selection for AI Assist"""
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ProperModelSelector:
    def __init__(self):
        self.models = {
            "quick": {
                "model": "tinyllama:latest",
                "max_time": 1.0,
                "use_for": ["greetings", "simple_questions"],
                "params": 1.1
            },
            "standard": {
                "model": "llama3.2:3b",
                "max_time": 3.0,
                "use_for": ["general_questions", "explanations"],
                "params": 3.2
            },
            "professional": {
                "model": "mistral:7b-instruct",
                "max_time": 5.0,
                "use_for": ["technical_questions", "analysis"],
                "params": 7.0
            },
            "expert": {
                "model": "qwen2.5-coder:32b",
                "max_time": 10.0,
                "use_for": ["coding", "complex_problems"],
                "params": 32.0
            },
            "genius": {
                "model": "llama3.1:70b",
                "max_time": 20.0,
                "use_for": ["research", "deep_analysis"],
                "params": 70.0
            }
        }

    def analyze_complexity(self, query: str) -> Tuple[str, str]:
        query_lower = query.lower()

        # Quick for simple
        if len(query) < 20 or any(w in query_lower for w in ["hi", "hello", "hey"]):
            return "quick", "Simple greeting"

        # Expert for coding
        if any(w in query_lower for w in ["code", "algorithm", "function", "implement"]):
            return "expert", "Coding task"

        # Genius for complex
        if len(query) > 200 and any(w in query_lower for w in ["analyze", "design", "architect"]):
            return "genius", "Complex analysis"

        # Professional for technical
        if any(w in query_lower for w in ["explain", "compare", "evaluate"]):
            return "professional", "Technical question"

        return "standard", "General query"

    def select_model(self, query: str, requested: Optional[str] = None) -> Dict:
        if requested and requested != "auto" and requested in self.models:
            level = requested
            reasoning = f"User requested {requested}"
        else:
            level, reasoning = self.analyze_complexity(query)

        config = self.models[level].copy()
        config["level"] = level
        config["reasoning"] = reasoning

        logger.info(f"Selected: {config['model']} ({reasoning})")
        return config
