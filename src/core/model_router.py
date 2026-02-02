#!/usr/bin/env python3
"""
Unified Model Router - Single Source of Truth
==============================================
Benchmark-optimized model routing for Echo Brain.

This module consolidates all model selection logic into one place,
using actual performance data to make routing decisions.

Based on benchmark results from 2026-01-06:
- qwen2.5:3b: Best for classification (94ms TTFT, 80% accuracy)
- qwen2.5-coder:7b: Best for coding tasks
"""

import logging
import os
import psycopg2
from psycopg2.extras import DictCursor
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelSelection:
    """Result of model selection process."""
    model: str
    reason: str
    confidence: float
    expected_ttft_ms: float
    expected_tps: float
    intent: str
    domain: str = "general"
    fallback_used: bool = False

class UnifiedModelRouter:
    """
    Single source of truth for model routing.

    All other modules should import from this router instead of
    hardcoding model names or using multiple routing systems.
    """

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
        }

        # Fallback mappings (used if database is unavailable)
        self.fallback_routing = {
            'classification': 'qwen2.5:3b',      # 94ms TTFT, 80% accuracy
            'greeting': 'qwen2.5:3b',            # Fast response
            'question': 'qwen2.5:3b',            # Quick answers
            'coding': 'qwen2.5-coder:7b',        # Specialized coding
            'code_generation': 'qwen2.5-coder:7b',
            'code_debugging': 'qwen2.5-coder:7b',
            'debugging': 'qwen2.5-coder:7b',
            'reasoning': 'deepseek-r1:8b',       # Available but not benchmarked
            'conversation': 'llama3.1:8b',       # Available
            'default': 'qwen2.5:3b'              # Safe fallback
        }

    def detect_intent(self, query: str) -> str:
        """Detect user intent from query using pattern matching."""
        query_lower = query.lower().strip()

        # Greeting patterns
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']):
            return 'greeting'

        # Coding patterns
        if any(word in query_lower for word in ['write', 'create', 'implement', 'build', 'code', 'function', 'script', 'class']):
            return 'coding'
        if any(word in query_lower for word in ['debug', 'fix', 'error', 'bug', 'issue', 'broken', 'exception', 'traceback']):
            return 'debugging'

        # Classification/routing patterns
        if any(word in query_lower for word in ['classify', 'categorize', 'intent', 'route', 'routing']):
            return 'classification'

        # Question patterns
        if query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')) or '?' in query:
            return 'question'

        # Reasoning patterns
        if any(word in query_lower for word in ['analyze', 'think', 'reason', 'plan', 'strategy', 'architecture', 'design']):
            return 'reasoning'

        # Default to conversation
        return 'conversation'

    def get_model_from_database(self, intent: str, domain: str = "general") -> Optional[ModelSelection]:
        """Get model recommendation from database based on benchmark data."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    # Query for exact intent match first
                    cur.execute("""
                        SELECT
                            im.recommended_model,
                            im.reason,
                            im.priority,
                            mc.avg_tokens_per_second,
                            mc.vram_required_gb
                        FROM intent_model_mapping im
                        JOIN model_capabilities mc ON im.recommended_model = mc.model_name
                        WHERE im.intent = %s AND im.domain = %s
                        AND mc.is_installed = true AND mc.is_active = true
                        ORDER BY im.priority ASC
                        LIMIT 1
                    """, (intent, domain))

                    result = cur.fetchone()
                    if result:
                        # Estimate TTFT based on model type (from benchmark data)
                        ttft_ms = 94.0 if 'qwen2.5:3b' in result['recommended_model'] else 100.0

                        return ModelSelection(
                            model=result['recommended_model'],
                            reason=result['reason'],
                            confidence=0.9,  # High confidence from database
                            expected_ttft_ms=ttft_ms,
                            expected_tps=result['avg_tokens_per_second'] or 8.0,
                            intent=intent,
                            domain=domain,
                            fallback_used=False
                        )

        except Exception as e:
            logger.warning(f"Database query failed: {e}")

        return None

    def select_model(self, query: str, domain: str = "general") -> ModelSelection:
        """
        Select the best model for a given query.

        This is the main entry point that all other modules should use.
        """
        # Detect intent
        intent = self.detect_intent(query)

        # Try database first (with benchmark data)
        db_result = self.get_model_from_database(intent, domain)
        if db_result:
            logger.info(f"Selected {db_result.model} for intent '{intent}' from database")
            return db_result

        # Fallback to hardcoded routing (benchmark-optimized)
        model = self.fallback_routing.get(intent, self.fallback_routing['default'])

        # Estimate performance based on benchmark data
        if model == 'qwen2.5:3b':
            ttft_ms, tps = 94.0, 9.7
        elif model == 'qwen2.5-coder:7b':
            ttft_ms, tps = 94.0, 8.7
        else:
            ttft_ms, tps = 100.0, 8.0

        logger.warning(f"Using fallback routing for intent '{intent}': {model}")

        return ModelSelection(
            model=model,
            reason=f"Fallback routing for {intent} (database unavailable)",
            confidence=0.7,  # Lower confidence for fallback
            expected_ttft_ms=ttft_ms,
            expected_tps=tps,
            intent=intent,
            domain=domain,
            fallback_used=True
        )

    def get_model_for_intent(self, intent: str, domain: str = "general") -> str:
        """Simple interface that just returns the model name."""
        selection = self.select_model(f"intent:{intent}", domain)
        return selection.model

    def get_model_for_category(self, category: str) -> str:
        """Get model for a specific category (alias for intent)."""
        return self.get_model_for_intent(category)

# Global instance (singleton pattern)
_router_instance = None

def get_router() -> UnifiedModelRouter:
    """Get the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = UnifiedModelRouter()
    return _router_instance

# Convenience functions for easy imports
def select_model_for_query(query: str, domain: str = "general") -> str:
    """Main function other modules should use."""
    return get_router().select_model(query, domain).model

def get_model_for_intent(intent: str, domain: str = "general") -> str:
    """Get model for specific intent."""
    return get_router().get_model_for_intent(intent, domain)

def get_model_for_category(category: str) -> str:
    """Get model for specific category."""
    return get_router().get_model_for_category(category)

# Legacy compatibility (for modules being migrated)
def get_best_model(query: str) -> str:
    """Legacy function for compatibility."""
    return select_model_for_query(query)

def route_query(query: str) -> str:
    """Legacy function for compatibility."""
    return select_model_for_query(query)

if __name__ == "__main__":
    # Test the router
    router = get_router()

    test_queries = [
        "Hello, how are you?",
        "Fix this Python bug in my code",
        "Write a function to sort a list",
        "What is the difference between lists and tuples?",
        "Classify this user intent: coding request"
    ]

    print("🧠 Unified Model Router Test")
    print("=" * 40)

    for query in test_queries:
        selection = router.select_model(query)
        print(f"Query: {query}")
        print(f"  → {selection.model} ({selection.intent})")
        print(f"    TTFT: {selection.expected_ttft_ms}ms, TPS: {selection.expected_tps}")
        print(f"    Reason: {selection.reason}")
        print()