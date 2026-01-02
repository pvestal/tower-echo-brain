#!/usr/bin/env python3
"""
Database-driven intelligent model router
Replaces the brain-dead word count complexity with actual intelligence
"""

import re
import time
import logging
import psycopg2
import psycopg2.extras
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    """Model routing decision with reasoning"""
    model: str
    complexity_score: int
    intent: Optional[str]
    domain: Optional[str]
    reasoning: str
    max_tokens: int = 2000
    temperature: float = 0.7
    requires_context: bool = True
    memory_window: int = 5


class DatabaseModelRouter:
    """Intelligent model router using database configuration"""

    def __init__(self, db_config: Dict[str, str]):
        """Initialize with database connection"""
        self.db_config = db_config
        self._cache = {}
        self._cache_timestamp = 0
        self.cache_duration = 60  # seconds

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    @lru_cache(maxsize=1)
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from database (cached for 60 seconds)"""
        current_time = time.time()

        # Check cache validity
        if self._cache_timestamp and (current_time - self._cache_timestamp < self.cache_duration):
            return self._cache

        logger.info("Loading configuration from database")

        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        try:
            # Load intent routing
            cursor.execute("""
                SELECT intent_name, base_complexity, keywords, preferred_model,
                       requires_context, requires_memory, requires_reasoning
                FROM intent_routing WHERE is_active = true
            """)
            intents = {row['intent_name']: dict(row) for row in cursor.fetchall()}

            # Load domain complexity
            cursor.execute("""
                SELECT domain_name, complexity_multiplier, keywords, preferred_model
                FROM domain_complexity WHERE is_active = true
            """)
            domains = {row['domain_name']: dict(row) for row in cursor.fetchall()}

            # Load complexity weights
            cursor.execute("""
                SELECT factor_name, weight FROM complexity_weights WHERE is_active = true
            """)
            weights = {row['factor_name']: row['weight'] for row in cursor.fetchall()}

            # Load active models
            cursor.execute("""
                SELECT model_name, min_complexity, max_complexity, priority, quality_tier
                FROM model_routing
                WHERE is_installed = true AND is_active = true
                ORDER BY priority ASC
            """)
            models = [dict(row) for row in cursor.fetchall()]

            self._cache = {
                'intents': intents,
                'domains': domains,
                'weights': weights,
                'models': models
            }
            self._cache_timestamp = current_time

        finally:
            cursor.close()
            conn.close()

        return self._cache

    def detect_intent(self, query: str) -> Optional[str]:
        """Detect intent from query using database patterns"""
        config = self._load_config()
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for intent_name, intent_config in config['intents'].items():
            keywords = intent_config.get('keywords', [])
            if not keywords:
                continue

            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)

            # Score based on match percentage
            if matches > 0:
                score = matches / len(keywords)
                if score > best_score:
                    best_score = score
                    best_match = intent_name

        # Need at least 20% keyword match
        if best_score >= 0.2:
            logger.info(f"Detected intent: {best_match} (score: {best_score:.2f})")
            return best_match

        return None

    def detect_domain(self, query: str) -> Optional[str]:
        """Detect domain from query using database patterns"""
        config = self._load_config()
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for domain_name, domain_config in config['domains'].items():
            keywords = domain_config.get('keywords', [])
            if not keywords:
                continue

            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)

            # Score based on match percentage
            if matches > 0:
                score = matches / len(keywords)
                if score > best_score:
                    best_score = score
                    best_match = domain_name

        # Need at least 15% keyword match for domain
        if best_score >= 0.15:
            logger.info(f"Detected domain: {best_match} (score: {best_score:.2f})")
            return best_match

        return 'general'  # default domain

    def calculate_complexity(self, query: str, has_context: bool = False,
                           intent: Optional[str] = None,
                           domain: Optional[str] = None) -> int:
        """Calculate complexity score using database weights"""
        config = self._load_config()
        weights = config['weights']
        score = 0

        # Base complexity from intent
        if intent and intent in config['intents']:
            score = config['intents'][intent].get('base_complexity', 20)
        else:
            score = 20

        # Word count factor (minor)
        word_count = len(query.split())
        score += word_count * weights.get('word_count', 0.05)

        # Question marks (multi-part queries)
        question_count = query.count('?')
        score += question_count * weights.get('question_marks', 8)

        # Context depth
        if has_context:
            score += weights.get('context_depth', 15)

        # Check for memory requirements
        memory_patterns = ['remember', 'previous', 'we discussed', 'earlier', 'last time']
        if any(pattern in query.lower() for pattern in memory_patterns):
            score += weights.get('memory_lookup', 20)

        # Check for reasoning requirements
        reasoning_patterns = ['why', 'because', 'if', 'then', 'therefore', 'implies']
        if any(pattern in query.lower() for pattern in reasoning_patterns):
            score += weights.get('reasoning_steps', 25)

        # Check for structured output
        structure_patterns = ['json', 'format', 'return only', 'structure', 'schema']
        if any(pattern in query.lower() for pattern in structure_patterns):
            score += weights.get('structured_output', 15)

        # Domain multiplier
        if domain and domain in config['domains']:
            multiplier = config['domains'][domain].get('complexity_multiplier', 1.0)
            score = int(score * multiplier)

        # Cap at 100
        return min(int(score), 100)

    def select_model(self, complexity: int, intent: Optional[str] = None,
                    domain: Optional[str] = None) -> Tuple[str, str]:
        """Select best model using database function"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT * FROM select_model_for_complexity(%s, %s, %s)",
                (complexity, intent, domain)
            )
            result = cursor.fetchone()

            if result:
                return result[0], result[1]  # model_name, reason
            else:
                return 'llama3.1:8b', 'Fallback model'

        finally:
            cursor.close()
            conn.close()

    def route_query(self, query: str, conversation_history: List[Dict] = None) -> RoutingDecision:
        """
        Main routing function - analyzes query and returns routing decision

        Args:
            query: User query text
            conversation_history: List of previous messages

        Returns:
            RoutingDecision with model selection and reasoning
        """
        # Detect intent and domain
        intent = self.detect_intent(query)
        domain = self.detect_domain(query)

        # Calculate complexity
        has_context = bool(conversation_history)
        complexity = self.calculate_complexity(query, has_context, intent, domain)

        # Select model
        model, reasoning = self.select_model(complexity, intent, domain)

        # Get additional config
        config = self._load_config()
        requires_context = False
        memory_window = 5

        if intent and intent in config['intents']:
            intent_config = config['intents'][intent]
            requires_context = intent_config.get('requires_context', False)
            memory_window = 10 if requires_context else 5

        # Build routing decision
        decision = RoutingDecision(
            model=model,
            complexity_score=complexity,
            intent=intent,
            domain=domain,
            reasoning=f"{reasoning}. Complexity: {complexity}, Intent: {intent or 'general'}, Domain: {domain or 'general'}",
            requires_context=requires_context or has_context,
            memory_window=memory_window
        )

        logger.info(f"Routing decision: {model} (complexity: {complexity}, intent: {intent}, domain: {domain})")

        return decision

    def log_performance(self, decision: RoutingDecision, response_time_ms: int,
                       success: bool, tokens_used: int = None, error: str = None):
        """Log model performance for learning"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO model_performance_log
                (intent_type, complexity_score, model_used, response_time_ms,
                 success, tokens_used, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                decision.intent,
                decision.complexity_score,
                decision.model,
                response_time_ms,
                success,
                tokens_used,
                error
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log performance: {e}")
        finally:
            cursor.close()
            conn.close()


# Global instance
db_config = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': 'tower_echo_brain_secret_key_2025'
}

intelligent_router = DatabaseModelRouter(db_config)


if __name__ == "__main__":
    # Test the router
    test_queries = [
        "Generate a scene where Kai hunts goblins in the Frankfurt subway",
        "Write Python code to sort a list",
        "Hi, how are you?",
        "Continue the story from where we left off with Mei confronting the yakuza",
        "Debug this error: TypeError: 'NoneType' object is not iterable",
        "Return only JSON: {scene, mood, action}"
    ]

    print("Testing Database Model Router")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query[:60]}...")
        decision = intelligent_router.route_query(query, conversation_history=[])
        print(f"  Model: {decision.model}")
        print(f"  Complexity: {decision.complexity_score}")
        print(f"  Intent: {decision.intent or 'None detected'}")
        print(f"  Domain: {decision.domain or 'general'}")
        print(f"  Reasoning: {decision.reasoning}")