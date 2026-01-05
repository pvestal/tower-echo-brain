"""Database-driven intelligent model routing"""

import logging
import hashlib
import json
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime

logger = logging.getLogger(__name__)

class RoutingDecision:
    """Represents a routing decision with full context"""

    def __init__(self, model: str = "llama3.2:3b", reason: str = "Default fallback", confidence: float = 0.5):
        self.model = model
        self.reason = reason
        self.confidence = confidence
        self.complexity_score = 50
        self.intent = "general"
        self.domain = "general"
        self.temperature = 0.7
        self.max_tokens = 2048
        self.output_format = "prose"

    def __str__(self):
        return f"RoutingDecision(model={self.model}, intent={self.intent}, confidence={self.confidence:.2f})"

class QueryAnalysis:
    """Full analysis of incoming query"""

    def __init__(self, query: str):
        self.query = query
        self.intent = self._detect_intent(query)
        self.domain = self._detect_domain(query)
        self.complexity = self._calculate_complexity(query)
        self.needs_vision = self._needs_vision(query)
        self.output_format = self._detect_output_format(query)

    def _detect_intent(self, query: str) -> str:
        """Detect intent from query using patterns"""
        query_lower = query.lower()

        # Code-related intents
        if any(word in query_lower for word in ['write', 'create', 'implement', 'build', 'code', 'function', 'script']):
            return 'code_generation'
        if any(word in query_lower for word in ['debug', 'fix', 'error', 'bug', 'issue', 'broken']):
            return 'code_debugging'
        if any(word in query_lower for word in ['explain', 'how', 'why', 'what']):
            return 'explanation'
        if any(word in query_lower for word in ['analyze', 'review', 'check', 'audit']):
            return 'error_analysis'

        # Creative and conversation
        if any(word in query_lower for word in ['story', 'creative', 'imagine', 'describe']):
            return 'creative'
        if any(word in query_lower for word in ['chat', 'talk', 'hello', 'hi']):
            return 'conversation'

        # System queries
        if any(word in query_lower for word in ['memory', 'remember', 'recall', 'history']):
            return 'memory_search'
        if any(word in query_lower for word in ['system', 'status', 'health', 'running']):
            return 'system_query'

        # Image analysis
        if any(word in query_lower for word in ['image', 'picture', 'photo', 'visual', 'screenshot']):
            return 'image_analysis'

        return 'question'  # Default intent

    def _detect_domain(self, query: str) -> str:
        """Detect domain from query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['echo', 'brain', 'routing', 'model']):
            return 'echo_brain'
        if any(word in query_lower for word in ['anime', 'comfy', 'lora', 'workflow']):
            return 'anime'
        if any(word in query_lower for word in ['tower', 'service', 'nginx', 'systemctl']):
            return 'system'
        if any(word in query_lower for word in ['database', 'postgres', 'sql', 'table']):
            return 'database'

        # Check for simple queries that don't need domain-specific knowledge
        if len(query.split()) < 10 and self.intent in ['question', 'conversation']:
            return 'simple'

        return 'general'

    def _calculate_complexity(self, query: str) -> int:
        """Calculate query complexity (0-100)"""
        words = query.split()
        word_count = len(words)

        # Base complexity from word count
        if word_count < 10:
            complexity = 10
        elif word_count < 30:
            complexity = 30
        elif word_count < 100:
            complexity = 50
        else:
            complexity = 70

        # Adjust for code blocks
        if '```' in query:
            complexity = min(100, complexity + 20)

        # Adjust for technical terms
        technical_terms = ['function', 'class', 'async', 'await', 'SELECT', 'FROM', 'WHERE']
        tech_count = sum(1 for term in technical_terms if term in query)
        complexity = min(100, complexity + (tech_count * 5))

        return complexity

    def _needs_vision(self, query: str) -> bool:
        """Check if query requires vision model"""
        vision_keywords = ['image', 'picture', 'photo', 'screenshot', 'visual', 'see', 'look at']
        return any(keyword in query.lower() for keyword in vision_keywords)

    def _detect_output_format(self, query: str) -> str:
        """Detect expected output format"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['code', 'function', 'script', 'implement']):
            return 'code'
        if any(word in query_lower for word in ['json', 'structured', 'format']):
            return 'json'
        if any(word in query_lower for word in ['creative', 'story', 'describe']):
            return 'creative'
        if any(word in query_lower for word in ['debug', 'fix', 'analyze']):
            return 'debug'

        return 'prose'

class DatabaseModelRouter:
    """Routes queries to models using database configuration"""

    def __init__(self):
        self.db_config = {
            'dbname': 'tower_consolidated',
            'user': 'patrick',
            'password': 'tower_echo_brain_secret_key_2025',
            'host': 'localhost'
        }
        self._query_cache = {}
        logger.info("DatabaseModelRouter initialized")

    def route_query(self, query: str, conversation_history: List[Dict] = None) -> RoutingDecision:
        """Route query using database-driven logic"""

        try:
            # Analyze the query
            analysis = QueryAnalysis(query)
            logger.info(f"Query analysis: intent={analysis.intent}, domain={analysis.domain}, "
                       f"complexity={analysis.complexity}, needs_vision={analysis.needs_vision}")

            # Check cache
            cache_key = hashlib.md5(f"{analysis.intent}:{analysis.domain}:{analysis.complexity}:{analysis.needs_vision}".encode()).hexdigest()
            if cache_key in self._query_cache:
                logger.info(f"Cache hit for routing decision")
                cached = self._query_cache[cache_key]
                return RoutingDecision(
                    model=cached['model'],
                    reason=cached['reason'],
                    confidence=cached['confidence']
                )

            # Query database for model selection
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    # Call the select_model function
                    cur.execute("""
                        SELECT * FROM select_model(
                            p_intent := %s,
                            p_domain := %s,
                            p_complexity := %s,
                            p_output_format := %s,
                            p_require_vision := %s
                        )
                    """, (analysis.intent, analysis.domain, analysis.complexity,
                         analysis.output_format, analysis.needs_vision))

                    result = cur.fetchone()

                    if result:
                        decision = RoutingDecision(
                            model=result['model_name'],
                            reason=result['reason'],
                            confidence=result['confidence']
                        )
                        decision.intent = analysis.intent
                        decision.domain = analysis.domain
                        decision.complexity_score = analysis.complexity
                        decision.output_format = analysis.output_format

                        # Cache the decision
                        self._query_cache[cache_key] = {
                            'model': decision.model,
                            'reason': decision.reason,
                            'confidence': decision.confidence
                        }

                        # Log the routing decision
                        self._log_routing_decision(query, analysis, decision)

                        logger.info(f"Database routing: {decision}")
                        return decision
                    else:
                        logger.warning("No model found in database, using fallback")
                        return RoutingDecision()

        except Exception as e:
            logger.error(f"Database routing failed: {e}")
            return RoutingDecision(reason=f"Database error: {str(e)}")

    def _log_routing_decision(self, query: str, analysis: QueryAnalysis, decision: RoutingDecision):
        """Log routing decision to database for analysis"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Store in cache table
                    cur.execute("""
                        INSERT INTO query_analysis_cache
                        (query_hash, query_text, intent, domain, complexity,
                         needs_codebase, needs_memory, needs_history, needs_system_state,
                         output_format, model_selected, confidence)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (query_hash) DO UPDATE SET
                            model_selected = EXCLUDED.model_selected,
                            confidence = EXCLUDED.confidence,
                            created_at = NOW()
                    """, (
                        hashlib.md5(query.encode()).hexdigest(),
                        query[:500],  # Truncate long queries
                        analysis.intent,
                        analysis.domain,
                        analysis.complexity,
                        False,  # needs_codebase - will implement later
                        False,  # needs_memory - will implement later
                        False,  # needs_history - will implement later
                        False,  # needs_system_state - will implement later
                        analysis.output_format,
                        decision.model,
                        decision.confidence
                    ))
                    conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log routing decision: {e}")

    def log_performance(self, model: str, response_time_ms: int, tokens_generated: int, success: bool):
        """Log model performance for future optimization"""
        # Temporarily disabled - table structure mismatch
        return
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO model_performance
                        (model_name, response_time_ms, tokens_generated, success, timestamp)
                        VALUES (%s, %s, %s, %s, NOW())
                    """, (model, response_time_ms, tokens_generated, success))
                    conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log performance: {e}")

    def get_user_preferences(self, category: str = None) -> Dict[str, str]:
        """Get user preferences from database"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    if category:
                        cur.execute("""
                            SELECT preference_key, preference_value
                            FROM user_preferences
                            WHERE category = %s
                        """, (category,))
                    else:
                        cur.execute("""
                            SELECT category, preference_key, preference_value
                            FROM user_preferences
                        """)

                    results = cur.fetchall()
                    preferences = {}
                    for row in results:
                        if category:
                            preferences[row['preference_key']] = row['preference_value']
                        else:
                            key = f"{row['category']}.{row['preference_key']}"
                            preferences[key] = row['preference_value']

                    return preferences
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return {}

# Create singleton instance
intelligent_router = DatabaseModelRouter()

# Backward compatibility
def get_model_for_query(query: str, conversation_history: List[Dict] = None) -> str:
    """Simple interface for model selection"""
    decision = intelligent_router.route_query(query, conversation_history)
    return decision.model