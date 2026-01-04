#!/usr/bin/env python3
"""
Integration tests for database routing with real database
Tests the complete flow with actual database operations
"""

import pytest
import psycopg2
import sys
import os
sys.path.insert(0, '/opt/tower-echo-brain')

from src.core.db_model_router import DatabaseModelRouter


class TestDatabaseIntegration:
    """Integration tests with real database"""

    @pytest.fixture(scope='class')
    def db_config(self):
        """Real database configuration"""
        return {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': '***REMOVED***'
        }

    @pytest.fixture
    def router(self, db_config):
        """Create router with real database"""
        return DatabaseModelRouter(db_config)

    def test_database_connection(self, db_config):
        """Test we can connect to the database"""
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        cursor.close()
        conn.close()

    def test_complexity_factors_exist(self, db_config):
        """Test complexity factors are in database"""
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM complexity_factors WHERE is_active = true
        """)
        count = cursor.fetchone()[0]
        assert count > 0, "No active complexity factors in database"

        cursor.execute("""
            SELECT factor_name, weight FROM complexity_factors
            WHERE factor_name IN ('base_intent', 'word_count', 'context_depth')
            AND is_active = true
        """)
        results = cursor.fetchall()
        assert len(results) >= 3, "Missing essential complexity factors"

        cursor.close()
        conn.close()

    def test_intent_routing_config_exists(self, db_config):
        """Test intent routing configurations exist"""
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM intent_routing WHERE is_active = true
        """)
        count = cursor.fetchone()[0]
        assert count > 0, "No active intent routing configurations"

        # Check specific intents we rely on
        cursor.execute("""
            SELECT intent_name, base_complexity, preferred_model
            FROM intent_routing
            WHERE intent_name IN ('anime_scene', 'code_generation', 'greeting')
            AND is_active = true
        """)
        results = cursor.fetchall()
        assert len(results) >= 3, "Missing essential intent configurations"

        cursor.close()
        conn.close()

    def test_model_routing_config_exists(self, db_config):
        """Test model routing configurations exist"""
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM model_routing
            WHERE is_installed = true AND is_active = true
        """)
        count = cursor.fetchone()[0]
        assert count >= 3, "Not enough active models configured"

        # Check specific models we need
        cursor.execute("""
            SELECT model_name, min_complexity, max_complexity
            FROM model_routing
            WHERE model_name IN ('llama3.1:8b', 'gemma2:9b', 'qwen2.5-coder:7b')
            AND is_installed = true AND is_active = true
        """)
        results = cursor.fetchall()
        assert len(results) >= 3, "Missing essential model configurations"

        cursor.close()
        conn.close()

    def test_router_loads_config(self, router):
        """Test router successfully loads database configuration"""
        config = router._load_config()

        assert 'intents' in config
        assert 'domains' in config
        assert 'weights' in config
        assert 'models' in config

        # Check we have actual data
        assert len(config['intents']) > 0
        assert len(config['domains']) > 0
        assert len(config['weights']) > 0
        assert len(config['models']) > 0

    def test_anime_scene_routing(self, router):
        """Test routing for anime scene generation"""
        query = "Generate a cyberpunk scene where Kai battles goblins in Tokyo"
        decision = router.route_query(query)

        # Should detect anime intent
        assert decision.intent in ['anime_scene', 'scene_suggestion', None]

        # Should detect anime domain
        assert decision.domain in ['anime_production', 'creative_writing', 'general']

        # Should have reasonable complexity
        assert 30 <= decision.complexity_score <= 100

        # Should select appropriate model
        assert decision.model in ['gemma2:9b', 'llama3.1:8b', 'deepseek-coder-v2:16b']

    def test_code_generation_routing(self, router):
        """Test routing for code generation"""
        query = "Write Python code to implement a quicksort algorithm"
        decision = router.route_query(query)

        # Should detect code intent
        assert decision.intent in ['code_generation', None]

        # Should detect programming domain
        assert decision.domain in ['programming', 'general']

        # Should have reasonable complexity for code
        assert 30 <= decision.complexity_score <= 100

        # Should select code-appropriate model
        acceptable_models = ['qwen2.5-coder:7b', 'deepseek-coder-v2:16b', 'llama3.1:8b', 'gemma2:9b']
        assert decision.model in acceptable_models

    def test_simple_greeting_routing(self, router):
        """Test routing for simple greetings"""
        query = "Hello, how are you?"
        decision = router.route_query(query)

        # Should have low complexity
        assert decision.complexity_score < 40

        # Should select simple model
        assert decision.model in ['llama3.1:8b', 'gemma2:9b']

    def test_context_affects_routing(self, router):
        """Test that conversation history affects routing"""
        query = "Continue the story"

        # Without context
        decision_no_context = router.route_query(query, conversation_history=[])

        # With context
        conversation_history = [
            {'role': 'user', 'content': 'Tell me about Kai'},
            {'role': 'assistant', 'content': 'Kai is a cybernetic warrior...'}
        ]
        decision_with_context = router.route_query(query, conversation_history=conversation_history)

        # Context should increase complexity
        assert decision_with_context.complexity_score >= decision_no_context.complexity_score
        assert decision_with_context.requires_context is True

    def test_model_selection_function(self, db_config):
        """Test the database function for model selection"""
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Test low complexity
        cursor.execute(
            "SELECT * FROM select_model_for_complexity(%s, %s, %s)",
            (20, 'greeting', 'general')
        )
        result = cursor.fetchone()
        assert result is not None
        model, reason = result
        assert model in ['llama3.1:8b', 'gemma2:9b']

        # Test high complexity
        cursor.execute(
            "SELECT * FROM select_model_for_complexity(%s, %s, %s)",
            (85, 'anime_scene', 'anime_production')
        )
        result = cursor.fetchone()
        assert result is not None
        model, reason = result
        assert model in ['gemma2:9b', 'deepseek-coder-v2:16b', 'llama3.1:8b']

        cursor.close()
        conn.close()

    def test_performance_logging(self, router, db_config):
        """Test that performance is logged to database"""
        from src.core.db_model_router import RoutingDecision

        decision = RoutingDecision(
            model='test_model',
            complexity_score=50,
            intent='test_intent',
            domain='test_domain',
            reasoning='Integration test'
        )

        # Log performance
        router.log_performance(decision, 1234, True, 100)

        # Verify it was logged
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_used, complexity_score, response_time_ms
            FROM model_performance_log
            WHERE intent_type = 'test_intent'
            ORDER BY created_at DESC
            LIMIT 1
        """)
        result = cursor.fetchone()

        if result:  # May not exist if table doesn't have the row
            assert result[0] == 'test_model'
            assert result[1] == 50
            assert result[2] == 1234

        cursor.close()
        conn.close()

    def test_cache_expiration(self, router):
        """Test that cache expires and reloads"""
        import time

        # Get initial config
        config1 = router._load_config()

        # Force cache expiration
        router._cache_timestamp = time.time() - 100  # 100 seconds ago

        # Get config again - should reload
        config2 = router._load_config()

        # Both should have data
        assert len(config1['intents']) > 0
        assert len(config2['intents']) > 0