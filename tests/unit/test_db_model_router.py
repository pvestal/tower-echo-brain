#!/usr/bin/env python3
"""
Unit tests for database-driven model router
Tests intelligent complexity scoring and model selection
"""

import pytest
import sys
import os
sys.path.insert(0, '/opt/tower-echo-brain')

from unittest.mock import Mock, patch, MagicMock
from src.core.db_model_router import DatabaseModelRouter, RoutingDecision


class TestDatabaseModelRouter:
    """Test suite for intelligent database-driven model routing"""

    @pytest.fixture
    def mock_db_config(self):
        """Mock database configuration"""
        return {
            'host': 'localhost',
            'database': 'test_db',
            'user': 'test',
            'password': 'test'
        }

    @pytest.fixture
    def router(self, mock_db_config):
        """Create router with mocked database"""
        with patch('psycopg2.connect') as mock_connect:
            router = DatabaseModelRouter(mock_db_config)
            # Mock the config loading
            router._cache = {
                'intents': {
                    'anime_scene': {
                        'intent_name': 'anime_scene',
                        'base_complexity': 60,
                        'keywords': ['scene', 'anime', 'generate', 'tokyo', 'cyberpunk'],
                        'preferred_model': 'gemma2:9b',
                        'requires_context': True,
                        'requires_memory': True,
                        'requires_reasoning': True
                    },
                    'code_generation': {
                        'intent_name': 'code_generation',
                        'base_complexity': 50,
                        'keywords': ['write', 'code', 'function', 'python', 'implement'],
                        'preferred_model': 'qwen2.5-coder:7b',
                        'requires_context': False,
                        'requires_memory': False,
                        'requires_reasoning': True
                    },
                    'greeting': {
                        'intent_name': 'greeting',
                        'base_complexity': 5,
                        'keywords': ['hello', 'hi', 'hey'],
                        'preferred_model': 'llama3.1:8b',
                        'requires_context': False,
                        'requires_memory': False,
                        'requires_reasoning': False
                    }
                },
                'domains': {
                    'anime_production': {
                        'domain_name': 'anime_production',
                        'complexity_multiplier': 1.4,
                        'keywords': ['anime', 'scene', 'mei', 'kai', 'tokyo'],
                        'preferred_model': 'gemma2:9b'
                    },
                    'programming': {
                        'domain_name': 'programming',
                        'complexity_multiplier': 1.3,
                        'keywords': ['code', 'python', 'function', 'debug'],
                        'preferred_model': 'deepseek-coder-v2:16b'
                    }
                },
                'weights': {
                    'base_intent': 1.0,
                    'word_count': 0.05,
                    'question_marks': 8.0,
                    'context_depth': 15.0,
                    'memory_lookup': 20.0,
                    'reasoning_steps': 25.0,
                    'structured_output': 15.0
                },
                'models': [
                    {'model_name': 'llama3.1:8b', 'min_complexity': 0, 'max_complexity': 60},
                    {'model_name': 'gemma2:9b', 'min_complexity': 20, 'max_complexity': 70},
                    {'model_name': 'qwen2.5-coder:7b', 'min_complexity': 30, 'max_complexity': 80},
                    {'model_name': 'deepseek-coder-v2:16b', 'min_complexity': 40, 'max_complexity': 100}
                ]
            }
            router._cache_timestamp = 9999999999  # Far future to avoid reload
            return router

    def test_intent_detection_anime(self, router):
        """Test intent detection for anime queries"""
        query = "Generate a cyberpunk anime scene in Tokyo"
        intent = router.detect_intent(query)

        assert intent == 'anime_scene'

    def test_intent_detection_code(self, router):
        """Test intent detection for code queries"""
        query = "Write Python code to implement a binary search"
        intent = router.detect_intent(query)

        assert intent == 'code_generation'

    def test_intent_detection_simple(self, router):
        """Test intent detection for simple queries"""
        query = "Hello, how are you?"
        intent = router.detect_intent(query)

        assert intent == 'greeting'

    def test_intent_detection_none(self, router):
        """Test intent detection when no match"""
        query = "Random gibberish that matches nothing"
        intent = router.detect_intent(query)

        assert intent is None

    def test_domain_detection_anime(self, router):
        """Test domain detection for anime content"""
        query = "Create a scene with Kai and Mei in Tokyo"
        domain = router.detect_domain(query)

        assert domain == 'anime_production'

    def test_domain_detection_programming(self, router):
        """Test domain detection for programming"""
        query = "Debug this Python function"
        domain = router.detect_domain(query)

        assert domain == 'programming'

    def test_domain_detection_general(self, router):
        """Test domain detection defaults to general"""
        query = "What's the weather today?"
        domain = router.detect_domain(query)

        assert domain == 'general'

    def test_complexity_calculation_simple(self, router):
        """Test complexity calculation for simple query"""
        query = "Hello"
        complexity = router.calculate_complexity(query, has_context=False)

        # Simple query should have low complexity
        assert complexity < 30
        assert complexity >= 0

    def test_complexity_calculation_with_context(self, router):
        """Test complexity increases with context"""
        query = "Continue the story"

        # Without context
        complexity_no_context = router.calculate_complexity(query, has_context=False)

        # With context
        complexity_with_context = router.calculate_complexity(query, has_context=True)

        # Context should increase complexity
        assert complexity_with_context > complexity_no_context
        assert complexity_with_context - complexity_no_context >= 10  # Context adds at least 10

    def test_complexity_calculation_with_memory(self, router):
        """Test complexity increases with memory requirements"""
        query = "Based on what we discussed earlier about the character"
        complexity = router.calculate_complexity(query, has_context=True)

        # Memory-requiring query should have higher complexity
        assert complexity >= 40

    def test_complexity_calculation_with_reasoning(self, router):
        """Test complexity increases with reasoning requirements"""
        query = "If we change this parameter, then what happens to the output and why?"
        complexity = router.calculate_complexity(query)

        # Reasoning query should have higher complexity
        assert complexity >= 35

    def test_complexity_calculation_structured_output(self, router):
        """Test complexity increases for structured output"""
        query = "Return only JSON format with the following structure"
        complexity = router.calculate_complexity(query)

        # Structured output should increase complexity
        assert complexity >= 30

    def test_complexity_with_domain_multiplier(self, router):
        """Test domain multiplier affects complexity"""
        query = "Generate an anime scene"

        # Calculate with anime domain (1.4x multiplier)
        complexity_anime = router.calculate_complexity(
            query,
            intent='anime_scene',
            domain='anime_production'
        )

        # Calculate without domain multiplier
        complexity_base = router.calculate_complexity(
            query,
            intent='anime_scene',
            domain='general'
        )

        # Anime domain should multiply complexity
        assert complexity_anime > complexity_base
        assert complexity_anime >= complexity_base * 1.3  # At least 1.3x

    def test_route_query_anime_scene(self, router):
        """Test complete routing for anime scene generation"""
        with patch.object(router, 'select_model', return_value=('gemma2:9b', 'Intent-based selection')):
            query = "Generate a cyberpunk scene where Mei confronts yakuza"
            decision = router.route_query(query, conversation_history=[])

            assert isinstance(decision, RoutingDecision)
            assert decision.model == 'gemma2:9b'
            assert decision.intent == 'anime_scene'
            assert decision.domain == 'anime_production'
            assert decision.complexity_score >= 60  # Should be high for anime scene
            assert decision.requires_context is True
            assert decision.memory_window >= 5

    def test_route_query_code_generation(self, router):
        """Test complete routing for code generation"""
        with patch.object(router, 'select_model', return_value=('qwen2.5-coder:7b', 'Intent-based selection')):
            query = "Write a Python function to sort a list"
            decision = router.route_query(query)

            assert decision.model == 'qwen2.5-coder:7b'
            assert decision.intent == 'code_generation'
            assert decision.domain == 'programming'
            assert decision.complexity_score >= 40

    def test_route_query_simple_greeting(self, router):
        """Test complete routing for simple greeting"""
        with patch.object(router, 'select_model', return_value=('llama3.1:8b', 'Complexity-based selection')):
            query = "Hello, how are you?"
            decision = router.route_query(query)

            assert decision.model == 'llama3.1:8b'
            assert decision.intent == 'greeting'
            assert decision.complexity_score < 30  # Should be low for greeting
            assert decision.requires_context is False

    def test_model_selection_database_query(self, router):
        """Test model selection via database function"""
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_connect.return_value.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = ('deepseek-coder-v2:16b', 'Complexity-based selection')

            model, reason = router.select_model(75, 'code_generation', 'programming')

            assert model == 'deepseek-coder-v2:16b'
            assert 'Complexity-based' in reason

            # Verify SQL was called correctly
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert 'select_model_for_complexity' in call_args[0]
            assert call_args[1] == (75, 'code_generation', 'programming')

    def test_performance_logging(self, router):
        """Test that performance is logged correctly"""
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_connect.return_value.cursor.return_value = mock_cursor

            decision = RoutingDecision(
                model='test_model',
                complexity_score=50,
                intent='test_intent',
                domain='test_domain',
                reasoning='Test reasoning'
            )

            router.log_performance(decision, 1500, True, 250, None)

            # Verify INSERT was called
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert 'INSERT INTO model_performance_log' in call_args[0]

            # Check parameters
            params = mock_cursor.execute.call_args[0][1]
            assert params[0] == 'test_intent'
            assert params[1] == 50
            assert params[2] == 'test_model'
            assert params[3] == 1500
            assert params[4] is True
            assert params[5] == 250
            assert params[6] is None


class TestRoutingDecision:
    """Test the RoutingDecision dataclass"""

    def test_routing_decision_creation(self):
        """Test creating a routing decision"""
        decision = RoutingDecision(
            model='gemma2:9b',
            complexity_score=75,
            intent='anime_scene',
            domain='anime_production',
            reasoning='Complex anime scene requiring creative model'
        )

        assert decision.model == 'gemma2:9b'
        assert decision.complexity_score == 75
        assert decision.intent == 'anime_scene'
        assert decision.domain == 'anime_production'
        assert decision.max_tokens == 2000  # Default
        assert decision.temperature == 0.7  # Default
        assert decision.requires_context is True  # Default
        assert decision.memory_window == 5  # Default

    def test_routing_decision_custom_values(self):
        """Test routing decision with custom values"""
        decision = RoutingDecision(
            model='llama3.1:8b',
            complexity_score=25,
            intent='simple_query',
            domain=None,
            reasoning='Simple query',
            max_tokens=500,
            temperature=0.5,
            requires_context=False,
            memory_window=1
        )

        assert decision.max_tokens == 500
        assert decision.temperature == 0.5
        assert decision.requires_context is False
        assert decision.memory_window == 1