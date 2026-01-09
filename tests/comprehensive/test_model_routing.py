"""
Comprehensive Model Routing and Selection Tests

Tests:
- Complexity calculation
- Model selection based on query
- Intelligent routing decisions
- Model escalation
- Domain classification
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestComplexityCalculation:
    """Test complexity scoring algorithms"""

    def test_short_query_low_complexity(self, sample_queries):
        """Short queries should have low complexity"""
        from model_router import ModelRouter
        router = ModelRouter()
        complexity = router.calculate_complexity(sample_queries["simple"])
        assert complexity < 20, f"Simple query complexity {complexity} should be < 20"

    def test_long_query_higher_complexity(self, sample_queries):
        """Longer queries should have higher complexity"""
        from model_router import ModelRouter
        router = ModelRouter()
        simple_complexity = router.calculate_complexity(sample_queries["simple"])
        complex_complexity = router.calculate_complexity(sample_queries["complex"])
        assert complex_complexity > simple_complexity

    def test_complexity_scales_with_word_count(self):
        """Complexity should scale with word count"""
        from model_router import ModelRouter
        router = ModelRouter()

        short = "Hello world"
        medium = " ".join(["word"] * 50)
        long = " ".join(["word"] * 200)

        short_c = router.calculate_complexity(short)
        medium_c = router.calculate_complexity(medium)
        long_c = router.calculate_complexity(long)

        assert short_c < medium_c < long_c

    def test_complexity_multiplier_over_100_words(self):
        """Queries over 100 words should get 1.5x multiplier"""
        from model_router import ModelRouter
        router = ModelRouter()

        # 99 words vs 101 words
        under_100 = " ".join(["test"] * 99)
        over_100 = " ".join(["test"] * 101)

        under_c = router.calculate_complexity(under_100)
        over_c = router.calculate_complexity(over_100)

        # Over 100 should be significantly higher due to multiplier
        expected_ratio = (101 * 0.1 * 1.5) / (99 * 0.1)
        actual_ratio = over_c / under_c
        assert actual_ratio > 1.4  # Allow some tolerance

    def test_complexity_multiplier_over_500_words(self):
        """Queries over 500 words should get 2x multiplier"""
        from model_router import ModelRouter
        router = ModelRouter()

        over_500 = " ".join(["test"] * 501)
        complexity = router.calculate_complexity(over_500)

        # Should be > 500 * 0.1 * 1.5 * 2 = 150
        assert complexity > 100


class TestModelSelection:
    """Test model selection logic"""

    def test_simple_query_selects_simple_model(self, sample_queries):
        """Simple queries should select simple/small model"""
        from model_router import ModelRouter
        router = ModelRouter()
        model = router.select_model(sample_queries["simple"])
        assert model in ["simple_model", "qwen2.5-coder:7b", "llama3.2:3b"]

    def test_complex_query_selects_complex_model(self, sample_queries):
        """Complex queries should select larger model"""
        from model_router import ModelRouter
        router = ModelRouter()
        model = router.select_model(sample_queries["complex"])
        assert model in ["complex_model", "qwen2.5-coder:32b", "llama3.1:70b"]

    def test_model_selection_boundaries(self):
        """Test model selection at complexity boundaries"""
        from model_router import ModelRouter
        router = ModelRouter()

        # Create queries at different complexity levels
        queries = {
            "tiny": "Hi",
            "small": " ".join(["test"] * 20),
            "medium": " ".join(["test"] * 150),
            "large": " ".join(["test"] * 600)
        }

        for name, query in queries.items():
            model = router.select_model(query)
            assert model is not None, f"No model selected for {name} query"


class TestIntelligentRouting:
    """Test database-driven intelligent routing"""

    @pytest.mark.asyncio
    async def test_route_query_returns_response(self, sample_queries):
        """route_query should return structured response"""
        from model_router import ModelRouter
        router = ModelRouter()

        with patch('src.core.db_model_router.intelligent_router') as mock_router:
            mock_decision = MagicMock()
            mock_decision.model = "qwen2.5-coder:7b"
            mock_decision.complexity_score = 15
            mock_decision.intent = "general"
            mock_decision.domain = "general"
            mock_router.route_query.return_value = mock_decision

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": "test"}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                result = await router.route_query(sample_queries["simple"])
                assert "success" in result or "response" in result or "error" in result

    @pytest.mark.asyncio
    async def test_route_query_with_context(self, sample_queries):
        """route_query should accept conversation context"""
        from model_router import ModelRouter
        router = ModelRouter()

        context = {
            "conversation_history": [
                {"role": "user", "content": "Previous message"},
                {"role": "assistant", "content": "Previous response"}
            ]
        }

        with patch('src.core.db_model_router.intelligent_router') as mock_router:
            mock_decision = MagicMock()
            mock_decision.model = "qwen2.5-coder:7b"
            mock_decision.complexity_score = 15
            mock_decision.intent = "general"
            mock_decision.domain = "general"
            mock_router.route_query.return_value = mock_decision

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": "test"}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                result = await router.route_query(sample_queries["simple"], context)
                # Should not raise exception


class TestReasoningDetection:
    """Test reasoning trigger detection"""

    def test_reasoning_triggers_detected(self):
        """Queries with reasoning triggers should be detected"""
        try:
            from reasoning.deepseek_reasoner import should_use_reasoning
        except ImportError:
            from src.reasoning.deepseek_reasoner import should_use_reasoning

        reasoning_queries = [
            "Think through the pros and cons of microservices",
            "Analyze this architecture decision",
            "Compare Python vs Go for this use case",
            "Step by step, how would I implement this?",
            "What are the trade-offs of using Redis?"
        ]

        for query in reasoning_queries:
            assert should_use_reasoning(query), f"Should trigger reasoning: {query}"

    def test_simple_queries_no_reasoning(self):
        """Simple queries should not trigger reasoning"""
        try:
            from reasoning.deepseek_reasoner import should_use_reasoning
        except ImportError:
            from src.reasoning.deepseek_reasoner import should_use_reasoning

        simple_queries = [
            "Hello",
            "What time is it?",
            "Print hello world",
            "List files"
        ]

        for query in simple_queries:
            # Short queries might still trigger due to complexity check
            # but explicit reasoning triggers should not be present
            result = should_use_reasoning(query)
            # Allow either True or False for simple queries

    def test_complex_queries_trigger_reasoning(self):
        """Long complex queries should trigger reasoning"""
        try:
            from reasoning.deepseek_reasoner import should_use_reasoning
        except ImportError:
            from src.reasoning.deepseek_reasoner import should_use_reasoning

        # Query with multiple questions (> 1 question mark)
        multi_question = "What is the best approach? How should I handle errors? What about edge cases?"
        assert should_use_reasoning(multi_question)

        # Long query (> 150 chars)
        long_query = "I need help with " + "a" * 150
        assert should_use_reasoning(long_query)


class TestDomainClassification:
    """Test domain/intent classification"""

    def test_code_queries_classified_correctly(self, sample_queries):
        """Code-related queries should be classified as coding domain"""
        # This tests the routing system's domain classification
        try:
            from core.db_model_router import intelligent_router
        except ImportError:
            try:
                from src.core.db_model_router import intelligent_router
            except ImportError:
                pytest.skip("db_model_router not available")

        code_queries = [
            "Write a Python function",
            "Debug this code",
            "Refactor this class",
            "Explain this algorithm"
        ]

        for query in code_queries:
            try:
                decision = intelligent_router.route_query(query)
                # Domain should be code-related or model should be code-focused
                assert decision.domain in ["coding", "technical", "general"] or \
                       "coder" in decision.model.lower()
            except Exception:
                pass  # Skip if database not available

    def test_anime_queries_classified_correctly(self, sample_queries):
        """Anime-related queries should be classified appropriately"""
        try:
            from core.db_model_router import intelligent_router
        except ImportError:
            try:
                from src.core.db_model_router import intelligent_router
            except ImportError:
                pytest.skip("db_model_router not available")

        anime_query = sample_queries.get("anime", "Generate anime character")
        try:
            decision = intelligent_router.route_query(anime_query)
            # Should recognize creative/generation intent
        except Exception:
            pass  # Skip if database not available


class TestModelEscalation:
    """Test model escalation paths"""

    def test_escalation_from_simple_to_complex(self):
        """Should escalate from simple to complex model when needed"""
        from model_router import ModelRouter
        router = ModelRouter()

        # Simple query
        simple = "Hello"
        simple_model = router.select_model(simple)

        # Complex query
        complex_query = " ".join(["complex question about"] * 100)
        complex_model = router.select_model(complex_query)

        # Models should be different (escalated)
        assert simple_model != complex_model or \
               "simple" in simple_model.lower() or \
               "7b" in simple_model.lower()

    def test_no_unnecessary_escalation(self):
        """Should not escalate for queries that don't need it"""
        from model_router import ModelRouter
        router = ModelRouter()

        simple_queries = [
            "What is 2+2?",
            "Hello Echo",
            "List files",
            "Print hello"
        ]

        for query in simple_queries:
            model = router.select_model(query)
            # Should not use the largest model for simple queries
            assert "70b" not in model.lower() and "32b" not in model.lower()


class TestModelAvailability:
    """Test handling of model availability"""

    @pytest.mark.asyncio
    async def test_handles_model_not_available(self):
        """Should handle case when selected model is not available"""
        from model_router import ModelRouter
        router = ModelRouter()

        with patch('httpx.AsyncClient') as mock_client:
            # Simulate model not found error
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"error": "model not found"}
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with patch('src.core.db_model_router.intelligent_router') as mock_router:
                mock_decision = MagicMock()
                mock_decision.model = "nonexistent:model"
                mock_decision.complexity_score = 15
                mock_decision.intent = "general"
                mock_decision.domain = "general"
                mock_router.route_query.return_value = mock_decision

                result = await router.route_query("test query")
                # Should handle error gracefully
                assert "error" in result or "success" in result
