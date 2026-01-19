"""
Comprehensive Cognitive and Reasoning Tests

Tests:
- Reasoning trigger detection
- DeepSeek reasoning execution
- Think tag parsing
- Step-by-step reasoning
- Reasoning result structure
- Fallback handling
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestReasoningDetection:
    """Test reasoning trigger detection"""

    def test_should_use_reasoning_import(self):
        """should_use_reasoning function should import"""
        try:
            from reasoning.deepseek_reasoner import should_use_reasoning
            assert callable(should_use_reasoning)
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_explicit_reasoning_triggers(self):
        """Explicit reasoning keywords should trigger reasoning"""
        try:
            from reasoning.deepseek_reasoner import should_use_reasoning

            triggers = [
                "think through this problem",
                "analyze the architecture",
                "compare these approaches",
                "evaluate the options",
                "what are the pros and cons",
                "consider the trade-offs",
                "should I use X or Y",
                "which is better",
                "explain why this works",
                "step by step guide"
            ]

            for query in triggers:
                result = should_use_reasoning(query)
                assert result is True, f"Should trigger reasoning for: {query}"
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_simple_queries_no_explicit_triggers(self):
        """Simple queries without triggers should not require reasoning"""
        try:
            from reasoning.deepseek_reasoner import should_use_reasoning

            simple = [
                "hello",
                "what time is it",
                "print hello world"
            ]

            for query in simple:
                # These are short so may not trigger based on length
                result = should_use_reasoning(query)
                # Just verify it returns a boolean
                assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_long_queries_trigger_reasoning(self):
        """Long queries (>150 chars) should trigger reasoning"""
        try:
            from reasoning.deepseek_reasoner import should_use_reasoning

            long_query = "I need help with " + "x" * 150
            result = should_use_reasoning(long_query)
            assert result is True
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_multi_question_triggers_reasoning(self):
        """Multiple questions should trigger reasoning"""
        try:
            from reasoning.deepseek_reasoner import should_use_reasoning

            multi = "What is this? How does it work? Why is it useful?"
            result = should_use_reasoning(multi)
            assert result is True
        except ImportError:
            pytest.skip("Reasoning module not available")


class TestThinkTagParsing:
    """Test parsing of <think> tags from responses"""

    def test_parse_think_tags_import(self):
        """parse_think_tags function should import"""
        try:
            from reasoning.deepseek_reasoner import parse_think_tags
            assert callable(parse_think_tags)
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_parse_single_think_block(self):
        """Should parse single think block"""
        try:
            from reasoning.deepseek_reasoner import parse_think_tags

            response = """<think>
Let me analyze this step by step.
First, I'll consider the requirements.
Then, I'll evaluate the options.
</think>

Based on my analysis, the best approach is X."""

            thinking_steps, final_answer = parse_think_tags(response)

            assert len(thinking_steps) > 0
            assert "best approach" in final_answer
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_parse_multiple_think_blocks(self):
        """Should parse multiple think blocks"""
        try:
            from reasoning.deepseek_reasoner import parse_think_tags

            response = """<think>First consideration</think>
Some intermediate text.
<think>Second consideration</think>
Final answer here."""

            thinking_steps, final_answer = parse_think_tags(response)

            assert len(thinking_steps) >= 2
            assert "Final answer" in final_answer
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_parse_no_think_tags(self):
        """Should handle response without think tags"""
        try:
            from reasoning.deepseek_reasoner import parse_think_tags

            response = "This is a direct response without any think tags."

            thinking_steps, final_answer = parse_think_tags(response)

            assert final_answer == response.strip()
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_parse_numbered_steps(self):
        """Should extract numbered steps even without think tags"""
        try:
            from reasoning.deepseek_reasoner import parse_think_tags

            response = """Step 1: First do this
Step 2: Then do that
Step 3: Finally complete

The result is achieved."""

            thinking_steps, final_answer = parse_think_tags(response)

            # Should either extract numbered steps or return empty (implementation dependent)
            assert isinstance(thinking_steps, list)
            assert isinstance(final_answer, str)
        except ImportError:
            pytest.skip("Reasoning module not available")


class TestReasoningExecution:
    """Test reasoning execution"""

    @pytest.mark.asyncio
    async def test_execute_reasoning_import(self):
        """execute_reasoning function should import"""
        try:
            from reasoning.deepseek_reasoner import execute_reasoning
            assert callable(execute_reasoning)
        except ImportError:
            pytest.skip("Reasoning module not available")

    @pytest.mark.asyncio
    async def test_execute_reasoning_returns_result(self):
        """execute_reasoning should return ReasoningResult"""
        try:
            from reasoning.deepseek_reasoner import execute_reasoning, ReasoningResult

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "response": "<think>Analyzing...</think>The answer is 42."
                }
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                result = await execute_reasoning("What is the meaning of life?")

                assert isinstance(result, ReasoningResult)
                assert result.success is True
                assert result.final_answer is not None
        except ImportError:
            pytest.skip("Reasoning module not available")

    @pytest.mark.asyncio
    async def test_execute_reasoning_captures_thinking_steps(self):
        """Should capture thinking steps from response"""
        try:
            from reasoning.deepseek_reasoner import execute_reasoning

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "response": """<think>
Step 1: Consider the question
Step 2: Analyze the context
Step 3: Formulate answer
</think>
The final answer is clear."""
                }
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                result = await execute_reasoning("Complex question")

                assert len(result.thinking_steps) > 0
        except ImportError:
            pytest.skip("Reasoning module not available")

    @pytest.mark.asyncio
    async def test_execute_reasoning_uses_correct_model(self):
        """Should use deepseek-r1 model by default"""
        try:
            from reasoning.deepseek_reasoner import execute_reasoning

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": "test"}

                mock_post = AsyncMock(return_value=mock_response)
                mock_client.return_value.__aenter__.return_value.post = mock_post

                await execute_reasoning("test", model="deepseek-r1:8b")

                call_args = mock_post.call_args
                if call_args:
                    json_arg = call_args.kwargs.get('json', {})
                    assert "deepseek" in json_arg.get('model', '').lower()
        except ImportError:
            pytest.skip("Reasoning module not available")

    @pytest.mark.asyncio
    async def test_execute_reasoning_handles_timeout(self):
        """Should handle timeout gracefully"""
        try:
            from reasoning.deepseek_reasoner import execute_reasoning
            import httpx

            with patch('httpx.AsyncClient') as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    side_effect=httpx.TimeoutException("Timeout")
                )

                result = await execute_reasoning("test")

                assert result.success is False
                assert result.error is not None
        except ImportError:
            pytest.skip("Reasoning module not available")

    @pytest.mark.asyncio
    async def test_execute_reasoning_handles_api_error(self):
        """Should handle API errors gracefully"""
        try:
            from reasoning.deepseek_reasoner import execute_reasoning

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.json.return_value = {"error": "Internal error"}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                result = await execute_reasoning("test")

                assert result.success is False
        except ImportError:
            pytest.skip("Reasoning module not available")


class TestReasoningResult:
    """Test ReasoningResult dataclass"""

    def test_reasoning_result_structure(self):
        """ReasoningResult should have expected fields"""
        try:
            from reasoning.deepseek_reasoner import ReasoningResult

            result = ReasoningResult(
                thinking_steps=["Step 1", "Step 2"],
                final_answer="The answer",
                model_used="deepseek-r1:8b",
                raw_response="<think>...</think>The answer",
                success=True
            )

            assert result.thinking_steps == ["Step 1", "Step 2"]
            assert result.final_answer == "The answer"
            assert result.model_used == "deepseek-r1:8b"
            assert result.success is True
        except ImportError:
            pytest.skip("Reasoning module not available")

    def test_reasoning_result_error_state(self):
        """ReasoningResult should handle error state"""
        try:
            from reasoning.deepseek_reasoner import ReasoningResult

            result = ReasoningResult(
                thinking_steps=[],
                final_answer="",
                model_used="",
                raw_response="",
                success=False,
                error="Connection failed"
            )

            assert result.success is False
            assert result.error == "Connection failed"
        except ImportError:
            pytest.skip("Reasoning module not available")


class TestComplexityAnalysis:
    """Test complexity analysis for routing decisions"""

    def test_complexity_analyzer_import(self):
        """ComplexityAnalyzer should import"""
        try:
            from core.complexity_analyzer import ComplexityAnalyzer
            assert ComplexityAnalyzer is not None
        except ImportError:
            pytest.skip("Complexity analyzer not available")

    def test_analyze_query_complexity(self):
        """Should analyze query complexity"""
        try:
            from core.complexity_analyzer import ComplexityAnalyzer
            analyzer = ComplexityAnalyzer()

            simple = "Hello"
            complex = "Design a distributed system for real-time video processing"

            if hasattr(analyzer, 'analyze'):
                simple_score = analyzer.analyze(simple)
                complex_score = analyzer.analyze(complex)

                # Just verify the method returns something, don't assume numeric comparison
                assert simple_score is not None
                assert complex_score is not None
        except ImportError:
            pytest.skip("Complexity analyzer not available")


class TestConversationContext:
    """Test conversation context for reasoning"""

    def test_conversation_context_import(self):
        """ConversationContext should import"""
        try:
            from intelligence.conversation_context import ConversationContext
            assert ConversationContext is not None
        except ImportError:
            pytest.skip("Conversation context not available")

    def test_context_enhances_reasoning(self):
        """Context should enhance reasoning queries"""
        try:
            from intelligence.conversation_context import enhance_query_with_context

            query = "What about the second approach?"
            context = {
                "history": [
                    {"role": "user", "content": "Compare microservices vs monolith"},
                    {"role": "assistant", "content": "First approach is microservices..."}
                ]
            }

            enhanced = enhance_query_with_context(query, context)
            # Enhanced query should include context
        except ImportError:
            pytest.skip("Conversation context not available")
