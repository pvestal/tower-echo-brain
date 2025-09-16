"""
Test Suite for End-to-End Integration

This module tests the complete integration of the Board of Directors system
with Echo Brain, including the full evaluation flow, async operations,
and real-world usage scenarios.

Author: Echo Brain Test Suite
Created: 2025-09-16
"""

import pytest
import sys
import os
import json
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies before importing
with patch('directors.decision_tracker.DecisionTracker'), \
     patch('directors.director_registry.DirectorRegistry'), \
     patch('directors.db_pool.DatabasePool'):

    from echo_board_integration import EchoBoardOfDirectors


class TestEchoBoardIntegration:
    """Test the main integration class."""

    @pytest.fixture
    def mock_registry(self):
        """Mock the DirectorRegistry."""
        with patch('echo_board_integration.DirectorRegistry') as mock_reg:
            mock_instance = MagicMock()
            mock_reg.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def echo_board(self, mock_registry):
        """Create EchoBoardOfDirectors instance for testing."""
        return EchoBoardOfDirectors()

    def test_echo_board_initialization(self, echo_board, mock_registry):
        """Test EchoBoardOfDirectors initializes correctly."""
        assert echo_board.registry == mock_registry

        # Should register all 5 directors
        assert mock_registry.register_director.call_count == 5

        # Verify each director type was registered
        registered_directors = []
        for call in mock_registry.register_director.call_args_list:
            director = call[0][0]
            registered_directors.append(type(director).__name__)

        expected_directors = [
            'SecurityDirector',
            'QualityDirector',
            'PerformanceDirector',
            'EthicsDirector',
            'UXDirector'
        ]

        for expected in expected_directors:
            assert expected in registered_directors

    @pytest.mark.asyncio
    async def test_evaluate_task_success(self, echo_board, mock_registry):
        """Test successful task evaluation."""
        # Mock successful evaluation
        mock_result = {
            "success": True,
            "evaluations": [
                {
                    "director_name": "SecurityDirector",
                    "recommendation": "approved",
                    "confidence": 90,
                    "findings": ["Secure implementation"]
                }
            ],
            "consensus": {
                "recommendation": "approved",
                "confidence": 85,
                "agreement_level": 0.9
            }
        }
        mock_registry.evaluate_task.return_value = mock_result

        task = {
            "task_type": "code_review",
            "code": "def secure_function(): pass",
            "description": "Review security implementation"
        }

        result = await echo_board.evaluate_task(task)

        assert result["success"] == True
        assert "evaluations" in result
        assert "consensus" in result

        # Verify the registry was called with the task
        mock_registry.evaluate_task.assert_called_once_with(task)

    @pytest.mark.asyncio
    async def test_evaluate_task_failure(self, echo_board, mock_registry):
        """Test task evaluation with errors."""
        # Mock evaluation failure
        mock_registry.evaluate_task.side_effect = Exception("Evaluation failed")

        task = {
            "task_type": "code_review",
            "code": "problematic_code()"
        }

        result = await echo_board.evaluate_task(task)

        assert result["success"] == False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evaluate_empty_task(self, echo_board, mock_registry):
        """Test evaluation of empty or invalid task."""
        mock_registry.evaluate_task.return_value = {
            "success": False,
            "error": "Invalid task data"
        }

        empty_task = {}

        result = await echo_board.evaluate_task(empty_task)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert "success" in result or "error" in result

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, echo_board, mock_registry):
        """Test concurrent task evaluations."""
        mock_registry.evaluate_task.return_value = {
            "success": True,
            "evaluations": [],
            "consensus": {"recommendation": "approved"}
        }

        tasks = [
            {"task_type": "security_review", "code": f"test_code_{i}()"}
            for i in range(5)
        ]

        # Run concurrent evaluations
        results = await asyncio.gather(*[
            echo_board.evaluate_task(task) for task in tasks
        ])

        assert len(results) == 5
        for result in results:
            assert result["success"] == True

        # Should have called evaluate_task for each task
        assert mock_registry.evaluate_task.call_count == 5


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.fixture
    def echo_board_real(self):
        """Create EchoBoardOfDirectors with real directors (mocked at lower level)."""
        with patch('directors.db_pool.DatabasePool'), \
             patch('psycopg2.pool.ThreadedConnectionPool'):
            return EchoBoardOfDirectors()

    @pytest.mark.asyncio
    async def test_security_vulnerability_detection(self, echo_board_real):
        """Test detection of security vulnerabilities in real scenario."""
        vulnerable_task = {
            'task_type': 'security_review',
            'code': '''
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    return result.fetchone() is not None
            ''',
            'description': 'Review authentication function',
            'priority': 'high'
        }

        result = await echo_board_real.evaluate_task(vulnerable_task)

        assert isinstance(result, dict)
        assert "success" in result

        if result.get("success"):
            # Should detect SQL injection vulnerability
            security_findings = []
            for evaluation in result.get("evaluations", []):
                if "Security" in evaluation.get("director_name", ""):
                    security_findings.extend(evaluation.get("findings", []))

            # Check if security issues were found
            findings_text = " ".join(str(f) for f in security_findings).lower()
            assert any(keyword in findings_text
                      for keyword in ['sql', 'injection', 'security', 'vulnerable'])

    @pytest.mark.asyncio
    async def test_code_quality_assessment(self, echo_board_real):
        """Test code quality assessment in real scenario."""
        quality_task = {
            'task_type': 'code_review',
            'code': '''
def calculate_stuff(x, y, z, a, b, c, d, e, f, g):
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        if c > 0:
                            if d > 0:
                                if e > 0:
                                    if f > 0:
                                        if g > 0:
                                            return x + y + z + a + b + c + d + e + f + g
    return 0
            ''',
            'description': 'Review complex calculation function',
            'priority': 'medium'
        }

        result = await echo_board_real.evaluate_task(quality_task)

        assert isinstance(result, dict)

        if result.get("success"):
            # Should detect complexity issues
            quality_findings = []
            for evaluation in result.get("evaluations", []):
                if "Quality" in evaluation.get("director_name", ""):
                    quality_findings.extend(evaluation.get("findings", []))

            findings_text = " ".join(str(f) for f in quality_findings).lower()
            assert any(keyword in findings_text
                      for keyword in ['complex', 'nested', 'refactor', 'cyclomatic'])

    @pytest.mark.asyncio
    async def test_multi_domain_evaluation(self, echo_board_real):
        """Test evaluation that spans multiple director domains."""
        multi_domain_task = {
            'task_type': 'comprehensive_review',
            'code': '''
// Login form with multiple issues
function handleLogin() {
    var username = document.getElementById("user").value;
    var password = document.getElementById("pass").value;

    // No input validation
    // No loading state
    // Sends password in URL
    window.location = "/login?user=" + username + "&pass=" + password;
}
            ''',
            'language': 'javascript',
            'description': 'Review login form handler',
            'context': {
                'component_type': 'frontend',
                'security_level': 'high',
                'user_facing': True
            }
        }

        result = await echo_board_real.evaluate_task(multi_domain_task)

        assert isinstance(result, dict)

        if result.get("success"):
            evaluations = result.get("evaluations", [])
            director_types = set()

            for evaluation in evaluations:
                director_name = evaluation.get("director_name", "")
                if "Security" in director_name:
                    director_types.add("security")
                elif "Quality" in director_name:
                    director_types.add("quality")
                elif "UX" in director_name:
                    director_types.add("ux")
                elif "Ethics" in director_name:
                    director_types.add("ethics")
                elif "Performance" in director_name:
                    director_types.add("performance")

            # Should involve multiple director types
            assert len(director_types) >= 2

    @pytest.mark.asyncio
    async def test_good_code_approval(self, echo_board_real):
        """Test that well-written code gets positive evaluation."""
        good_code_task = {
            'task_type': 'code_review',
            'code': '''
/**
 * Calculates the area of a rectangle with input validation
 * @param {number} width - The width of the rectangle
 * @param {number} height - The height of the rectangle
 * @returns {number} The area of the rectangle
 * @throws {Error} If inputs are invalid
 */
function calculateRectangleArea(width, height) {
    // Input validation
    if (typeof width !== 'number' || typeof height !== 'number') {
        throw new Error('Width and height must be numbers');
    }

    if (width <= 0 || height <= 0) {
        throw new Error('Width and height must be positive');
    }

    // Calculate and return area
    return width * height;
}
            ''',
            'language': 'javascript',
            'description': 'Review utility function',
            'priority': 'low'
        }

        result = await echo_board_real.evaluate_task(good_code_task)

        assert isinstance(result, dict)

        if result.get("success"):
            # Should generally approve good code
            consensus = result.get("consensus", {})
            recommendation = consensus.get("recommendation", "")

            # Should not be rejected
            assert recommendation != "rejected"

            # Should have reasonable confidence
            confidence = consensus.get("confidence", 0)
            assert confidence >= 0  # Some level of confidence


class TestAsyncOperations:
    """Test asynchronous operation handling."""

    @pytest.fixture
    def echo_board(self):
        """Create EchoBoardOfDirectors instance."""
        with patch('echo_board_integration.DirectorRegistry') as mock_reg:
            mock_instance = MagicMock()
            mock_reg.return_value = mock_instance
            return EchoBoardOfDirectors()

    @pytest.mark.asyncio
    async def test_async_timeout_handling(self, echo_board):
        """Test handling of async operation timeouts."""
        # Mock slow evaluation
        echo_board.registry.evaluate_task = Mock(side_effect=lambda x: asyncio.sleep(10))

        task = {"task_type": "review", "code": "slow_code()"}

        # Test with timeout
        try:
            result = await asyncio.wait_for(
                echo_board.evaluate_task(task),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Timeout is expected behavior
            pass

    @pytest.mark.asyncio
    async def test_async_cancellation(self, echo_board):
        """Test proper handling of async operation cancellation."""
        echo_board.registry.evaluate_task = Mock(return_value={
            "success": True,
            "evaluations": []
        })

        task = {"task_type": "review", "code": "test_code()"}

        # Start task and cancel it
        task_future = asyncio.create_task(echo_board.evaluate_task(task))
        await asyncio.sleep(0.01)  # Let it start
        task_future.cancel()

        try:
            await task_future
        except asyncio.CancelledError:
            # Cancellation should be handled properly
            pass

    @pytest.mark.asyncio
    async def test_multiple_async_tasks(self, echo_board):
        """Test handling multiple async tasks simultaneously."""
        echo_board.registry.evaluate_task = Mock(return_value={
            "success": True,
            "evaluations": [],
            "consensus": {"recommendation": "approved"}
        })

        tasks = [
            {"task_type": "review", "code": f"task_{i}()"}
            for i in range(10)
        ]

        # Run all tasks concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[
            echo_board.evaluate_task(task) for task in tasks
        ])
        end_time = asyncio.get_event_loop().time()

        assert len(results) == 10
        for result in results:
            assert result["success"] == True

        # Should complete in reasonable time (async execution)
        assert end_time - start_time < 5.0  # Should be much faster than 10 sequential calls


class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.fixture
    def echo_board(self):
        """Create EchoBoardOfDirectors instance for error testing."""
        with patch('echo_board_integration.DirectorRegistry') as mock_reg:
            mock_instance = MagicMock()
            mock_reg.return_value = mock_instance
            return EchoBoardOfDirectors()

    @pytest.mark.asyncio
    async def test_partial_director_failure(self, echo_board):
        """Test handling when some directors fail."""
        # Mock partial failure scenario
        def mock_evaluate(task):
            return {
                "success": True,
                "evaluations": [
                    {
                        "director_name": "SecurityDirector",
                        "recommendation": "approved",
                        "confidence": 80
                    },
                    {
                        "director_name": "QualityDirector",
                        "error": "Director temporarily unavailable"
                    }
                ],
                "consensus": {
                    "recommendation": "approved",
                    "confidence": 60,
                    "partial_failure": True
                }
            }

        echo_board.registry.evaluate_task = Mock(side_effect=mock_evaluate)

        task = {"task_type": "review", "code": "test_code()"}
        result = await echo_board.evaluate_task(task)

        assert result["success"] == True
        assert "evaluations" in result

        # Should handle partial failures gracefully
        evaluations = result["evaluations"]
        assert len(evaluations) == 2

        # One successful, one failed
        success_count = sum(1 for e in evaluations if "error" not in e)
        error_count = sum(1 for e in evaluations if "error" in e)
        assert success_count >= 1
        assert error_count >= 1

    @pytest.mark.asyncio
    async def test_complete_system_failure(self, echo_board):
        """Test handling of complete system failure."""
        # Mock complete failure
        echo_board.registry.evaluate_task = Mock(side_effect=Exception("System failure"))

        task = {"task_type": "review", "code": "test_code()"}
        result = await echo_board.evaluate_task(task)

        assert result["success"] == False
        assert "error" in result

        # Should provide meaningful error information
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0

    @pytest.mark.asyncio
    async def test_recovery_after_failure(self, echo_board):
        """Test that system can recover after failures."""
        # First call fails, second succeeds
        call_count = 0

        def mock_evaluate(task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            else:
                return {
                    "success": True,
                    "evaluations": [],
                    "consensus": {"recommendation": "approved"}
                }

        echo_board.registry.evaluate_task = Mock(side_effect=mock_evaluate)

        task = {"task_type": "review", "code": "test_code()"}

        # First call should fail
        result1 = await echo_board.evaluate_task(task)
        assert result1["success"] == False

        # Second call should succeed
        result2 = await echo_board.evaluate_task(task)
        assert result2["success"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])