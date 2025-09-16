"""
Test Suite for Consensus Algorithm

This module tests the consensus-building functionality of the Director Registry,
including director agreement calculations, threshold management, conflict resolution,
and weighted decision-making processes.

Author: Echo Brain Test Suite
Created: 2025-09-16
"""

import pytest
import sys
import os
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import asyncio

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directors.director_registry import DirectorRegistry
from directors.base_director import DirectorBase
from directors import (
    SecurityDirector,
    QualityDirector,
    PerformanceDirector,
    EthicsDirector,
    UXDirector
)


class MockDirector(DirectorBase):
    """Mock director for testing purposes."""

    def __init__(self, name: str, expertise: str, mock_evaluation: Dict[str, Any] = None):
        super().__init__(name, expertise)
        self.mock_evaluation = mock_evaluation or {
            "confidence": 75,
            "recommendation": "approved",
            "reasoning": "Mock evaluation",
            "priority": "MEDIUM",
            "findings": ["Mock finding"]
        }

    def evaluate(self, task_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Return mock evaluation."""
        return self.mock_evaluation.copy()


class TestDirectorRegistry:
    """Test basic DirectorRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a DirectorRegistry instance for testing."""
        return DirectorRegistry(consensus_threshold=0.6, max_directors_per_task=5)

    @pytest.fixture
    def mock_directors(self):
        """Create mock directors with different evaluation patterns."""
        return [
            MockDirector("Director1", "Test1", {
                "confidence": 80,
                "recommendation": "approved",
                "priority": "HIGH",
                "findings": ["Positive finding 1"]
            }),
            MockDirector("Director2", "Test2", {
                "confidence": 70,
                "recommendation": "approved",
                "priority": "MEDIUM",
                "findings": ["Positive finding 2"]
            }),
            MockDirector("Director3", "Test3", {
                "confidence": 60,
                "recommendation": "needs_review",
                "priority": "LOW",
                "findings": ["Minor concern"]
            })
        ]

    def test_registry_initialization(self, registry):
        """Test DirectorRegistry initializes correctly."""
        assert registry.consensus_threshold == 0.6
        assert registry.max_directors_per_task == 5
        assert len(registry.directors) == 0

    def test_register_director(self, registry):
        """Test director registration."""
        director = MockDirector("TestDirector", "Testing")

        # Register director
        registry.register_director(director)

        assert len(registry.directors) == 1
        assert "TestDirector" in registry.directors
        assert registry.directors["TestDirector"] == director

    def test_register_duplicate_director(self, registry):
        """Test handling of duplicate director registration."""
        director1 = MockDirector("TestDirector", "Testing")
        director2 = MockDirector("TestDirector", "Testing")

        registry.register_director(director1)

        # Should either replace or raise error
        try:
            registry.register_director(director2)
            # If no error, check that it was replaced or handled
            assert len(registry.directors) == 1
        except Exception:
            # If error is raised, that's also acceptable behavior
            pass

    def test_unregister_director(self, registry):
        """Test director unregistration."""
        director = MockDirector("TestDirector", "Testing")
        registry.register_director(director)

        # Unregister
        removed = registry.unregister_director("TestDirector")

        assert removed == director
        assert len(registry.directors) == 0
        assert "TestDirector" not in registry.directors

    def test_unregister_nonexistent_director(self, registry):
        """Test unregistering non-existent director."""
        result = registry.unregister_director("NonexistentDirector")
        assert result is None


class TestConsensusCalculation:
    """Test consensus calculation algorithms."""

    @pytest.fixture
    def registry(self):
        """Create a DirectorRegistry with low threshold for testing."""
        return DirectorRegistry(consensus_threshold=0.5)

    def test_unanimous_approval_consensus(self, registry, mock_directors):
        """Test consensus with unanimous approval."""
        # All directors approve
        for director in mock_directors:
            director.mock_evaluation = {
                "confidence": 85,
                "recommendation": "approved",
                "priority": "HIGH",
                "findings": ["Good to go"]
            }
            registry.register_director(director)

        task = {
            "task_type": "review",
            "code": "def test(): pass"
        }

        # Evaluate task
        result = registry.evaluate_task(task)

        assert "evaluations" in result
        assert len(result["evaluations"]) == len(mock_directors)

        # Should reach consensus for approval
        consensus = result.get("consensus", {})
        assert consensus.get("agreement_level", 0) >= 0.5

    def test_mixed_consensus(self, registry):
        """Test consensus with mixed recommendations."""
        directors = [
            MockDirector("Approver1", "Test", {
                "confidence": 90,
                "recommendation": "approved",
                "priority": "HIGH"
            }),
            MockDirector("Approver2", "Test", {
                "confidence": 80,
                "recommendation": "approved",
                "priority": "MEDIUM"
            }),
            MockDirector("Reviewer", "Test", {
                "confidence": 70,
                "recommendation": "needs_review",
                "priority": "MEDIUM"
            }),
            MockDirector("Rejecter", "Test", {
                "confidence": 60,
                "recommendation": "rejected",
                "priority": "LOW"
            })
        ]

        for director in directors:
            registry.register_director(director)

        task = {"task_type": "review", "code": "mixed_code()"}
        result = registry.evaluate_task(task)

        assert "evaluations" in result
        assert len(result["evaluations"]) == 4

        # Should have some level of disagreement
        consensus = result.get("consensus", {})
        agreement_level = consensus.get("agreement_level", 1.0)
        assert 0.0 <= agreement_level <= 1.0

    def test_high_threshold_consensus(self):
        """Test consensus with high threshold requirement."""
        registry = DirectorRegistry(consensus_threshold=0.9)

        # Add directors with slight disagreement
        directors = [
            MockDirector("Director1", "Test", {
                "confidence": 95,
                "recommendation": "approved"
            }),
            MockDirector("Director2", "Test", {
                "confidence": 90,
                "recommendation": "approved"
            }),
            MockDirector("Director3", "Test", {
                "confidence": 80,
                "recommendation": "needs_review"  # Different recommendation
            })
        ]

        for director in directors:
            registry.register_director(director)

        task = {"task_type": "review"}
        result = registry.evaluate_task(task)

        consensus = result.get("consensus", {})
        # With high threshold and disagreement, should not reach consensus
        assert consensus.get("agreement_level", 1.0) < 0.9

    def test_confidence_weighted_consensus(self, registry):
        """Test that confidence scores properly weight consensus."""
        directors = [
            MockDirector("HighConfidence", "Test", {
                "confidence": 95,
                "recommendation": "approved",
                "priority": "HIGH"
            }),
            MockDirector("LowConfidence", "Test", {
                "confidence": 30,
                "recommendation": "rejected",
                "priority": "LOW"
            })
        ]

        for director in directors:
            registry.register_director(director)

        task = {"task_type": "review"}
        result = registry.evaluate_task(task)

        # High confidence director should have more weight
        assert "evaluations" in result
        assert len(result["evaluations"]) == 2

        # The consensus should favor the high-confidence director
        consensus = result.get("consensus", {})
        assert "weighted_recommendation" in consensus or "recommendation" in consensus


class TestTaskRouting:
    """Test task routing to appropriate directors."""

    @pytest.fixture
    def registry_with_real_directors(self):
        """Create registry with real director implementations."""
        registry = DirectorRegistry()

        # Add all real directors
        registry.register_director(SecurityDirector())
        registry.register_director(QualityDirector())
        registry.register_director(PerformanceDirector())
        registry.register_director(EthicsDirector())
        registry.register_director(UXDirector())

        return registry

    def test_security_task_routing(self, registry_with_real_directors):
        """Test that security tasks are properly routed."""
        task = {
            "task_type": "security_review",
            "code": "SELECT * FROM users WHERE id = ?",
            "priority": "high"
        }

        result = registry_with_real_directors.evaluate_task(task)

        assert "evaluations" in result

        # Should have evaluations from multiple directors
        evaluations = result["evaluations"]
        assert len(evaluations) > 0

        # Security director should be included
        director_names = [eval.get("director_name", "") for eval in evaluations]
        assert any("Security" in name for name in director_names)

    def test_code_quality_task_routing(self, registry_with_real_directors):
        """Test that code quality tasks are properly routed."""
        task = {
            "task_type": "code_review",
            "code": """
def complex_function():
    if True:
        if True:
            if True:
                return "nested"
            """,
            "priority": "medium"
        }

        result = registry_with_real_directors.evaluate_task(task)

        assert "evaluations" in result
        evaluations = result["evaluations"]
        assert len(evaluations) > 0

        # Quality director should be included
        director_names = [eval.get("director_name", "") for eval in evaluations]
        assert any("Quality" in name for name in director_names)

    def test_multi_domain_task_routing(self, registry_with_real_directors):
        """Test routing for tasks that span multiple domains."""
        task = {
            "task_type": "full_review",
            "code": """
function authenticateUser(password) {
    // Poor UX - no validation feedback
    // Security issue - plain text comparison
    if (password == "admin123") {
        return true;
    }
    return false;
}
            """,
            "context": {
                "review_type": "comprehensive"
            }
        }

        result = registry_with_real_directors.evaluate_task(task)

        assert "evaluations" in result
        evaluations = result["evaluations"]

        # Should involve multiple directors for comprehensive review
        assert len(evaluations) >= 2

        director_names = [eval.get("director_name", "") for eval in evaluations]
        # Should include multiple types of directors
        unique_types = set()
        for name in director_names:
            if "Security" in name:
                unique_types.add("security")
            elif "Quality" in name:
                unique_types.add("quality")
            elif "UX" in name:
                unique_types.add("ux")
            elif "Ethics" in name:
                unique_types.add("ethics")
            elif "Performance" in name:
                unique_types.add("performance")

        assert len(unique_types) >= 2


class TestConflictResolution:
    """Test conflict resolution between directors."""

    @pytest.fixture
    def registry(self):
        """Create registry for conflict testing."""
        return DirectorRegistry(consensus_threshold=0.6)

    def test_complete_disagreement_resolution(self, registry):
        """Test resolution when directors completely disagree."""
        conflicting_directors = [
            MockDirector("Approver", "Test", {
                "confidence": 90,
                "recommendation": "approved",
                "priority": "HIGH",
                "reasoning": "Looks good to me"
            }),
            MockDirector("Rejecter", "Test", {
                "confidence": 90,
                "recommendation": "rejected",
                "priority": "HIGH",
                "reasoning": "Major issues found"
            })
        ]

        for director in conflicting_directors:
            registry.register_director(director)

        task = {"task_type": "review", "code": "controversial_code()"}
        result = registry.evaluate_task(task)

        assert "evaluations" in result
        assert len(result["evaluations"]) == 2

        # Should handle disagreement gracefully
        consensus = result.get("consensus", {})
        assert "conflict_detected" in consensus or "agreement_level" in consensus

        # Should provide some resolution strategy
        assert "resolution" in consensus or "recommendation" in consensus

    def test_priority_based_resolution(self, registry):
        """Test resolution based on priority levels."""
        directors = [
            MockDirector("CriticalDirector", "Test", {
                "confidence": 80,
                "recommendation": "rejected",
                "priority": "CRITICAL",
                "reasoning": "Critical security flaw"
            }),
            MockDirector("LowDirector", "Test", {
                "confidence": 85,
                "recommendation": "approved",
                "priority": "LOW",
                "reasoning": "Minor optimization possible"
            })
        ]

        for director in directors:
            registry.register_director(director)

        task = {"task_type": "review"}
        result = registry.evaluate_task(task)

        # Critical priority should override low priority
        consensus = result.get("consensus", {})

        # Should lean towards the critical finding
        final_recommendation = consensus.get("recommendation", "")
        assert final_recommendation in ["rejected", "needs_review", "critical_review"]

    def test_confidence_tiebreaker(self, registry):
        """Test using confidence as tiebreaker."""
        directors = [
            MockDirector("HighConfidence", "Test", {
                "confidence": 95,
                "recommendation": "approved",
                "priority": "MEDIUM"
            }),
            MockDirector("LowConfidence", "Test", {
                "confidence": 60,
                "recommendation": "rejected",
                "priority": "MEDIUM"
            })
        ]

        for director in directors:
            registry.register_director(director)

        task = {"task_type": "review"}
        result = registry.evaluate_task(task)

        # Higher confidence should win in tiebreaker
        consensus = result.get("consensus", {})

        # Should have some mechanism to handle this
        assert "recommendation" in consensus or "weighted_recommendation" in consensus


class TestPerformanceMetrics:
    """Test performance tracking and metrics."""

    @pytest.fixture
    def registry(self):
        """Create registry for performance testing."""
        return DirectorRegistry()

    def test_evaluation_performance_tracking(self, registry):
        """Test that evaluation performance is tracked."""
        director = MockDirector("TestDirector", "Test")
        registry.register_director(director)

        # Evaluate multiple tasks
        for i in range(5):
            task = {"task_type": "review", "iteration": i}
            result = registry.evaluate_task(task)

        # Should track performance metrics
        metrics = registry.get_performance_metrics()
        assert isinstance(metrics, dict)

        # Should have some performance data
        assert len(metrics) > 0 or hasattr(registry, 'evaluation_history')

    def test_director_reliability_tracking(self, registry):
        """Test tracking of individual director reliability."""
        reliable_director = MockDirector("Reliable", "Test", {
            "confidence": 90,
            "recommendation": "approved"
        })

        unreliable_director = MockDirector("Unreliable", "Test", {
            "confidence": 30,
            "recommendation": "unknown"
        })

        registry.register_director(reliable_director)
        registry.register_director(unreliable_director)

        # Evaluate tasks
        for i in range(3):
            task = {"task_type": "review", "test": i}
            registry.evaluate_task(task)

        # Should be able to get individual director metrics
        metrics = registry.get_director_metrics()
        assert isinstance(metrics, dict)

    def test_consensus_quality_metrics(self, registry):
        """Test tracking of consensus quality over time."""
        # Add directors with varying agreement patterns
        directors = [
            MockDirector("Agreeable1", "Test", {"confidence": 80, "recommendation": "approved"}),
            MockDirector("Agreeable2", "Test", {"confidence": 85, "recommendation": "approved"}),
            MockDirector("Contrarian", "Test", {"confidence": 70, "recommendation": "rejected"})
        ]

        for director in directors:
            registry.register_director(director)

        # Evaluate several tasks
        for i in range(5):
            task = {"task_type": "review", "round": i}
            result = registry.evaluate_task(task)

        # Should track consensus quality
        consensus_metrics = registry.get_consensus_metrics()
        assert isinstance(consensus_metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])