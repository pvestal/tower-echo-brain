"""
Basic Functionality Tests for Echo Brain Board of Directors

This module contains simple tests to verify the core functionality
and that the test infrastructure is working correctly.

Author: Echo Brain CI/CD Pipeline
Created: 2025-09-16
"""

import pytest
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directors.base_director import DirectorBase
from directors.security_director import SecurityDirector
from directors.quality_director import QualityDirector
from directors.performance_director import PerformanceDirector
from directors.ethics_director import EthicsDirector
from directors.ux_director import UXDirector


class TestBasicFunctionality:
    """Test basic functionality to verify test infrastructure."""

    def test_python_version(self):
        """Test that we're running on a supported Python version."""
        assert sys.version_info >= (3, 9), "Python 3.9+ required"
        assert sys.version_info < (4, 0), "Python 4.0+ not yet supported"

    def test_imports_work(self):
        """Test that all director classes can be imported."""
        # Test that all classes are available
        assert DirectorBase is not None
        assert SecurityDirector is not None
        assert QualityDirector is not None
        assert PerformanceDirector is not None
        assert EthicsDirector is not None
        assert UXDirector is not None

    def test_director_instantiation(self):
        """Test that director classes can be instantiated."""
        # Test SecurityDirector instantiation
        security_director = SecurityDirector()
        assert security_director.name == "SecurityDirector"
        assert "security" in security_director.expertise.lower() or "cybersecurity" in security_director.expertise.lower()

        # Test QualityDirector instantiation
        quality_director = QualityDirector()
        assert quality_director.name == "QualityDirector"
        assert "quality" in quality_director.expertise.lower()

    def test_director_evaluation_interface(self):
        """Test that directors implement the evaluation interface."""
        directors = [
            SecurityDirector(),
            QualityDirector(),
            PerformanceDirector(),
            EthicsDirector(),
            UXDirector()
        ]

        simple_task = {
            "task_id": "test_001",
            "code": "def hello(): return 'world'",
            "language": "python",
            "description": "Simple test function"
        }

        for director in directors:
            # Test that evaluate method exists and is callable
            assert hasattr(director, 'evaluate'), f"{director.name} missing evaluate method"
            assert callable(getattr(director, 'evaluate')), f"{director.name} evaluate is not callable"

            # Test that evaluation returns expected structure
            result = director.evaluate(simple_task)

            # Check required fields
            required_fields = ['confidence', 'recommendation', 'findings', 'reasoning']
            for field in required_fields:
                assert field in result, f"{director.name} missing {field} in result"

            # Check data types
            assert isinstance(result['confidence'], (int, float))
            assert 0 <= result['confidence'] <= 100
            assert result['recommendation'] in ['approved', 'needs_review', 'rejected']
            assert isinstance(result['findings'], list)
            assert isinstance(result['reasoning'], str)

    def test_pytest_markers(self):
        """Test that pytest markers are working."""
        # This test should be automatically marked as 'unit'
        # We can't easily test this within the test itself, but this
        # verifies the test discovery and execution is working
        assert True

    @pytest.mark.unit
    def test_explicit_unit_marker(self):
        """Test with explicit unit marker."""
        assert True

    def test_mock_functionality(self):
        """Test that mocking works in our test environment."""
        # Test basic mocking
        mock_obj = Mock()
        mock_obj.test_method.return_value = "mocked_value"

        result = mock_obj.test_method()
        assert result == "mocked_value"
        mock_obj.test_method.assert_called_once()

    def test_patch_functionality(self):
        """Test that patching works in our test environment."""
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, 0)

            # This would normally return current time, but now returns mocked time
            assert mock_datetime.now() == datetime(2025, 1, 1, 12, 0, 0)

    def test_fixtures_available(self, mock_user, sample_task_data):
        """Test that our test fixtures are available and working."""
        # Test mock_user fixture
        assert mock_user is not None
        assert "user_id" in mock_user
        assert "permissions" in mock_user
        assert isinstance(mock_user["permissions"], list)

        # Test sample_task_data fixture
        assert sample_task_data is not None
        assert "task_description" in sample_task_data
        assert "user_id" in sample_task_data
        assert "code" in sample_task_data["context"]

    def test_vulnerable_code_samples(self, vulnerable_code_sample):
        """Test that vulnerable code samples are available."""
        assert vulnerable_code_sample is not None
        assert "sql_injection" in vulnerable_code_sample
        assert "xss_vulnerability" in vulnerable_code_sample
        assert "hardcoded_secrets" in vulnerable_code_sample

        # Verify SQL injection sample contains dangerous patterns
        sql_injection_code = vulnerable_code_sample["sql_injection"]
        assert "SELECT" in sql_injection_code or "select" in sql_injection_code
        assert "user_id" in sql_injection_code

    def test_quality_issues_samples(self, quality_issues_sample):
        """Test that quality issue samples are available."""
        assert quality_issues_sample is not None
        assert "high_complexity" in quality_issues_sample
        assert "poor_naming" in quality_issues_sample
        assert "no_documentation" in quality_issues_sample

        # Verify high complexity sample has nested conditions
        complex_code = quality_issues_sample["high_complexity"]
        assert complex_code.count("if") >= 3  # Multiple nested if statements


class TestDirectorSpecificBasics:
    """Test basic functionality specific to each director type."""

    def test_security_director_basics(self):
        """Test SecurityDirector basic functionality."""
        director = SecurityDirector()

        # Test secure code evaluation
        secure_code = {
            "task_id": "secure_test",
            "code": "def safe_function(data): return data.strip()",
            "language": "python"
        }

        result = director.evaluate(secure_code)
        assert result["confidence"] > 0
        assert result["recommendation"] in ["approved", "needs_review", "rejected"]

    def test_quality_director_basics(self):
        """Test QualityDirector basic functionality."""
        director = QualityDirector()

        # Test well-written code evaluation
        quality_code = {
            "task_id": "quality_test",
            "code": """
def calculate_total(items):
    \"\"\"Calculate total price of items.\"\"\"
    return sum(item.price for item in items)
            """,
            "language": "python"
        }

        result = director.evaluate(quality_code)
        assert result["confidence"] > 0
        assert result["recommendation"] in ["approved", "needs_review", "rejected"]

    def test_performance_director_basics(self):
        """Test PerformanceDirector basic functionality."""
        director = PerformanceDirector()

        # Test performance-conscious code
        performance_code = {
            "task_id": "performance_test",
            "code": "def efficient_search(data, target): return target in set(data)",
            "language": "python"
        }

        result = director.evaluate(performance_code)
        assert result["confidence"] > 0
        assert result["recommendation"] in ["approved", "needs_review", "rejected"]

    def test_ethics_director_basics(self):
        """Test EthicsDirector basic functionality."""
        director = EthicsDirector()

        # Test ethically neutral code
        ethical_code = {
            "task_id": "ethics_test",
            "code": "def process_public_data(data): return data.process()",
            "language": "python"
        }

        result = director.evaluate(ethical_code)
        assert result["confidence"] > 0
        assert result["recommendation"] in ["approved", "needs_review", "rejected"]

    def test_ux_director_basics(self):
        """Test UXDirector basic functionality."""
        director = UXDirector()

        # Test user-friendly interface code
        ux_code = {
            "task_id": "ux_test",
            "code": """
<form>
    <label for="email">Email Address:</label>
    <input type="email" id="email" name="email" required>
    <button type="submit">Submit</button>
</form>
            """,
            "language": "html"
        }

        result = director.evaluate(ux_code)
        assert result["confidence"] > 0
        assert result["recommendation"] in ["approved", "needs_review", "rejected"]


class TestErrorHandling:
    """Test error handling in the test environment."""

    def test_missing_required_imports(self):
        """Test that missing imports are handled gracefully."""
        try:
            # Try to import a non-existent module
            import nonexistent_module  # noqa: F401
            assert False, "Should have raised ImportError"
        except ImportError:
            # This is expected
            assert True

    def test_invalid_director_input(self):
        """Test that directors handle invalid input gracefully."""
        director = SecurityDirector()

        # Test with empty input
        try:
            result = director.evaluate({})
            # If it doesn't raise an exception, it should return a valid result
            assert "confidence" in result or "error" in result
        except (ValueError, KeyError, TypeError):
            # It's acceptable to raise an exception for invalid input
            assert True

    def test_none_input_handling(self):
        """Test handling of None inputs."""
        director = QualityDirector()

        try:
            result = director.evaluate(None)
            # If it doesn't raise an exception, it should return a valid result
            assert result is not None
        except (ValueError, TypeError):
            # It's acceptable to raise an exception for None input
            assert True


# Run a simple test if this file is executed directly
if __name__ == "__main__":
    print("Running basic functionality tests...")
    test_basic = TestBasicFunctionality()
    test_basic.test_python_version()
    test_basic.test_imports_work()
    test_basic.test_director_instantiation()
    print("✅ All basic tests passed!")