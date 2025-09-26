"""
Comprehensive Unit Tests for All Board Directors

This module provides thorough unit testing for every director class:
- BaseDirector (abstract functionality)
- SecurityDirector (cybersecurity evaluation)
- QualityDirector (code quality assessment)
- PerformanceDirector (performance optimization)
- EthicsDirector (ethical compliance)
- UXDirector (user experience evaluation)

Author: Echo Brain CI/CD Pipeline
Created: 2025-09-16
"""

import pytest
import sys
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.base_director import DirectorBase
from routing.security_director import SecurityDirector
from routing.quality_director import QualityDirector
from routing.performance_director import PerformanceDirector
from routing.ethics_director import EthicsDirector
from routing.ux_director import UXDirector


# ============================================================================
# Test Base Director Functionality
# ============================================================================

class TestDirectorBase:
    """Test the abstract base director functionality."""

    def test_base_director_initialization(self):
        """Test base director initialization with various parameters."""
        class ConcreteDirector(DirectorBase):
            def evaluate(self, task_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                return {"result": "test_evaluation"}

        # Test with default version
        director = ConcreteDirector("TestDirector", "Testing Expertise")
        assert director.name == "TestDirector"
        assert director.expertise == "Testing Expertise"
        assert director.version == "1.0.0"
        assert isinstance(director.created_at, datetime)
        assert director.evaluation_history == []

        # Test with custom version
        director_v2 = ConcreteDirector("TestDirector2", "Advanced Testing", "2.1.0")
        assert director_v2.version == "2.1.0"

    def test_base_director_abstract_methods(self):
        """Test that base director enforces abstract methods."""
        with pytest.raises(TypeError):
            # Should not be able to instantiate abstract class
            DirectorBase("Invalid", "Should Fail")

    def test_base_director_evaluation_history(self):
        """Test evaluation history tracking."""
        class ConcreteDirector(DirectorBase):
            def evaluate(self, task_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                result = {"confidence": 85, "recommendation": "approved"}
                self.evaluation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task_data.get("task_id", "unknown"),
                    "result": result
                })
                return result

        director = ConcreteDirector("HistoryTestDirector", "History Testing")
        
        # Perform multiple evaluations
        task1 = {"task_id": "task_001", "code": "print('hello')"}
        task2 = {"task_id": "task_002", "code": "def func(): pass"}
        
        director.evaluate(task1)
        director.evaluate(task2)
        
        assert len(director.evaluation_history) == 2
        assert director.evaluation_history[0]["task_id"] == "task_001"
        assert director.evaluation_history[1]["task_id"] == "task_002"


# ============================================================================
# Test Security Director
# ============================================================================

class TestSecurityDirector:
    """Test the SecurityDirector functionality."""

    @pytest.fixture
    def security_director(self):
        """Provide a SecurityDirector instance for testing."""
        return SecurityDirector()

    def test_security_director_initialization(self, security_director):
        """Test SecurityDirector initialization."""
        assert security_director.name == "SecurityDirector"
        assert "cybersecurity" in security_director.expertise.lower()
        assert security_director.version.startswith("1.")
        assert security_director.vulnerability_patterns is not None

    def test_security_sql_injection_detection(self, security_director, vulnerable_code_sample):
        """Test SQL injection vulnerability detection."""
        task_data = {
            "task_id": "security_test_001",
            "code": vulnerable_code_sample["sql_injection"],
            "language": "python"
        }
        
        result = security_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert result["recommendation"] in ["needs_review", "rejected"]
        assert "sql injection" in result["reasoning"].lower() or any(
            "sql" in finding.lower() for finding in result["findings"]
        )
        assert result["priority"] in ["HIGH", "CRITICAL"]

    def test_security_xss_detection(self, security_director, vulnerable_code_sample):
        """Test XSS vulnerability detection."""
        task_data = {
            "task_id": "security_test_002",
            "code": vulnerable_code_sample["xss_vulnerability"],
            "language": "javascript"
        }
        
        result = security_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert result["recommendation"] in ["needs_review", "rejected"]
        assert any("xss" in finding.lower() or "script" in finding.lower() 
                  for finding in result["findings"])

    def test_security_hardcoded_secrets_detection(self, security_director, vulnerable_code_sample):
        """Test hardcoded secrets detection."""
        task_data = {
            "task_id": "security_test_003",
            "code": vulnerable_code_sample["hardcoded_secrets"],
            "language": "python"
        }
        
        result = security_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert result["recommendation"] in ["needs_review", "rejected"]
        assert any("secret" in finding.lower() or "password" in finding.lower() 
                  for finding in result["findings"])

    def test_security_safe_code_approval(self, security_director):
        """Test that secure code gets approved."""
        safe_code = """
def authenticate_user(username, password):
    from werkzeug.security import check_password_hash
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        return user
    return None
        """
        
        task_data = {
            "task_id": "security_test_004",
            "code": safe_code,
            "language": "python"
        }
        
        result = security_director.evaluate(task_data)
        
        assert result["confidence"] >= 70
        assert result["recommendation"] == "approved"
        assert result["priority"] in ["LOW", "MEDIUM"]

    def test_security_evaluation_structure(self, security_director, sample_task_data):
        """Test that security evaluation returns proper structure."""
        result = security_director.evaluate(sample_task_data)
        
        # Check required fields
        required_fields = ["confidence", "recommendation", "priority", "findings", 
                          "reasoning", "evaluation_time", "timestamp"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 100
        assert result["recommendation"] in ["approved", "needs_review", "rejected"]
        assert result["priority"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert isinstance(result["findings"], list)
        assert isinstance(result["reasoning"], str)
        assert isinstance(result["evaluation_time"], (int, float))


# ============================================================================
# Test Quality Director
# ============================================================================

class TestQualityDirector:
    """Test the QualityDirector functionality."""

    @pytest.fixture
    def quality_director(self):
        """Provide a QualityDirector instance for testing."""
        return QualityDirector()

    def test_quality_director_initialization(self, quality_director):
        """Test QualityDirector initialization."""
        assert quality_director.name == "QualityDirector"
        assert "code quality" in quality_director.expertise.lower()
        assert quality_director.version.startswith("1.")

    def test_quality_high_complexity_detection(self, quality_director, quality_issues_sample):
        """Test detection of high cyclomatic complexity."""
        task_data = {
            "task_id": "quality_test_001",
            "code": quality_issues_sample["high_complexity"],
            "language": "python"
        }
        
        result = quality_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert result["recommendation"] in ["needs_review", "rejected"]
        assert any("complexity" in finding.lower() for finding in result["findings"])

    def test_quality_poor_naming_detection(self, quality_director, quality_issues_sample):
        """Test detection of poor naming conventions."""
        task_data = {
            "task_id": "quality_test_002",
            "code": quality_issues_sample["poor_naming"],
            "language": "python"
        }
        
        result = quality_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("naming" in finding.lower() or "variable" in finding.lower() 
                  for finding in result["findings"])

    def test_quality_documentation_check(self, quality_director, quality_issues_sample):
        """Test documentation requirements checking."""
        task_data = {
            "task_id": "quality_test_003",
            "code": quality_issues_sample["no_documentation"],
            "language": "python"
        }
        
        result = quality_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("documentation" in finding.lower() or "comment" in finding.lower() 
                  for finding in result["findings"])

    def test_quality_good_code_approval(self, quality_director):
        """Test that well-written code gets approved."""
        good_code = """
def calculate_compound_interest(principal: float, rate: float, time: int, 
                              compounds_per_year: int = 12) -> float:
    """
    Calculate compound interest using the standard formula.
    
    Args:
        principal: Initial investment amount
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Investment period in years
        compounds_per_year: Number of times interest compounds per year
    
    Returns:
        Final amount after compound interest
    """
    if principal <= 0 or rate < 0 or time < 0 or compounds_per_year <= 0:
        raise ValueError("Invalid input parameters")
    
    return principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time)
        """
        
        task_data = {
            "task_id": "quality_test_004",
            "code": good_code,
            "language": "python"
        }
        
        result = quality_director.evaluate(task_data)
        
        assert result["confidence"] >= 75
        assert result["recommendation"] == "approved"


# ============================================================================
# Test Performance Director
# ============================================================================

class TestPerformanceDirector:
    """Test the PerformanceDirector functionality."""

    @pytest.fixture
    def performance_director(self):
        """Provide a PerformanceDirector instance for testing."""
        return PerformanceDirector()

    def test_performance_director_initialization(self, performance_director):
        """Test PerformanceDirector initialization."""
        assert performance_director.name == "PerformanceDirector"
        assert "performance" in performance_director.expertise.lower()
        assert performance_director.version.startswith("1.")

    def test_performance_inefficient_algorithm_detection(self, performance_director):
        """Test detection of inefficient algorithms."""
        inefficient_code = """
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates
        """
        
        task_data = {
            "task_id": "performance_test_001",
            "code": inefficient_code,
            "language": "python"
        }
        
        result = performance_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("efficiency" in finding.lower() or "performance" in finding.lower() 
                  or "algorithm" in finding.lower() for finding in result["findings"])

    def test_performance_memory_leak_detection(self, performance_director):
        """Test detection of potential memory issues."""
        memory_issue_code = """
def process_large_data():
    all_data = []
    for i in range(1000000):
        data = expensive_operation(i)
        all_data.append(data)  # Never cleaned up
    return all_data
        """
        
        task_data = {
            "task_id": "performance_test_002",
            "code": memory_issue_code,
            "language": "python"
        }
        
        result = performance_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("memory" in finding.lower() for finding in result["findings"])

    def test_performance_efficient_code_approval(self, performance_director):
        """Test that efficient code gets approved."""
        efficient_code = """
def find_duplicates_efficient(arr):
    seen = set()
    duplicates = set()
    for item in arr:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)
        """
        
        task_data = {
            "task_id": "performance_test_003",
            "code": efficient_code,
            "language": "python"
        }
        
        result = performance_director.evaluate(task_data)
        
        assert result["confidence"] >= 70
        assert result["recommendation"] == "approved"


# ============================================================================
# Test Ethics Director
# ============================================================================

class TestEthicsDirector:
    """Test the EthicsDirector functionality."""

    @pytest.fixture
    def ethics_director(self):
        """Provide an EthicsDirector instance for testing."""
        return EthicsDirector()

    def test_ethics_director_initialization(self, ethics_director):
        """Test EthicsDirector initialization."""
        assert ethics_director.name == "EthicsDirector"
        assert "ethics" in ethics_director.expertise.lower()
        assert ethics_director.version.startswith("1.")

    def test_ethics_privacy_violation_detection(self, ethics_director):
        """Test detection of privacy violations."""
        privacy_violation_code = """
def log_user_activity(user_id, activity):
    with open('/var/log/user_activity.log', 'a') as f:
        f.write(f"{user_id}: {activity} - {user.personal_info} - {user.ssn}\n")
        """
        
        task_data = {
            "task_id": "ethics_test_001",
            "code": privacy_violation_code,
            "language": "python"
        }
        
        result = ethics_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("privacy" in finding.lower() or "personal" in finding.lower() 
                  for finding in result["findings"])

    def test_ethics_bias_detection(self, ethics_director):
        """Test detection of potential algorithmic bias."""
        biased_code = """
def evaluate_loan_application(applicant):
    score = applicant.credit_score
    if applicant.zip_code in ['90210', '10001']:  # Wealthy areas
        score += 50  # Bias based on location
    if applicant.name in common_names_by_ethnicity['european']:
        score += 25  # Ethnic bias
    return score > 700
        """
        
        task_data = {
            "task_id": "ethics_test_002",
            "code": biased_code,
            "language": "python"
        }
        
        result = ethics_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("bias" in finding.lower() or "discrimination" in finding.lower() 
                  or "fair" in finding.lower() for finding in result["findings"])

    def test_ethics_accessibility_check(self, ethics_director):
        """Test accessibility compliance checking."""
        inaccessible_code = """
<button onclick="submitForm()" style="color: #ccc; background: #ddd;">
    Submit
</button>
        """
        
        task_data = {
            "task_id": "ethics_test_003",
            "code": inaccessible_code,
            "language": "html"
        }
        
        result = ethics_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("accessibility" in finding.lower() or "contrast" in finding.lower() 
                  for finding in result["findings"])

    def test_ethics_compliant_code_approval(self, ethics_director):
        """Test that ethically compliant code gets approved."""
        compliant_code = """
def evaluate_loan_application(applicant):
    """
    Evaluate loan application based on financial factors only.
    Ensures fair and unbiased assessment.
    """
    score = 0
    
    # Credit history (40% weight)
    score += applicant.credit_score * 0.4
    
    # Income to debt ratio (30% weight)
    if applicant.debt_to_income_ratio < 0.3:
        score += 30
    
    # Employment stability (30% weight)
    if applicant.employment_years > 2:
        score += 30
    
    return score > 70
        """
        
        task_data = {
            "task_id": "ethics_test_004",
            "code": compliant_code,
            "language": "python"
        }
        
        result = ethics_director.evaluate(task_data)
        
        assert result["confidence"] >= 70
        assert result["recommendation"] == "approved"


# ============================================================================
# Test UX Director
# ============================================================================

class TestUXDirector:
    """Test the UXDirector functionality."""

    @pytest.fixture
    def ux_director(self):
        """Provide a UXDirector instance for testing."""
        return UXDirector()

    def test_ux_director_initialization(self, ux_director):
        """Test UXDirector initialization."""
        assert ux_director.name == "UXDirector"
        assert "user experience" in ux_director.expertise.lower() or "ux" in ux_director.expertise.lower()
        assert ux_director.version.startswith("1.")

    def test_ux_poor_usability_detection(self, ux_director):
        """Test detection of poor usability patterns."""
        poor_ux_code = """
<form>
    <input type="text" placeholder="Enter data">
    <input type="text" placeholder="Enter more data">
    <input type="text" placeholder="Enter even more data">
    <button>Submit</button>
</form>
        """
        
        task_data = {
            "task_id": "ux_test_001",
            "code": poor_ux_code,
            "language": "html"
        }
        
        result = ux_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("usability" in finding.lower() or "user" in finding.lower() 
                  or "label" in finding.lower() for finding in result["findings"])

    def test_ux_confusing_interface_detection(self, ux_director):
        """Test detection of confusing interface elements."""
        confusing_code = """
function deleteAccount() {
    if (confirm("Are you sure?")) {
        // No additional confirmation for destructive action
        destroyUserData();
    }
}
        """
        
        task_data = {
            "task_id": "ux_test_002",
            "code": confusing_code,
            "language": "javascript"
        }
        
        result = ux_director.evaluate(task_data)
        
        assert result["confidence"] > 0
        assert any("confirmation" in finding.lower() or "destructive" in finding.lower() 
                  or "user experience" in finding.lower() for finding in result["findings"])

    def test_ux_good_interface_approval(self, ux_director):
        """Test that well-designed interfaces get approved."""
        good_ux_code = """
<form aria-label="Contact Form">
    <div class="form-group">
        <label for="name">Full Name *</label>
        <input type="text" id="name" name="name" required 
               aria-describedby="name-help">
        <div id="name-help" class="form-help">Enter your full legal name</div>
    </div>
    <div class="form-group">
        <label for="email">Email Address *</label>
        <input type="email" id="email" name="email" required
               aria-describedby="email-help">
        <div id="email-help" class="form-help">We'll use this to contact you</div>
    </div>
    <button type="submit" class="btn-primary">Send Message</button>
</form>
        """
        
        task_data = {
            "task_id": "ux_test_003",
            "code": good_ux_code,
            "language": "html"
        }
        
        result = ux_director.evaluate(task_data)
        
        assert result["confidence"] >= 70
        assert result["recommendation"] == "approved"


# ============================================================================
# Cross-Director Integration Tests
# ============================================================================

class TestDirectorInteractions:
    """Test interactions between different directors."""

    def test_all_directors_same_task(self, sample_task_data):
        """Test that all directors can evaluate the same task."""
        directors = [
            SecurityDirector(),
            QualityDirector(),
            PerformanceDirector(),
            EthicsDirector(),
            UXDirector()
        ]
        
        results = []
        for director in directors:
            result = director.evaluate(sample_task_data)
            results.append((director.name, result))
        
        # All directors should return valid evaluations
        assert len(results) == 5
        for director_name, result in results:
            assert "confidence" in result
            assert "recommendation" in result
            assert "reasoning" in result
            assert isinstance(result["findings"], list)

    def test_director_consistency(self):
        """Test that directors give consistent results for identical inputs."""
        task_data = {
            "task_id": "consistency_test",
            "code": "def simple_function(): return True",
            "language": "python"
        }
        
        director = SecurityDirector()
        
        # Run evaluation multiple times
        results = [director.evaluate(task_data.copy()) for _ in range(3)]
        
        # Results should be consistent (allowing for small timing variations)
        base_result = results[0]
        for result in results[1:]:
            assert result["confidence"] == base_result["confidence"]
            assert result["recommendation"] == base_result["recommendation"]
            assert result["priority"] == base_result["priority"]

    def test_director_performance_timing(self):
        """Test that directors complete evaluations in reasonable time."""
        task_data = {
            "task_id": "timing_test",
            "code": "def test(): pass",
            "language": "python"
        }
        
        directors = [
            SecurityDirector(),
            QualityDirector(),
            PerformanceDirector(),
            EthicsDirector(),
            UXDirector()
        ]
        
        for director in directors:
            start_time = datetime.now()
            result = director.evaluate(task_data)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Evaluation should complete in reasonable time (less than 5 seconds)
            assert execution_time < 5.0, f"{director.name} took {execution_time} seconds"
            
            # Reported evaluation time should be reasonable
            assert result["evaluation_time"] < 5.0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestDirectorErrorHandling:
    """Test error handling and edge cases for all directors."""

    @pytest.mark.parametrize("director_class", [
        SecurityDirector, QualityDirector, PerformanceDirector, 
        EthicsDirector, UXDirector
    ])
    def test_invalid_input_handling(self, director_class):
        """Test that directors handle invalid input gracefully."""
        director = director_class()
        
        # Test with empty task data
        with pytest.raises((ValueError, KeyError)) or True:
            director.evaluate({})
        
        # Test with None
        with pytest.raises((ValueError, TypeError)) or True:
            director.evaluate(None)
        
        # Test with missing required fields
        incomplete_task = {"task_id": "test"}
        try:
            result = director.evaluate(incomplete_task)
            # If it doesn't raise an exception, it should return a valid result
            assert "confidence" in result
            assert "recommendation" in result
        except (ValueError, KeyError):
            # Acceptable to raise an exception for invalid input
            pass

    @pytest.mark.parametrize("director_class", [
        SecurityDirector, QualityDirector, PerformanceDirector, 
        EthicsDirector, UXDirector
    ])
    def test_large_input_handling(self, director_class):
        """Test that directors can handle large input data."""
        director = director_class()
        
        # Create a large code sample
        large_code = "\n".join([f"def function_{i}(): pass" for i in range(1000)])
        
        task_data = {
            "task_id": "large_input_test",
            "code": large_code,
            "language": "python"
        }
        
        # Should handle large input without crashing
        result = director.evaluate(task_data)
        assert "confidence" in result
        assert "recommendation" in result

    @pytest.mark.parametrize("director_class", [
        SecurityDirector, QualityDirector, PerformanceDirector, 
        EthicsDirector, UXDirector
    ])
    def test_special_characters_handling(self, director_class):
        """Test that directors handle special characters and Unicode."""
        director = director_class()
        
        special_code = """
# Function with special characters: ñáéíóú, 中文, العربية, 🚀
def función_especial():
    message = "Hello 世界! مرحبا 🌍"
    return message
        """
        
        task_data = {
            "task_id": "unicode_test",
            "code": special_code,
            "language": "python"
        }
        
        # Should handle Unicode without crashing
        result = director.evaluate(task_data)
        assert "confidence" in result
        assert "recommendation" in result