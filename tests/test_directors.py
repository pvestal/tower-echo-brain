"""
Test Suite for Board of Directors System - Individual Directors

This module tests all five specialized directors:
- SecurityDirector: Cybersecurity and vulnerability assessment
- QualityDirector: Code quality and best practices
- PerformanceDirector: Performance optimization and efficiency
- EthicsDirector: Ethical considerations and compliance
- UXDirector: User experience and interface evaluation

Author: Echo Brain Test Suite
Created: 2025-09-16
"""

import pytest
import sys
import os
import json
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to Python path to import directors
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directors import (
    SecurityDirector,
    QualityDirector,
    PerformanceDirector,
    EthicsDirector,
    UXDirector,
    DirectorBase
)


class TestDirectorBase:
    """Test the base director functionality."""

    def test_director_initialization(self):
        """Test that directors initialize correctly."""
        # Test with minimal director
        class TestDirector(DirectorBase):
            def evaluate(self, task_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                return {"test": True}

        director = TestDirector("TestDirector", "Testing")
        assert director.name == "TestDirector"
        assert director.expertise == "Testing"
        assert director.version == "1.0.0"
        assert director.created_at is not None


class TestSecurityDirector:
    """Test Security Director functionality."""

    @pytest.fixture
    def security_director(self):
        """Create a SecurityDirector instance for testing."""
        return SecurityDirector()

    def test_security_director_initialization(self, security_director):
        """Test SecurityDirector initializes correctly."""
        assert security_director.name == "SecurityDirector"
        assert "Cybersecurity" in security_director.expertise
        assert security_director.version == "1.0.0"

    def test_sql_injection_detection(self, security_director):
        """Test detection of SQL injection vulnerabilities."""
        vulnerable_code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id={user_id}"
    return execute_query(query)
        """

        task_data = {
            'code': vulnerable_code,
            'task_type': 'security_review',
            'description': 'Review authentication code'
        }

        result = security_director.evaluate(task_data)

        # Should detect SQL injection risk
        assert result['confidence'] > 50
        assert result['priority'] in ['HIGH', 'CRITICAL']

        # Check for SQL injection in findings or recommendations
        findings_text = json.dumps(result.get('findings', [])).lower()
        recommendations_text = json.dumps(result.get('recommendations', [])).lower()
        reasoning_text = result.get('reasoning', '').lower()

        assert any('sql' in text and 'injection' in text
                  for text in [findings_text, recommendations_text, reasoning_text])

    def test_xss_vulnerability_detection(self, security_director):
        """Test detection of XSS vulnerabilities."""
        vulnerable_code = """
function displayMessage(msg) {
    document.getElementById('output').innerHTML = msg;
}
        """

        task_data = {
            'code': vulnerable_code,
            'task_type': 'security_review',
            'language': 'javascript'
        }

        result = security_director.evaluate(task_data)

        # Should detect XSS risk
        assert result['confidence'] > 30

        # Check for XSS-related findings
        result_text = json.dumps(result).lower()
        assert 'xss' in result_text or 'cross-site' in result_text or 'innerhtml' in result_text

    def test_secure_code_approval(self, security_director):
        """Test that secure code gets good evaluation."""
        secure_code = """
import hashlib
import secrets
from sqlalchemy import text

def authenticate_user(username, password):
    # Use parameterized queries
    query = text("SELECT id, password_hash FROM users WHERE username = :username")
    user = db.execute(query, username=username).fetchone()

    if user and verify_password(password, user.password_hash):
        return user.id
    return None

def verify_password(password, hash):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000) == hash
        """

        task_data = {
            'code': secure_code,
            'task_type': 'security_review'
        }

        result = security_director.evaluate(task_data)

        # Should approve secure code
        assert result['confidence'] >= 0
        assert result['priority'] in ['LOW', 'MEDIUM', 'HIGH']  # Not CRITICAL


class TestQualityDirector:
    """Test Quality Director functionality."""

    @pytest.fixture
    def quality_director(self):
        """Create a QualityDirector instance for testing."""
        return QualityDirector()

    def test_quality_director_initialization(self, quality_director):
        """Test QualityDirector initializes correctly."""
        assert quality_director.name == "QualityDirector"
        assert "Code Quality" in quality_director.expertise or "Quality" in quality_director.expertise

    def test_code_complexity_detection(self, quality_director):
        """Test detection of high code complexity."""
        complex_code = """
def complex_function(x, y, z):
    if x > 10:
        if y < 5:
            if z == 0:
                return "case1"
            elif z == 1:
                return "case2"
            elif z == 2:
                return "case3"
            else:
                return "case4"
        elif y > 20:
            if z > 100:
                return "case5"
            else:
                return "case6"
        else:
            return "case7"
    elif x < 0:
        if y > 0:
            return "case8"
        else:
            return "case9"
    else:
        return "case10"
        """

        task_data = {
            'code': complex_code,
            'task_type': 'code_review'
        }

        result = quality_director.evaluate(task_data)

        # Should detect complexity issues
        assert result['confidence'] > 40

        result_text = json.dumps(result).lower()
        assert any(keyword in result_text
                  for keyword in ['complex', 'nested', 'refactor', 'simplify'])

    def test_naming_conventions(self, quality_director):
        """Test detection of poor naming conventions."""
        poor_code = """
def a(b, c):
    d = b + c
    if d > 10:
        e = d * 2
        return e
    return d
        """

        task_data = {
            'code': poor_code,
            'task_type': 'code_review'
        }

        result = quality_director.evaluate(task_data)

        # Should detect naming issues
        result_text = json.dumps(result).lower()
        assert any(keyword in result_text
                  for keyword in ['naming', 'variable', 'descriptive', 'readable'])

    def test_good_quality_code(self, quality_director):
        """Test that well-written code gets positive evaluation."""
        good_code = """
def calculate_total_price(item_price: float, quantity: int, tax_rate: float = 0.08) -> float:
    \"\"\"
    Calculate the total price including tax for a given item.

    Args:
        item_price: The base price of the item
        quantity: Number of items
        tax_rate: Tax rate as decimal (default 8%)

    Returns:
        Total price including tax
    \"\"\"
    if quantity <= 0:
        raise ValueError("Quantity must be positive")

    subtotal = item_price * quantity
    tax_amount = subtotal * tax_rate
    total = subtotal + tax_amount

    return round(total, 2)
        """

        task_data = {
            'code': good_code,
            'task_type': 'code_review'
        }

        result = quality_director.evaluate(task_data)

        # Should approve good code
        assert result['confidence'] >= 0

        # May have minor suggestions but should be generally positive
        if result.get('findings'):
            result_text = json.dumps(result).lower()
            assert any(positive in result_text
                      for positive in ['good', 'well', 'clear', 'documented'])


class TestPerformanceDirector:
    """Test Performance Director functionality."""

    @pytest.fixture
    def performance_director(self):
        """Create a PerformanceDirector instance for testing."""
        return PerformanceDirector()

    def test_performance_director_initialization(self, performance_director):
        """Test PerformanceDirector initializes correctly."""
        assert performance_director.name == "PerformanceDirector"
        assert "Performance" in performance_director.expertise

    def test_inefficient_algorithm_detection(self, performance_director):
        """Test detection of inefficient algorithms."""
        inefficient_code = """
def find_duplicates(numbers):
    duplicates = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j] and numbers[i] not in duplicates:
                duplicates.append(numbers[i])
    return duplicates
        """

        task_data = {
            'code': inefficient_code,
            'task_type': 'performance_review'
        }

        result = performance_director.evaluate(task_data)

        # Should detect O(n²) complexity
        result_text = json.dumps(result).lower()
        assert any(keyword in result_text
                  for keyword in ['o(n', 'complexity', 'inefficient', 'optimize', 'nested'])

    def test_database_query_optimization(self, performance_director):
        """Test detection of database performance issues."""
        poor_query_code = """
def get_user_posts(user_id):
    posts = []
    for post_id in get_user_post_ids(user_id):
        post = db.query("SELECT * FROM posts WHERE id = ?", post_id)
        posts.append(post)
    return posts
        """

        task_data = {
            'code': poor_query_code,
            'task_type': 'performance_review'
        }

        result = performance_director.evaluate(task_data)

        # Should detect N+1 query problem
        result_text = json.dumps(result).lower()
        assert any(keyword in result_text
                  for keyword in ['n+1', 'query', 'join', 'batch', 'database'])


class TestEthicsDirector:
    """Test Ethics Director functionality."""

    @pytest.fixture
    def ethics_director(self):
        """Create an EthicsDirector instance for testing."""
        return EthicsDirector()

    def test_ethics_director_initialization(self, ethics_director):
        """Test EthicsDirector initializes correctly."""
        assert ethics_director.name == "EthicsDirector"
        assert "Ethics" in ethics_director.expertise or "Ethical" in ethics_director.expertise

    def test_privacy_violation_detection(self, ethics_director):
        """Test detection of privacy violations."""
        privacy_violating_code = """
def log_user_activity(user_id, activity):
    # Log everything including personal data
    log_entry = {
        'user_id': user_id,
        'activity': activity,
        'ip_address': get_client_ip(),
        'personal_data': get_user_personal_info(user_id),
        'browsing_history': get_user_browsing_history(user_id)
    }
    analytics_service.send_to_third_party(log_entry)
        """

        task_data = {
            'code': privacy_violating_code,
            'task_type': 'ethics_review'
        }

        result = ethics_director.evaluate(task_data)

        # Should detect privacy concerns
        result_text = json.dumps(result).lower()
        assert any(keyword in result_text
                  for keyword in ['privacy', 'personal', 'gdpr', 'data', 'consent'])

    def test_bias_detection(self, ethics_director):
        """Test detection of potential bias in algorithms."""
        biased_code = """
def calculate_loan_score(applicant):
    score = 100

    # Problematic bias factors
    if applicant['zip_code'] in ['12345', '67890']:  # redlining
        score -= 20

    if applicant['age'] > 50:
        score -= 10

    if applicant['gender'] == 'female':
        score -= 5

    return score
        """

        task_data = {
            'code': biased_code,
            'task_type': 'ethics_review'
        }

        result = ethics_director.evaluate(task_data)

        # Should detect bias
        result_text = json.dumps(result).lower()
        assert any(keyword in result_text
                  for keyword in ['bias', 'discriminat', 'fair', 'equal', 'age', 'gender'])


class TestUXDirector:
    """Test UX Director functionality."""

    @pytest.fixture
    def ux_director(self):
        """Create a UXDirector instance for testing."""
        return UXDirector()

    def test_ux_director_initialization(self, ux_director):
        """Test UXDirector initializes correctly."""
        assert ux_director.name == "UXDirector"
        assert "UX" in ux_director.expertise or "User Experience" in ux_director.expertise

    def test_accessibility_issues(self, ux_director):
        """Test detection of accessibility problems."""
        poor_accessibility_code = """
<div onclick="doSomething()">
    <img src="photo.jpg">
    <input type="text" placeholder="Enter value">
    <button>Click</button>
</div>
        """

        task_data = {
            'code': poor_accessibility_code,
            'task_type': 'ux_review',
            'language': 'html'
        }

        result = ux_director.evaluate(task_data)

        # Should detect accessibility issues
        result_text = json.dumps(result).lower()
        assert any(keyword in result_text
                  for keyword in ['accessibility', 'alt', 'aria', 'label', 'wcag', 'keyboard'])

    def test_usability_concerns(self, ux_director):
        """Test detection of usability problems."""
        poor_usability_code = """
function processForm() {
    // No validation feedback
    if (!validateEmail(email)) {
        return false;
    }

    // No loading state
    submitData();

    // No success/error feedback
}
        """

        task_data = {
            'code': poor_usability_code,
            'task_type': 'ux_review',
            'language': 'javascript'
        }

        result = ux_director.evaluate(task_data)

        # Should detect usability issues
        result_text = json.dumps(result).lower()
        assert any(keyword in result_text
                  for keyword in ['feedback', 'validation', 'loading', 'error', 'user'])


class TestDirectorErrorHandling:
    """Test error handling across all directors."""

    @pytest.fixture(params=[
        SecurityDirector, QualityDirector, PerformanceDirector,
        EthicsDirector, UXDirector
    ])
    def director(self, request):
        """Parameterized fixture for all director types."""
        return request.param()

    def test_empty_task_handling(self, director):
        """Test handling of empty or minimal task data."""
        empty_task = {}
        result = director.evaluate(empty_task)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'confidence' in result or 'error' in result

    def test_invalid_task_type(self, director):
        """Test handling of invalid task types."""
        invalid_task = {
            'task_type': 'invalid_type',
            'code': 'print("test")'
        }

        result = director.evaluate(invalid_task)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_large_code_input(self, director):
        """Test handling of very large code inputs."""
        large_code = "print('test')\n" * 10000

        task_data = {
            'code': large_code,
            'task_type': 'review'
        }

        result = director.evaluate(task_data)

        # Should handle without crashing
        assert isinstance(result, dict)


# Integration tests for director interactions
class TestDirectorIntegration:
    """Test interactions between directors and common scenarios."""

    def test_all_directors_available(self):
        """Test that all directors can be instantiated."""
        directors = [
            SecurityDirector(),
            QualityDirector(),
            PerformanceDirector(),
            EthicsDirector(),
            UXDirector()
        ]

        assert len(directors) == 5
        for director in directors:
            assert hasattr(director, 'evaluate')
            assert hasattr(director, 'name')
            assert hasattr(director, 'expertise')

    def test_consistent_evaluation_interface(self):
        """Test that all directors have consistent interfaces."""
        directors = [
            SecurityDirector(),
            QualityDirector(),
            PerformanceDirector(),
            EthicsDirector(),
            UXDirector()
        ]

        test_task = {
            'code': 'def test(): pass',
            'task_type': 'review'
        }

        for director in directors:
            result = director.evaluate(test_task)

            # All should return dict with these basic fields
            assert isinstance(result, dict)

            # Should have some form of assessment
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])