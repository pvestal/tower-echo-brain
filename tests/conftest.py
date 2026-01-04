"""
Pytest Configuration and Fixtures for Echo Brain Anime Generation Testing
Provides reusable test fixtures, mock configurations, and test data setup.

Author: Echo Brain Anime Testing Framework
Created: 2025-11-19
"""

import pytest
import sys
import os
import json
import asyncio
import tempfile
import shutil
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests that require database"
    )
    config.addinivalue_line(
        "markers", "async_test: marks tests that test async functionality"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark slow tests
        if "slow" in item.nodeid or any(
            keyword in item.name.lower()
            for keyword in ["performance", "stress", "load", "concurrent"]
        ):
            item.add_marker(pytest.mark.slow)

        # Auto-mark integration tests
        if "integration" in item.nodeid or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Auto-mark API tests
        if "api" in item.nodeid or "test_board_api" in item.nodeid:
            item.add_marker(pytest.mark.api)

        # Auto-mark database tests
        if any(keyword in item.nodeid.lower() for keyword in ["database", "db_pool", "pool"]):
            item.add_marker(pytest.mark.database)

        # Auto-mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.async_test)


# ============================================================================
# Common Fixtures
# ============================================================================

@pytest.fixture
def mock_db_config():
    """Provide mock database configuration for testing."""
    return {
        "host": "localhost",
        "database": "test_echo_brain",
        "user": "test_user",
        "password": "test_password",
        "port": 5432,
        "connect_timeout": 10,
        "application_name": "echo_brain_test"
    }


@pytest.fixture
def auth_token():
    """Provide a mock JWT token for authentication testing."""
    return "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidGVzdF91c2VyIiwicGVybWlzc2lvbnMiOlsiYm9hcmQ6c3VibWl0IiwiYm9hcmQ6dmlldyIsImJvYXJkOmZlZWRiYWNrIl0sImV4cCI6OTk5OTk5OTk5OX0.test_signature"


@pytest.fixture
def mock_user():
    """Provide a mock authenticated user for testing."""
    return {
        "user_id": "test_user_123",
        "username": "test_user",
        "email": "test@example.com",
        "permissions": [
            "board:submit",
            "board:view",
            "board:feedback",
            "board:admin"
        ],
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat()
    }


@pytest.fixture
def limited_user():
    """Provide a mock user with limited permissions."""
    return {
        "user_id": "limited_user_456",
        "username": "limited_user",
        "email": "limited@example.com",
        "permissions": ["board:view"],  # Only view permission
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat()
    }


@pytest.fixture
def sample_task_data():
    """Provide sample task data for testing."""
    return {
        "task_description": "Review authentication system",
        "user_id": "test_user",
        "priority": "high",
        "context": {
            "code": """
def authenticate_user(username, password):
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        return user
    return None
            """,
            "task_type": "security_review",
            "language": "python",
            "framework": "flask"
        },
        "expected_completion_time": (datetime.now() + timedelta(hours=1)).isoformat()
    }


@pytest.fixture
def sample_evaluation_result():
    """Provide sample evaluation result for testing."""
    return {
        "task_id": "test-task-123",
        "status": "completed",
        "success": True,
        "evaluations": [
            {
                "director_name": "SecurityDirector",
                "recommendation": "approved",
                "confidence": 90,
                "priority": "HIGH",
                "findings": [
                    "Uses secure password hashing",
                    "Proper user lookup implementation",
                    "No obvious security vulnerabilities"
                ],
                "reasoning": "The authentication implementation follows security best practices with proper password hashing and user validation.",
                "evaluation_time": 0.25,
                "timestamp": datetime.now().isoformat()
            },
            {
                "director_name": "QualityDirector",
                "recommendation": "approved",
                "confidence": 85,
                "priority": "MEDIUM",
                "findings": [
                    "Good function structure",
                    "Clear naming conventions",
                    "Could benefit from type hints"
                ],
                "reasoning": "Code quality is good with room for minor improvements in type annotations.",
                "evaluation_time": 0.18,
                "timestamp": datetime.now().isoformat()
            }
        ],
        "consensus": {
            "recommendation": "approved",
            "confidence": 87.5,
            "agreement_level": 0.95,
            "weighted_confidence": 88.2,
            "reasoning": "Strong consensus for approval with high confidence from security and quality directors."
        },
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
        "processing_time": 0.43
    }


@pytest.fixture
def vulnerable_code_sample():
    """Provide sample vulnerable code for security testing."""
    return {
        "sql_injection": """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return database.execute(query).fetchall()
        """,
        "xss_vulnerability": """
function displayMessage(msg) {
    document.getElementById('output').innerHTML = msg;
}
        """,
        "hardcoded_secrets": """
API_KEY = "sk-1234567890abcdef"
DATABASE_PASSWORD = "admin123"

def connect_to_api():
    return requests.get(f"https://api.example.com/data?key={API_KEY}")
        """,
        "weak_authentication": """
def login(username, password):
    if username == "admin" and password == "password":
        return True
    return False
        """
    }


@pytest.fixture
def quality_issues_sample():
    """Provide sample code with quality issues for testing."""
    return {
        "high_complexity": """
def process_data(x, y, z, a, b, c):
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        if c > 0:
                            return x + y + z + a + b + c
                        else:
                            return x + y + z + a + b
                    else:
                        return x + y + z + a
                else:
                    return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0
        """,
        "poor_naming": """
def calc(a, b, c):
    d = a + b
    e = d * c
    f = e / 2
    return f
        """,
        "no_documentation": """
def mysterious_function(data):
    result = []
    for item in data:
        if len(item) > 5:
            result.append(item.upper())
    return result
        """
    }


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_director():
    """Provide a mock director for testing."""
    class MockDirector:
        def __init__(self, name="MockDirector", expertise="Testing"):
            self.name = name
            self.expertise = expertise
            self.version = "1.0.0"
            self.created_at = datetime.now()
            self.evaluation_history = []

        def evaluate(self, task_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
            return {
                "confidence": 75,
                "recommendation": "approved",
                "priority": "MEDIUM",
                "findings": ["Mock evaluation finding"],
                "reasoning": "Mock evaluation reasoning",
                "evaluation_time": 0.1,
                "timestamp": datetime.now().isoformat()
            }

    return MockDirector


@pytest.fixture
def mock_database_pool():
    """Provide a mock database pool for testing."""
    with patch('directors.db_pool.DatabasePool') as mock_pool_class:
        mock_instance = MagicMock()
        mock_connection = MagicMock()

        # Configure mock connection
        mock_connection.cursor.return_value = MagicMock()
        mock_connection.commit = MagicMock()
        mock_connection.rollback = MagicMock()

        # Configure mock pool
        mock_instance.get_connection.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_instance.get_connection.return_value.__exit__ = Mock(return_value=None)
        mock_instance.close = MagicMock()

        mock_pool_class.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def mock_request_logger():
    """Provide a mock decision tracker for testing."""
    with patch('directors.request_logger.DecisionTracker') as mock_tracker_class:
        mock_instance = AsyncMock()

        # Configure async methods
        mock_instance.submit_task = AsyncMock(return_value="test-task-123")
        mock_instance.get_decision = AsyncMock(return_value={
            "task_id": "test-task-123",
            "status": "completed",
            "decision": "approved"
        })
        mock_instance.list_tasks = AsyncMock(return_value=[])
        mock_instance.add_feedback = AsyncMock(return_value=True)

        mock_tracker_class.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def mock_auth_middleware():
    """Provide mock authentication middleware."""
    with patch('directors.auth_middleware.get_current_user') as mock_get_user, \
         patch('directors.auth_middleware.require_permission') as mock_require_perm, \
         patch('directors.auth_middleware.authenticate_websocket') as mock_auth_ws:

        # Configure mocks
        mock_get_user.return_value = {
            "user_id": "test_user",
            "permissions": ["board:submit", "board:view", "board:feedback"]
        }
        mock_require_perm.return_value = True
        mock_auth_ws.return_value = {
            "user_id": "test_user",
            "permissions": ["board:view"]
        }

        yield {
            "get_current_user": mock_get_user,
            "require_permission": mock_require_perm,
            "authenticate_websocket": mock_auth_ws
        }


# ============================================================================
# Temporary Resources
# ============================================================================

@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="echo_brain_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_file():
    """Provide a temporary file for testing."""
    temp_fd, temp_path = tempfile.mkstemp(prefix="echo_brain_test_", suffix=".txt")
    os.close(temp_fd)  # Close the file descriptor
    yield temp_path
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def temp_config_file():
    """Provide a temporary configuration file for testing."""
    config_data = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db"
        },
        "directors": {
            "consensus_threshold": 0.6,
            "max_directors_per_task": 5
        },
        "api": {
            "port": 8000,
            "debug": True
        }
    }

    temp_fd, temp_path = tempfile.mkstemp(prefix="echo_brain_config_", suffix=".json")
    with os.fdopen(temp_fd, 'w') as f:
        json.dump(config_data, f, indent=2)

    yield temp_path

    try:
        os.unlink(temp_path)
    except OSError:
        pass


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def clean_environment(monkeypatch):
    """Provide a clean environment for testing."""
    # Clear relevant environment variables
    env_vars_to_clear = [
        "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT",
        "ECHO_DEBUG", "ECHO_LOG_LEVEL", "JWT_SECRET_KEY"
    ]

    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    # Set test-specific environment
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_NAME", "test_echo_brain")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")

    yield


@pytest.fixture
def mock_environment(monkeypatch):
    """Provide mock environment variables for testing."""
    test_env = {
        "DB_HOST": "test.example.com",
        "DB_NAME": "test_database",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "DB_PORT": "5433",
        "JWT_SECRET_KEY": "test_secret_key",
        "ECHO_DEBUG": "true",
        "ECHO_LOG_LEVEL": "DEBUG"
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    yield test_env


# ============================================================================
# Async Test Utilities
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_test_client():
    """Provide an async test client for API testing."""
    from fastapi.testclient import TestClient

    # Mock the app import to avoid dependency issues
    with patch('board_api.app') as mock_app:
        mock_app.debug = True
        client = TestClient(mock_app)
        yield client


# ============================================================================
# Performance Testing Utilities
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities."""
    import time
    import psutil
    import threading

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.peak_memory = None
            self._monitoring = False
            self._monitor_thread = None

        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.virtual_memory().used
            self.peak_memory = self.start_memory
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_memory)
            self._monitor_thread.start()

        def stop(self):
            self.end_time = time.time()
            self.end_memory = psutil.virtual_memory().used
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join()

        def _monitor_memory(self):
            while self._monitoring:
                current_memory = psutil.virtual_memory().used
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                time.sleep(0.1)

        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

        @property
        def memory_delta(self):
            if self.start_memory and self.end_memory:
                return self.end_memory - self.start_memory
            return None

        @property
        def peak_memory_delta(self):
            if self.start_memory and self.peak_memory:
                return self.peak_memory - self.start_memory
            return None

    return PerformanceMonitor()


# ============================================================================
# Logging Utilities
# ============================================================================

@pytest.fixture
def capture_logs(caplog):
    """Provide enhanced log capture utilities."""
    import logging

    # Set logging level to capture all messages
    caplog.set_level(logging.DEBUG)

    class LogCapture:
        def __init__(self, caplog):
            self.caplog = caplog

        def get_logs_containing(self, text):
            return [record for record in self.caplog.records if text in record.message]

        def get_logs_by_level(self, level):
            return [record for record in self.caplog.records if record.levelno == level]

        def has_error_logs(self):
            return any(record.levelno >= logging.ERROR for record in self.caplog.records)

        def has_warning_logs(self):
            return any(record.levelno >= logging.WARNING for record in self.caplog.records)

        @property
        def all_messages(self):
            return [record.message for record in self.caplog.records]

    return LogCapture(caplog)


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def generate_test_tasks():
    """Provide a generator for test task data."""
    def generator(count=10, task_type="review"):
        tasks = []
        for i in range(count):
            task = {
                "task_id": f"test-task-{i:03d}",
                "task_type": task_type,
                "description": f"Test task {i+1}",
                "code": f"def test_function_{i}(): pass",
                "priority": ["low", "medium", "high"][i % 3],
                "created_at": datetime.now().isoformat(),
                "user_id": f"user_{i % 3}"
            }
            tasks.append(task)
        return tasks

    return generator


@pytest.fixture
def generate_test_evaluations():
    """Provide a generator for test evaluation data."""
    def generator(count=5):
        evaluations = []
        directors = ["SecurityDirector", "QualityDirector", "PerformanceDirector", "EthicsDirector", "UXDirector"]
        recommendations = ["approved", "needs_review", "rejected"]

        for i in range(count):
            evaluation = {
                "director_name": directors[i % len(directors)],
                "recommendation": recommendations[i % len(recommendations)],
                "confidence": 50 + (i * 10) % 50,  # 50-99
                "priority": ["LOW", "MEDIUM", "HIGH", "CRITICAL"][i % 4],
                "findings": [f"Finding {i+1}", f"Additional finding {i+1}"],
                "reasoning": f"Evaluation reasoning for test {i+1}",
                "evaluation_time": 0.1 + (i * 0.05),
                "timestamp": datetime.now().isoformat()
            }
            evaluations.append(evaluation)

        return evaluations

    return generator