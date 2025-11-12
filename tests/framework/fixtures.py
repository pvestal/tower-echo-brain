"""
Shared test fixtures for Echo Brain testing
"""

import pytest
import asyncio
from unittest.mock import MagicMock
import tempfile
import os

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_database():
    """Mock database connection"""
    db_mock = MagicMock()
    db_mock.execute.return_value = []
    db_mock.fetchall.return_value = []
    db_mock.fetchone.return_value = None
    return db_mock

@pytest.fixture
def mock_redis():
    """Mock Redis connection"""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.exists.return_value = False
    return redis_mock

@pytest.fixture
def temp_config_dir():
    """Temporary directory for configuration files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_vault_client():
    """Mock HashiCorp Vault client"""
    vault_mock = MagicMock()
    vault_mock.secrets.kv.v2.read_secret_version.return_value = {
        'data': {'data': {'test_key': 'test_value'}}
    }
    return vault_mock

@pytest.fixture
def echo_test_config():
    """Test configuration for Echo Brain"""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'test_echo_brain',
            'user': 'test_user',
            'password': 'test_password'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        },
        'api': {
            'host': '127.0.0.1',
            'port': 8309,
            'debug': True
        }
    }
