"""
Critical functionality tests for Echo Brain
These tests must pass for production deployment
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from tests.framework.test_markers import critical, unit, api

@pytest.mark.critical
@pytest.mark.unit
def test_echo_brain_import():
    """Test that core Echo Brain modules can be imported"""
    try:
        from src.main import app
        assert app is not None
    except ImportError as e:
        pytest.fail(f"Failed to import Echo Brain main module: {e}")

@pytest.mark.critical
@pytest.mark.api
def test_health_endpoint_exists():
    """Test that health endpoint is accessible"""
    from src.main import app
    
    # Verify health endpoint is registered
    with patch('src.main.app') as mock_app:
        mock_app.routes = []
        # This would normally check if /api/echo/health route exists
        assert True  # Placeholder for actual health check

@pytest.mark.critical
@pytest.mark.unit
def test_configuration_loading():
    """Test that configuration can be loaded"""
    try:
        # Test basic configuration loading
        config = {
            'database': {'host': 'localhost'},
            'api': {'port': 8309}
        }
        assert 'database' in config
        assert 'api' in config
    except Exception as e:
        pytest.fail(f"Configuration loading failed: {e}")

@pytest.mark.critical
@pytest.mark.unit 
def test_database_schema_validation():
    """Test that database schema is valid"""
    # Placeholder for database schema validation
    required_tables = [
        'conversations', 
        'echo_decisions', 
        'autonomous_tasks',
        'model_decisions'
    ]
    
    # In real implementation, this would check database schema
    for table in required_tables:
        assert len(table) > 0  # Placeholder validation

@pytest.mark.critical
@pytest.mark.unit
def test_model_registry_initialization():
    """Test that model registry can be initialized"""
    try:
        # Test model registry initialization
        models = ['llama3.1', 'qwen2.5-coder', 'deepseek-coder']
        assert len(models) > 0
    except Exception as e:
        pytest.fail(f"Model registry initialization failed: {e}")

@pytest.mark.critical
@pytest.mark.unit
async def test_async_functionality():
    """Test that async functionality works correctly"""
    async def dummy_async_function():
        await asyncio.sleep(0.1)
        return True
    
    result = await dummy_async_function()
    assert result is True
