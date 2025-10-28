"""
Test suite for autonomous authentication system
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

def test_auth_middleware_imports():
    """Test that auth middleware imports successfully"""
    try:
        from src.security.auth_middleware import get_current_user, require_admin_user, get_auth_status
        assert callable(get_current_user)
        assert callable(require_admin_user)
        assert callable(get_auth_status)
    except ImportError as e:
        pytest.fail(f"Auth middleware import failed: {e}")

def test_autonomous_routes_imports():
    """Test that autonomous routes import successfully"""
    try:
        from src.api.autonomous_routes import router, SafeExecutor, AutonomousRequest, AutonomousResponse
        assert router is not None
        assert SafeExecutor is not None
        assert AutonomousRequest is not None
        assert AutonomousResponse is not None
    except ImportError as e:
        pytest.fail(f"Autonomous routes import failed: {e}")

def test_jwt_token_creation():
    """Test JWT token creation functionality"""
    try:
        from src.security.auth_middleware import create_auth_token_for_patrick
        token = create_auth_token_for_patrick()
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are long strings
        assert '.' in token  # JWT tokens contain dots
    except Exception as e:
        pytest.fail(f"JWT token creation failed: {e}")

@pytest.mark.asyncio
async def test_auth_status_endpoint():
    """Test auth status function"""
    try:
        from src.security.auth_middleware import get_auth_status
        with patch('httpx.AsyncClient') as mock_client:
            # Mock auth service unavailable
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Service unavailable")

            status = await get_auth_status()
            assert isinstance(status, dict)
            assert 'auth_service_available' in status
            assert 'local_jwt_verification' in status
            assert status['local_jwt_verification'] is True
    except Exception as e:
        pytest.fail(f"Auth status test failed: {e}")

def test_safe_executor_initialization():
    """Test SafeExecutor can be initialized"""
    try:
        from src.api.autonomous_routes import SafeExecutor
        executor = SafeExecutor()
        assert executor.db_config is not None
        assert 'host' in executor.db_config
        assert 'database' in executor.db_config
    except Exception as e:
        pytest.fail(f"SafeExecutor initialization failed: {e}")

def test_request_response_models():
    """Test Pydantic models for autonomous requests"""
    try:
        from src.api.autonomous_routes import AutonomousRequest, AutonomousResponse

        # Test AutonomousRequest
        request = AutonomousRequest(action="test_action", parameters={}, require_proof=True)
        assert request.action == "test_action"
        assert request.require_proof is True

        # Test AutonomousResponse
        response = AutonomousResponse(
            success=True,
            result="test_result",
            proof="test_proof",
            execution_time=0.1,
            timestamp="2025-10-28T21:00:00Z"
        )
        assert response.success is True
        assert response.result == "test_result"
        assert response.proof == "test_proof"
    except Exception as e:
        pytest.fail(f"Request/Response model test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])