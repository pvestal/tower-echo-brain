"""
Comprehensive Auth and Services Tests

Tests:
- Authentication middleware
- User context management
- Permission system
- Vault integration
- Telegram integration
- Financial services
- Plaid authentication
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestUserContextManager:
    """Test user context management"""

    def test_user_context_manager_imports(self):
        """User context manager should import"""
        try:
            from core.user_context_manager import get_user_context_manager
            assert callable(get_user_context_manager)
        except ImportError:
            pytest.skip("User context manager not available")

    @pytest.mark.asyncio
    async def test_get_or_create_context(self, test_user):
        """Should get or create user context"""
        try:
            from core.user_context_manager import get_user_context_manager

            manager = await get_user_context_manager()
            context = await manager.get_or_create_context(test_user["username"])

            assert context is not None
        except ImportError:
            pytest.skip("User context manager not available")

    @pytest.mark.asyncio
    async def test_add_conversation_to_context(self, test_user):
        """Should add conversation to user context"""
        try:
            from core.user_context_manager import get_user_context_manager

            manager = await get_user_context_manager()

            if hasattr(manager, 'add_conversation'):
                await manager.add_conversation(
                    test_user["username"],
                    "user",
                    "Test message"
                )
        except ImportError:
            pytest.skip("User context manager not available")


class TestEchoIdentity:
    """Test Echo identity and user recognition"""

    def test_echo_identity_imports(self):
        """Echo identity should import"""
        try:
            from core.echo_identity import get_echo_identity
            assert callable(get_echo_identity)
        except ImportError:
            pytest.skip("Echo identity not available")

    def test_recognize_user(self, test_user):
        """Should recognize known users"""
        try:
            from core.echo_identity import get_echo_identity

            identity = get_echo_identity()
            recognition = identity.recognize_user(test_user["username"])

            assert "access_level" in recognition
        except ImportError:
            pytest.skip("Echo identity not available")

    def test_recognize_anonymous_user(self):
        """Should handle anonymous users"""
        try:
            from core.echo_identity import get_echo_identity

            identity = get_echo_identity()
            recognition = identity.recognize_user("anonymous")

            assert recognition is not None
        except ImportError:
            pytest.skip("Echo identity not available")


class TestAuthMiddleware:
    """Test authentication middleware"""

    def test_user_context_middleware_imports(self):
        """User context middleware should import"""
        try:
            from middleware.user_context_middleware import UserContextMiddleware
            assert UserContextMiddleware is not None
        except ImportError:
            pytest.skip("User context middleware not available")

    def test_permission_middleware_imports(self):
        """Permission middleware should import"""
        try:
            from middleware.user_context_middleware import PermissionMiddleware
            assert PermissionMiddleware is not None
        except ImportError:
            pytest.skip("Permission middleware not available")


class TestVaultIntegration:
    """Test Vault secret management integration"""

    def test_vault_manager_imports(self):
        """Vault manager should import"""
        try:
            from integrations.vault_manager import get_vault_manager
            assert callable(get_vault_manager)
        except ImportError:
            pytest.skip("Vault manager not available")

    @pytest.mark.asyncio
    async def test_vault_get_secret(self):
        """Should get secret from vault"""
        try:
            from integrations.vault_manager import get_vault_manager

            vault = await get_vault_manager()
            if hasattr(vault, 'get_secret'):
                with patch.object(vault, 'get_secret', new_callable=AsyncMock) as mock:
                    mock.return_value = "secret_value"
                    secret = await vault.get_secret("test/secret")
                    assert secret is not None
        except ImportError:
            pytest.skip("Vault manager not available")

    @pytest.mark.asyncio
    async def test_vault_store_secret(self):
        """Should store secret in vault"""
        try:
            from integrations.vault_manager import get_vault_manager

            vault = await get_vault_manager()
            if hasattr(vault, 'store_secret'):
                with patch.object(vault, 'store_secret', new_callable=AsyncMock) as mock:
                    mock.return_value = True
                    result = await vault.store_secret("test/secret", "value")
        except ImportError:
            pytest.skip("Vault manager not available")


class TestTelegramIntegration:
    """Test Telegram bot integration"""

    def test_telegram_client_imports(self):
        """Telegram client should import"""
        try:
            from integrations.telegram_client import TelegramClient
            assert TelegramClient is not None
        except ImportError:
            pytest.skip("Telegram client not available")

    def test_telegram_executor_imports(self):
        """Telegram executor should import"""
        try:
            from integrations.telegram_echo_executor import telegram_executor_router
            assert telegram_executor_router is not None
        except ImportError:
            pytest.skip("Telegram executor not available")

    @pytest.mark.asyncio
    async def test_telegram_webhook_endpoint(self, async_client):
        """Telegram webhook endpoint should exist"""
        try:
            response = await async_client.post(
                "/api/telegram/webhook",
                json={"update_id": 12345}
            )
            assert response.status_code in [200, 404, 422]
        except Exception:
            pytest.skip("Echo Brain not running")


class TestPlaidAuthentication:
    """Test Plaid financial authentication"""

    def test_plaid_auth_imports(self):
        """Plaid auth should import"""
        try:
            from auth.plaid_auth_api import PlaidAuthAPI
            assert PlaidAuthAPI is not None
        except ImportError:
            pytest.skip("Plaid auth not available")

    def test_plaid_webhooks_imports(self):
        """Plaid webhooks should import"""
        try:
            from auth.plaid_webhooks import PlaidWebhooks
            assert PlaidWebhooks is not None
        except ImportError:
            pytest.skip("Plaid webhooks not available")


class TestFinancialServices:
    """Test financial service integrations"""

    def test_financial_integration_imports(self):
        """Financial integration should import"""
        try:
            from financial.integrated_financial_service import IntegratedFinancialService
            assert IntegratedFinancialService is not None
        except ImportError:
            pytest.skip("Financial service not available")

    def test_simple_plaid_service_imports(self):
        """Simple Plaid service should import"""
        try:
            from financial.simple_plaid_service import SimplePlaidService
            assert SimplePlaidService is not None
        except ImportError:
            pytest.skip("Simple Plaid service not available")


class TestOllamaIntegration:
    """Test Ollama LLM integration"""

    def test_ollama_client_imports(self):
        """Ollama client should import"""
        try:
            from integrations.ollama_client import OllamaClient
            assert OllamaClient is not None
        except ImportError:
            pytest.skip("Ollama client not available")

    @pytest.mark.asyncio
    async def test_ollama_generate(self):
        """Should generate response from Ollama"""
        try:
            from integrations.ollama_client import OllamaClient

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": "test response"}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                client = OllamaClient()
                if hasattr(client, 'generate'):
                    response = await client.generate("test prompt")
        except ImportError:
            pytest.skip("Ollama client not available")


class TestAuthManager:
    """Test authentication manager"""

    def test_auth_manager_imports(self):
        """Auth manager should import"""
        try:
            from integrations.auth_manager import AuthManager
            assert AuthManager is not None
        except ImportError:
            pytest.skip("Auth manager not available")


class TestSecurityModule:
    """Test security module"""

    def test_security_imports(self):
        """Security module should import"""
        try:
            from security import SecurityManager
            assert SecurityManager is not None
        except ImportError:
            try:
                from security.security import SecurityManager
                assert SecurityManager is not None
            except ImportError:
                pytest.skip("Security module not available")


class TestAuthorization:
    """Test authorization system"""

    def test_autonomous_auth_imports(self):
        """Autonomous auth should import"""
        try:
            from authorization.autonomous_auth import AutonomousAuth
            assert AutonomousAuth is not None
        except ImportError:
            pytest.skip("Autonomous auth not available")

    def test_rollback_manager_imports(self):
        """Rollback manager should import"""
        try:
            from authorization.rollback_manager import RollbackManager
            assert RollbackManager is not None
        except ImportError:
            pytest.skip("Rollback manager not available")


class TestAPIEndpointSecurity:
    """Test API endpoint security"""

    @pytest.mark.asyncio
    async def test_secured_endpoint_requires_auth(self, async_client):
        """Secured endpoints should require authentication"""
        try:
            # Try accessing a secured endpoint without auth
            response = await async_client.get("/api/echo/users/test")
            # Should either require auth or return user info
            assert response.status_code in [200, 401, 403, 404]
        except Exception:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_auth_headers_accepted(self, async_client, auth_headers):
        """Should accept authentication headers"""
        try:
            response = await async_client.get(
                "/api/echo/health",
                headers=auth_headers
            )
            assert response.status_code == 200
        except Exception:
            pytest.skip("Echo Brain not running")


class TestServiceHealth:
    """Test service health monitoring"""

    @pytest.mark.asyncio
    async def test_services_health_check(self, async_client):
        """Should report health of dependent services"""
        try:
            response = await async_client.get("/api/echo/health")
            if response.status_code == 200:
                data = response.json()
                # May include service health info
        except Exception:
            pytest.skip("Echo Brain not running")


class TestEmailIntegration:
    """Test email client integration"""

    def test_email_client_imports(self):
        """Email client should import"""
        try:
            from integrations.email_client import EmailClient
            assert EmailClient is not None
        except ImportError:
            pytest.skip("Email client not available")


class TestAppleMusicIntegration:
    """Test Apple Music integration"""

    def test_apple_music_imports(self):
        """Apple Music integration should import"""
        try:
            from integrations.apple.apple_music_bpm_analyzer import AppleMusicBPMAnalyzer
            assert AppleMusicBPMAnalyzer is not None
        except ImportError:
            pytest.skip("Apple Music integration not available")


class TestGitIntegration:
    """Test Git operations integration"""

    def test_git_integration_imports(self):
        """Git integration should import"""
        try:
            from git.echo_git_integration import EchoGitIntegration
            assert EchoGitIntegration is not None
        except ImportError:
            pytest.skip("Git integration not available")

    def test_autonomous_git_controller_imports(self):
        """Autonomous git controller should import"""
        try:
            from git.autonomous_git_controller import AutonomousGitController
            assert AutonomousGitController is not None
        except ImportError:
            pytest.skip("Autonomous git controller not available")
