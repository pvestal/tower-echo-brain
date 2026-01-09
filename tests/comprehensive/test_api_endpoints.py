"""
Comprehensive API Endpoint Tests for Echo Brain

Tests all API endpoints:
- Health checks
- Chat/Query endpoints
- Delegation endpoints
- Model management
- User preferences
- Knowledge base
- Git operations
"""

import pytest
import httpx
from unittest.mock import patch, AsyncMock, MagicMock
import json


class TestHealthEndpoints:
    """Test health check endpoints"""

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self, async_client, test_config):
        """GET /api/echo/health should return healthy status"""
        try:
            response = await async_client.get("/api/echo/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] in ["healthy", "ok", "running"]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running - run with live server")

    @pytest.mark.asyncio
    async def test_health_check_includes_timestamp(self, async_client):
        """Health check should include timestamp"""
        try:
            response = await async_client.get("/api/echo/health")
            if response.status_code == 200:
                data = response.json()
                assert "timestamp" in data or "time" in data
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_stats_endpoint(self, async_client):
        """GET /api/echo/stats should return system statistics"""
        try:
            response = await async_client.get("/api/echo/stats")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_ready_endpoint(self, async_client):
        """GET /ready should indicate service readiness"""
        try:
            response = await async_client.get("/ready")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_alive_endpoint(self, async_client):
        """GET /alive should indicate service is alive"""
        try:
            response = await async_client.get("/alive")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")


class TestChatQueryEndpoints:
    """Test main chat and query endpoints"""

    @pytest.mark.asyncio
    async def test_chat_endpoint_accepts_valid_request(self, async_client, chat_request):
        """POST /api/echo/chat should accept valid request"""
        try:
            response = await async_client.post("/api/echo/chat", json=chat_request)
            assert response.status_code in [200, 201, 202]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_chat_endpoint_returns_response(self, async_client, chat_request):
        """Chat endpoint should return a response field"""
        try:
            response = await async_client.post("/api/echo/chat", json=chat_request)
            if response.status_code == 200:
                data = response.json()
                assert "response" in data
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_chat_endpoint_returns_model_used(self, async_client, chat_request):
        """Chat endpoint should indicate which model was used"""
        try:
            response = await async_client.post("/api/echo/chat", json=chat_request)
            if response.status_code == 200:
                data = response.json()
                assert "model_used" in data or "model" in data
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_query_endpoint_alias(self, async_client, chat_request):
        """POST /api/echo/query should work same as chat"""
        try:
            response = await async_client.post("/api/echo/query", json=chat_request)
            assert response.status_code in [200, 201, 202]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_chat_requires_query_field(self, async_client):
        """Chat should fail without query field"""
        try:
            response = await async_client.post("/api/echo/chat", json={"user_id": "test"})
            assert response.status_code in [400, 422]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_chat_with_conversation_id(self, async_client):
        """Chat should accept and return conversation_id for context"""
        try:
            request = {
                "query": "Remember that my name is TestUser",
                "user_id": "test",
                "conversation_id": "conv_test_123"
            }
            response = await async_client.post("/api/echo/chat", json=request)
            if response.status_code == 200:
                data = response.json()
                assert "conversation_id" in data
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_chat_intelligence_levels(self, async_client):
        """Chat should accept different intelligence levels"""
        levels = ["auto", "basic", "advanced", "reasoning"]
        for level in levels:
            try:
                request = {
                    "query": "Test query",
                    "user_id": "test",
                    "intelligence_level": level
                }
                response = await async_client.post("/api/echo/chat", json=request)
                assert response.status_code in [200, 201, 202, 422], f"Failed for level: {level}"
            except httpx.ConnectError:
                pytest.skip("Echo Brain not running")


class TestDelegationEndpoints:
    """Test delegation to Tower LLMs endpoints"""

    @pytest.mark.asyncio
    async def test_delegation_endpoint_exists(self, async_client, delegation_request):
        """POST /api/echo/delegate/to-tower should exist"""
        try:
            response = await async_client.post("/api/echo/delegate/to-tower", json=delegation_request)
            assert response.status_code in [200, 201, 202, 500]  # 500 if Ollama not running
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_delegation_capabilities_endpoint(self, async_client):
        """GET /api/echo/delegate/capabilities should return available models"""
        try:
            response = await async_client.get("/api/echo/delegate/capabilities")
            assert response.status_code == 200
            data = response.json()
            assert "available_models" in data or "capabilities" in data or "model" in data
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_delegation_history_endpoint(self, async_client):
        """GET /api/echo/delegate/history should return execution history"""
        try:
            response = await async_client.get("/api/echo/delegate/history")
            assert response.status_code == 200
            data = response.json()
            assert "history" in data or "executions" in data or isinstance(data, list)
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_delegation_requires_task(self, async_client):
        """Delegation should fail without task field"""
        try:
            response = await async_client.post("/api/echo/delegate/to-tower", json={})
            assert response.status_code in [400, 422]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")


class TestConversationEndpoints:
    """Test conversation management endpoints"""

    @pytest.mark.asyncio
    async def test_get_conversation(self, async_client):
        """GET /api/echo/conversation/{id} should return conversation"""
        try:
            response = await async_client.get("/api/echo/conversation/test_conv_001")
            # 404 is acceptable if conversation doesn't exist
            assert response.status_code in [200, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_list_conversations(self, async_client):
        """GET /api/echo/conversations should list conversations"""
        try:
            response = await async_client.get("/api/echo/conversations")
            assert response.status_code in [200, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")


class TestUserEndpoints:
    """Test user management endpoints"""

    @pytest.mark.asyncio
    async def test_get_user(self, async_client, test_user):
        """GET /api/echo/users/{username} should return user info"""
        try:
            response = await async_client.get(f"/api/echo/users/{test_user['username']}")
            assert response.status_code in [200, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_set_user_preferences(self, async_client, test_user):
        """POST /api/echo/users/{username}/preferences should set preferences"""
        try:
            prefs = {"theme": "dark", "language": "en"}
            response = await async_client.post(
                f"/api/echo/users/{test_user['username']}/preferences",
                json=prefs
            )
            assert response.status_code in [200, 201, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")


class TestGitOperationsEndpoints:
    """Test Git operations endpoints"""

    @pytest.mark.asyncio
    async def test_git_status(self, async_client):
        """GET /api/echo/git/status should return git status"""
        try:
            response = await async_client.get("/api/echo/git/status")
            assert response.status_code in [200, 404, 500]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_git_health(self, async_client):
        """GET /api/echo/git/health should return git health"""
        try:
            response = await async_client.get("/api/echo/git/health")
            assert response.status_code in [200, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")


class TestCodeEndpoints:
    """Test code generation/debugging endpoints"""

    @pytest.mark.asyncio
    async def test_code_generate(self, async_client):
        """POST /api/echo/code/generate should generate code"""
        try:
            request = {
                "prompt": "Write a function to add two numbers",
                "language": "python"
            }
            response = await async_client.post("/api/echo/code/generate", json=request)
            assert response.status_code in [200, 201, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_code_debug(self, async_client):
        """POST /api/echo/code/debug should debug code"""
        try:
            request = {
                "code": "def add(a, b):\n    return a + c",  # intentional bug
                "language": "python"
            }
            response = await async_client.post("/api/echo/code/debug", json=request)
            assert response.status_code in [200, 201, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")


class TestAgentEndpoints:
    """Test agent-related endpoints"""

    @pytest.mark.asyncio
    async def test_list_agents(self, async_client):
        """GET /api/theater/agents should list available agents"""
        try:
            response = await async_client.get("/api/theater/agents")
            assert response.status_code in [200, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_coding_agent_endpoint(self, async_client):
        """POST /api/coding-agent should invoke coding agent"""
        try:
            request = {"task": "Refactor this code", "code": "x=1\ny=2\nprint(x+y)"}
            response = await async_client.post("/api/coding-agent/execute", json=request)
            assert response.status_code in [200, 201, 404, 422]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")


class TestMetricsEndpoints:
    """Test metrics and monitoring endpoints"""

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, async_client):
        """GET /metrics should return Prometheus metrics"""
        try:
            response = await async_client.get("/metrics")
            assert response.status_code in [200, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_db_metrics(self, async_client):
        """GET /api/db/metrics should return database metrics"""
        try:
            response = await async_client.get("/api/db/metrics")
            assert response.status_code in [200, 404]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")


class TestErrorHandling:
    """Test API error handling"""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, async_client):
        """Invalid JSON should return 400"""
        try:
            response = await async_client.post(
                "/api/echo/chat",
                content="not valid json",
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code in [400, 422]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, async_client):
        """Non-existent endpoint should return 404"""
        try:
            response = await async_client.get("/api/echo/nonexistent_endpoint_xyz")
            assert response.status_code == 404
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, async_client):
        """Wrong HTTP method should return 405"""
        try:
            response = await async_client.delete("/api/echo/health")
            assert response.status_code in [404, 405]
        except httpx.ConnectError:
            pytest.skip("Echo Brain not running")
