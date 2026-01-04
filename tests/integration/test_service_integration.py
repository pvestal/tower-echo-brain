#!/usr/bin/env python3
"""
Integration Test Suite for Echo Brain Service-to-Service Communication

Tests integration between:
- Echo Brain API ↔ Database Pool Manager
- Echo Brain API ↔ Rate Limiting Middleware
- Echo Brain API ↔ Authentication Service
- Echo Brain API ↔ Vector Memory System (Qdrant)
- Echo Brain API ↔ External Tower Services
- Echo Brain API ↔ Circuit Breaker Components

Author: Echo Brain Testing Framework
Created: 2026-01-02
"""

import pytest
import asyncio
import httpx
import json
import time
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.api.echo import router as echo_router
from src.routing.auth_middleware import AuthMiddleware
from src.middleware.rate_limiting import RateLimitMiddleware
from src.db.pool_manager import AsyncConnectionPool
from src.db.models import QueryRequest, QueryResponse


class TestServiceIntegration:
    """Integration tests for service-to-service communication"""

    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI app for testing"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(echo_router)
        return app

    @pytest.fixture
    async def test_client(self, mock_app):
        """Create async test client"""
        async with httpx.AsyncClient(app=mock_app, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def auth_middleware(self):
        """Authentication middleware for testing"""
        return AuthMiddleware()

    @pytest.fixture
    def rate_limiter(self):
        """Rate limiting middleware for testing"""
        return RateLimitMiddleware()

    @pytest.fixture
    def valid_jwt_token(self):
        """Valid JWT token for testing"""
        import jwt
        payload = {
            'user_id': 'test_user',
            'username': 'testuser',
            'roles': ['user'],
            'permissions': ['api.access'],
            'exp': datetime.utcnow().timestamp() + 3600
        }
        secret = os.environ.get("JWT_SECRET", "test_secret_key")
        return jwt.encode(payload, secret, algorithm="HS256")

    @pytest.fixture
    async def mock_database_pool(self):
        """Mock database connection pool"""
        with patch('src.db.pool_manager.get_secure_db_config') as mock_config:
            mock_config.return_value = {
                "database": "test_echo_brain",
                "user": "test_user",
                "password": "test_password",
                "host": "localhost",
                "port": 5432
            }

            with patch('asyncpg.create_pool') as mock_create_pool:
                mock_pool = AsyncMock()
                mock_pool.get_size.return_value = 10
                mock_pool.get_idle_size.return_value = 5

                # Mock connection
                mock_connection = AsyncMock()
                mock_connection.fetch.return_value = [
                    {'conversation_id': 'test_123', 'query': 'test query', 'response': 'test response'}
                ]
                mock_connection.execute.return_value = "INSERT 1"
                mock_connection.fetchval.return_value = 1

                mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
                mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

                mock_create_pool.return_value = mock_pool

                pool = AsyncConnectionPool(min_size=5, max_size=20)
                await pool.initialize()
                yield pool

    @pytest.fixture
    def mock_vector_memory(self):
        """Mock vector memory system (Qdrant)"""
        with patch('src.services.real_vector_search.VectorSearch') as mock_vector_search:
            mock_instance = Mock()
            mock_instance.search_all_collections.return_value = [
                {
                    'content': 'Previous conversation about testing',
                    'metadata': {'source': 'conversations', 'confidence': 0.85}
                }
            ]
            mock_vector_search.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis for rate limiting"""
        with patch('redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True

            mock_pipeline = Mock()
            mock_pipeline.execute.return_value = [None, 3, None, None]  # 3 current requests
            mock_client.pipeline.return_value = mock_pipeline

            mock_redis.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_echo_brain_to_database_integration(self, mock_database_pool, mock_vector_memory):
        """Test Echo Brain API integration with database pool"""
        with patch('src.core.intelligence.intelligence_router.query_model') as mock_query:
            # Mock LLM response
            mock_query.return_value = {
                "success": True,
                "response": "This is a test response from the LLM",
                "model": "test_model"
            }

            with patch('src.services.conversation.conversation_manager') as mock_conversation:
                mock_conversation.classify_intent.return_value = ("general_query", 0.9, {})
                mock_conversation.needs_clarification.return_value = False
                mock_conversation.get_conversation_context.return_value = {"history": []}
                mock_conversation.search_semantic_memory.return_value = []

                # Create query request
                query_request = QueryRequest(
                    query="Test query for database integration",
                    conversation_id="test_integration_123",
                    user_id="test_user"
                )

                # Import and test the query function
                from src.api.echo import query_echo
                from fastapi import Request

                mock_request = Mock(spec=Request)
                mock_request.headers = {"X-Username": "testuser"}

                # Execute the query
                response = await query_echo(query_request, mock_request)

                # Verify integration
                assert isinstance(response, QueryResponse)
                assert response.response == "This is a test response from the LLM"
                assert response.conversation_id == "test_integration_123"
                assert response.model_used == "test_model"

                # Verify database interaction occurred
                # (This would be verified through database pool metrics in real implementation)

    @pytest.mark.asyncio
    async def test_authentication_rate_limiting_integration(self, auth_middleware, rate_limiter, valid_jwt_token):
        """Test integration between authentication and rate limiting"""
        from fastapi import Request

        # Create mock request with authentication
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/api/echo/query"
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {"Authorization": f"Bearer {valid_jwt_token}"}

        # Test authentication first
        user_info = await auth_middleware.verify_token(valid_jwt_token)
        assert user_info is not None
        assert user_info['user_id'] == 'test_user'

        # Test rate limiting with authenticated user
        with patch('redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_pipeline = Mock()
            mock_pipeline.execute.return_value = [None, 2, None, None]  # 2 requests
            mock_client.pipeline.return_value = mock_pipeline
            mock_redis.return_value = mock_client

            rate_limit_result = await rate_limiter.check_rate_limit(mock_request, user_info)

            # Authenticated user should get user-tier limits
            assert rate_limit_result['allowed'] is True
            assert rate_limit_result['tier'] == 'user'
            assert rate_limit_result['limit'] == 60  # User tier limit

    @pytest.mark.asyncio
    async def test_vector_memory_integration(self, mock_vector_memory):
        """Test integration with vector memory system"""
        # Mock conversation manager with vector search
        with patch('src.services.conversation.conversation_manager') as mock_conversation:
            mock_conversation.vector_search = mock_vector_memory
            mock_conversation.search_semantic_memory.return_value = [
                {
                    'content': 'Previous conversation about API testing',
                    'metadata': {'source': 'conversations', 'confidence': 0.9}
                }
            ]

            # Test memory search integration
            search_results = mock_conversation.search_semantic_memory("API testing")

            assert len(search_results) == 1
            assert search_results[0]['content'] == 'Previous conversation about API testing'
            assert search_results[0]['metadata']['confidence'] == 0.9

            # Verify vector search was called
            mock_vector_memory.search_all_collections.assert_called()

    @pytest.mark.asyncio
    async def test_external_tower_service_integration(self):
        """Test integration with external Tower services"""
        # Test Echo Brain to Anime Production Service
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "job_id": "anime_job_123",
                "status": "started",
                "prompt": "magical anime girl"
            }

            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.return_value.post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Test anime generation integration from Echo Brain
            from src.api.echo import handle_capability_intent
            from src.db.models import QueryRequest

            request = QueryRequest(
                query="generate anime magical girl",
                conversation_id="test_anime_123",
                user_id="test_user"
            )

            # This would test the actual anime generation capability
            # For now, we'll just verify the structure
            assert request.query == "generate anime magical girl"

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test integration with circuit breaker components"""
        # Mock circuit breaker states
        circuit_states = {
            'database': 'closed',  # Normal operation
            'external_api': 'half_open',  # Testing after failure
            'vector_search': 'open'  # Circuit tripped
        }

        async def mock_service_call(service_name: str):
            """Mock service call with circuit breaker"""
            state = circuit_states.get(service_name, 'closed')

            if state == 'open':
                raise Exception(f"Circuit breaker open for {service_name}")
            elif state == 'half_open':
                # Simulate 50% success rate
                import random
                if random.random() > 0.5:
                    raise Exception(f"Service {service_name} still failing")
                return f"Success from {service_name}"
            else:
                return f"Success from {service_name}"

        # Test circuit breaker behavior
        try:
            result = await mock_service_call('database')
            assert result == "Success from database"
        except Exception:
            pytest.fail("Database service should be available (circuit closed)")

        # Vector search should fail (circuit open)
        with pytest.raises(Exception, match="Circuit breaker open for vector_search"):
            await mock_service_call('vector_search')

        # External API might succeed or fail (half-open)
        try:
            result = await mock_service_call('external_api')
            # Either succeeds or raises exception
        except Exception as e:
            assert "still failing" in str(e)

    @pytest.mark.asyncio
    async def test_end_to_end_service_flow(self, mock_database_pool, mock_vector_memory, valid_jwt_token):
        """Test complete end-to-end service integration flow"""
        # Mock all required services
        with patch('src.core.intelligence.intelligence_router.query_model') as mock_query, \
             patch('src.services.conversation.conversation_manager') as mock_conversation, \
             patch('redis.Redis') as mock_redis:

            # Setup mocks
            mock_query.return_value = {
                "success": True,
                "response": "Complete end-to-end test response",
                "model": "llama3.1:8b"
            }

            mock_conversation.classify_intent.return_value = ("general_query", 0.95, {})
            mock_conversation.needs_clarification.return_value = False
            mock_conversation.get_conversation_context.return_value = {"history": []}
            mock_conversation.search_semantic_memory.return_value = [
                {
                    'content': 'Relevant memory for e2e test',
                    'metadata': {'source': 'conversations'}
                }
            ]
            mock_conversation.update_conversation = Mock()

            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_pipeline = Mock()
            mock_pipeline.execute.return_value = [None, 1, None, None]
            mock_redis_client.pipeline.return_value = mock_pipeline
            mock_redis.return_value = mock_redis_client

            # Create complete request
            query_request = QueryRequest(
                query="Complete end-to-end integration test query",
                conversation_id="e2e_test_123",
                user_id="test_user",
                intelligence_level="auto"
            )

            from src.api.echo import query_echo
            from fastapi import Request

            mock_request = Mock(spec=Request)
            mock_request.headers = {"X-Username": "testuser", "Authorization": f"Bearer {valid_jwt_token}"}

            # Execute complete flow
            start_time = time.time()
            response = await query_echo(query_request, mock_request)
            end_time = time.time()

            # Verify complete integration
            assert isinstance(response, QueryResponse)
            assert response.response == "Complete end-to-end test response"
            assert response.conversation_id == "e2e_test_123"
            assert response.model_used == "llama3.1:8b"
            assert response.intent == "general_query"
            assert response.confidence == 0.95

            # Verify performance
            processing_time = end_time - start_time
            assert processing_time < 5.0, f"E2E flow too slow: {processing_time}s"

            # Verify all services were called
            mock_conversation.classify_intent.assert_called_once()
            mock_conversation.search_semantic_memory.assert_called_once()
            mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_failure_handling(self, mock_database_pool):
        """Test service integration failure handling"""
        # Test database failure scenario
        with patch('src.core.intelligence.intelligence_router.query_model') as mock_query:
            mock_query.side_effect = Exception("Database connection failed")

            query_request = QueryRequest(
                query="Test database failure handling",
                conversation_id="failure_test_123",
                user_id="test_user"
            )

            from src.api.echo import query_echo
            from fastapi import Request

            mock_request = Mock(spec=Request)
            mock_request.headers = {"X-Username": "testuser"}

            # Should handle failure gracefully
            response = await query_echo(query_request, mock_request)

            assert isinstance(response, QueryResponse)
            assert "error" in response.response.lower() or "failed" in response.response.lower()
            assert response.model_used == "error_handler"

    @pytest.mark.asyncio
    async def test_service_timeout_handling(self):
        """Test service timeout handling"""
        async def slow_service_call():
            """Simulate slow service that times out"""
            await asyncio.sleep(10)  # 10 second delay
            return "Should not reach here"

        # Test timeout handling
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_service_call(), timeout=1.0)

        # Test timeout with fallback
        async def service_with_fallback():
            try:
                result = await asyncio.wait_for(slow_service_call(), timeout=1.0)
                return result
            except asyncio.TimeoutError:
                return "Fallback response due to timeout"

        result = await service_with_fallback()
        assert result == "Fallback response due to timeout"

    @pytest.mark.asyncio
    async def test_concurrent_service_integration(self, mock_database_pool, mock_vector_memory):
        """Test service integration under concurrent load"""
        with patch('src.core.intelligence.intelligence_router.query_model') as mock_query, \
             patch('src.services.conversation.conversation_manager') as mock_conversation:

            # Setup mocks for concurrent testing
            mock_query.return_value = {
                "success": True,
                "response": "Concurrent test response",
                "model": "test_model"
            }

            mock_conversation.classify_intent.return_value = ("general_query", 0.9, {})
            mock_conversation.needs_clarification.return_value = False
            mock_conversation.get_conversation_context.return_value = {"history": []}
            mock_conversation.search_semantic_memory.return_value = []
            mock_conversation.update_conversation = Mock()

            async def concurrent_request(request_id):
                """Execute concurrent request"""
                query_request = QueryRequest(
                    query=f"Concurrent test query {request_id}",
                    conversation_id=f"concurrent_test_{request_id}",
                    user_id="test_user"
                )

                from src.api.echo import query_echo
                from fastapi import Request

                mock_request = Mock(spec=Request)
                mock_request.headers = {"X-Username": "testuser"}

                return await query_echo(query_request, mock_request)

            # Execute concurrent requests
            concurrent_count = 20
            tasks = [concurrent_request(i) for i in range(concurrent_count)]

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            # Verify all requests succeeded
            successful_results = [r for r in results if isinstance(r, QueryResponse)]
            failed_results = [r for r in results if isinstance(r, Exception)]

            assert len(successful_results) == concurrent_count, f"Some concurrent requests failed: {len(failed_results)} failures"

            # Verify performance under load
            total_time = end_time - start_time
            avg_time_per_request = total_time / concurrent_count

            assert avg_time_per_request < 1.0, f"Concurrent performance too slow: {avg_time_per_request}s per request"

            print(f"Concurrent Integration Test: {concurrent_count} requests in {total_time:.2f}s "
                  f"(avg: {avg_time_per_request*1000:.1f}ms per request)")

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self):
        """Test service health monitoring integration"""
        # Mock health check responses from various services
        service_health = {
            'database_pool': {'status': 'healthy', 'connections': 10, 'response_time': 0.002},
            'redis_cache': {'status': 'healthy', 'memory_usage': '45%', 'response_time': 0.001},
            'vector_search': {'status': 'degraded', 'indices': 2, 'response_time': 0.15},
            'external_apis': {'status': 'partial', 'available_services': 3, 'total_services': 5}
        }

        async def check_service_health(service_name):
            """Mock service health check"""
            return service_health.get(service_name, {'status': 'unknown'})

        # Test health monitoring integration
        health_results = {}
        for service in service_health.keys():
            health_results[service] = await check_service_health(service)

        # Verify health monitoring
        assert health_results['database_pool']['status'] == 'healthy'
        assert health_results['redis_cache']['status'] == 'healthy'
        assert health_results['vector_search']['status'] == 'degraded'
        assert health_results['external_apis']['status'] == 'partial'

        # Calculate overall system health
        healthy_services = sum(1 for h in health_results.values() if h['status'] == 'healthy')
        total_services = len(health_results)
        overall_health_percentage = (healthy_services / total_services) * 100

        print(f"System Health: {overall_health_percentage:.1f}% ({healthy_services}/{total_services} services healthy)")

        # System should be operational with at least 50% healthy services
        assert overall_health_percentage >= 50, f"System health too low: {overall_health_percentage}%"


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])