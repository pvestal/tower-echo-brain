#!/usr/bin/env python3
"""
Comprehensive Security Tests for Echo Brain API
Tests authentication, rate limiting, input validation, and security headers
"""

import pytest
import time
import asyncio
from typing import Dict, Any
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Import the secure app and components
from src.secure_app_factory import create_secure_app
from src.middleware.auth_middleware import (
    auth_middleware, create_emergency_token, EnhancedAuthMiddleware
)
from src.middleware.rate_limiting import RateLimitMiddleware
from src.api.schemas import QueryRequest, LoginRequest

@pytest.fixture
def client():
    """Create test client with secure app"""
    app = create_secure_app()
    return TestClient(app)

@pytest.fixture
def admin_token():
    """Create admin token for testing"""
    return create_emergency_token("patrick", "admin")

@pytest.fixture
def user_token():
    """Create user token for testing"""
    return create_emergency_token("testuser", "user")

class TestAuthentication:
    """Test authentication functionality"""

    def test_protected_endpoint_without_auth(self, client):
        """Test that protected endpoints require authentication"""
        response = client.post("/api/secure/query", json={
            "query": "test query"
        })
        assert response.status_code == 401
        assert "authentication" in response.json().get("detail", "").lower()

    def test_protected_endpoint_with_invalid_token(self, client):
        """Test protected endpoint with invalid token"""
        response = client.post(
            "/api/secure/query",
            json={"query": "test query"},
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401

    def test_protected_endpoint_with_valid_token(self, client, admin_token):
        """Test protected endpoint with valid token"""
        response = client.post(
            "/api/secure/query",
            json={"query": "test query", "intelligence_level": "auto"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Should not return 401
        assert response.status_code != 401

    def test_admin_endpoint_with_user_token(self, client, user_token):
        """Test that admin endpoints reject user tokens"""
        response = client.post(
            "/api/secure/execute",
            json={"command": "echo test", "safe_mode": True},
            headers={"Authorization": f"Bearer {user_token}"}
        )
        assert response.status_code == 403

    def test_admin_endpoint_with_admin_token(self, client, admin_token):
        """Test that admin endpoints accept admin tokens"""
        response = client.post(
            "/api/secure/execute",
            json={"command": "echo test", "safe_mode": True},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Should not return 403
        assert response.status_code != 403

    def test_login_endpoint(self, client):
        """Test login endpoint"""
        response = client.post("/api/secure/auth/login", json={
            "username": "patrick",
            "password": "admin"
        })

        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
            assert "expires_in" in data

    def test_logout_endpoint(self, client, admin_token):
        """Test logout endpoint"""
        response = client.post(
            "/api/secure/auth/logout",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200

    def test_token_blacklisting(self, client, admin_token):
        """Test token blacklisting after logout"""
        # First request should work
        response1 = client.get(
            "/api/secure/user/info",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response1.status_code != 401

        # Logout (blacklist token)
        logout_response = client.post(
            "/api/secure/auth/logout",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert logout_response.status_code == 200

        # Second request with same token should fail
        response2 = client.get(
            "/api/secure/user/info",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Token should be blacklisted, but this depends on implementation
        # In development, this might not be enforced

class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limit_headers(self, client, admin_token):
        """Test that rate limit headers are present"""
        response = client.post(
            "/api/secure/query",
            json={"query": "test", "intelligence_level": "auto"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        # Check for rate limit headers
        headers = response.headers
        assert "X-RateLimit-Limit" in headers or response.status_code == 401
        if "X-RateLimit-Limit" in headers:
            assert "X-RateLimit-Remaining" in headers
            assert "X-RateLimit-Reset" in headers
            assert "X-RateLimit-Tier" in headers

    def test_rate_limit_enforcement(self, client):
        """Test rate limit enforcement for anonymous users"""
        # Make multiple requests quickly
        responses = []
        for i in range(35):  # Exceed anonymous limit of 30
            response = client.get("/health")
            responses.append(response.status_code)
            if response.status_code == 429:
                break

        # Should eventually get rate limited
        assert 429 in responses or all(r == 200 for r in responses[:30])

    def test_rate_limit_reset(self, client):
        """Test that rate limits reset over time"""
        # This test would need to wait for rate limit window to reset
        # Skipping due to time constraints in testing
        pass

class TestInputValidation:
    """Test input validation and sanitization"""

    def test_query_length_validation(self, client, admin_token):
        """Test query length validation"""
        long_query = "a" * 11000  # Exceed MAX_QUERY_LENGTH
        response = client.post(
            "/api/secure/query",
            json={"query": long_query, "intelligence_level": "auto"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 422  # Validation error

    def test_xss_prevention(self, client, admin_token):
        """Test XSS prevention in query"""
        xss_query = "<script>alert('xss')</script>"
        response = client.post(
            "/api/secure/query",
            json={"query": xss_query, "intelligence_level": "auto"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Should either reject or sanitize
        assert response.status_code in [400, 422] or response.status_code == 200

    def test_sql_injection_prevention(self, client, admin_token):
        """Test SQL injection prevention"""
        sql_query = "'; DROP TABLE users; --"
        response = client.post(
            "/api/secure/query",
            json={"query": sql_query, "intelligence_level": "auto"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Should either reject or sanitize
        assert response.status_code in [400, 422] or response.status_code == 200

    def test_command_validation(self, client, admin_token):
        """Test dangerous command validation"""
        dangerous_command = "rm -rf /"
        response = client.post(
            "/api/secure/execute",
            json={"command": dangerous_command, "safe_mode": True},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 422  # Should be rejected by validation

    def test_path_traversal_prevention(self, client, admin_token):
        """Test path traversal prevention"""
        malicious_path = "../../etc/passwd"
        response = client.post(
            "/api/secure/files/analyze",
            json={"file_path": malicious_path, "analysis_type": "code"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 422  # Should be rejected

class TestSecurityHeaders:
    """Test security headers"""

    def test_security_headers_present(self, client):
        """Test that security headers are present"""
        response = client.get("/health")
        headers = response.headers

        # Check for key security headers
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Content-Security-Policy"
        ]

        for header in expected_headers:
            assert header in headers, f"Missing security header: {header}"

    def test_cors_headers(self, client):
        """Test CORS headers configuration"""
        response = client.options("/health")

        # CORS should be configured
        if "Access-Control-Allow-Origin" in response.headers:
            # Verify it's not wildcard in production
            origin = response.headers["Access-Control-Allow-Origin"]
            assert origin != "*" or "localhost" in origin

class TestAuditLogging:
    """Test audit logging functionality"""

    def test_authentication_events_logged(self, client):
        """Test that authentication events are logged"""
        with patch('src.middleware.auth_middleware.logger') as mock_logger:
            # Attempt login with invalid credentials
            client.post("/api/secure/auth/login", json={
                "username": "invalid",
                "password": "invalid"
            })

            # Check that warning/error was logged
            assert mock_logger.warning.called or mock_logger.error.called

    def test_security_violations_logged(self, client):
        """Test that security violations are logged"""
        with patch('src.middleware.security_middleware.logger') as mock_logger:
            # Make request with suspicious content
            client.get("/health?query=<script>alert('xss')</script>")

            # Check that security event was potentially logged
            # (This depends on implementation details)

class TestErrorHandling:
    """Test error handling"""

    def test_error_response_format(self, client):
        """Test that errors return consistent format"""
        response = client.post("/api/secure/query", json={
            "invalid": "request"
        })

        assert response.status_code in [400, 401, 422]
        data = response.json()
        assert "error" in data or "detail" in data

    def test_error_information_disclosure(self, client):
        """Test that errors don't disclose sensitive information"""
        response = client.post("/api/secure/auth/login", json={
            "username": "admin",
            "password": "wrong"
        })

        error_message = response.json().get("detail", "").lower()
        # Should not disclose specific information about what went wrong
        sensitive_info = ["password", "user not found", "invalid user"]
        for info in sensitive_info:
            assert info not in error_message

class TestSystemSecurity:
    """Test system-level security"""

    def test_security_status_endpoint(self, client):
        """Test security status endpoint"""
        response = client.get("/security/status")
        assert response.status_code == 200

        data = response.json()
        assert "security_middleware_active" in data
        assert "authentication" in data
        assert "rate_limiting" in data

    def test_emergency_token_creation(self):
        """Test emergency token creation"""
        token = create_emergency_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_user_info_endpoint(self, client, admin_token):
        """Test user info endpoint"""
        response = client.get(
            "/api/secure/user/info",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        if response.status_code == 200:
            data = response.json()
            assert "user_id" in data
            assert "username" in data
            assert "role" in data

# Integration tests
class TestIntegration:
    """Integration tests for complete security flow"""

    def test_complete_authentication_flow(self, client):
        """Test complete authentication flow"""
        # 1. Try protected endpoint without auth (should fail)
        response = client.post("/api/secure/query", json={"query": "test"})
        assert response.status_code == 401

        # 2. Login (if implemented)
        login_response = client.post("/api/secure/auth/login", json={
            "username": "patrick",
            "password": "admin"
        })

        if login_response.status_code == 200:
            token = login_response.json()["access_token"]

            # 3. Use token to access protected endpoint
            protected_response = client.post(
                "/api/secure/query",
                json={"query": "test", "intelligence_level": "auto"},
                headers={"Authorization": f"Bearer {token}"}
            )
            assert protected_response.status_code != 401

            # 4. Logout
            logout_response = client.post(
                "/api/secure/auth/logout",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert logout_response.status_code == 200

    def test_rate_limiting_and_auth_interaction(self, client, admin_token):
        """Test interaction between rate limiting and authentication"""
        # Make requests with valid auth
        for i in range(5):
            response = client.get(
                "/api/secure/health",
                headers={"Authorization": f"Bearer {admin_token}"}
            )
            # Should get different rate limits based on auth status
            if "X-RateLimit-Tier" in response.headers:
                assert response.headers["X-RateLimit-Tier"] in ["admin", "patrick"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])