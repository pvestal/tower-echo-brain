#!/usr/bin/env python3
"""
Comprehensive Security Middleware for Echo Brain API
Integrates authentication, rate limiting, input validation, and audit logging
"""

import time
import logging
import json
from typing import Dict, Any, Optional, Callable, List
from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.middleware.auth_middleware import (
    get_current_user_optional,
    auth_middleware,
    security
)
from src.middleware.rate_limiting import rate_limiter
from src.api.schemas import ErrorResponse

logger = logging.getLogger(__name__)

# Security configuration
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

# Endpoints that require authentication
PROTECTED_ENDPOINTS = [
    "/api/echo/query",
    "/api/echo/chat",
    "/api/echo/models/manage",
    "/api/echo/models/pull",
    "/api/echo/models/delete",
    "/code/",
    "/terminal/",
    "/files/analyze",
    "/anime/generate"
]

# Endpoints that require admin privileges
ADMIN_ENDPOINTS = [
    "/api/echo/models/pull",
    "/api/echo/models/delete",
    "/git/",
    "/api/echo/system/",
    "/api/echo/config/",
    "/terminal/execute"
]

# Endpoints exempt from rate limiting (system health checks)
RATE_LIMIT_EXEMPT = [
    "/health",
    "/api/echo/health",
    "/api/echo/status",
    "/api/echo/metrics/system"
]

class SecurityValidationError(Exception):
    """Security validation error"""
    pass

class SecurityAuditLogger:
    """Security audit logging"""

    @staticmethod
    def log_security_event(
        event_type: str,
        request: Request,
        user_data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        details: str = "",
        risk_level: str = "low"
    ):
        """Log security event with comprehensive context"""

        client_ip = SecurityMiddleware._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")[:200]
        endpoint = request.url.path
        method = request.method

        user_id = user_data.get('user', 'anonymous') if user_data else 'anonymous'
        user_role = user_data.get('role', 'anonymous') if user_data else 'anonymous'

        log_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "user_id": user_id,
            "user_role": user_role,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "endpoint": endpoint,
            "method": method,
            "success": success,
            "risk_level": risk_level,
            "details": details
        }

        log_level = {
            "low": logger.info,
            "medium": logger.warning,
            "high": logger.error,
            "critical": logger.critical
        }.get(risk_level, logger.info)

        if success:
            log_level(f"🔐 SECURITY EVENT [{risk_level.upper()}]: {event_type} - {user_id} from {client_ip} to {endpoint}")
        else:
            log_level(f"🚨 SECURITY VIOLATION [{risk_level.upper()}]: {event_type} - {user_id} from {client_ip} to {endpoint} - {details}")

class RequestValidator:
    """Request validation and sanitization"""

    @staticmethod
    def validate_request_size(request: Request, max_size: int = 1024 * 1024) -> bool:
        """Validate request body size"""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise SecurityValidationError(f"Request body too large: {content_length} bytes")
        return True

    @staticmethod
    def validate_headers(request: Request) -> bool:
        """Validate request headers for security"""
        # Check for suspicious headers
        suspicious_headers = [
            "x-forwarded-host",
            "x-cluster-client-ip",
            "x-original-url",
            "x-rewrite-url"
        ]

        for header in suspicious_headers:
            if header in request.headers:
                logger.warning(f"Suspicious header detected: {header}")

        # Validate User-Agent
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 500:
            raise SecurityValidationError("User-Agent header too long")

        return True

    @staticmethod
    def validate_query_parameters(request: Request) -> bool:
        """Validate query parameters"""
        query_params = str(request.query_params)

        # Check for SQL injection patterns
        sql_patterns = [
            "union select", "drop table", "insert into", "delete from",
            "update set", "create table", "alter table", "--", "/*", "*/"
        ]

        query_lower = query_params.lower()
        for pattern in sql_patterns:
            if pattern in query_lower:
                raise SecurityValidationError(f"Potential SQL injection detected: {pattern}")

        # Check for XSS patterns
        xss_patterns = [
            "<script", "javascript:", "onload=", "onerror=",
            "onclick=", "onmouseover=", "eval(", "alert("
        ]

        for pattern in xss_patterns:
            if pattern in query_lower:
                raise SecurityValidationError(f"Potential XSS attack detected: {pattern}")

        return True

class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware"""

    def __init__(self, app, enforce_https: bool = True, audit_all_requests: bool = False):
        super().__init__(app)
        self.enforce_https = enforce_https
        self.audit_all_requests = audit_all_requests
        self.validator = RequestValidator()
        self.audit_logger = SecurityAuditLogger()

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP from request"""
        # Check for proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        if hasattr(request.client, 'host'):
            return request.client.host

        return "unknown"

    def _is_protected_endpoint(self, path: str) -> bool:
        """Check if endpoint requires authentication"""
        return any(path.startswith(protected) for protected in PROTECTED_ENDPOINTS)

    def _is_admin_endpoint(self, path: str) -> bool:
        """Check if endpoint requires admin privileges"""
        return any(path.startswith(admin) for admin in ADMIN_ENDPOINTS)

    def _is_rate_limit_exempt(self, path: str) -> bool:
        """Check if endpoint is exempt from rate limiting"""
        return any(path.startswith(exempt) for exempt in RATE_LIMIT_EXEMPT)

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response"""
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value

    async def _handle_security_error(self, request: Request, error: Exception, error_type: str) -> JSONResponse:
        """Handle security errors consistently"""
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        # Log security violation
        self.audit_logger.log_security_event(
            event_type=error_type,
            request=request,
            success=False,
            details=str(error),
            risk_level="high"
        )

        # Determine appropriate error response
        if isinstance(error, HTTPException):
            status_code = error.status_code
            detail = error.detail
        elif isinstance(error, SecurityValidationError):
            status_code = status.HTTP_400_BAD_REQUEST
            detail = str(error)
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            detail = "Internal security error"

        error_response = ErrorResponse(
            error=error_type,
            message=detail,
            timestamp=time.time()
        )

        response = JSONResponse(
            status_code=status_code,
            content=error_response.dict()
        )

        self._add_security_headers(response)
        return response

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main security middleware dispatch"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        method = request.method

        try:
            # 1. HTTPS enforcement (skip for local development)
            if self.enforce_https and not request.url.scheme == "https" and not client_ip.startswith("192.168."):
                raise SecurityValidationError("HTTPS required")

            # 2. Request validation
            self.validator.validate_request_size(request)
            self.validator.validate_headers(request)
            self.validator.validate_query_parameters(request)

            # 3. Authentication check
            user_data = None
            if self._is_protected_endpoint(endpoint):
                # Extract credentials for protected endpoints
                auth_header = request.headers.get("authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    return await self._handle_security_error(
                        request,
                        HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing or invalid authentication token",
                            headers={"WWW-Authenticate": "Bearer"}
                        ),
                        "authentication_missing"
                    )

                try:
                    # Verify token
                    token = auth_header.split(" ")[1]
                    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
                    user_data = await auth_middleware.authenticate_user(request, credentials)
                except Exception as e:
                    return await self._handle_security_error(request, e, "authentication_failed")

                # Check admin privileges for admin endpoints
                if self._is_admin_endpoint(endpoint):
                    user_role = user_data.get('role', 'user')
                    if user_role not in ['admin', 'patrick']:
                        return await self._handle_security_error(
                            request,
                            HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="Admin privileges required"
                            ),
                            "authorization_failed"
                        )

            # 4. Rate limiting (skip for exempt endpoints)
            if not self._is_rate_limit_exempt(endpoint):
                try:
                    rate_limit_info = await rate_limiter.check_rate_limit(request, user_data)
                except Exception as e:
                    return await self._handle_security_error(request, e, "rate_limit_exceeded")

            # 5. Audit logging for all requests if enabled
            if self.audit_all_requests:
                self.audit_logger.log_security_event(
                    event_type="request_processed",
                    request=request,
                    user_data=user_data,
                    success=True,
                    details=f"{method} {endpoint}",
                    risk_level="low"
                )

            # 6. Process the request
            response = await call_next(request)

            # 7. Add security headers
            self._add_security_headers(response)

            # 8. Add rate limit headers if applicable
            if not self._is_rate_limit_exempt(endpoint) and 'rate_limit_info' in locals():
                response.headers["X-RateLimit-Limit"] = str(rate_limit_info['limit'])
                response.headers["X-RateLimit-Remaining"] = str(rate_limit_info['remaining'])
                response.headers["X-RateLimit-Reset"] = str(rate_limit_info['reset_time'])
                response.headers["X-RateLimit-Tier"] = rate_limit_info['tier']

            # 9. Log successful request processing
            processing_time = time.time() - start_time
            if processing_time > 5.0:  # Log slow requests
                self.audit_logger.log_security_event(
                    event_type="slow_request",
                    request=request,
                    user_data=user_data,
                    success=True,
                    details=f"Processing time: {processing_time:.2f}s",
                    risk_level="medium"
                )

            return response

        except SecurityValidationError as e:
            return await self._handle_security_error(request, e, "validation_error")
        except HTTPException as e:
            return await self._handle_security_error(request, e, "http_error")
        except Exception as e:
            logger.exception(f"Unexpected security middleware error: {e}")
            return await self._handle_security_error(request, e, "unexpected_error")

# Dependency functions for manual security application
async def apply_authentication(request: Request) -> Dict[str, Any]:
    """Apply authentication dependency"""
    return await get_current_user_optional(request)

async def require_authentication(request: Request) -> Dict[str, Any]:
    """Require authentication dependency"""
    user_data = await get_current_user_optional(request)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user_data

async def require_admin(request: Request) -> Dict[str, Any]:
    """Require admin privileges dependency"""
    user_data = await require_authentication(request)
    user_role = user_data.get('role', 'user')
    if user_role not in ['admin', 'patrick']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user_data

async def apply_rate_limiting(request: Request, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply rate limiting dependency"""
    return await rate_limiter.check_rate_limit(request, user_data)

def create_security_dependency(require_auth: bool = True, require_admin: bool = False):
    """Create a security dependency with specified requirements"""
    async def security_dependency(request: Request) -> Dict[str, Any]:
        user_data = None

        # Apply authentication if required
        if require_auth:
            user_data = await require_authentication(request)

            # Check admin privileges if required
            if require_admin:
                user_role = user_data.get('role', 'user')
                if user_role not in ['admin', 'patrick']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin privileges required"
                    )

        # Apply rate limiting
        await apply_rate_limiting(request, user_data)

        return user_data or {}

    return security_dependency

# Pre-configured security dependencies
RequireAuth = create_security_dependency(require_auth=True, require_admin=False)
RequireAdmin = create_security_dependency(require_auth=True, require_admin=True)
OptionalAuth = create_security_dependency(require_auth=False, require_admin=False)

# Global security status check
async def get_security_status() -> Dict[str, Any]:
    """Get comprehensive security system status"""
    from src.middleware.auth_middleware import get_auth_status
    from src.middleware.rate_limiting import get_rate_limit_status

    auth_status = await get_auth_status()
    rate_limit_status = get_rate_limit_status()

    return {
        "security_middleware_active": True,
        "https_enforcement": True,
        "request_validation": True,
        "security_headers": True,
        "audit_logging": True,
        "authentication": auth_status,
        "rate_limiting": rate_limit_status,
        "protected_endpoints": len(PROTECTED_ENDPOINTS),
        "admin_endpoints": len(ADMIN_ENDPOINTS),
        "rate_limit_exempt": len(RATE_LIMIT_EXEMPT)
    }