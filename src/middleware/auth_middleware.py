#!/usr/bin/env python3
"""
Enhanced JWT Authentication Middleware for Echo Brain API Security
Provides comprehensive authentication, session management, and audit logging
"""

import jwt
import time
import logging
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import os
import redis
import json

logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', '***REMOVED***')
AUTH_SERVICE_URL = os.getenv('AUTH_SERVICE_URL', "https://***REMOVED***/api/auth")
REDIS_HOST = os.getenv('REDIS_HOST', '***REMOVED***')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_AUTH_DB', '1'))

# Security configuration
TOKEN_EXPIRY_SECONDS = 30 * 60  # 30 minutes
REFRESH_TOKEN_EXPIRY_SECONDS = 7 * 24 * 60 * 60  # 7 days
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 15 * 60  # 15 minutes

# Create security scheme
security = HTTPBearer(auto_error=False)

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class TokenBlacklistError(Exception):
    """Token is blacklisted"""
    pass

class SessionManager:
    """Manages user sessions and token blacklisting"""

    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis session manager connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, using memory cache: {e}")
            self.redis_available = False
            self.memory_cache = {}

    def _get_key(self, key_type: str, identifier: str) -> str:
        """Generate Redis key"""
        return f"echo_auth:{key_type}:{identifier}"

    def store_session(self, user_id: str, session_data: Dict[str, Any], expiry_seconds: int = TOKEN_EXPIRY_SECONDS):
        """Store user session data"""
        try:
            if self.redis_available:
                key = self._get_key("session", user_id)
                self.redis_client.setex(key, expiry_seconds, json.dumps(session_data))
            else:
                self.memory_cache[f"session:{user_id}"] = {
                    'data': session_data,
                    'expires': time.time() + expiry_seconds
                }
        except Exception as e:
            logger.error(f"Failed to store session: {e}")

    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user session data"""
        try:
            if self.redis_available:
                key = self._get_key("session", user_id)
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            else:
                session = self.memory_cache.get(f"session:{user_id}")
                if session and session['expires'] > time.time():
                    return session['data']
                return None
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    def blacklist_token(self, token_jti: str, expiry_seconds: int = TOKEN_EXPIRY_SECONDS):
        """Add token to blacklist"""
        try:
            if self.redis_available:
                key = self._get_key("blacklist", token_jti)
                self.redis_client.setex(key, expiry_seconds, "blacklisted")
            else:
                self.memory_cache[f"blacklist:{token_jti}"] = {
                    'blacklisted': True,
                    'expires': time.time() + expiry_seconds
                }
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")

    def is_token_blacklisted(self, token_jti: str) -> bool:
        """Check if token is blacklisted"""
        try:
            if self.redis_available:
                key = self._get_key("blacklist", token_jti)
                return self.redis_client.exists(key) > 0
            else:
                blacklist_entry = self.memory_cache.get(f"blacklist:{token_jti}")
                if blacklist_entry and blacklist_entry['expires'] > time.time():
                    return blacklist_entry.get('blacklisted', False)
                return False
        except Exception as e:
            logger.error(f"Failed to check token blacklist: {e}")
            return False

    def track_failed_attempt(self, ip_address: str):
        """Track failed authentication attempt"""
        try:
            if self.redis_available:
                key = self._get_key("failed_attempts", ip_address)
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, LOCKOUT_DURATION_SECONDS)
                pipe.execute()
            else:
                current_time = time.time()
                attempt_key = f"failed:{ip_address}"
                if attempt_key not in self.memory_cache:
                    self.memory_cache[attempt_key] = {'count': 0, 'expires': current_time + LOCKOUT_DURATION_SECONDS}
                self.memory_cache[attempt_key]['count'] += 1
        except Exception as e:
            logger.error(f"Failed to track failed attempt: {e}")

    def is_ip_locked_out(self, ip_address: str) -> bool:
        """Check if IP is locked out due to failed attempts"""
        try:
            if self.redis_available:
                key = self._get_key("failed_attempts", ip_address)
                attempts = self.redis_client.get(key)
                return int(attempts or 0) >= MAX_FAILED_ATTEMPTS
            else:
                attempt_data = self.memory_cache.get(f"failed:{ip_address}")
                if attempt_data and attempt_data['expires'] > time.time():
                    return attempt_data['count'] >= MAX_FAILED_ATTEMPTS
                return False
        except Exception as e:
            logger.error(f"Failed to check IP lockout: {e}")
            return False

    def clear_failed_attempts(self, ip_address: str):
        """Clear failed attempts for IP"""
        try:
            if self.redis_available:
                key = self._get_key("failed_attempts", ip_address)
                self.redis_client.delete(key)
            else:
                self.memory_cache.pop(f"failed:{ip_address}", None)
        except Exception as e:
            logger.error(f"Failed to clear failed attempts: {e}")

class AuditLogger:
    """Audit logging for authentication events"""

    @staticmethod
    def log_auth_event(event_type: str, user_id: str, ip_address: str, user_agent: str, success: bool, details: str = ""):
        """Log authentication event for security audit"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent[:200],  # Limit length
            "success": success,
            "details": details
        }

        if success:
            logger.info(f"🔐 AUTH SUCCESS: {event_type} - {user_id} from {ip_address}")
        else:
            logger.warning(f"🚨 AUTH FAILURE: {event_type} - {user_id} from {ip_address} - {details}")

class EnhancedAuthMiddleware:
    """Enhanced authentication middleware with comprehensive security features"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.audit_logger = AuditLogger()
        self.auth_service_url = AUTH_SERVICE_URL
        self.jwt_secret = JWT_SECRET_KEY

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for proxy headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        if hasattr(request.client, 'host'):
            return request.client.host

        return "unknown"

    def _generate_token_jti(self, payload: Dict[str, Any]) -> str:
        """Generate unique token identifier"""
        jti_data = f"{payload.get('user', '')}{payload.get('iat', 0)}{JWT_SECRET_KEY}"
        return hashlib.sha256(jti_data.encode()).hexdigest()[:16]

    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token with enhanced security"""
        now = datetime.utcnow()
        payload = {
            'user': user_data.get('username', ''),
            'user_id': user_data.get('user_id', ''),
            'role': user_data.get('role', 'user'),
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(seconds=TOKEN_EXPIRY_SECONDS)).timestamp()),
            'iss': 'echo-brain-auth',
            'aud': 'echo-brain-api'
        }

        # Add unique token identifier
        payload['jti'] = self._generate_token_jti(payload)

        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token"""
        now = datetime.utcnow()
        payload = {
            'user_id': user_id,
            'type': 'refresh',
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(seconds=REFRESH_TOKEN_EXPIRY_SECONDS)).timestamp()),
            'iss': 'echo-brain-auth',
            'aud': 'echo-brain-refresh'
        }

        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    async def verify_token_with_auth_service(self, token: str) -> Dict[str, Any]:
        """Verify token with Tower auth service"""
        try:
            async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
                response = await client.post(
                    f"{self.auth_service_url}/verify",
                    headers={"Authorization": f"Bearer {token}"},
                )

                if response.status_code == 200:
                    user_info = response.json()
                    logger.debug(f"✅ Token verified with auth service for user: {user_info.get('user', 'unknown')}")
                    return user_info
                else:
                    logger.warning(f"❌ Auth service rejected token: {response.status_code}")
                    raise AuthenticationError("Token verification failed with auth service")

        except httpx.TimeoutException:
            logger.warning("⚠️ Auth service timeout, falling back to local verification")
            raise AuthenticationError("Auth service timeout")
        except Exception as e:
            logger.warning(f"⚠️ Auth service error: {e}, falling back to local verification")
            raise AuthenticationError(f"Auth service error: {e}")

    def verify_token_locally(self, token: str) -> Dict[str, Any]:
        """Local JWT token verification with enhanced security checks"""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                audience="echo-brain-api",
                issuer="echo-brain-auth"
            )

            # Check token blacklist
            token_jti = payload.get('jti')
            if token_jti and self.session_manager.is_token_blacklisted(token_jti):
                raise TokenBlacklistError("Token has been revoked")

            # Check expiration (already done by jwt.decode, but explicit check)
            current_time = datetime.utcnow().timestamp()
            if payload.get('exp', 0) <= current_time:
                raise AuthenticationError("Token expired")

            # Validate required fields
            required_fields = ['user', 'user_id', 'role', 'iat', 'exp']
            for field in required_fields:
                if field not in payload:
                    raise AuthenticationError(f"Token missing required field: {field}")

            logger.debug(f"✅ Local token verified for user: {payload.get('user', 'unknown')}")
            return payload

        except TokenBlacklistError:
            logger.error("❌ Token is blacklisted")
            raise AuthenticationError("Token has been revoked")
        except jwt.ExpiredSignatureError:
            logger.error("❌ Token expired")
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"❌ Invalid JWT token: {e}")
            raise AuthenticationError(f"Invalid token: {e}")

    async def authenticate_user(self, request: Request, credentials: Optional[HTTPAuthorizationCredentials]) -> Dict[str, Any]:
        """Main authentication method with comprehensive security"""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")[:200]

        # Check IP lockout
        if self.session_manager.is_ip_locked_out(client_ip):
            self.audit_logger.log_auth_event(
                "authentication_attempt",
                "unknown",
                client_ip,
                user_agent,
                False,
                "IP locked out due to excessive failed attempts"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed authentication attempts. Try again later.",
                headers={"Retry-After": str(LOCKOUT_DURATION_SECONDS)}
            )

        # Check for missing credentials
        if not credentials or not credentials.credentials:
            self.session_manager.track_failed_attempt(client_ip)
            self.audit_logger.log_auth_event(
                "authentication_attempt",
                "unknown",
                client_ip,
                user_agent,
                False,
                "Missing authentication token"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        try:
            # Try auth service first
            try:
                user_info = await self.verify_token_with_auth_service(credentials.credentials)
            except AuthenticationError:
                # Fallback to local verification
                user_info = self.verify_token_locally(credentials.credentials)

            # Clear failed attempts on successful auth
            self.session_manager.clear_failed_attempts(client_ip)

            # Store/update session
            session_data = {
                "last_activity": datetime.utcnow().isoformat(),
                "ip_address": client_ip,
                "user_agent": user_agent
            }
            self.session_manager.store_session(user_info.get('user_id', ''), session_data)

            # Log successful authentication
            self.audit_logger.log_auth_event(
                "authentication_success",
                user_info.get('user', 'unknown'),
                client_ip,
                user_agent,
                True,
                "Token verified successfully"
            )

            return user_info

        except AuthenticationError as e:
            # Track failed attempt
            self.session_manager.track_failed_attempt(client_ip)

            # Log failed authentication
            self.audit_logger.log_auth_event(
                "authentication_failure",
                "unknown",
                client_ip,
                user_agent,
                False,
                str(e)
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )

    async def logout_user(self, token: str) -> bool:
        """Logout user by blacklisting token"""
        try:
            payload = self.verify_token_locally(token)
            token_jti = payload.get('jti')
            if token_jti:
                remaining_time = payload.get('exp', 0) - int(datetime.utcnow().timestamp())
                if remaining_time > 0:
                    self.session_manager.blacklist_token(token_jti, remaining_time)
                return True
        except Exception as e:
            logger.error(f"Failed to logout user: {e}")
        return False

# Global instances
auth_middleware = EnhancedAuthMiddleware()

async def get_current_user(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Dependency for protecting API endpoints"""
    return await auth_middleware.authenticate_user(request, credentials)

async def get_current_user_optional(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Optional authentication - returns None if no valid token"""
    try:
        if credentials and credentials.credentials:
            return await auth_middleware.authenticate_user(request, credentials)
        return None
    except HTTPException:
        return None

async def require_admin_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency for admin-only endpoints"""
    user_role = user.get('role', 'user')
    if user_role not in ['admin', 'patrick']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user

def create_emergency_token(username: str = "patrick", role: str = "admin") -> str:
    """Emergency function to create admin token"""
    user_data = {
        'username': username,
        'user_id': username,
        'role': role
    }
    return auth_middleware.create_access_token(user_data)

async def get_auth_status() -> Dict[str, Any]:
    """Get comprehensive authentication system status"""
    try:
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            response = await client.get(f"{AUTH_SERVICE_URL}/health")
            auth_service_status = response.status_code == 200
    except:
        auth_service_status = False

    return {
        "auth_service_available": auth_service_status,
        "local_jwt_verification": True,
        "session_management": auth_middleware.session_manager.redis_available,
        "audit_logging": True,
        "rate_limiting": True,
        "token_blacklisting": True,
        "emergency_token_available": True,
        "lockout_protection": True
    }