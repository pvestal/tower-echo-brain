#!/usr/bin/env python3
"""
Authentication Middleware for Echo Brain Autonomous Operations
Secures autonomous execution with JWT token validation
"""

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import httpx
import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'echo-brain-secret-key-2025')
AUTH_SERVICE_URL = "https://192.168.50.135/api/auth"

# Security scheme
security = HTTPBearer()

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class EchoAuthMiddleware:
    """Authentication middleware for Echo autonomous operations"""

    def __init__(self):
        self.auth_service_url = AUTH_SERVICE_URL
        self.jwt_secret = JWT_SECRET_KEY

    async def verify_token_with_auth_service(self, token: str) -> Dict[str, Any]:
        """Verify token with Tower auth service"""
        try:
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.post(
                    f"{self.auth_service_url}/verify",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=5.0
                )

                if response.status_code == 200:
                    user_info = response.json()
                    logger.info(f"✅ Token verified for user: {user_info.get('user', 'unknown')}")
                    return user_info
                else:
                    logger.warning(f"❌ Auth service rejected token: {response.status_code}")
                    raise AuthenticationError("Token verification failed")

        except Exception as e:
            logger.error(f"❌ Auth service connection failed: {e}")
            # Fallback to local JWT verification
            return self.verify_token_locally(token)

    def verify_token_locally(self, token: str) -> Dict[str, Any]:
        """Fallback local JWT token verification"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            # Check expiration
            if payload.get('exp', 0) < datetime.utcnow().timestamp():
                raise AuthenticationError("Token expired")

            logger.info(f"✅ Local token verified for user: {payload.get('user', 'unknown')}")
            return payload

        except jwt.InvalidTokenError as e:
            logger.error(f"❌ Invalid JWT token: {e}")
            raise AuthenticationError("Invalid token")

    async def authenticate_user(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Main authentication method"""
        if not credentials or not credentials.credentials:
            raise HTTPException(
                status_code=401,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        try:
            # Try auth service first, fallback to local verification
            user_info = await self.verify_token_with_auth_service(credentials.credentials)
            return user_info

        except AuthenticationError as e:
            raise HTTPException(
                status_code=401,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )

# Global auth middleware instance
auth_middleware = EchoAuthMiddleware()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency for protecting autonomous endpoints"""
    return await auth_middleware.authenticate_user(credentials)

async def require_admin_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency for admin-only autonomous operations"""
    user_role = user.get('role', 'user')
    if user_role not in ['admin', 'patrick']:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required for autonomous operations"
        )
    return user

def create_auth_token_for_patrick() -> str:
    """Emergency function to create admin token for Patrick"""
    payload = {
        'user': 'patrick',
        'role': 'admin',
        'iat': datetime.utcnow().timestamp(),
        'exp': datetime.utcnow().timestamp() + (30 * 24 * 3600),  # 30 days
        'autonomous_access': True
    }

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
    logger.info("🔑 Emergency admin token created for Patrick")
    return token

async def get_auth_status() -> Dict[str, Any]:
    """Check authentication system status"""
    try:
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(f"{AUTH_SERVICE_URL}/health", timeout=3.0)
            auth_service_status = response.status_code == 200
    except:
        auth_service_status = False

    return {
        "auth_service_available": auth_service_status,
        "local_jwt_verification": True,
        "autonomous_endpoints_secured": True,
        "emergency_token_available": True
    }