#!/usr/bin/env python3
"""
Authentication Middleware for Echo Brain Board of Directors
Integrates with Tower's centralized auth service for JWT validation
"""

import os
import logging
import aiohttp
import json
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# JWT Bearer token extraction
security = HTTPBearer()

class AuthMiddleware:
    """Authentication middleware for Echo Brain Board"""

    def __init__(self):
        self.auth_service_url = os.environ.get("AUTH_SERVICE_URL", "http://***REMOVED***:8088")
        self.jwt_secret = os.environ.get("JWT_SECRET", "")
        self.token_expiry_minutes = int(os.environ.get("TOKEN_EXPIRY_MINUTES", "30"))

        if not self.jwt_secret:
            logger.warning("JWT_SECRET not configured - using fallback authentication")

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token with Tower's auth service

        Args:
            token: JWT token string

        Returns:
            User information if valid, None if invalid
        """
        try:
            # First try local JWT verification
            if self.jwt_secret:
                try:
                    payload = jwt.decode(
                        token,
                        self.jwt_secret,
                        algorithms=["HS256"]
                    )

                    # Check expiration
                    exp = payload.get('exp', 0)
                    if datetime.utcnow().timestamp() > exp:
                        logger.warning("Token expired")
                        return None

                    return {
                        'user_id': payload.get('user_id', 'unknown'),
                        'username': payload.get('username', 'unknown'),
                        'roles': payload.get('roles', []),
                        'permissions': payload.get('permissions', [])
                    }
                except jwt.InvalidTokenError as e:
                    logger.warning(f"JWT verification failed: {e}")

            # Fallback to Tower auth service verification
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.auth_service_url}/api/auth/verify",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Auth service verification failed: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """
        FastAPI dependency to get current authenticated user

        Args:
            credentials: HTTP Bearer credentials

        Returns:
            User information

        Raises:
            HTTPException: If authentication fails
        """
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication credentials required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_info = await self.verify_token(credentials.credentials)

        if not user_info:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user_info

    async def get_optional_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Get user information if present, but don't require authentication
        Used for optional auth endpoints

        Args:
            request: FastAPI request object

        Returns:
            User information if authenticated, None otherwise
        """
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]
        return await self.verify_token(token)

    def require_permission(self, permission: str):
        """
        Decorator factory for permission-based access control

        Args:
            permission: Required permission string

        Returns:
            FastAPI dependency function
        """
        async def permission_checker(user: Dict[str, Any] = Depends(self.get_current_user)):
            user_permissions = user.get('permissions', [])
            user_roles = user.get('roles', [])

            # Check direct permission
            if permission in user_permissions:
                return user

            # Check role-based permissions
            admin_roles = ['admin', 'board_admin', 'system_admin']
            if any(role in user_roles for role in admin_roles):
                return user

            # Check specific board permissions
            board_permissions = [
                'board.submit_task',
                'board.view_decisions',
                'board.provide_feedback',
                'board.override_decisions'
            ]

            if permission in board_permissions and 'board_user' in user_roles:
                return user

            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )

        return permission_checker

    def require_role(self, role: str):
        """
        Decorator factory for role-based access control

        Args:
            role: Required role string

        Returns:
            FastAPI dependency function
        """
        async def role_checker(user: Dict[str, Any] = Depends(self.get_current_user)):
            user_roles = user.get('roles', [])

            if role not in user_roles and 'admin' not in user_roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{role}' required"
                )

            return user

        return role_checker

# Global auth middleware instance
auth_middleware = AuthMiddleware()

# Convenience dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    return await auth_middleware.get_current_user(credentials)

async def get_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get user if present, don't require auth"""
    return await auth_middleware.get_optional_user(request)

def require_permission(permission: str):
    """Require specific permission"""
    return auth_middleware.require_permission(permission)

def require_role(role: str):
    """Require specific role"""
    return auth_middleware.require_role(role)

# WebSocket authentication
async def authenticate_websocket(websocket) -> Optional[Dict[str, Any]]:
    """
    Authenticate WebSocket connection using token from query params or headers

    Args:
        websocket: WebSocket connection

    Returns:
        User information if valid, None otherwise
    """
    try:
        # Try to get token from query parameters
        token = websocket.query_params.get("token")

        if not token:
            # Try to get from headers (if client supports it)
            auth_header = websocket.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            logger.warning("WebSocket connection attempted without token")
            return None

        user_info = await auth_middleware.verify_token(token)

        if user_info:
            logger.info(f"WebSocket authenticated for user: {user_info.get('user_id', 'unknown')}")
        else:
            logger.warning("WebSocket authentication failed")

        return user_info

    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None