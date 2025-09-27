#!/usr/bin/env python3
"""
Authentication Middleware for AI Assist Board of Directors
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
    """Authentication middleware for AI Assist Board"""

    def __init__(self):
        self.auth_service_url = os.environ.get("AUTH_SERVICE_URL", "http://localhost:8088")
        self.jwt_secret = os.environ.get("JWT_SECRET", "")
        self.token_expiry_minutes = int(os.environ.get("TOKEN_EXPIRY_MINUTES", "30"))

        # SECURITY FIX: Require JWT_SECRET to be configured - no fallback authentication
        if not self.jwt_secret:
            logger.critical("JWT_SECRET environment variable is required for secure authentication")
            raise ValueError("JWT_SECRET must be configured. Fallback authentication is disabled for security.")

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token with Tower's auth service

        Args:
            token: JWT token string

        Returns:
            User information if valid, None if invalid
        """
        try:
            # SECURITY FIX: Only use JWT verification - no fallback authentication
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

            # Validate required fields
            user_id = payload.get('user_id')
            if not user_id:
                logger.warning("Token missing required user_id field")
                return None

            return {
                'user_id': user_id,
                'username': payload.get('username', 'unknown'),
                'roles': payload.get('roles', []),
                'permissions': payload.get('permissions', [])
            }

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"JWT verification failed: {e}")
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

            # SECURITY FIX: Strengthen role-based permission checks
            # Check direct permission first
            if permission in user_permissions:
                return user

            # Strict admin role verification with additional checks
            admin_roles = ['system_admin']  # Reduced admin scope
            if any(role in user_roles for role in admin_roles):
                # Additional verification for admin actions
                admin_permissions = [
                    'system.admin', 'board.admin', 'security.admin'
                ]
                if any(admin_perm in user_permissions for admin_perm in admin_permissions):
                    return user

            # Specific board permissions must be explicitly granted
            board_permissions_map = {
                'board.submit_task': ['board_user', 'board_contributor'],
                'board.view_decisions': ['board_user', 'board_viewer', 'board_contributor'],
                'board.provide_feedback': ['board_contributor', 'board_reviewer'],
                'board.override_decisions': ['board_admin', 'system_admin']
            }

            required_roles = board_permissions_map.get(permission, [])
            if permission in board_permissions_map and any(role in user_roles for role in required_roles):
                return user

            logger.warning(f"Access denied for user {user.get('user_id')} to permission '{permission}'")
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: '{permission}'"
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