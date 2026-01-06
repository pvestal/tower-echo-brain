#!/usr/bin/env python3
"""
User Context Middleware for Echo Brain
Handles user identification, context loading, and permission checking
"""

import logging
from typing import Optional, Callable
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.echo_identity import get_echo_identity
from src.core.user_context_manager import get_user_context_manager

logger = logging.getLogger(__name__)

class UserContextMiddleware(BaseHTTPMiddleware):
    """Middleware to inject user context into requests"""

    async def dispatch(self, request: Request, call_next: Callable):
        """Process requests to add user context"""

        # Skip middleware for non-API routes
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        # Extract username from various sources
        username = await self._extract_username(request)

        # Get user context and identity
        try:
            user_manager = await get_user_context_manager()
            user_context = await user_manager.get_or_create_context(username)
            echo_identity = get_echo_identity()
            user_recognition = echo_identity.recognize_user(username)

            # Attach to request state for use in endpoints
            request.state.username = username
            request.state.user_context = user_context
            request.state.user_recognition = user_recognition
            request.state.user_manager = user_manager
            request.state.echo_identity = echo_identity

            logger.debug(f"User context loaded for {username}: {user_recognition['access_level']}")

        except Exception as e:
            logger.error(f"Failed to load user context: {e}")
            # Continue without context rather than failing
            request.state.username = "anonymous"
            request.state.user_context = None
            request.state.user_recognition = {"access_level": "none"}

        # Process request
        response = await call_next(request)

        # Log interaction if we have a user context
        if hasattr(request.state, 'user_context') and request.state.user_context:
            request.state.user_context.update_interaction()

        return response

    async def _extract_username(self, request: Request) -> str:
        """Extract username from request"""
        # Priority 1: Header
        username = request.headers.get("X-Username")
        if username:
            return username

        # Priority 2: Query parameter
        username = request.query_params.get("user")
        if username:
            return username

        # Priority 3: Authorization header (if using tokens)
        auth_header = request.headers.get("Authorization", "")
        if "Bearer" in auth_header:
            # This would normally decode a JWT token
            # For now, just check for known patterns
            if "patrick" in auth_header.lower():
                return "patrick"

        # Priority 4: Request body (for POST requests)
        if request.method == "POST":
            try:
                # Note: This requires reading the body which can only be done once
                # In production, use a proper solution
                if hasattr(request, '_body'):
                    import json
                    body = json.loads(request._body)
                    username = body.get("username")
                    if username:
                        return username
            except:
                pass

        return "anonymous"


class PermissionMiddleware(BaseHTTPMiddleware):
    """Middleware to check permissions for protected endpoints"""

    # Define protected endpoint patterns and required permissions
    PROTECTED_ENDPOINTS = {
        "/api/echo/system/": "system_commands",
        "/api/echo/oversight/": "creator",
        "/api/echo/users/": "creator",
        "/api/echo/vault/": "creator",
        "/api/tower/control/": "service_control",
        "/api/files/": "file_access",
        "/api/database/": "database_access"
    }

    async def dispatch(self, request: Request, call_next: Callable):
        """Check permissions before processing protected endpoints"""

        path = request.url.path

        # Check if this is a protected endpoint
        for pattern, required_permission in self.PROTECTED_ENDPOINTS.items():
            if path.startswith(pattern):
                # Check if user has permission
                if not await self._check_permission(request, required_permission):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Permission denied",
                            "detail": f"You don't have permission to access {pattern}",
                            "required": required_permission
                        }
                    )

        return await call_next(request)

    async def _check_permission(self, request: Request, permission: str) -> bool:
        """Check if user has required permission"""

        # Get user context from request state (set by UserContextMiddleware)
        if not hasattr(request.state, 'user_context'):
            return False

        user_context = request.state.user_context
        if not user_context:
            return False

        # Special case for creator-only endpoints
        if permission == "creator":
            return request.state.username == "patrick"

        # Check standard permissions
        return user_context.permissions.get(permission, False)