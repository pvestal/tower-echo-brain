#!/usr/bin/env python3
"""
Secure WebSocket Manager with Authentication and Rate Limiting
Replaces insecure WebSocket implementation in main.py
"""

import asyncio
import time
import jwt
import logging
from typing import Dict, Set, Optional
from collections import defaultdict, deque
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for WebSocket connections"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        now = time.time()
        client_requests = self.requests[client_id]

        # Remove old requests outside the window
        while client_requests and client_requests[0] <= now - self.window_seconds:
            client_requests.popleft()

        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client"""
        return max(0, self.max_requests - len(self.requests[client_id]))

class SecureConnectionManager:
    """Secure WebSocket connection manager with auth and rate limiting"""

    def __init__(self, jwt_secret: str = "echo-brain-jwt-secret", max_connections: int = 100):
        self.active_connections: Dict[str, Dict] = {}
        self.jwt_secret = jwt_secret
        self.max_connections = max_connections
        self.rate_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 req/min
        self.connection_timestamps: Dict[str, float] = {}

    def _generate_client_id(self, websocket: WebSocket) -> str:
        """Generate unique client ID from websocket"""
        client_host = websocket.client.host if websocket.client else "unknown"
        client_port = websocket.client.port if websocket.client else 0
        return f"{client_host}:{client_port}:{time.time()}"

    def _validate_jwt_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and extract user info"""
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])

            # Check expiration
            if payload.get('exp', 0) < time.time():
                return None

            return payload

        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logger.error(f"JWT validation error: {e}")
            return None

    async def authenticate_connection(self, websocket: WebSocket, token: Optional[str]) -> Optional[str]:
        """Authenticate WebSocket connection"""

        # Get client IP for rate limiting (consistent across connections)
        client_ip = websocket.client.host if websocket.client else "unknown"

        # For development, allow connections without auth but log warning
        if not token:
            logger.warning("WebSocket connection without authentication token")
            user_info = {"user_id": "anonymous", "permissions": ["read"]}
            # Use IP for rate limiting, but timestamp for unique connection ID
            rate_limit_key = f"anonymous@{client_ip}"
            client_id = f"anonymous@{client_ip}:{time.time()}"
        else:
            # Validate JWT token
            user_info = self._validate_jwt_token(token)
            if not user_info:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return None

            # Use user_id for rate limiting, but timestamp for unique connection ID
            user_id = user_info.get('user_id', 'unknown')
            rate_limit_key = user_id
            client_id = f"{user_id}:{time.time()}"

        # Check connection limits
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Server overloaded")
            return None

        # Check rate limits using consistent key (not unique connection ID)
        if not self.rate_limiter.is_allowed(rate_limit_key):
            await websocket.close(code=1008, reason="Rate limit exceeded")
            return None

        return client_id

    def get_rate_limit_key(self, client_id: str) -> str:
        """Extract rate limit key from client_id"""
        # client_id format: "user_id:timestamp" or "anonymous@ip:timestamp"
        if ":" in client_id:
            return client_id.split(":")[0]  # Remove timestamp
        return client_id

    async def connect(self, websocket: WebSocket, token: Optional[str] = None) -> Optional[str]:
        """Establish secure WebSocket connection"""

        # Authenticate first
        client_id = await self.authenticate_connection(websocket, token)
        if not client_id:
            return None

        try:
            await websocket.accept()

            # Store connection info
            self.active_connections[client_id] = {
                "websocket": websocket,
                "connected_at": time.time(),
                "last_activity": time.time(),
                "message_count": 0
            }

            logger.info(f"Secure WebSocket connection established: {client_id}")
            return client_id

        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            return None

    def disconnect(self, client_id: str):
        """Disconnect client"""
        if client_id in self.active_connections:
            connection_info = self.active_connections[client_id]
            duration = time.time() - connection_info["connected_at"]

            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id} (duration: {duration:.1f}s, messages: {connection_info['message_count']})")

    async def send_personal_message(self, message: str, client_id: str) -> bool:
        """Send message to specific client"""
        if client_id not in self.active_connections:
            return False

        try:
            websocket = self.active_connections[client_id]["websocket"]
            await websocket.send_text(message)

            # Update activity
            self.active_connections[client_id]["last_activity"] = time.time()
            self.active_connections[client_id]["message_count"] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
            self.disconnect(client_id)
            return False

    async def send_json_message(self, data: Dict, client_id: str) -> bool:
        """Send JSON message to specific client"""
        if client_id not in self.active_connections:
            return False

        try:
            websocket = self.active_connections[client_id]["websocket"]
            await websocket.send_json(data)

            # Update activity
            self.active_connections[client_id]["last_activity"] = time.time()
            self.active_connections[client_id]["message_count"] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to send JSON to {client_id}: {e}")
            self.disconnect(client_id)
            return False

    async def broadcast(self, message: str, exclude: Optional[str] = None):
        """Broadcast message to all connected clients"""
        disconnected_clients = []

        for client_id, connection_info in self.active_connections.items():
            if client_id == exclude:
                continue

            try:
                websocket = connection_info["websocket"]
                await websocket.send_text(message)
                connection_info["message_count"] += 1

            except Exception as e:
                logger.error(f"Broadcast failed for {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        now = time.time()
        connections = []

        for client_id, connection_info in self.active_connections.items():
            connections.append({
                "client_id": client_id,
                "connected_duration": now - connection_info["connected_at"],
                "last_activity": now - connection_info["last_activity"],
                "message_count": connection_info["message_count"]
            })

        return {
            "total_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "connections": connections,
            "timestamp": datetime.now().isoformat()
        }

    async def cleanup_stale_connections(self, max_idle_time: int = 300):
        """Clean up connections idle for too long"""
        now = time.time()
        stale_clients = []

        for client_id, connection_info in self.active_connections.items():
            if now - connection_info["last_activity"] > max_idle_time:
                stale_clients.append(client_id)

        for client_id in stale_clients:
            try:
                websocket = self.active_connections[client_id]["websocket"]
                await websocket.close(code=1000, reason="Connection idle timeout")
            except:
                pass  # Connection may already be closed
            self.disconnect(client_id)

        if stale_clients:
            logger.info(f"Cleaned up {len(stale_clients)} stale WebSocket connections")

# Global secure connection manager
secure_manager = SecureConnectionManager()