#!/usr/bin/env python3
"""
Advanced Rate Limiting Middleware for Echo Brain API
Implements sliding window rate limiting with different tiers and protection levels
"""

import time
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
from fastapi import HTTPException, Request, Response, status
from datetime import datetime, timedelta
import redis
import json
import os
import ipaddress

logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', '***REMOVED***')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_RATE_LIMIT_DB', '2'))

# Rate limiting tiers (requests per minute)
RATE_LIMITS = {
    'anonymous': 30,    # Unauthenticated requests
    'user': 60,         # Regular authenticated users
    'admin': 120,       # Admin users
    'patrick': 300,     # Patrick (unlimited practically)
    'system': 1000      # System-to-system calls
}

# Special endpoint limits (requests per minute)
ENDPOINT_LIMITS = {
    '/api/echo/query': 20,          # Main query endpoint
    '/api/echo/chat': 20,           # Chat endpoint
    '/api/echo/models/pull': 5,     # Model pulling (heavy operation)
    '/api/echo/models/manage': 10,  # Model management
    '/health': 100,                 # Health checks
    '/api/echo/health': 100,        # Echo health checks
    '/api/echo/status': 100,        # Status checks
}

# Burst protection (requests per second)
BURST_LIMITS = {
    'anonymous': 5,
    'user': 10,
    'admin': 20,
    'patrick': 50,
    'system': 100
}

# Window sizes in seconds
WINDOW_SIZE = 60  # 1 minute sliding window
BURST_WINDOW_SIZE = 1  # 1 second burst window

class RateLimitExceeded(Exception):
    """Rate limit exceeded exception"""
    pass

class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation"""

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
            logger.info("✅ Redis rate limiter connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, using memory-based rate limiting: {e}")
            self.redis_available = False
            self.memory_windows = defaultdict(lambda: deque())
            self.burst_windows = defaultdict(lambda: deque())

    def _get_client_identifier(self, request: Request, user_data: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Get client identifier and rate limit tier"""
        # Determine user tier
        if user_data:
            role = user_data.get('role', 'user')
            username = user_data.get('user', '')

            if username == 'patrick':
                tier = 'patrick'
            elif role == 'admin':
                tier = 'admin'
            elif role == 'system':
                tier = 'system'
            else:
                tier = 'user'

            # Use user ID for authenticated users
            identifier = f"user:{user_data.get('user_id', username)}"
        else:
            tier = 'anonymous'
            # Use IP for anonymous users
            ip = self._get_client_ip(request)
            identifier = f"ip:{ip}"

        return identifier, tier

    def _get_client_ip(self, request: Request) -> str:
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

    def _is_whitelisted_ip(self, ip: str) -> bool:
        """Check if IP is whitelisted (local network, etc.)"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            # Local network ranges
            local_ranges = [
                ipaddress.ip_network('192.168.0.0/16'),
                ipaddress.ip_network('10.0.0.0/8'),
                ipaddress.ip_network('172.16.0.0/12'),
                ipaddress.ip_network('127.0.0.0/8'),
            ]
            return any(ip_addr in network for network in local_ranges)
        except:
            return False

    def _get_endpoint_limit(self, endpoint: str, tier: str) -> int:
        """Get specific endpoint limit or default tier limit"""
        # Check for exact endpoint match
        if endpoint in ENDPOINT_LIMITS:
            return ENDPOINT_LIMITS[endpoint]

        # Check for pattern matches
        for pattern, limit in ENDPOINT_LIMITS.items():
            if pattern.endswith('*') and endpoint.startswith(pattern[:-1]):
                return limit

        # Return tier default
        return RATE_LIMITS.get(tier, RATE_LIMITS['anonymous'])

    def _redis_check_rate_limit(self, key: str, limit: int, window_size: int) -> Tuple[bool, int, int]:
        """Check rate limit using Redis sliding window"""
        try:
            current_time = int(time.time())
            window_start = current_time - window_size

            pipe = self.redis_client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current requests
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(current_time * 1000000 + hash(key) % 1000000): current_time})

            # Set expiration
            pipe.expire(key, window_size + 1)

            results = pipe.execute()
            current_count = results[1] + 1  # +1 for the request we just added

            allowed = current_count <= limit
            remaining = max(0, limit - current_count)

            return allowed, remaining, limit

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to allow request
            return True, limit, limit

    def _memory_check_rate_limit(self, key: str, limit: int, window_size: int) -> Tuple[bool, int, int]:
        """Check rate limit using memory-based sliding window"""
        try:
            current_time = time.time()
            window_start = current_time - window_size

            # Get or create window
            if window_size == BURST_WINDOW_SIZE:
                window = self.burst_windows[key]
            else:
                window = self.memory_windows[key]

            # Remove old entries
            while window and window[0] <= window_start:
                window.popleft()

            # Check limit
            current_count = len(window) + 1  # +1 for current request
            allowed = current_count <= limit

            if allowed:
                window.append(current_time)

            remaining = max(0, limit - current_count)
            return allowed, remaining, limit

        except Exception as e:
            logger.error(f"Memory rate limit check failed: {e}")
            return True, limit, limit

    def check_rate_limit(self, request: Request, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check if request is within rate limits"""
        identifier, tier = self._get_client_identifier(request, user_data)
        endpoint = request.url.path

        # Skip rate limiting for whitelisted IPs in local network
        client_ip = self._get_client_ip(request)
        if self._is_whitelisted_ip(client_ip) and tier in ['patrick', 'admin']:
            return {
                'allowed': True,
                'tier': tier,
                'limit': RATE_LIMITS[tier],
                'remaining': RATE_LIMITS[tier],
                'reset_time': int(time.time()) + WINDOW_SIZE,
                'whitelisted': True
            }

        # Get limits for this endpoint and tier
        minute_limit = self._get_endpoint_limit(endpoint, tier)
        burst_limit = BURST_LIMITS.get(tier, BURST_LIMITS['anonymous'])

        # Check burst limit (requests per second)
        burst_key = f"burst:{identifier}:{endpoint}"
        if self.redis_available:
            burst_allowed, burst_remaining, _ = self._redis_check_rate_limit(
                burst_key, burst_limit, BURST_WINDOW_SIZE
            )
        else:
            burst_allowed, burst_remaining, _ = self._memory_check_rate_limit(
                burst_key, burst_limit, BURST_WINDOW_SIZE
            )

        if not burst_allowed:
            raise RateLimitExceeded(
                f"Burst limit exceeded: {burst_limit} requests per second"
            )

        # Check minute limit (requests per minute)
        minute_key = f"rate:{identifier}:{endpoint}"
        if self.redis_available:
            minute_allowed, minute_remaining, _ = self._redis_check_rate_limit(
                minute_key, minute_limit, WINDOW_SIZE
            )
        else:
            minute_allowed, minute_remaining, _ = self._memory_check_rate_limit(
                minute_key, minute_limit, WINDOW_SIZE
            )

        if not minute_allowed:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {minute_limit} requests per minute"
            )

        return {
            'allowed': True,
            'tier': tier,
            'limit': minute_limit,
            'remaining': minute_remaining,
            'burst_remaining': burst_remaining,
            'reset_time': int(time.time()) + WINDOW_SIZE,
            'whitelisted': False
        }

class RateLimitMiddleware:
    """Rate limiting middleware for FastAPI"""

    def __init__(self):
        self.limiter = SlidingWindowRateLimiter()

    def _add_rate_limit_headers(self, response: Response, rate_limit_info: Dict[str, Any]):
        """Add rate limit headers to response"""
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info['remaining'])
        response.headers["X-RateLimit-Reset"] = str(rate_limit_info['reset_time'])
        response.headers["X-RateLimit-Tier"] = rate_limit_info['tier']

        if 'burst_remaining' in rate_limit_info:
            response.headers["X-RateLimit-Burst-Remaining"] = str(rate_limit_info['burst_remaining'])

    async def check_rate_limit(self, request: Request, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check rate limit for incoming request"""
        try:
            rate_limit_info = self.limiter.check_rate_limit(request, user_data)

            # Log rate limit check for monitoring
            if rate_limit_info['remaining'] < 5:  # Low remaining requests
                client_ip = self.limiter._get_client_ip(request)
                logger.warning(
                    f"⚠️ Rate limit approaching: {rate_limit_info['tier']} user from {client_ip} "
                    f"has {rate_limit_info['remaining']} requests remaining"
                )

            return rate_limit_info

        except RateLimitExceeded as e:
            client_ip = self.limiter._get_client_ip(request)
            endpoint = request.url.path
            tier = user_data.get('role', 'anonymous') if user_data else 'anonymous'

            logger.warning(
                f"🚨 Rate limit exceeded: {tier} user from {client_ip} "
                f"for endpoint {endpoint} - {str(e)}"
            )

            # Determine retry after time
            retry_after = BURST_WINDOW_SIZE if "per second" in str(e) else WINDOW_SIZE

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {str(e)}",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Error": str(e)
                }
            )

    async def __call__(self, request: Request, call_next, user_data: Optional[Dict[str, Any]] = None):
        """Middleware call method"""
        # Check rate limit
        rate_limit_info = await self.check_rate_limit(request, user_data)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        self._add_rate_limit_headers(response, rate_limit_info)

        return response

# Global rate limiter instance
rate_limiter = RateLimitMiddleware()

async def apply_rate_limit(request: Request, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dependency function to apply rate limiting"""
    return await rate_limiter.check_rate_limit(request, user_data)

def get_rate_limit_status() -> Dict[str, Any]:
    """Get rate limiting system status"""
    return {
        "redis_available": rate_limiter.limiter.redis_available,
        "rate_limits": RATE_LIMITS,
        "endpoint_limits": ENDPOINT_LIMITS,
        "burst_limits": BURST_LIMITS,
        "window_size_minutes": WINDOW_SIZE // 60,
        "burst_window_seconds": BURST_WINDOW_SIZE
    }

async def get_user_rate_limit_status(request: Request, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get current user's rate limit status"""
    identifier, tier = rate_limiter.limiter._get_client_identifier(request, user_data)
    endpoint = request.url.path

    minute_limit = rate_limiter.limiter._get_endpoint_limit(endpoint, tier)
    burst_limit = BURST_LIMITS.get(tier, BURST_LIMITS['anonymous'])

    # Get current usage (without incrementing)
    current_time = int(time.time())
    minute_key = f"rate:{identifier}:{endpoint}"
    burst_key = f"burst:{identifier}:{endpoint}"

    try:
        if rate_limiter.limiter.redis_available:
            pipe = rate_limiter.limiter.redis_client.pipeline()
            pipe.zcount(minute_key, current_time - WINDOW_SIZE, current_time)
            pipe.zcount(burst_key, current_time - BURST_WINDOW_SIZE, current_time)
            results = pipe.execute()
            minute_used = results[0]
            burst_used = results[1]
        else:
            # Memory-based counting is harder without side effects
            minute_used = 0
            burst_used = 0
    except:
        minute_used = 0
        burst_used = 0

    return {
        "tier": tier,
        "endpoint": endpoint,
        "minute_limit": minute_limit,
        "minute_used": minute_used,
        "minute_remaining": max(0, minute_limit - minute_used),
        "burst_limit": burst_limit,
        "burst_used": burst_used,
        "burst_remaining": max(0, burst_limit - burst_used),
        "reset_time": current_time + WINDOW_SIZE
    }