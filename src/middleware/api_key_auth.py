#!/usr/bin/env python3
"""
API Key Authentication Middleware for Echo Brain
Provides programmatic access via API keys with role-based permissions
"""

import os
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, Request, Depends, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import json

logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', '192.168.50.135')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_API_KEY_DB = int(os.getenv('REDIS_API_KEY_DB', '2'))

# API Key configuration
API_KEY_LENGTH = 32
API_KEY_PREFIX = "eb_"  # Echo Brain prefix
KEY_EXPIRY_DAYS = 90

class APIKeyManager:
    """Manages API key generation, validation, and storage"""

    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_API_KEY_DB,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis API key manager connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable for API keys: {e}")
            self.redis_available = False
            # Fallback to environment variables for static keys
            self.static_keys = self._load_static_keys()

    def _load_static_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load static API keys from environment variables"""
        static_keys = {}

        # Load system API keys
        system_key = os.getenv('ECHO_BRAIN_SYSTEM_API_KEY')
        if system_key:
            static_keys[system_key] = {
                'name': 'System Key',
                'role': 'system',
                'permissions': ['read', 'write', 'admin'],
                'created_by': 'system',
                'created_at': datetime.now().isoformat(),
                'last_used': None,
                'usage_count': 0
            }

        # Load read-only key
        readonly_key = os.getenv('ECHO_BRAIN_READONLY_API_KEY')
        if readonly_key:
            static_keys[readonly_key] = {
                'name': 'Read-Only Key',
                'role': 'readonly',
                'permissions': ['read'],
                'created_by': 'system',
                'created_at': datetime.now().isoformat(),
                'last_used': None,
                'usage_count': 0
            }

        logger.info(f"Loaded {len(static_keys)} static API keys")
        return static_keys

    def _get_key(self, key_type: str, identifier: str) -> str:
        """Generate Redis key"""
        return f"echo_api_keys:{key_type}:{identifier}"

    def generate_api_key(self, name: str, role: str, permissions: List[str], created_by: str) -> Dict[str, Any]:
        """Generate a new API key"""
        # Generate secure random key
        key_bytes = secrets.token_bytes(API_KEY_LENGTH)
        api_key = API_KEY_PREFIX + key_bytes.hex()

        # Create key metadata
        key_data = {
            'name': name,
            'role': role,
            'permissions': permissions,
            'created_by': created_by,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=KEY_EXPIRY_DAYS)).isoformat(),
            'last_used': None,
            'usage_count': 0,
            'active': True
        }

        # Store in Redis if available
        if self.redis_available:
            try:
                key_redis_key = self._get_key("data", api_key)
                self.redis_client.setex(
                    key_redis_key,
                    KEY_EXPIRY_DAYS * 24 * 60 * 60,
                    json.dumps(key_data)
                )

                # Store in key index for management
                index_key = self._get_key("index", created_by)
                self.redis_client.sadd(index_key, api_key)

                logger.info(f"Generated new API key for {name} by {created_by}")

            except Exception as e:
                logger.error(f"Failed to store API key in Redis: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate API key")

        return {
            'api_key': api_key,
            'metadata': key_data
        }

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return metadata"""
        if not api_key or not api_key.startswith(API_KEY_PREFIX):
            return None

        try:
            # Check Redis first
            if self.redis_available:
                key_redis_key = self._get_key("data", api_key)
                key_data_str = self.redis_client.get(key_redis_key)

                if key_data_str:
                    key_data = json.loads(key_data_str)

                    # Check if key is active and not expired
                    if not key_data.get('active', False):
                        return None

                    expires_at = datetime.fromisoformat(key_data['expires_at'])
                    if expires_at < datetime.now():
                        # Key expired, remove it
                        self.revoke_api_key(api_key)
                        return None

                    # Update last used timestamp
                    key_data['last_used'] = datetime.now().isoformat()
                    key_data['usage_count'] = key_data.get('usage_count', 0) + 1

                    # Update in Redis
                    self.redis_client.setex(
                        key_redis_key,
                        KEY_EXPIRY_DAYS * 24 * 60 * 60,
                        json.dumps(key_data)
                    )

                    return key_data

            # Fallback to static keys
            else:
                if api_key in self.static_keys:
                    key_data = self.static_keys[api_key].copy()
                    key_data['last_used'] = datetime.now().isoformat()
                    key_data['usage_count'] = key_data.get('usage_count', 0) + 1
                    return key_data

            return None

        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        try:
            if self.redis_available:
                key_redis_key = self._get_key("data", api_key)
                key_data_str = self.redis_client.get(key_redis_key)

                if key_data_str:
                    key_data = json.loads(key_data_str)
                    key_data['active'] = False
                    key_data['revoked_at'] = datetime.now().isoformat()

                    # Store updated data with shorter expiry
                    self.redis_client.setex(key_redis_key, 3600, json.dumps(key_data))  # 1 hour

                    logger.info(f"Revoked API key: {api_key}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False

    def list_api_keys(self, user: str) -> List[Dict[str, Any]]:
        """List API keys for a user"""
        keys = []

        try:
            if self.redis_available:
                index_key = self._get_key("index", user)
                api_keys = self.redis_client.smembers(index_key)

                for api_key in api_keys:
                    key_redis_key = self._get_key("data", api_key)
                    key_data_str = self.redis_client.get(key_redis_key)

                    if key_data_str:
                        key_data = json.loads(key_data_str)
                        # Don't return the actual key, just metadata
                        safe_data = key_data.copy()
                        safe_data['api_key_prefix'] = api_key[:8] + "..."
                        keys.append(safe_data)

        except Exception as e:
            logger.error(f"Failed to list API keys: {e}")

        return keys

# Global API key manager
api_key_manager = APIKeyManager()

def get_api_key_from_header(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Extract API key from X-API-Key header"""
    return x_api_key

async def validate_api_key_auth(request: Request, api_key: Optional[str] = Depends(get_api_key_from_header)) -> Optional[Dict[str, Any]]:
    """Validate API key authentication"""
    if not api_key:
        return None

    # Validate the API key
    key_data = api_key_manager.validate_api_key(api_key)

    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # Log API key usage
    client_ip = request.headers.get("X-Real-IP") or \
               request.headers.get("X-Forwarded-For", "").split(",")[0] or \
               getattr(request.client, 'host', 'unknown')

    logger.info(f"🔑 API Key used: {key_data['name']} from {client_ip}")

    return key_data

async def require_api_key_permission(permission: str):
    """Dependency to require specific API key permission"""
    async def permission_checker(key_data: Dict[str, Any] = Depends(validate_api_key_auth)):
        if not key_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )

        if permission not in key_data.get('permissions', []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key missing permission: {permission}"
            )

        return key_data

    return permission_checker

async def get_api_key_or_jwt_auth(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key_from_header)
) -> Dict[str, Any]:
    """Allow either API key or JWT authentication"""

    # Try API key first
    if api_key:
        key_data = api_key_manager.validate_api_key(api_key)
        if key_data:
            return {
                'auth_type': 'api_key',
                'user': key_data['name'],
                'role': key_data['role'],
                'permissions': key_data['permissions'],
                'key_data': key_data
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

    # Fallback to JWT auth
    try:
        from src.middleware.auth_middleware import get_current_user
        user_data = await get_current_user(request)
        return {
            'auth_type': 'jwt',
            'user': user_data.get('user'),
            'role': user_data.get('role'),
            'permissions': ['read', 'write'] if user_data.get('role') == 'admin' else ['read'],
            'user_data': user_data
        }
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required (JWT or API key)",
            headers={"WWW-Authenticate": "Bearer, ApiKey"}
        )

def create_emergency_api_key(name: str = "Emergency Key") -> str:
    """Create emergency API key for system recovery"""
    result = api_key_manager.generate_api_key(
        name=name,
        role="admin",
        permissions=["read", "write", "admin"],
        created_by="system"
    )
    return result['api_key']

async def get_api_key_status() -> Dict[str, Any]:
    """Get API key system status"""
    return {
        "api_key_system_available": api_key_manager.redis_available,
        "static_keys_loaded": len(api_key_manager.static_keys) if hasattr(api_key_manager, 'static_keys') else 0,
        "key_prefix": API_KEY_PREFIX,
        "key_expiry_days": KEY_EXPIRY_DAYS,
        "emergency_key_generation": True
    }