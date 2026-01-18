#!/usr/bin/env python3
"""
Centralized async database connection pooling for Echo Brain
Prevents connection exhaustion and improves performance
"""

import asyncpg
import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DatabasePool:
    """Singleton database connection pool manager"""

    _instance: Optional['DatabasePool'] = None
    _pool: Optional[asyncpg.Pool] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self, **kwargs):
        """Initialize the connection pool"""
        async with self._lock:
            if self._pool is None:
                # Default configuration
                config = {
                    'host': 'localhost',
                    'port': 5432,
                    'database': kwargs.get('database', 'echo_brain'),
                    'user': kwargs.get('user', 'patrick'),
                    'password': kwargs.get('password', 'RP78eIrW7cI2jYvL5akt1yurE'),
                    'min_size': kwargs.get('min_size', 5),
                    'max_size': kwargs.get('max_size', 20),
                    'max_queries': kwargs.get('max_queries', 50000),
                    'max_inactive_connection_lifetime': kwargs.get('max_inactive_lifetime', 300.0),
                    'command_timeout': kwargs.get('command_timeout', 60),
                }

                try:
                    logger.info(f"Initializing database pool: {config['database']}@{config['host']}")
                    self._pool = await asyncpg.create_pool(**config)
                    logger.info(f"✅ Database pool initialized with {config['min_size']}-{config['max_size']} connections")
                except Exception as e:
                    logger.error(f"Failed to initialize database pool: {e}")
                    raise

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if self._pool is None:
            await self.initialize()

        async with self._pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args, timeout: float = None):
        """Execute a query using a pooled connection"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetch(self, query: str, *args, timeout: float = None):
        """Fetch results using a pooled connection"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query: str, *args, timeout: float = None):
        """Fetch a single row using a pooled connection"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def fetchval(self, query: str, *args, column: int = 0, timeout: float = None):
        """Fetch a single value using a pooled connection"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def close(self):
        """Close the connection pool"""
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
                logger.info("Database pool closed")

    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized"""
        return self._pool is not None

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        if self._pool:
            return {
                'size': self._pool.get_size(),
                'min_size': self._pool.get_min_size(),
                'max_size': self._pool.get_max_size(),
                'free_connections': self._pool.get_idle_size(),
                'used_connections': self._pool.get_size() - self._pool.get_idle_size(),
            }
        return {}

# Global pool instance
db_pool = DatabasePool()

# Convenience functions for backward compatibility
async def get_pool() -> DatabasePool:
    """Get the global database pool instance"""
    if not db_pool.is_initialized:
        await db_pool.initialize()
    return db_pool

async def execute(query: str, *args, **kwargs):
    """Execute a query using the global pool"""
    pool = await get_pool()
    return await pool.execute(query, *args, **kwargs)

async def fetch(query: str, *args, **kwargs):
    """Fetch results using the global pool"""
    pool = await get_pool()
    return await pool.fetch(query, *args, **kwargs)

async def fetchrow(query: str, *args, **kwargs):
    """Fetch a single row using the global pool"""
    pool = await get_pool()
    return await pool.fetchrow(query, *args, **kwargs)

async def fetchval(query: str, *args, **kwargs):
    """Fetch a single value using the global pool"""
    pool = await get_pool()
    return await pool.fetchval(query, *args, **kwargs)