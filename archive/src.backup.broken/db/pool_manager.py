#!/usr/bin/env python3
"""
AsyncPG Connection Pool Manager for Echo Brain System

Enterprise-grade database connection pooling with:
- Async/await support for FastAPI
- Health monitoring and connection recycling
- Performance metrics and observability
- Automatic connection recovery
- Query optimization and caching
"""

import asyncio
import asyncpg
import logging
import os
import time
import json
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass, field
import weakref
import psutil

# Import secure credential manager
try:
    from ..security.credential_validator import get_secure_db_config
except ImportError:
    # Fallback for development environments
    def get_secure_db_config():
        logger.warning("⚠️ Secure credential validator not available, using basic fallback")
        db_password = os.environ.get("DB_PASSWORD")
        if not db_password:
            raise RuntimeError("Database password not configured. Set DB_PASSWORD environment variable.")
        return {
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick"),
            "host": os.environ.get("DB_HOST", "localhost"),
            "password": db_password,
            "port": int(os.environ.get("DB_PORT", 5432))
        }

logger = logging.getLogger(__name__)

@dataclass
class PoolMetrics:
    """Connection pool performance metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_queries: int = 0
    total_query_time: float = 0.0
    slow_queries: int = 0
    failed_queries: int = 0
    connection_errors: int = 0
    pool_exhausted: int = 0
    last_health_check: Optional[datetime] = None
    queries_per_second: float = 0.0
    avg_query_time: float = 0.0
    connection_recycled: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class QueryResult:
    """Enhanced query result with performance data"""
    data: Any
    execution_time: float
    cached: bool = False
    query_id: Optional[str] = None

class QueryCache:
    """Simple query result cache with TTL"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry['expires']:
                    return entry['data']
                else:
                    del self._cache[key]
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        async with self._lock:
            # Simple LRU eviction
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k]['created'])
                del self._cache[oldest_key]

            self._cache[key] = {
                'data': value,
                'created': time.time(),
                'expires': time.time() + (ttl or self.default_ttl)
            }

    async def invalidate(self, pattern: str = None):
        """Invalidate cache entries matching pattern"""
        async with self._lock:
            if pattern:
                keys_to_remove = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self._cache[key]
            else:
                self._cache.clear()

class AsyncConnectionPool:
    """
    Enterprise-grade AsyncPG connection pool with monitoring and optimization
    """

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 50,
        command_timeout: float = 60.0,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0,
        enable_cache: bool = True,
        cache_ttl: int = 300
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime

        self._pool: Optional[asyncpg.Pool] = None
        self._db_config = None
        self._metrics = PoolMetrics()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

        # Query optimization features
        self._cache = QueryCache() if enable_cache else None
        self._slow_query_threshold = 1.0  # Log queries slower than 1 second
        self._query_stats: Dict[str, Dict] = {}

        # Performance monitoring
        self._start_time = time.time()
        self._query_count_last_minute = []

    async def initialize(self) -> bool:
        """Initialize the connection pool with secure credentials"""
        async with self._lock:
            if self._initialized:
                return True

            try:
                # Get secure database configuration
                self._db_config = get_secure_db_config()
                logger.info("✅ Database configuration loaded securely")

                # Create connection pool
                self._pool = await asyncpg.create_pool(
                    database=self._db_config["database"],
                    user=self._db_config["user"],
                    password=self._db_config["password"],
                    host=self._db_config["host"],
                    port=self._db_config["port"],
                    min_size=self.min_size,
                    max_size=self.max_size,
                    command_timeout=self.command_timeout,
                    max_queries=self.max_queries,
                    max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
                    # Connection setup
                    init=self._init_connection,
                    # Server settings for performance
                    server_settings={
                        'application_name': 'echo_brain_asyncpg_pool',
                        'tcp_keepalives_idle': '600',
                        'tcp_keepalives_interval': '30',
                        'tcp_keepalives_count': '3',
                    }
                )

                self._initialized = True
                logger.info(f"✅ AsyncPG pool initialized: {self.min_size}-{self.max_size} connections")

                # Start background tasks
                await self._start_background_tasks()

                return True

            except Exception as e:
                logger.error(f"❌ Failed to initialize connection pool: {e}")
                self._initialized = False
                return False

    async def _init_connection(self, connection: asyncpg.Connection):
        """Initialize each connection with performance optimizations"""
        try:
            # Set connection-level optimizations
            await connection.execute("SET synchronous_commit = off")
            await connection.execute("SET wal_writer_delay = '200ms'")
            await connection.execute("SET checkpoint_completion_target = 0.9")

            # Register custom types if needed
            # await connection.set_type_codec('jsonb', encoder=json.dumps, decoder=json.loads, schema='pg_catalog')

        except Exception as e:
            logger.warning(f"Connection initialization warning: {e}")

    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        if not self._health_check_task or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())

        if not self._metrics_task or self._metrics_task.done():
            self._metrics_task = asyncio.create_task(self._metrics_loop())

    async def _health_check_loop(self):
        """Periodic health check and connection maintenance"""
        while self._initialized:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _metrics_loop(self):
        """Periodic metrics calculation and cleanup"""
        while self._initialized:
            try:
                await asyncio.sleep(60)  # Update metrics every minute
                await self._update_metrics()
                await self._cleanup_query_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")

    async def _update_metrics(self):
        """Update performance metrics"""
        now = time.time()

        # Calculate queries per second
        self._query_count_last_minute = [
            t for t in self._query_count_last_minute if now - t < 60
        ]
        self._metrics.queries_per_second = len(self._query_count_last_minute) / 60.0

        # Calculate average query time
        if self._metrics.total_queries > 0:
            self._metrics.avg_query_time = self._metrics.total_query_time / self._metrics.total_queries

        # Update pool status
        if self._pool:
            self._metrics.total_connections = self._pool.get_size()
            self._metrics.idle_connections = self._pool.get_idle_size()
            self._metrics.active_connections = self._metrics.total_connections - self._metrics.idle_connections

    async def _cleanup_query_stats(self):
        """Clean up old query statistics"""
        cutoff = time.time() - 3600  # Keep last hour
        self._query_stats = {
            k: v for k, v in self._query_stats.items()
            if v.get('last_seen', 0) > cutoff
        }

    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire a connection from the pool with automatic cleanup"""
        if not self._initialized:
            await self.initialize()

        connection = None
        try:
            connection = await self._pool.acquire()
            yield connection

        except asyncpg.PoolAcquireTimeoutError:
            self._metrics.pool_exhausted += 1
            logger.warning("Connection pool exhausted")
            raise
        except Exception as e:
            self._metrics.connection_errors += 1
            logger.error(f"Connection error: {e}")
            raise
        finally:
            if connection:
                try:
                    await self._pool.release(connection)
                except Exception as e:
                    logger.error(f"Error releasing connection: {e}")

    async def execute_query(
        self,
        query: str,
        *args,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> QueryResult:
        """
        Execute a query with caching, monitoring, and optimization
        """
        query_start = time.time()
        query_id = f"query_{int(query_start * 1000)}"

        try:
            # Check cache first
            if cache_key and self._cache:
                cached_result = await self._cache.get(cache_key)
                if cached_result is not None:
                    self._metrics.cache_hits += 1
                    return QueryResult(
                        data=cached_result,
                        execution_time=0.001,  # Minimal cache lookup time
                        cached=True,
                        query_id=query_id
                    )
                else:
                    self._metrics.cache_misses += 1

            # Execute query
            async with self.acquire_connection() as connection:
                if timeout:
                    result = await asyncio.wait_for(
                        connection.fetch(query, *args),
                        timeout=timeout
                    )
                else:
                    result = await connection.fetch(query, *args)

            execution_time = time.time() - query_start

            # Convert asyncpg Records to dict for JSON serialization
            if result:
                data = [dict(record) for record in result]
            else:
                data = []

            # Cache result if requested
            if cache_key and self._cache:
                await self._cache.set(cache_key, data, cache_ttl)

            # Update metrics
            self._metrics.total_queries += 1
            self._metrics.total_query_time += execution_time
            self._query_count_last_minute.append(query_start)

            # Log slow queries
            if execution_time > self._slow_query_threshold:
                self._metrics.slow_queries += 1
                logger.warning(
                    f"Slow query detected: {execution_time:.3f}s - {query[:100]}..."
                )

            # Update query statistics
            query_hash = str(hash(query))
            if query_hash in self._query_stats:
                stats = self._query_stats[query_hash]
                stats['count'] += 1
                stats['total_time'] += execution_time
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['last_seen'] = query_start
            else:
                self._query_stats[query_hash] = {
                    'query': query[:200] + "..." if len(query) > 200 else query,
                    'count': 1,
                    'total_time': execution_time,
                    'avg_time': execution_time,
                    'first_seen': query_start,
                    'last_seen': query_start
                }

            return QueryResult(
                data=data,
                execution_time=execution_time,
                cached=False,
                query_id=query_id
            )

        except Exception as e:
            self._metrics.failed_queries += 1
            execution_time = time.time() - query_start
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")
            raise

    async def execute_one(self, query: str, *args, **kwargs) -> QueryResult:
        """Execute query expecting single result"""
        result = await self.execute_query(query, *args, **kwargs)
        if result.data:
            result.data = result.data[0] if result.data else None
        return result

    async def execute_command(self, query: str, *args, timeout: Optional[float] = None) -> int:
        """Execute INSERT/UPDATE/DELETE command, return affected rows"""
        query_start = time.time()

        try:
            async with self.acquire_connection() as connection:
                if timeout:
                    result = await asyncio.wait_for(
                        connection.execute(query, *args),
                        timeout=timeout
                    )
                else:
                    result = await connection.execute(query, *args)

            execution_time = time.time() - query_start

            # Update metrics
            self._metrics.total_queries += 1
            self._metrics.total_query_time += execution_time
            self._query_count_last_minute.append(query_start)

            # Extract affected row count from result
            if result.startswith('INSERT'):
                return int(result.split()[-1])
            elif result.startswith('UPDATE'):
                return int(result.split()[-1])
            elif result.startswith('DELETE'):
                return int(result.split()[-1])
            else:
                return 0

        except Exception as e:
            self._metrics.failed_queries += 1
            logger.error(f"Command failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_data = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "pool_initialized": self._initialized,
            "connection_test": False,
            "metrics": None,
            "errors": []
        }

        try:
            if not self._initialized:
                health_data["status"] = "not_initialized"
                return health_data

            # Test connection
            async with self.acquire_connection() as connection:
                result = await connection.fetchval("SELECT 1")
                health_data["connection_test"] = (result == 1)

            # Pool metrics
            if self._pool:
                pool_size = self._pool.get_size()
                idle_size = self._pool.get_idle_size()

                health_data["metrics"] = {
                    "total_connections": pool_size,
                    "active_connections": pool_size - idle_size,
                    "idle_connections": idle_size,
                    "total_queries": self._metrics.total_queries,
                    "queries_per_second": self._metrics.queries_per_second,
                    "avg_query_time": self._metrics.avg_query_time,
                    "slow_queries": self._metrics.slow_queries,
                    "failed_queries": self._metrics.failed_queries,
                    "pool_exhausted": self._metrics.pool_exhausted,
                    "cache_hit_ratio": (
                        self._metrics.cache_hits /
                        max(1, self._metrics.cache_hits + self._metrics.cache_misses)
                    ) if self._cache else 0.0
                }

            self._metrics.last_health_check = datetime.now()
            health_data["status"] = "healthy"

        except Exception as e:
            health_data["status"] = "unhealthy"
            health_data["errors"].append(str(e))
            logger.error(f"Health check failed: {e}")

        return health_data

    async def get_query_stats(self, limit: int = 20) -> List[Dict]:
        """Get top query statistics"""
        sorted_stats = sorted(
            self._query_stats.values(),
            key=lambda x: x['avg_time'],
            reverse=True
        )
        return sorted_stats[:limit]

    async def invalidate_cache(self, pattern: str = None):
        """Invalidate query cache"""
        if self._cache:
            await self._cache.invalidate(pattern)

    async def close(self):
        """Clean shutdown of the connection pool"""
        self._initialized = False

        # Cancel background tasks
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()

        # Close pool
        if self._pool:
            await self._pool.close()
            logger.info("✅ AsyncPG connection pool closed")


# Global pool instance (singleton)
_global_pool: Optional[AsyncConnectionPool] = None
_pool_lock = asyncio.Lock()

async def get_pool(
    min_size: int = None,
    max_size: int = None,
    **kwargs
) -> AsyncConnectionPool:
    """
    Get or create the global connection pool instance
    """
    global _global_pool

    async with _pool_lock:
        if _global_pool is None or not _global_pool._initialized:
            # Get configuration from environment
            pool_min = min_size or int(os.environ.get("DB_POOL_MIN_SIZE", "5"))
            pool_max = max_size or int(os.environ.get("DB_POOL_MAX_SIZE", "50"))

            _global_pool = AsyncConnectionPool(
                min_size=pool_min,
                max_size=pool_max,
                **kwargs
            )

            if not await _global_pool.initialize():
                raise RuntimeError("Failed to initialize connection pool")

    return _global_pool

async def close_global_pool():
    """Close the global connection pool"""
    global _global_pool

    if _global_pool:
        async with _pool_lock:
            if _global_pool:
                await _global_pool.close()
                _global_pool = None

# Convenience functions for direct usage
async def execute_query(query: str, *args, **kwargs):
    """Execute query using global pool"""
    pool = await get_pool()
    return await pool.execute_query(query, *args, **kwargs)

async def execute_one(query: str, *args, **kwargs):
    """Execute query expecting single result using global pool"""
    pool = await get_pool()
    return await pool.execute_one(query, *args, **kwargs)

async def execute_command(query: str, *args, **kwargs):
    """Execute command using global pool"""
    pool = await get_pool()
    return await pool.execute_command(query, *args, **kwargs)

async def health_check():
    """Get health status of global pool"""
    pool = await get_pool()
    return await pool.health_check()

async def get_query_stats(limit: int = 20):
    """Get query statistics from global pool"""
    pool = await get_pool()
    return await pool.get_query_stats(limit)

# Context manager for connection acquisition
async def acquire_connection():
    """Acquire connection from global pool"""
    pool = await get_pool()
    return pool.acquire_connection()