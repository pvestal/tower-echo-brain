#!/usr/bin/env python3
"""
Database Connection Pool for AI Assist Board of Directors
Implements secure, efficient database connections with proper resource management
"""

import os
import logging
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import threading
from typing import Optional

logger = logging.getLogger(__name__)

class DatabasePool:
    """
    Thread-safe database connection pool with proper resource management
    """

    def __init__(self, min_conn: int = 1, max_conn: int = 20):
        """
        Initialize the database connection pool

        Args:
            min_conn: Minimum number of connections to maintain
            max_conn: Maximum number of connections allowed
        """
        self._pool = None
        self._lock = threading.Lock()
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool with proper error handling"""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                self.min_conn,
                self.max_conn,
                host=os.environ.get("DB_HOST", "localhost"),
                database=os.environ.get("DB_NAME", "echo_brain"),
                user=os.environ.get("DB_USER", os.getenv("TOWER_USER", "patrick")),
                password=os.environ.get("DB_PASSWORD", ""),
                port=int(os.environ.get("DB_PORT", "5432")),
                # Connection parameters for better reliability
                connect_timeout=10,
                application_name="echo_brain_board",
                # Security: SSL settings
                sslmode=os.environ.get("DB_SSLMODE", "prefer")
            )
            logger.info(f"Database pool initialized with {self.min_conn}-{self.max_conn} connections")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool with automatic cleanup

        Yields:
            psycopg2.connection: Database connection

        Raises:
            psycopg2.Error: Database connection or operation errors
        """
        conn = None
        try:
            with self._lock:
                if self._pool is None:
                    raise Exception("Database pool not initialized")
                conn = self._pool.getconn()

            if conn is None:
                raise Exception("Failed to get connection from pool")

            # Test connection health
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")

            yield conn
            conn.commit()

        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass  # Connection might be closed
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                try:
                    with self._lock:
                        if self._pool:
                            self._pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Failed to return connection to pool: {e}")

    @contextmanager
    def get_cursor(self, dict_cursor: bool = False):
        """
        Get a cursor with automatic connection and transaction management

        Args:
            dict_cursor: If True, returns a DictCursor for named column access

        Yields:
            psycopg2.cursor: Database cursor
        """
        with self.get_connection() as conn:
            cursor_factory = psycopg2.extras.DictCursor if dict_cursor else None
            with conn.cursor(cursor_factory=cursor_factory) as cursor:
                yield cursor

    def execute_query(self, query: str, params: tuple = None, fetch: str = None) -> Optional[list]:
        """
        Execute a query with proper parameter sanitization

        Args:
            query: SQL query with %s placeholders
            params: Parameters to substitute safely
            fetch: 'one', 'all', or None for no fetch

        Returns:
            Query results or None
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())

            if fetch == 'one':
                return cursor.fetchone()
            elif fetch == 'all':
                return cursor.fetchall()
            else:
                return None

    def execute_dict_query(self, query: str, params: tuple = None, fetch: str = None) -> Optional[list]:
        """
        Execute a query returning dictionary results

        Args:
            query: SQL query with %s placeholders
            params: Parameters to substitute safely
            fetch: 'one', 'all', or None for no fetch

        Returns:
            Query results as dictionaries or None
        """
        with self.get_cursor(dict_cursor=True) as cursor:
            cursor.execute(query, params or ())

            if fetch == 'one':
                return cursor.fetchone()
            elif fetch == 'all':
                return cursor.fetchall()
            else:
                return None

    def health_check(self) -> bool:
        """
        Check if the database pool is healthy

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_pool_status(self) -> dict:
        """
        Get current pool statistics

        Returns:
            dict: Pool status information
        """
        if not self._pool:
            return {"status": "not_initialized"}

        try:
            return {
                "status": "active",
                "min_connections": self.min_conn,
                "max_connections": self.max_conn,
                "current_connections": len(self._pool._pool),
                "available_connections": len([c for c in self._pool._pool if not c.closed])
            }
        except Exception as e:
            logger.error(f"Failed to get pool status: {e}")
            return {"status": "error", "error": str(e)}

    def close_all_connections(self):
        """
        Close all connections in the pool (for cleanup)
        """
        try:
            with self._lock:
                if self._pool:
                    self._pool.closeall()
                    self._pool = None
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global pool instance with lazy initialization
_db_pool: Optional[DatabasePool] = None
_pool_lock = threading.Lock()

def get_db_pool() -> DatabasePool:
    """
    Get the global database pool instance (singleton pattern)

    Returns:
        DatabasePool: The global database pool
    """
    global _db_pool

    if _db_pool is None:
        with _pool_lock:
            if _db_pool is None:
                _db_pool = DatabasePool()

    return _db_pool

def close_db_pool():
    """
    Close the global database pool
    """
    global _db_pool

    if _db_pool:
        with _pool_lock:
            if _db_pool:
                _db_pool.close_all_connections()
                _db_pool = None

# Convenience functions for direct usage
def execute_query(query: str, params: tuple = None, fetch: str = None):
    """Execute query using the global pool"""
    return get_db_pool().execute_query(query, params, fetch)

def execute_dict_query(query: str, params: tuple = None, fetch: str = None):
    """Execute query returning dictionaries using the global pool"""
    return get_db_pool().execute_dict_query(query, params, fetch)

def get_connection():
    """Get connection from the global pool"""
    return get_db_pool().get_connection()

def get_cursor(dict_cursor: bool = False):
    """Get cursor from the global pool"""
    return get_db_pool().get_cursor(dict_cursor)