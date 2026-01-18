#!/usr/bin/env python3
"""
Enterprise Database Connection Pool Manager
Centralized database connection management for Echo Brain
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncpg
import psycopg2
from psycopg2 import pool
from threading import Lock

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """
    Singleton database connection pool manager
    Provides both async (asyncpg) and sync (psycopg2) connections
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.config = None
            self.async_pool = None
            self.sync_pool = None
            self.is_connected = False

    def _get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration from a single source of truth
        Priority: Environment > Vault > Defaults
        """

        # Try environment variables first
        db_password = os.environ.get("DB_PASSWORD") or os.environ.get("DATABASE_PASSWORD")

        # If no password in environment, try vault
        if not db_password:
            try:
                # Try Tower vault
                import json
                vault_path = "/home/patrick/.tower_credentials/vault.json"
                if os.path.exists(vault_path):
                    with open(vault_path, 'r') as f:
                        vault_data = json.load(f)
                        if "postgresql" in vault_data:
                            db_password = vault_data["postgresql"].get("password")
                            logger.info("🔐 Database password loaded from Tower vault")
            except Exception as e:
                logger.warning(f"⚠️ Could not load from vault: {e}")

        # If still no password, check .env file
        if not db_password:
            try:
                env_path = "/opt/tower-echo-brain/.env"
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.startswith("DB_PASSWORD="):
                                db_password = line.split("=", 1)[1].strip()
                                logger.info("🔐 Database password loaded from .env file")
                                break
            except Exception as e:
                logger.warning(f"⚠️ Could not load from .env: {e}")

        if not db_password:
            raise ValueError("Database password not configured. Set DB_PASSWORD environment variable.")

        config = {
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": int(os.environ.get("DB_PORT", 5432)),
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick"),
            "password": db_password
        }

        logger.info(f"📊 Database config: {config['user']}@{config['host']}:{config['port']}/{config['database']}")
        return config

    async def initialize_async(self):
        """Initialize async connection pool (asyncpg)"""
        if self.async_pool is not None:
            return

        if self.config is None:
            self.config = self._get_database_config()

        try:
            self.async_pool = await asyncpg.create_pool(
                host=self.config["host"],
                port=self.config["port"],
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("✅ Async database pool initialized")
            self.is_connected = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize async pool: {e}")
            raise

    def initialize_sync(self):
        """Initialize sync connection pool (psycopg2)"""
        if self.sync_pool is not None:
            return

        if self.config is None:
            self.config = self._get_database_config()

        try:
            self.sync_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=self.config["host"],
                port=self.config["port"],
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"]
            )
            logger.info("✅ Sync database pool initialized")
            self.is_connected = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize sync pool: {e}")
            raise

    @asynccontextmanager
    async def acquire_async(self):
        """Acquire async connection from pool"""
        if self.async_pool is None:
            await self.initialize_async()

        async with self.async_pool.acquire() as connection:
            yield connection

    def get_sync_connection(self):
        """Get sync connection from pool"""
        if self.sync_pool is None:
            self.initialize_sync()

        return self.sync_pool.getconn()

    def return_sync_connection(self, connection):
        """Return sync connection to pool"""
        if self.sync_pool:
            self.sync_pool.putconn(connection)

    def get_config_for_legacy(self) -> Dict[str, Any]:
        """
        Get config dict for legacy code that expects psycopg2 config
        """
        if self.config is None:
            self.config = self._get_database_config()

        return {
            "host": self.config["host"],
            "port": self.config["port"],
            "database": self.config["database"],
            "user": self.config["user"],
            "password": self.config["password"]
        }

    async def close_async(self):
        """Close async pool"""
        if self.async_pool:
            await self.async_pool.close()
            self.async_pool = None
            logger.info("🛑 Async database pool closed")

    def close_sync(self):
        """Close sync pool"""
        if self.sync_pool:
            self.sync_pool.closeall()
            self.sync_pool = None
            logger.info("🛑 Sync database pool closed")

    async def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            async with self.acquire_async() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

# Global connection pool instance
_connection_pool = None

def get_connection_pool() -> DatabaseConnectionPool:
    """Get the global connection pool instance"""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = DatabaseConnectionPool()
    return _connection_pool

async def initialize_database():
    """Initialize the database connection pool"""
    pool = get_connection_pool()
    await pool.initialize_async()
    pool.initialize_sync()

    # Test the connection
    if await pool.test_connection():
        logger.info("✅ Database connection pool ready")
    else:
        raise ConnectionError("Failed to establish database connection")

def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration for legacy components
    This ensures all components use the same configuration
    """
    pool = get_connection_pool()
    return pool.get_config_for_legacy()

# Context manager for sync connections
class SyncConnection:
    """Context manager for sync database connections"""

    def __init__(self):
        self.pool = get_connection_pool()
        self.connection = None

    def __enter__(self):
        self.connection = self.pool.get_sync_connection()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.pool.return_sync_connection(self.connection)