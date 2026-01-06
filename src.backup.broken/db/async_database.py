#!/usr/bin/env python3
"""
Async Database operations for Echo Brain system using AsyncPG Connection Pool

This replaces the old psycopg2 direct connection patterns with enterprise-grade
async connection pooling, query optimization, and monitoring.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from .pool_manager import get_pool, execute_query, execute_one, execute_command
from .query_optimizer import analyze_query, detect_n_plus_one
from ..interfaces.database import AsyncDatabaseInterface, ConversationDatabaseInterface

logger = logging.getLogger(__name__)

class AsyncEchoDatabase(AsyncDatabaseInterface, ConversationDatabaseInterface):
    """
    Unified async database manager for Echo Brain learning with connection pooling
    """

    def __init__(self):
        self._pool = None
        self._optimizer_enabled = True
        self._query_cache = {}
        logger.info("✅ AsyncEchoDatabase initialized with connection pooling")

    async def initialize(self):
        """Initialize the database connection pool"""
        try:
            self._pool = await get_pool()
            # Ensure tables exist
            await self.create_tables_if_needed()
            logger.info("✅ AsyncEchoDatabase pool initialized and tables verified")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize AsyncEchoDatabase: {e}")
            return False

    async def log_interaction(
        self,
        query: str,
        response: str,
        model_used: str,
        processing_time: float,
        escalation_path: List[str],
        conversation_id: Optional[str] = None,
        user_id: str = "default",
        intent: Optional[str] = None,
        confidence: float = 0.0,
        requires_clarification: bool = False,
        clarifying_questions: Optional[List[str]] = None,
        complexity_score: Optional[float] = None,
        tier: Optional[str] = None
    ) -> bool:
        """Log interaction for learning improvement with async pooling and retry logic"""

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Build metadata with complexity info
                metadata = {}
                if complexity_score is not None:
                    metadata["complexity_score"] = complexity_score
                if tier is not None:
                    metadata["tier"] = tier

                # Use async pool for database operation
                sql = """
                    INSERT INTO echo_unified_interactions
                    (query, response, model_used, processing_time, escalation_path,
                     conversation_id, user_id, intent, confidence, requires_clarification,
                     clarifying_questions, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """

                await execute_command(
                    sql,
                    query,
                    response,
                    model_used,
                    processing_time,
                    json.dumps(escalation_path),
                    conversation_id or "",
                    user_id,
                    intent or "",
                    confidence,
                    requires_clarification,
                    json.dumps(clarifying_questions or []),
                    json.dumps(metadata),
                    timeout=30.0
                )

                # Log success
                logger.info(f"✅ Interaction logged: conversation_id={conversation_id}, intent={intent}")

                # Optional query optimization analysis
                if self._optimizer_enabled and attempt == 0:
                    try:
                        analysis = await analyze_query(sql, processing_time)
                        if analysis.optimization_suggestions:
                            logger.info(f"💡 Query optimization suggestions available for interaction logging")
                    except Exception:
                        pass  # Don't fail on optimization analysis

                return True

            except Exception as e:
                logger.warning(f"⚠️ Database log attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"❌ CRITICAL: Interaction logging FAILED after {max_retries} attempts. Conversation lost: {conversation_id}")
                    return False

        return False

    async def create_tables_if_needed(self):
        """Create unified Echo tables with async pool"""
        try:
            # Create main interactions table
            await execute_command("""
                CREATE TABLE IF NOT EXISTS echo_unified_interactions (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(100),
                    user_id VARCHAR(100) DEFAULT 'default',
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model_used VARCHAR(100) NOT NULL,
                    processing_time FLOAT NOT NULL,
                    escalation_path JSONB,
                    intent VARCHAR(50),
                    confidence FLOAT,
                    requires_clarification BOOLEAN DEFAULT FALSE,
                    clarifying_questions JSONB,
                    metadata JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create conversations tracking table
            await execute_command("""
                CREATE TABLE IF NOT EXISTS echo_conversations (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(100) UNIQUE NOT NULL,
                    user_id VARCHAR(100) DEFAULT 'default',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    intent_history JSONB,
                    context JSONB
                )
            """)

            # Create performance indexes for better query performance
            await execute_command("""
                CREATE INDEX IF NOT EXISTS idx_interactions_conversation_id
                ON echo_unified_interactions(conversation_id)
            """)

            await execute_command("""
                CREATE INDEX IF NOT EXISTS idx_interactions_user_id_timestamp
                ON echo_unified_interactions(user_id, timestamp DESC)
            """)

            await execute_command("""
                CREATE INDEX IF NOT EXISTS idx_interactions_intent_timestamp
                ON echo_unified_interactions(intent, timestamp DESC)
            """)

            # Create query performance monitoring table
            await execute_command("""
                CREATE TABLE IF NOT EXISTS echo_query_performance (
                    id SERIAL PRIMARY KEY,
                    query_hash VARCHAR(32) UNIQUE NOT NULL,
                    query_sample TEXT NOT NULL,
                    execution_count INTEGER DEFAULT 1,
                    total_execution_time FLOAT DEFAULT 0.0,
                    avg_execution_time FLOAT DEFAULT 0.0,
                    max_execution_time FLOAT DEFAULT 0.0,
                    min_execution_time FLOAT DEFAULT 0.0,
                    last_executed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    optimization_suggestions JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            logger.info("✅ Echo unified database tables and indexes created/verified")

        except Exception as e:
            logger.error(f"❌ Database table creation failed: {e}")
            raise

    async def get_conversation_history(self, conversation_id: str, limit: int = 100) -> List[Dict]:
        """Get conversation history for a specific conversation with caching"""
        try:
            cache_key = f"conversation_history:{conversation_id}:{limit}"

            result = await execute_query(
                """
                SELECT query, response, model_used, processing_time, timestamp, intent, confidence
                FROM echo_unified_interactions
                WHERE conversation_id = $1
                ORDER BY timestamp ASC
                LIMIT $2
                """,
                conversation_id,
                limit,
                cache_key=cache_key,
                cache_ttl=300  # 5 minute cache
            )

            if result.data:
                return [
                    {
                        "query": row["query"],
                        "response": row["response"],
                        "model_used": row["model_used"],
                        "processing_time": row["processing_time"],
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                        "intent": row["intent"],
                        "confidence": row["confidence"]
                    }
                    for row in result.data
                ]
            else:
                return []

        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            return []

    async def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all conversations for a user with performance optimization"""
        try:
            cache_key = f"user_conversations:{user_id}:{limit}"

            result = await execute_query(
                """
                SELECT DISTINCT conversation_id, MIN(timestamp) as first_interaction,
                       MAX(timestamp) as last_interaction, COUNT(*) as interaction_count
                FROM echo_unified_interactions
                WHERE user_id = $1
                GROUP BY conversation_id
                ORDER BY last_interaction DESC
                LIMIT $2
                """,
                user_id,
                limit,
                cache_key=cache_key,
                cache_ttl=600  # 10 minute cache
            )

            if result.data:
                return [
                    {
                        "conversation_id": row["conversation_id"],
                        "first_interaction": row["first_interaction"].isoformat() if row["first_interaction"] else None,
                        "last_interaction": row["last_interaction"].isoformat() if row["last_interaction"] else None,
                        "interaction_count": row["interaction_count"]
                    }
                    for row in result.data
                ]
            else:
                return []

        except Exception as e:
            logger.error(f"❌ Failed to get user conversations: {e}")
            return []

    async def get_interaction_analytics(
        self,
        user_id: str = None,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get interaction analytics with performance insights"""
        try:
            where_clause = ""
            params = [hours_back]

            if user_id:
                where_clause = "AND user_id = $2"
                params.append(user_id)

            result = await execute_query(
                f"""
                SELECT
                    COUNT(*) as total_interactions,
                    AVG(processing_time) as avg_processing_time,
                    MAX(processing_time) as max_processing_time,
                    COUNT(DISTINCT conversation_id) as unique_conversations,
                    COUNT(DISTINCT model_used) as models_used,
                    array_agg(DISTINCT intent) as intents_used
                FROM echo_unified_interactions
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                {where_clause}
                """,
                *params,
                cache_key=f"analytics:{user_id}:{hours_back}",
                cache_ttl=300
            )

            if result.data and result.data[0]:
                row = result.data[0]
                return {
                    "total_interactions": row["total_interactions"],
                    "avg_processing_time": float(row["avg_processing_time"]) if row["avg_processing_time"] else 0.0,
                    "max_processing_time": float(row["max_processing_time"]) if row["max_processing_time"] else 0.0,
                    "unique_conversations": row["unique_conversations"],
                    "models_used": row["models_used"],
                    "intents_used": [intent for intent in row["intents_used"] if intent]
                }
            else:
                return {
                    "total_interactions": 0,
                    "avg_processing_time": 0.0,
                    "max_processing_time": 0.0,
                    "unique_conversations": 0,
                    "models_used": 0,
                    "intents_used": []
                }

        except Exception as e:
            logger.error(f"❌ Failed to get interaction analytics: {e}")
            return {}

    async def record_query_performance(self, query: str, execution_time: float):
        """Record query performance for monitoring and optimization"""
        try:
            # Analyze query for optimization
            analysis = await analyze_query(query, execution_time)

            # Upsert query performance record
            await execute_command(
                """
                INSERT INTO echo_query_performance
                (query_hash, query_sample, execution_count, total_execution_time,
                 avg_execution_time, max_execution_time, min_execution_time,
                 optimization_suggestions, last_executed)
                VALUES ($1, $2, 1, $3, $3, $3, $3, $4, NOW())
                ON CONFLICT (query_hash) DO UPDATE SET
                    execution_count = echo_query_performance.execution_count + 1,
                    total_execution_time = echo_query_performance.total_execution_time + $3,
                    avg_execution_time = (echo_query_performance.total_execution_time + $3) / (echo_query_performance.execution_count + 1),
                    max_execution_time = GREATEST(echo_query_performance.max_execution_time, $3),
                    min_execution_time = LEAST(echo_query_performance.min_execution_time, $3),
                    last_executed = NOW(),
                    optimization_suggestions = $4
                """,
                analysis.query_hash,
                query[:500],  # Store sample of query
                execution_time,
                json.dumps(analysis.optimization_suggestions)
            )

        except Exception as e:
            logger.warning(f"Failed to record query performance: {e}")

    async def get_slow_queries(self, limit: int = 20, min_execution_time: float = 1.0) -> List[Dict]:
        """Get slowest queries for optimization analysis"""
        try:
            result = await execute_query(
                """
                SELECT query_hash, query_sample, execution_count, avg_execution_time,
                       max_execution_time, optimization_suggestions, last_executed
                FROM echo_query_performance
                WHERE avg_execution_time >= $1
                ORDER BY avg_execution_time DESC
                LIMIT $2
                """,
                min_execution_time,
                limit,
                cache_key=f"slow_queries:{limit}:{min_execution_time}",
                cache_ttl=600
            )

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []

    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old interaction data to maintain performance"""
        try:
            cutoff_date = f"{days_to_keep} days"

            # Delete old interactions
            deleted_count = await execute_command(
                """
                DELETE FROM echo_unified_interactions
                WHERE timestamp < NOW() - INTERVAL %s
                """,
                cutoff_date
            )

            # Clean up orphaned conversations
            await execute_command(
                """
                DELETE FROM echo_conversations
                WHERE conversation_id NOT IN (
                    SELECT DISTINCT conversation_id FROM echo_unified_interactions
                )
                """
            )

            # Clean up old query performance data (keep more for analysis)
            await execute_command(
                """
                DELETE FROM echo_query_performance
                WHERE last_executed < NOW() - INTERVAL %s
                """,
                f"{days_to_keep * 2} days"
            )

            logger.info(f"✅ Cleaned up {deleted_count} old interactions (>{days_to_keep} days)")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def get_pool_health(self) -> Dict[str, Any]:
        """Get connection pool health status"""
        try:
            pool = await get_pool()
            return await pool.health_check()
        except Exception as e:
            logger.error(f"Failed to get pool health: {e}")
            return {"status": "error", "error": str(e)}

    async def optimize_tables(self):
        """Run database optimization tasks"""
        try:
            # Update table statistics
            await execute_command("ANALYZE echo_unified_interactions")
            await execute_command("ANALYZE echo_conversations")
            await execute_command("ANALYZE echo_query_performance")

            # Vacuum if needed (light vacuum)
            await execute_command("VACUUM (ANALYZE) echo_unified_interactions")

            logger.info("✅ Database tables optimized")

        except Exception as e:
            logger.error(f"Failed to optimize tables: {e}")

    # Additional methods required by AsyncDatabaseInterface

    async def close(self) -> bool:
        """Close database connection pool"""
        try:
            if self._pool:
                await self._pool.close()
                self._pool = None
            logger.info("✅ Database connection pool closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close database pool: {e}")
            return False

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute async SELECT query"""
        try:
            if params:
                # Convert dict params to positional for asyncpg
                param_values = list(params.values()) if isinstance(params, dict) else params
                return await execute_query(query, *param_values)
            else:
                return await execute_query(query)
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []

    async def execute_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Execute query and return single result"""
        try:
            if params:
                param_values = list(params.values()) if isinstance(params, dict) else params
                return await execute_one(query, *param_values)
            else:
                return await execute_one(query)
        except Exception as e:
            logger.error(f"Failed to execute query (single result): {e}")
            return None

    async def execute_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Execute async INSERT/UPDATE/DELETE command"""
        try:
            if params:
                param_values = list(params.values()) if isinstance(params, dict) else params
                result = await execute_command(command, *param_values)
                return result is not None
            else:
                result = await execute_command(command)
                return result is not None
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return False

    async def execute_batch(self, commands: List[str], params_list: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Execute multiple commands in a batch"""
        try:
            for i, command in enumerate(commands):
                params = params_list[i] if params_list and i < len(params_list) else None
                success = await self.execute_command(command, params)
                if not success:
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to execute batch commands: {e}")
            return False

    async def transaction(self):
        """Get async transaction context manager"""
        pool = await get_pool()
        return pool.transaction()

    async def get_connection(self):
        """Get a database connection from the pool"""
        return await get_pool()

    async def health_check(self) -> bool:
        """Perform database health check"""
        try:
            result = await self.execute_one("SELECT 1 AS health_check")
            return result is not None and result.get("health_check") == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        try:
            return await self.get_pool_health()
        except Exception as e:
            logger.error(f"Failed to get pool stats: {e}")
            return {"error": str(e)}

    # Additional methods for ConversationDatabaseInterface compliance

    async def create_conversation(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new conversation"""
        try:
            conversation_id = f"conv_{int(time.time() * 1000)}"
            await execute_command(
                """
                INSERT INTO echo_conversations (conversation_id, user_id, title, created_at)
                VALUES ($1, $2, $3, NOW())
                """,
                conversation_id,
                user_id,
                title or f"Conversation {conversation_id}"
            )
            logger.info(f"✅ Created conversation: {conversation_id}")
            return conversation_id
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise Exception("Failed to create conversation")

    async def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title"""
        try:
            await execute_command(
                "UPDATE echo_conversations SET title = $1 WHERE conversation_id = $2",
                title,
                conversation_id
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update conversation title: {e}")
            return False

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its interactions"""
        try:
            # Delete interactions first (foreign key constraint)
            await execute_command(
                "DELETE FROM echo_unified_interactions WHERE conversation_id = $1",
                conversation_id
            )
            # Delete conversation record
            await execute_command(
                "DELETE FROM echo_conversations WHERE conversation_id = $1",
                conversation_id
            )
            logger.info(f"✅ Deleted conversation: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return False

    async def search_conversations(self, user_id: str, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        try:
            results = await execute_query(
                """
                SELECT DISTINCT c.conversation_id, c.title, c.created_at, c.updated_at
                FROM echo_conversations c
                JOIN echo_unified_interactions i ON c.conversation_id = i.conversation_id
                WHERE c.user_id = $1 AND (
                    c.title ILIKE $2 OR
                    i.query ILIKE $2 OR
                    i.response ILIKE $2
                )
                ORDER BY c.updated_at DESC
                LIMIT $3
                """,
                user_id,
                f"%{search_term}%",
                limit
            )
            return results or []
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []


# Global database instance with lazy initialization
_async_database: Optional[AsyncEchoDatabase] = None

async def get_async_database() -> AsyncEchoDatabase:
    """Get or create the global async database instance"""
    global _async_database

    if _async_database is None:
        _async_database = AsyncEchoDatabase()
        await _async_database.initialize()

    return _async_database

# Convenience functions for easy migration
async def log_interaction(*args, **kwargs) -> bool:
    """Log interaction using async database"""
    db = await get_async_database()
    return await db.log_interaction(*args, **kwargs)

async def get_conversation_history(conversation_id: str, limit: int = 100) -> List[Dict]:
    """Get conversation history using async database"""
    db = await get_async_database()
    return await db.get_conversation_history(conversation_id, limit)

async def get_user_conversations(user_id: str, limit: int = 50) -> List[Dict]:
    """Get user conversations using async database"""
    db = await get_async_database()
    return await db.get_user_conversations(user_id, limit)

# Migration utility function
async def migrate_from_sync_to_async():
    """Migrate any existing sync database usage to async"""
    logger.info("🔄 Migrating to async database with connection pooling...")

    # Initialize async database
    db = await get_async_database()
    health = await db.get_pool_health()

    if health["status"] == "healthy":
        logger.info("✅ Async database migration successful - connection pool active")
        return True
    else:
        logger.error(f"❌ Async database migration failed: {health}")
        return False