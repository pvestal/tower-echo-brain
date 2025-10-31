#!/usr/bin/env python3
"""
Async Database operations for Echo Brain system using asyncpg
Replaces synchronous psycopg2 implementation for better performance
"""

import asyncpg
import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class AsyncEchoDatabase:
    """Async database manager for Echo learning with connection pooling"""

    def __init__(self):
        # SECURITY: Use environment variables for database credentials
        self.db_config = {
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick"),
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": int(os.environ.get("DB_PORT", "5432"))
        }
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the connection pool"""
        if self._initialized:
            return

        try:
            self.pool = await asyncpg.create_pool(
                database=self.db_config["database"],
                user=self.db_config["user"],
                host=self.db_config["host"],
                port=self.db_config["port"],
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'application_name': 'echo_brain_async',
                }
            )
            self._initialized = True
            logger.info("✅ Async database pool initialized")

            # Create tables if needed
            await self.create_tables_if_needed()

        except Exception as e:
            logger.error(f"❌ Failed to initialize async database pool: {e}")
            raise

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("Database pool closed")

    async def create_tables_if_needed(self):
        """Create tables if they don't exist"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            # Check if echo_unified_interactions table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'echo_unified_interactions'
                );
            """)

            if not table_exists:
                await conn.execute("""
                    CREATE TABLE echo_unified_interactions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        model_used VARCHAR(100),
                        processing_time FLOAT,
                        escalation_path JSONB,
                        conversation_id VARCHAR(100),
                        user_id VARCHAR(50) DEFAULT 'default',
                        intent VARCHAR(100),
                        confidence FLOAT DEFAULT 0.0,
                        requires_clarification BOOLEAN DEFAULT FALSE,
                        clarifying_questions JSONB,
                        metadata JSONB DEFAULT '{}'::jsonb
                    );
                """)

                # Create indexes for better performance
                await conn.execute("CREATE INDEX idx_interactions_timestamp ON echo_unified_interactions(timestamp);")
                await conn.execute("CREATE INDEX idx_interactions_conversation ON echo_unified_interactions(conversation_id);")
                await conn.execute("CREATE INDEX idx_interactions_user ON echo_unified_interactions(user_id);")
                await conn.execute("CREATE INDEX idx_interactions_model ON echo_unified_interactions(model_used);")

                logger.info("✅ Created echo_unified_interactions table with indexes")

    async def log_interaction(self, query: str, response: str, model_used: str,
                            processing_time: float, escalation_path: List[str],
                            conversation_id: Optional[str] = None, user_id: str = "default",
                            intent: Optional[str] = None, confidence: float = 0.0,
                            requires_clarification: bool = False,
                            clarifying_questions: Optional[List[str]] = None,
                            complexity_score: Optional[float] = None,
                            tier: Optional[str] = None):
        """Log interaction for learning improvement - async version"""
        if not self.pool:
            await self.initialize()

        try:
            # Build metadata with complexity info
            metadata = {}
            if complexity_score is not None:
                metadata["complexity_score"] = complexity_score
            if tier is not None:
                metadata["tier"] = tier

            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO echo_unified_interactions
                    (query, response, model_used, processing_time, escalation_path,
                     conversation_id, user_id, intent, confidence, requires_clarification,
                     clarifying_questions, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, query, response, model_used, processing_time,
                json.dumps(escalation_path), conversation_id, user_id, intent,
                confidence, requires_clarification,
                json.dumps(clarifying_questions) if clarifying_questions else None,
                json.dumps(metadata))

            logger.debug(f"Logged interaction: {query[:50]}... with {model_used}")

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
            # Don't raise exception for logging failures

    async def get_conversation_history(self, conversation_id: str, limit: int = 50, user_id: str = None) -> List[Dict]:
        """Get conversation history - async version with user isolation"""
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                if user_id:
                    # User-isolated query - CRITICAL for privacy
                    rows = await conn.fetch("""
                        SELECT query, response, model_used, processing_time,
                               intent, confidence, timestamp, metadata
                        FROM echo_unified_interactions
                        WHERE conversation_id = $1 AND user_id = $2
                        ORDER BY timestamp DESC
                        LIMIT $3
                    """, conversation_id, user_id, limit)
                else:
                    # Fallback for system queries (no user context)
                    rows = await conn.fetch("""
                        SELECT query, response, model_used, processing_time,
                               intent, confidence, timestamp, metadata
                        FROM echo_unified_interactions
                        WHERE conversation_id = $1
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """, conversation_id, limit)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def get_learning_insights(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Get learning insights from recent interactions - async version"""
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                # Get model usage stats
                model_stats = await conn.fetch("""
                    SELECT model_used, COUNT(*) as usage_count,
                           AVG(processing_time) as avg_time,
                           AVG(confidence) as avg_confidence
                    FROM echo_unified_interactions
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                    GROUP BY model_used
                    ORDER BY usage_count DESC
                """, timeframe_hours)

                # Get intent distribution
                intent_stats = await conn.fetch("""
                    SELECT intent, COUNT(*) as count
                    FROM echo_unified_interactions
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                      AND intent IS NOT NULL
                    GROUP BY intent
                    ORDER BY count DESC
                """, timeframe_hours)

                # Get escalation patterns
                escalation_stats = await conn.fetch("""
                    SELECT jsonb_array_length(escalation_path) as escalation_depth,
                           COUNT(*) as count
                    FROM echo_unified_interactions
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                      AND escalation_path IS NOT NULL
                    GROUP BY escalation_depth
                    ORDER BY escalation_depth
                """, timeframe_hours)

                return {
                    "model_usage": [dict(row) for row in model_stats],
                    "intent_distribution": [dict(row) for row in intent_stats],
                    "escalation_patterns": [dict(row) for row in escalation_stats],
                    "timeframe_hours": timeframe_hours,
                    "generated_at": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {"error": str(e)}

    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a custom query - async version"""
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                if params:
                    rows = await conn.fetch(query, *params)
                else:
                    rows = await conn.fetch(query)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    async def fetch_one(self, query: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Fetch one row - async version"""
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                # Convert named parameters to positional for asyncpg
                param_values = []
                param_placeholders = []
                param_counter = 1

                for key, value in params.items():
                    param_values.append(value)
                    query = query.replace(f":{key}", f"${param_counter}")
                    param_counter += 1

                row = await conn.fetchrow(query, *param_values)
                return dict(row) if row else None

        except Exception as e:
            logger.error(f"Fetch one failed: {e}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics - async version"""
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                # Get table count
                table_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)

                # Get total interactions
                interaction_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM echo_unified_interactions
                """)

                # Get recent activity
                recent_activity = await conn.fetchval("""
                    SELECT COUNT(*) FROM echo_unified_interactions
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)

                return {
                    "table_count": table_count,
                    "total_interactions": interaction_count,
                    "recent_activity_1h": recent_activity,
                    "pool_size": len(self.pool._holders) if self.pool else 0,
                    "pool_max_size": self.pool._maxsize if self.pool else 0,
                    "database": self.db_config["database"]
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

# Create global instance
async_database = AsyncEchoDatabase()