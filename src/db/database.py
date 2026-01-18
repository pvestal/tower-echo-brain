#!/usr/bin/env python3
"""
Database operations for Echo Brain system
"""

import psycopg2
import psycopg2.extras
import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime

# Use centralized connection pool
from .connection_pool import get_database_config, SyncConnection

logger = logging.getLogger(__name__)

class EchoDatabase:
    """Unified database manager for Echo learning"""

    def __init__(self):
        # Use centralized connection pool configuration
        self.db_config = get_database_config()
        logger.info("✅ Database connection configured via connection pool")


    async def log_interaction(self, query: str, response: str, model_used: str,
                            processing_time: float, escalation_path: List[str],
                            conversation_id: Optional[str] = None, user_id: str = "default",
                            intent: Optional[str] = None, confidence: float = 0.0,
                            requires_clarification: bool = False,
                            clarifying_questions: Optional[List[str]] = None,
                            complexity_score: Optional[float] = None,
                            tier: Optional[str] = None):
        """Log interaction for learning improvement with robust error handling and retry"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()

                # Build metadata with complexity info
                metadata = {}
                if complexity_score is not None:
                    metadata["complexity_score"] = complexity_score
                if tier is not None:
                    metadata["tier"] = tier

                cursor.execute("""
                    INSERT INTO echo_unified_interactions
                    (query, response, model_used, processing_time, escalation_path,
                     conversation_id, user_id, intent, confidence, requires_clarification,
                     clarifying_questions, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (query, response, model_used, processing_time,
                      psycopg2.extras.Json(escalation_path), conversation_id or "", user_id, intent or "",
                      confidence, requires_clarification, psycopg2.extras.Json(clarifying_questions or []),
                      psycopg2.extras.Json(metadata)))

                conn.commit()
                cursor.close()
                conn.close()

                # Success - log interaction saved
                logger.info(f"✅ Interaction logged: conversation_id={conversation_id}, intent={intent}")
                return True

            except psycopg2.OperationalError as e:
                logger.warning(f"⚠️ Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"❌ CRITICAL: Database logging FAILED after {max_retries} attempts. Conversation lost: {conversation_id}")
                    return False

            except psycopg2.DatabaseError as e:
                logger.error(f"❌ Database error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"❌ CRITICAL: Database schema error. Conversation lost: {conversation_id}")
                    return False

            except Exception as e:
                logger.error(f"❌ Unexpected database error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"❌ CRITICAL: Unknown database error. Conversation lost: {conversation_id}")
                    return False

        return False

    async def create_tables_if_needed(self):
        """Create unified Echo tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
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
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
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

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Echo unified database tables initialized")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    async def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history for a specific conversation"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT query, response, model_used, processing_time, timestamp, intent, confidence
                FROM echo_unified_interactions
                WHERE conversation_id = %s
                ORDER BY timestamp ASC
            """, (conversation_id,))

            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            return [
                {
                    "query": row[0],
                    "response": row[1],
                    "model_used": row[2],
                    "processing_time": row[3],
                    "timestamp": row[4].isoformat() if row[4] else None,
                    "intent": row[5],
                    "confidence": row[6]
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all conversations for a user"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT conversation_id, MIN(timestamp) as first_interaction,
                       MAX(timestamp) as last_interaction, COUNT(*) as interaction_count
                FROM echo_unified_interactions
                WHERE user_id = %s
                GROUP BY conversation_id
                ORDER BY last_interaction DESC
                LIMIT %s
            """, (user_id, limit))

            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            return [
                {
                    "conversation_id": row[0],
                    "first_interaction": row[1].isoformat() if row[1] else None,
                    "last_interaction": row[2].isoformat() if row[2] else None,
                    "interaction_count": row[3]
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get user conversations: {e}")
            return []

# Global database instance
database = EchoDatabase()