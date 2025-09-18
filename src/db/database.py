#!/usr/bin/env python3
"""
Database operations for Echo Brain system
"""

import psycopg2
import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class EchoDatabase:
    """Unified database manager for Echo learning"""

    def __init__(self):
        # SECURITY FIX: Use environment variables for sensitive database credentials
        self.db_config = {
            "host": os.environ.get("DB_HOST", "192.168.50.135"),
            "database": os.environ.get("DB_NAME", "tower_consolidated"),
            "user": os.environ.get("DB_USER", "patrick")
        }
        # Note: No password required for local Tower database with user 'patrick'

    async def log_interaction(self, query: str, response: str, model_used: str,
                            processing_time: float, escalation_path: List[str],
                            conversation_id: Optional[str] = None, user_id: str = "default",
                            intent: Optional[str] = None, confidence: float = 0.0,
                            requires_clarification: bool = False,
                            clarifying_questions: Optional[List[str]] = None):
        """Log interaction for learning improvement"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO echo_unified_interactions
                (query, response, model_used, processing_time, escalation_path,
                 conversation_id, user_id, intent, confidence, requires_clarification,
                 clarifying_questions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (query, response, model_used, processing_time,
                  json.dumps(escalation_path), conversation_id or "", user_id, intent or "",
                  confidence, requires_clarification, json.dumps(clarifying_questions or [])))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Database logging failed: {e}")

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