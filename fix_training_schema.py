#!/usr/bin/env python3
"""Fix database schema for real training data storage"""

import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_schema():
    conn = psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="***REMOVED***"
    )
    cur = conn.cursor()

    # Create proper training tables
    queries = [
        # Training feedback table
        """
        CREATE TABLE IF NOT EXISTS training_feedback (
            id SERIAL PRIMARY KEY,
            conversation_id VARCHAR(255),
            query_text TEXT,
            response_text TEXT,
            was_useful BOOLEAN,
            user_edited BOOLEAN DEFAULT FALSE,
            edited_response TEXT,
            feedback_score FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """,

        # Model training status
        """
        CREATE TABLE IF NOT EXISTS model_training_status (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255),
            training_phase VARCHAR(50),
            epochs_completed INTEGER DEFAULT 0,
            total_epochs INTEGER,
            loss FLOAT,
            accuracy FLOAT,
            dataset_size INTEGER,
            started_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """,

        # Pattern recognition table
        """
        CREATE TABLE IF NOT EXISTS learned_patterns (
            id SERIAL PRIMARY KEY,
            pattern_type VARCHAR(100),
            pattern_data JSONB,
            frequency INTEGER DEFAULT 1,
            confidence FLOAT,
            last_seen TIMESTAMP DEFAULT NOW(),
            created_at TIMESTAMP DEFAULT NOW()
        )
        """,

        # Add missing columns to learning_history
        """
        ALTER TABLE learning_history
        ADD COLUMN IF NOT EXISTS learning_type VARCHAR(100),
        ADD COLUMN IF NOT EXISTS model_version VARCHAR(50),
        ADD COLUMN IF NOT EXISTS training_loss FLOAT
        """
    ]

    for query in queries:
        try:
            cur.execute(query)
            conn.commit()
            logger.info(f"✅ Executed: {query[:50]}...")
        except Exception as e:
            conn.rollback()
            logger.warning(f"⚠️  Skipped (may already exist): {e}")

    # Create indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_feedback_useful ON training_feedback(was_useful)",
        "CREATE INDEX IF NOT EXISTS idx_patterns_type ON learned_patterns(pattern_type)",
        "CREATE INDEX IF NOT EXISTS idx_training_model ON model_training_status(model_name)"
    ]

    for idx in indexes:
        try:
            cur.execute(idx)
            conn.commit()
            logger.info(f"✅ Created index: {idx[20:60]}...")
        except Exception as e:
            logger.warning(f"⚠️  Index issue: {e}")

    cur.close()
    conn.close()
    logger.info("✅ Schema fixed!")

if __name__ == "__main__":
    fix_schema()