#!/usr/bin/env python3
"""
Simple Training Loop for Echo Brain
Continuously learns from conversation history
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import psycopg2
from pathlib import Path

# Setup logging
log_path = Path("/opt/tower-echo-brain/retraining/logs/simple_training.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleTrainer:
    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "***REMOVED***"
        }
        self.training_data = []
        self.model_metrics = {
            "conversations_processed": 0,
            "patterns_learned": 0,
            "embeddings_created": 0,
            "last_update": None
        }

    def connect_db(self):
        """Connect to database"""
        return psycopg2.connect(**self.db_config)

    def fetch_learning_data(self):
        """Fetch recent conversations for learning"""
        try:
            conn = self.connect_db()
            cur = conn.cursor()

            # Get recent conversations
            cur.execute("""
                SELECT id, query_text, response_text, created_at
                FROM echo_conversations
                WHERE created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 100
            """)

            conversations = cur.fetchall()
            logger.info(f"📚 Fetched {len(conversations)} recent conversations")

            cur.close()
            conn.close()

            return conversations
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return []

    def process_conversations(self, conversations):
        """Process conversations to extract patterns"""
        patterns = []

        for conv_id, query, response, created_at in conversations:
            # Simple pattern extraction
            if query and response:
                pattern = {
                    "query_length": len(query),
                    "response_length": len(response),
                    "query_tokens": query.lower().split(),
                    "timestamp": created_at.isoformat() if created_at else None
                }
                patterns.append(pattern)

        logger.info(f"🔍 Extracted {len(patterns)} patterns")
        return patterns

    def update_learning_history(self, patterns):
        """Store learned patterns in database"""
        try:
            conn = self.connect_db()
            cur = conn.cursor()

            for pattern in patterns[:10]:  # Store top 10 patterns
                cur.execute("""
                    INSERT INTO learning_history (
                        learning_type,
                        learning_data,
                        created_at
                    ) VALUES (%s, %s, NOW())
                    ON CONFLICT DO NOTHING
                """, ('pattern_extraction', json.dumps(pattern)))

            conn.commit()
            rows_added = cur.rowcount

            cur.close()
            conn.close()

            logger.info(f"💾 Stored {rows_added} new patterns in learning history")
            return rows_added
        except Exception as e:
            logger.error(f"❌ Failed to update learning history: {e}")
            return 0

    async def training_loop(self):
        """Main training loop"""
        logger.info("🚀 Starting simple training loop")
        iteration = 0

        while True:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Training iteration #{iteration}")

            try:
                # Fetch recent conversations
                conversations = self.fetch_learning_data()

                if conversations:
                    # Process conversations
                    patterns = self.process_conversations(conversations)

                    # Update learning history
                    stored = self.update_learning_history(patterns)

                    # Update metrics
                    self.model_metrics["conversations_processed"] += len(conversations)
                    self.model_metrics["patterns_learned"] += len(patterns)
                    self.model_metrics["embeddings_created"] += stored
                    self.model_metrics["last_update"] = datetime.now().isoformat()

                    # Log metrics
                    logger.info(f"📊 Training Metrics:")
                    logger.info(f"   Conversations: {self.model_metrics['conversations_processed']}")
                    logger.info(f"   Patterns: {self.model_metrics['patterns_learned']}")
                    logger.info(f"   Embeddings: {self.model_metrics['embeddings_created']}")
                else:
                    logger.info("😴 No new conversations to learn from")

                # Check GPU usage
                import subprocess
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True
                    )
                    gpu_usage = result.stdout.strip()
                    logger.info(f"🎮 GPU Usage: {gpu_usage}%")
                except:
                    pass

                # Sleep for 5 minutes
                logger.info("💤 Sleeping for 5 minutes...")
                await asyncio.sleep(300)

            except KeyboardInterrupt:
                logger.info("⏹️ Training stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in training loop: {e}")
                await asyncio.sleep(60)

async def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("🧠 ECHO BRAIN SIMPLE TRAINING SYSTEM")
    logger.info("="*60)
    logger.info(f"Started: {datetime.now().isoformat()}")

    trainer = SimpleTrainer()

    try:
        await trainer.training_loop()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Training stopped")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")