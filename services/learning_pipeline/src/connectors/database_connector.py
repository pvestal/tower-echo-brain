"""
PostgreSQL database connector for the learning pipeline.
Handles connection pooling, transactions, and data operations.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncpg
from asyncpg.pool import Pool

from ..config.settings import DatabaseConfig
from ..models.learning_item import LearningItem
from ..models.pipeline_state import PipelineRun

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Async PostgreSQL database connector with connection pooling.

    Features:
    - Connection pooling for performance
    - Automatic reconnection handling
    - Transaction support
    - Health checking
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool: Optional[Pool] = None
        self.is_connected = False

    async def connect(self) -> None:
        """Establish database connection pool."""
        try:
            logger.info(f"Connecting to PostgreSQL at {self.config.host}:{self.config.port}")

            # Create connection pool with retry logic
            for attempt in range(3):
                try:
                    self.connection_pool = await asyncpg.create_pool(
                        host=self.config.host,
                        port=self.config.port,
                        user=self.config.user,
                        password=self.config.password,
                        database=self.config.name,
                        min_size=5,
                        max_size=self.config.pool_size,
                        max_inactive_connection_lifetime=300,
                        command_timeout=self.config.command_timeout
                    )
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    await asyncio.sleep(2)

            self.is_connected = True

            # Initialize database schema if needed
            await self._ensure_schema()

            logger.info("Database connection pool established successfully")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.is_connected = False
            raise

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self.connection_pool:
            logger.info("Closing database connection pool")
            await self.connection_pool.close()
            self.connection_pool = None
            self.is_connected = False

    async def _ensure_schema(self) -> None:
        """Ensure required database tables exist."""
        async with self.connection_pool.acquire() as connection:
            # Create pipeline_runs table
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id VARCHAR(36) PRIMARY KEY,
                    started_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ,
                    status VARCHAR(20) NOT NULL,
                    conversations_processed INTEGER DEFAULT 0,
                    articles_processed INTEGER DEFAULT 0,
                    learning_items_extracted INTEGER DEFAULT 0,
                    vectors_updated INTEGER DEFAULT 0,
                    errors_encountered INTEGER DEFAULT 0,
                    error_message TEXT,
                    performance_metrics JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create learning_items table
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS learning_items (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    item_type VARCHAR(50) NOT NULL,
                    title VARCHAR(500),
                    source_file VARCHAR(1000),
                    confidence_score FLOAT,
                    tags TEXT[],
                    category VARCHAR(100),
                    importance_score FLOAT,
                    unique_id VARCHAR(100),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes for performance
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_items_source_file
                ON learning_items(source_file)
            """)

            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_items_type
                ON learning_items(item_type)
            """)

            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_items_unique_id
                ON learning_items(unique_id) WHERE unique_id IS NOT NULL
            """)

            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
                ON pipeline_runs(status)
            """)

            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at
                ON pipeline_runs(started_at)
            """)

    async def save_pipeline_run(self, pipeline_run: PipelineRun) -> None:
        """Save pipeline run to database."""
        async with self.connection_pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO pipeline_runs (
                    run_id, started_at, completed_at, status,
                    conversations_processed, articles_processed,
                    learning_items_extracted, vectors_updated,
                    errors_encountered, error_message, performance_metrics
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (run_id) DO UPDATE SET
                    completed_at = EXCLUDED.completed_at,
                    status = EXCLUDED.status,
                    conversations_processed = EXCLUDED.conversations_processed,
                    articles_processed = EXCLUDED.articles_processed,
                    learning_items_extracted = EXCLUDED.learning_items_extracted,
                    vectors_updated = EXCLUDED.vectors_updated,
                    errors_encountered = EXCLUDED.errors_encountered,
                    error_message = EXCLUDED.error_message,
                    performance_metrics = EXCLUDED.performance_metrics
            """,
                pipeline_run.run_id,
                pipeline_run.started_at,
                pipeline_run.completed_at,
                pipeline_run.status.value,
                pipeline_run.conversations_processed,
                pipeline_run.articles_processed,
                pipeline_run.learning_items_extracted,
                pipeline_run.vectors_updated,
                pipeline_run.errors_encountered,
                pipeline_run.error_message,
                pipeline_run.performance_metrics
            )

    async def update_pipeline_run(self, pipeline_run: PipelineRun) -> None:
        """Update existing pipeline run."""
        async with self.connection_pool.acquire() as connection:
            await connection.execute("""
                UPDATE pipeline_runs SET
                    completed_at = $2,
                    status = $3,
                    conversations_processed = $4,
                    articles_processed = $5,
                    learning_items_extracted = $6,
                    vectors_updated = $7,
                    errors_encountered = $8,
                    error_message = $9,
                    performance_metrics = $10
                WHERE run_id = $1
            """,
                pipeline_run.run_id,
                pipeline_run.completed_at,
                pipeline_run.status.value,
                pipeline_run.conversations_processed,
                pipeline_run.articles_processed,
                pipeline_run.learning_items_extracted,
                pipeline_run.vectors_updated,
                pipeline_run.errors_encountered,
                pipeline_run.error_message,
                pipeline_run.performance_metrics
            )

    async def get_pipeline_run(self, run_id: str) -> Optional[PipelineRun]:
        """Retrieve pipeline run by ID."""
        async with self.connection_pool.acquire() as connection:
            row = await connection.fetchrow("""
                SELECT * FROM pipeline_runs WHERE run_id = $1
            """, run_id)

            if not row:
                return None

            from ..models.pipeline_state import ProcessingStatus
            return PipelineRun(
                run_id=row['run_id'],
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                status=ProcessingStatus(row['status']),
                conversations_processed=row['conversations_processed'],
                articles_processed=row['articles_processed'],
                learning_items_extracted=row['learning_items_extracted'],
                vectors_updated=row['vectors_updated'],
                errors_encountered=row['errors_encountered'],
                error_message=row['error_message'],
                performance_metrics=row['performance_metrics']
            )

    async def save_learning_items(self, learning_items: List[LearningItem]) -> int:
        """Save learning items to database."""
        if not learning_items:
            return 0

        async with self.connection_pool.acquire() as connection:
            async with connection.transaction():
                saved_count = 0

                for item in learning_items:
                    # Check if item already exists (by unique_id if provided)
                    if hasattr(item, 'unique_id') and item.unique_id:
                        existing = await connection.fetchval("""
                            SELECT id FROM learning_items WHERE unique_id = $1
                        """, item.unique_id)

                        if existing:
                            # Update existing item
                            await connection.execute("""
                                UPDATE learning_items SET
                                    content = $2, title = $3, confidence_score = $4,
                                    tags = $5, category = $6, importance_score = $7,
                                    metadata = $8, updated_at = NOW()
                                WHERE unique_id = $1
                            """,
                                item.unique_id, item.content, item.title,
                                getattr(item, 'confidence_score', None),
                                getattr(item, 'tags', None),
                                getattr(item, 'category', None),
                                getattr(item, 'importance_score', None),
                                getattr(item, 'metadata', None)
                            )
                        else:
                            # Insert new item
                            await connection.execute("""
                                INSERT INTO learning_items (
                                    content, item_type, title, source_file,
                                    confidence_score, tags, category, importance_score,
                                    unique_id, metadata
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            """,
                                item.content, item.item_type.value, item.title,
                                getattr(item, 'source_file', None),
                                getattr(item, 'confidence_score', None),
                                getattr(item, 'tags', None),
                                getattr(item, 'category', None),
                                getattr(item, 'importance_score', None),
                                item.unique_id,
                                getattr(item, 'metadata', None)
                            )
                    else:
                        # Insert new item without unique_id
                        await connection.execute("""
                            INSERT INTO learning_items (
                                content, item_type, title, source_file,
                                confidence_score, tags, category, importance_score,
                                metadata
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                            item.content, item.item_type.value, item.title,
                            getattr(item, 'source_file', None),
                            getattr(item, 'confidence_score', None),
                            getattr(item, 'tags', None),
                            getattr(item, 'category', None),
                            getattr(item, 'importance_score', None),
                            getattr(item, 'metadata', None)
                        )

                    saved_count += 1

                return saved_count

    async def get_learning_items_by_source(self, source_file: str) -> List[LearningItem]:
        """Get learning items by source file."""
        async with self.connection_pool.acquire() as connection:
            rows = await connection.fetch("""
                SELECT * FROM learning_items WHERE source_file = $1
                ORDER BY created_at DESC
            """, source_file)

            items = []
            for row in rows:
                from ..models.learning_item import LearningItemType
                item = LearningItem(
                    content=row['content'],
                    item_type=LearningItemType(row['item_type']),
                    title=row['title']
                )
                # Set additional attributes
                item.source_file = row['source_file']
                item.confidence_score = row['confidence_score']
                item.tags = row['tags']
                item.category = row['category']
                item.importance_score = row['importance_score']
                item.unique_id = row['unique_id']
                item.metadata = row['metadata']
                items.append(item)

            return items

    async def get_last_successful_run_time(self) -> Optional[datetime]:
        """Get timestamp of last successful pipeline run."""
        async with self.connection_pool.acquire() as connection:
            row = await connection.fetchrow("""
                SELECT completed_at FROM pipeline_runs
                WHERE status = 'completed' AND completed_at IS NOT NULL
                ORDER BY completed_at DESC
                LIMIT 1
            """)

            return row['completed_at'] if row else None

    async def health_check(self) -> bool:
        """Check database connection health."""
        if not self.is_connected or not self.connection_pool:
            return False

        try:
            async with self.connection_pool.acquire() as connection:
                await connection.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self.connection_pool:
            return {'connected': False}

        return {
            'connected': self.is_connected,
            'pool_size': self.connection_pool.get_size(),
            'idle_connections': self.connection_pool.get_idle_size(),
            'max_size': self.config.pool_size
        }