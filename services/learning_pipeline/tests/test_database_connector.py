"""
Test suite for DatabaseConnector - comprehensive failing tests.
These tests will fail until DatabaseConnector is implemented.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from pathlib import Path
import asyncpg
import os

from src.connectors.database_connector import DatabaseConnector
from src.models.learning_item import LearningItem, LearningItemType
from src.models.pipeline_state import PipelineRun, ProcessingStatus
from src.config.settings import DatabaseConfig


@pytest.fixture
def localhost_db_config():
    """Database config using localhost (not 192.168.50.135)."""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        name="echo_brain",
        user="patrick",
        password_env="ECHO_BRAIN_DB_PASSWORD",
        connection_timeout=30
    )


@pytest.mark.asyncio
class TestDatabaseConnector:
    """Test suite for PostgreSQL database connector with localhost."""

    async def test_database_connector_initialization(self, localhost_db_config):
        """Test connector initializes with localhost configuration."""
        connector = DatabaseConnector(localhost_db_config)

        assert connector.config.host == "localhost"
        assert connector.config.port == 5432
        assert connector.config.user == "patrick"
        assert connector.connection_pool is None  # Not connected yet

    async def test_connect_to_localhost_database(self, localhost_db_config):
        """Test connector connects to localhost database properly."""
        # Mock password environment variable
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'tower_echo_brain_secret_key_2025'}):
            connector = DatabaseConnector(localhost_db_config)

            # Should connect without errors
            await connector.connect()

            assert connector.connection_pool is not None
            assert connector.is_connected is True

            # Clean up
            await connector.disconnect()

    async def test_connection_string_uses_localhost(self, localhost_db_config):
        """Test connection string uses localhost, not 192.168.50.135."""
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'test_password'}):
            connector = DatabaseConnector(localhost_db_config)

            connection_string = localhost_db_config.async_connection_string
            assert "localhost" in connection_string
            assert "192.168.50.135" not in connection_string
            assert "postgresql+asyncpg://patrick:test_password@localhost:5432/echo_brain" == connection_string

    async def test_save_pipeline_run(self, localhost_db_config):
        """Test saving pipeline run to database."""
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'tower_echo_brain_secret_key_2025'}):
            connector = DatabaseConnector(localhost_db_config)
            await connector.connect()

            # Create test pipeline run
            pipeline_run = PipelineRun(
                run_id="test-run-123",
                started_at=datetime.now(),
                status=ProcessingStatus.RUNNING,
                conversations_processed=5,
                learning_items_extracted=12,
                vectors_updated=10
            )

            # Should save without errors
            await connector.save_pipeline_run(pipeline_run)

            # Should be able to retrieve it
            retrieved = await connector.get_pipeline_run(pipeline_run.run_id)
            assert retrieved is not None
            assert retrieved.run_id == pipeline_run.run_id
            assert retrieved.conversations_processed == 5

            await connector.disconnect()

    async def test_update_pipeline_run(self, localhost_db_config):
        """Test updating pipeline run status."""
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'tower_echo_brain_secret_key_2025'}):
            connector = DatabaseConnector(localhost_db_config)
            await connector.connect()

            # Create and save pipeline run
            pipeline_run = PipelineRun(
                run_id="test-update-456",
                started_at=datetime.now(),
                status=ProcessingStatus.RUNNING
            )
            await connector.save_pipeline_run(pipeline_run)

            # Update status
            pipeline_run.status = ProcessingStatus.COMPLETED
            pipeline_run.completed_at = datetime.now()
            pipeline_run.vectors_updated = 20

            await connector.update_pipeline_run(pipeline_run)

            # Verify update
            updated = await connector.get_pipeline_run(pipeline_run.run_id)
            assert updated.status == ProcessingStatus.COMPLETED
            assert updated.vectors_updated == 20
            assert updated.completed_at is not None

            await connector.disconnect()

    async def test_save_learning_items(self, localhost_db_config):
        """Test saving learning items to database."""
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'tower_echo_brain_secret_key_2025'}):
            connector = DatabaseConnector(localhost_db_config)
            await connector.connect()

            # Create test learning items
            learning_items = [
                LearningItem(
                    content="Test insight about database connections",
                    item_type=LearningItemType.INSIGHT,
                    title="Database Connection Best Practice",
                    source_file="test_conversation.md",
                    confidence_score=0.9
                ),
                LearningItem(
                    content="SELECT * FROM pipeline_runs WHERE status = 'completed'",
                    item_type=LearningItemType.CODE_EXAMPLE,
                    title="Query Pipeline Status",
                    source_file="test_conversation.md",
                    confidence_score=0.8
                )
            ]

            # Should save without errors
            saved_count = await connector.save_learning_items(learning_items)
            assert saved_count == 2

            # Should be able to retrieve them
            retrieved_items = await connector.get_learning_items_by_source("test_conversation.md")
            assert len(retrieved_items) == 2
            assert retrieved_items[0].item_type in [LearningItemType.INSIGHT, LearningItemType.CODE_EXAMPLE]

            await connector.disconnect()

    async def test_get_last_successful_run_time(self, localhost_db_config):
        """Test retrieving last successful pipeline run time."""
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'tower_echo_brain_secret_key_2025'}):
            connector = DatabaseConnector(localhost_db_config)
            await connector.connect()

            # Should return None when no successful runs exist
            last_run = await connector.get_last_successful_run_time()

            # Create successful run
            completed_time = datetime.now() - timedelta(hours=1)
            pipeline_run = PipelineRun(
                run_id="successful-run-789",
                started_at=completed_time - timedelta(minutes=30),
                completed_at=completed_time,
                status=ProcessingStatus.COMPLETED
            )
            await connector.save_pipeline_run(pipeline_run)

            # Should return the completion time
            last_run = await connector.get_last_successful_run_time()
            assert last_run is not None
            assert abs((last_run - completed_time).total_seconds()) < 1  # Within 1 second

            await connector.disconnect()

    async def test_health_check(self, localhost_db_config):
        """Test database health check functionality."""
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'tower_echo_brain_secret_key_2025'}):
            connector = DatabaseConnector(localhost_db_config)

            # Health check should fail when not connected
            health = await connector.health_check()
            assert health is False

            # Connect and health check should pass
            await connector.connect()
            health = await connector.health_check()
            assert health is True

            await connector.disconnect()

    async def test_connection_error_handling(self, localhost_db_config):
        """Test proper error handling for connection failures."""
        # Use wrong password
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'wrong_password'}):
            connector = DatabaseConnector(localhost_db_config)

            # Should raise appropriate exception
            with pytest.raises(Exception):  # Could be asyncpg.ConnectionError or similar
                await connector.connect()

    async def test_transaction_rollback(self, localhost_db_config):
        """Test transaction rollback on errors."""
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'tower_echo_brain_secret_key_2025'}):
            connector = DatabaseConnector(localhost_db_config)
            await connector.connect()

            # Mock a database error during save
            original_save = connector.save_learning_items

            async def failing_save(*args, **kwargs):
                # Start transaction, then fail
                async with connector.connection_pool.acquire() as connection:
                    async with connection.transaction():
                        # Do some work that should be rolled back
                        await connection.execute("UPDATE learning_items SET title = 'TEMP' WHERE id = -1")
                        # Simulate error
                        raise Exception("Simulated database error")

            connector.save_learning_items = failing_save

            # Should handle error gracefully
            test_items = [LearningItem(
                content="Test content",
                item_type=LearningItemType.INSIGHT,
                title="Test"
            )]

            with pytest.raises(Exception):
                await connector.save_learning_items(test_items)

            await connector.disconnect()

    async def test_concurrent_connections(self, localhost_db_config):
        """Test handling multiple concurrent connections."""
        with patch.dict(os.environ, {'ECHO_BRAIN_DB_PASSWORD': 'tower_echo_brain_secret_key_2025'}):
            connector = DatabaseConnector(localhost_db_config)
            await connector.connect()

            # Perform multiple concurrent operations
            tasks = []
            for i in range(5):
                pipeline_run = PipelineRun(
                    run_id=f"concurrent-{i}",
                    started_at=datetime.now(),
                    status=ProcessingStatus.RUNNING
                )
                tasks.append(connector.save_pipeline_run(pipeline_run))

            # Should handle concurrent operations
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed (no exceptions)
            for result in results:
                assert not isinstance(result, Exception)

            await connector.disconnect()