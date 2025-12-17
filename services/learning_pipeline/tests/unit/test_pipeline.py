"""
Unit tests for learning pipeline core functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from pathlib import Path

from src.core.pipeline import LearningPipeline, PipelineMetrics
from src.models.learning_item import LearningItem, LearningItemType, ProcessingResult
from src.models.pipeline_state import PipelineRun, ProcessingStatus
from src.config.settings import PipelineConfig


@pytest.fixture
def mock_config():
    """Create mock pipeline configuration."""
    config = Mock(spec=PipelineConfig)
    config.database = Mock()
    config.vector_database = Mock()
    config.pipeline = Mock()
    config.pipeline.batch_size = 10
    config.pipeline.max_concurrent_processors = 2
    config.circuit_breaker = Mock()
    config.circuit_breaker.failure_threshold = 5
    config.circuit_breaker.reset_timeout = 60
    config.circuit_breaker.half_open_max_calls = 3
    return config


@pytest.fixture
async def pipeline(mock_config):
    """Create learning pipeline instance."""
    pipeline = LearningPipeline(mock_config)

    # Mock the connectors
    pipeline.db_connector = AsyncMock()
    pipeline.vector_connector = AsyncMock()
    pipeline.claude_connector = AsyncMock()
    pipeline.conversation_processor = AsyncMock()

    return pipeline


@pytest.mark.asyncio
class TestLearningPipeline:
    """Test suite for LearningPipeline class."""

    async def test_pipeline_initialization(self, mock_config):
        """Test pipeline initializes all components correctly."""
        pipeline = LearningPipeline(mock_config)

        assert pipeline.config == mock_config
        assert pipeline.run_id is not None
        assert isinstance(pipeline.metrics, PipelineMetrics)
        assert len(pipeline.processed_files) == 0
        assert len(pipeline.failed_files) == 0

    async def test_successful_learning_cycle(self, pipeline):
        """Test successful execution of complete learning cycle."""
        # Setup test data
        test_learning_items = [
            LearningItem(
                content="Test insight about Python",
                item_type=LearningItemType.INSIGHT,
                title="Python Best Practice"
            ),
            LearningItem(
                content="def test(): pass",
                item_type=LearningItemType.CODE_EXAMPLE,
                title="Test Function"
            )
        ]

        test_result = ProcessingResult(learning_items=test_learning_items)

        # Mock conversation processing
        pipeline.conversation_processor.process_file = AsyncMock(return_value=test_result)
        pipeline.claude_connector.get_new_conversations = AsyncMock(
            return_value=[Path("test_conversation.md")]
        )

        # Mock vector database update
        pipeline.vector_connector.add_learning_items = AsyncMock(return_value=2)

        # Mock database operations
        pipeline.db_connector.save_pipeline_run = AsyncMock()
        pipeline.db_connector.update_pipeline_run = AsyncMock()
        pipeline.db_connector.save_learning_items = AsyncMock()

        # Execute learning cycle
        result = await pipeline.run_learning_cycle()

        # Verify results
        assert isinstance(result, PipelineRun)
        assert result.status == ProcessingStatus.COMPLETED
        assert result.conversations_processed == 1
        assert result.learning_items_extracted == 2
        assert result.vectors_updated == 2
        assert result.errors_encountered == 0

        # Verify metrics
        assert pipeline.metrics.conversations_processed == 1
        assert pipeline.metrics.learning_items_extracted == 2
        assert pipeline.metrics.vectors_updated == 2
        assert pipeline.metrics.start_time is not None
        assert pipeline.metrics.end_time is not None

    async def test_learning_cycle_with_processing_errors(self, pipeline):
        """Test learning cycle handles processing errors gracefully."""
        # Setup error scenario
        error_result = ProcessingResult()
        error_result.add_error("Processing failed")

        pipeline.claude_connector.get_new_conversations = AsyncMock(
            return_value=[Path("failed_conversation.md")]
        )
        pipeline.conversation_processor.process_file = AsyncMock(return_value=error_result)

        # Mock database operations
        pipeline.db_connector.save_pipeline_run = AsyncMock()
        pipeline.db_connector.update_pipeline_run = AsyncMock()
        pipeline.vector_connector.add_learning_items = AsyncMock(return_value=0)
        pipeline.db_connector.save_learning_items = AsyncMock()

        # Execute learning cycle
        result = await pipeline.run_learning_cycle()

        # Verify error handling
        assert result.status == ProcessingStatus.COMPLETED  # Pipeline completes despite processing errors
        assert result.errors_encountered == 1
        assert result.learning_items_extracted == 0

    async def test_circuit_breaker_failure(self, pipeline):
        """Test pipeline handles circuit breaker failures."""
        from src.core.circuit_breaker import CircuitBreakerError

        # Mock circuit breaker to fail
        pipeline.circuit_breaker.call = AsyncMock(side_effect=CircuitBreakerError("Circuit open"))

        # Mock minimal required operations
        pipeline.claude_connector.get_new_conversations = AsyncMock(return_value=[])

        # Execute and expect failure
        with pytest.raises(CircuitBreakerError):
            await pipeline.run_learning_cycle()

    async def test_batch_processing(self, pipeline):
        """Test conversation files are processed in batches."""
        # Create test files
        test_files = [Path(f"test_{i}.md") for i in range(25)]

        pipeline.claude_connector.get_new_conversations = AsyncMock(return_value=test_files)
        pipeline.config.pipeline.batch_size = 10

        # Mock successful processing
        test_result = ProcessingResult(learning_items=[])
        pipeline.conversation_processor.process_file = AsyncMock(return_value=test_result)

        # Mock other operations
        pipeline.db_connector.save_pipeline_run = AsyncMock()
        pipeline.db_connector.update_pipeline_run = AsyncMock()
        pipeline.vector_connector.add_learning_items = AsyncMock(return_value=0)
        pipeline.db_connector.save_learning_items = AsyncMock()

        # Execute
        result = await pipeline.run_learning_cycle()

        # Verify all files processed
        assert result.conversations_processed == 25
        assert pipeline.conversation_processor.process_file.call_count == 25

    async def test_no_new_conversations(self, pipeline):
        """Test pipeline handles case with no new conversations."""
        pipeline.claude_connector.get_new_conversations = AsyncMock(return_value=[])

        # Mock database operations
        pipeline.db_connector.save_pipeline_run = AsyncMock()
        pipeline.db_connector.update_pipeline_run = AsyncMock()

        # Execute
        result = await pipeline.run_learning_cycle()

        # Verify results
        assert result.status == ProcessingStatus.COMPLETED
        assert result.conversations_processed == 0
        assert result.learning_items_extracted == 0
        assert result.vectors_updated == 0

    async def test_vector_database_update_failure(self, pipeline):
        """Test pipeline handles vector database update failures."""
        # Setup test data
        test_items = [LearningItem(content="test", item_type=LearningItemType.INSIGHT)]
        test_result = ProcessingResult(learning_items=test_items)

        pipeline.claude_connector.get_new_conversations = AsyncMock(
            return_value=[Path("test.md")]
        )
        pipeline.conversation_processor.process_file = AsyncMock(return_value=test_result)

        # Mock vector update failure
        pipeline.vector_connector.add_learning_items = AsyncMock(
            side_effect=Exception("Vector DB connection failed")
        )

        # Mock database operations
        pipeline.db_connector.save_pipeline_run = AsyncMock()
        pipeline.db_connector.update_pipeline_run = AsyncMock()

        # Execute and expect failure
        with pytest.raises(Exception) as exc_info:
            await pipeline.run_learning_cycle()

        assert "Vector DB connection failed" in str(exc_info.value)

    async def test_health_check(self, pipeline):
        """Test pipeline health check functionality."""
        # Mock healthy components
        pipeline.db_connector.health_check = AsyncMock(return_value=True)
        pipeline.vector_connector.health_check = AsyncMock(return_value=True)
        pipeline.claude_connector.health_check = AsyncMock(return_value=True)

        # Execute health check
        health_status = await pipeline.health_check()

        # Verify results
        assert health_status['healthy'] is True
        assert 'database' in health_status['components']
        assert 'vector_database' in health_status['components']
        assert 'claude_connector' in health_status['components']
        assert 'circuit_breaker' in health_status['components']

    async def test_health_check_with_failures(self, pipeline):
        """Test health check detects component failures."""
        # Mock failing components
        pipeline.db_connector.health_check = AsyncMock(
            side_effect=Exception("DB connection failed")
        )
        pipeline.vector_connector.health_check = AsyncMock(return_value=True)
        pipeline.claude_connector.health_check = AsyncMock(return_value=True)

        # Execute health check
        health_status = await pipeline.health_check()

        # Verify failure detection
        assert health_status['healthy'] is False
        assert health_status['components']['database']['healthy'] is False
        assert 'error' in health_status['components']['database']

    async def test_pipeline_status(self, pipeline):
        """Test pipeline status reporting."""
        status = await pipeline.get_pipeline_status()

        # Verify status structure
        assert 'run_id' in status
        assert 'status' in status
        assert 'metrics' in status
        assert 'circuit_breaker' in status
        assert 'config' in status

        # Verify run ID matches
        assert status['run_id'] == pipeline.run_id

    async def test_cleanup(self, pipeline):
        """Test pipeline cleanup functionality."""
        # Mock connectors
        pipeline.db_connector.disconnect = AsyncMock()
        pipeline.vector_connector.disconnect = AsyncMock()

        # Execute cleanup
        await pipeline.cleanup()

        # Verify cleanup called
        pipeline.db_connector.disconnect.assert_called_once()
        pipeline.vector_connector.disconnect.assert_called_once()


@pytest.mark.asyncio
class TestPipelineMetrics:
    """Test suite for PipelineMetrics class."""

    def test_metrics_initialization(self):
        """Test metrics initialize with default values."""
        metrics = PipelineMetrics()

        assert metrics.conversations_processed == 0
        assert metrics.articles_processed == 0
        assert metrics.learning_items_extracted == 0
        assert metrics.vectors_updated == 0
        assert metrics.errors_encountered == 0
        assert metrics.processing_time == 0.0
        assert metrics.start_time is None
        assert metrics.end_time is None

    def test_metrics_to_dict(self):
        """Test metrics serialization to dictionary."""
        metrics = PipelineMetrics()
        metrics.conversations_processed = 10
        metrics.start_time = datetime.now()

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result['conversations_processed'] == 10
        assert 'start_time' in result
        assert result['start_time'] is not None