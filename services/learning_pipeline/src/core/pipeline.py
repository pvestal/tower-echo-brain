"""
Main learning pipeline orchestrator with circuit breaker pattern.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import json
import uuid

from ..models.learning_item import LearningItem, ProcessingResult
from ..models.pipeline_state import PipelineRun, ProcessingStatus
from ..connectors.database_connector import DatabaseConnector
from ..connectors.vector_connector import VectorConnector
from ..connectors.claude_connector import ClaudeConnector
from ..processors.conversation_processor import ConversationProcessor
from ..config.settings import PipelineConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    conversations_processed: int = 0
    articles_processed: int = 0
    learning_items_extracted: int = 0
    vectors_updated: int = 0
    errors_encountered: int = 0
    processing_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'conversations_processed': self.conversations_processed,
            'articles_processed': self.articles_processed,
            'learning_items_extracted': self.learning_items_extracted,
            'vectors_updated': self.vectors_updated,
            'errors_encountered': self.errors_encountered,
            'processing_time': self.processing_time,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class LearningPipeline:
    """
    Main orchestrator for the learning pipeline with circuit breaker pattern.

    Coordinates the entire learning process:
    1. Discover new/modified conversations and articles
    2. Process content to extract learning items
    3. Generate embeddings for learning items
    4. Store in vector database and PostgreSQL
    5. Track metrics and handle errors gracefully
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_id = str(uuid.uuid4())

        # Initialize circuit breaker
        cb_config = CircuitBreakerConfig(
            failure_threshold=config.circuit_breaker.failure_threshold,
            reset_timeout=config.circuit_breaker.reset_timeout,
            half_open_max_calls=config.circuit_breaker.half_open_max_calls
        )
        self.circuit_breaker = CircuitBreaker(cb_config)

        # Initialize connectors (will be created in async init)
        self.db_connector: Optional[DatabaseConnector] = None
        self.vector_connector: Optional[VectorConnector] = None
        self.claude_connector: Optional[ClaudeConnector] = None

        # Initialize processors
        self.conversation_processor = ConversationProcessor(config)

        # Metrics tracking
        self.metrics = PipelineMetrics()

        # Processing state
        self.processed_files: Set[str] = set()
        self.failed_files: Set[str] = set()

    async def initialize(self):
        """Initialize async components."""
        logger.info(f"Initializing learning pipeline {self.run_id}")

        try:
            # Initialize connectors with circuit breaker protection
            self.db_connector = await self.circuit_breaker.call(
                self._init_db_connector
            )
            self.vector_connector = await self.circuit_breaker.call(
                self._init_vector_connector
            )
            self.claude_connector = ClaudeConnector(self.config)

            logger.info("Learning pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize learning pipeline: {e}")
            raise

    async def _init_db_connector(self) -> DatabaseConnector:
        """Initialize database connector."""
        connector = DatabaseConnector(self.config.database)
        await connector.connect()
        return connector

    async def _init_vector_connector(self) -> VectorConnector:
        """Initialize vector database connector."""
        connector = VectorConnector(self.config.vector_database)
        await connector.connect()
        return connector

    async def run_learning_cycle(self) -> PipelineRun:
        """
        Execute a complete learning cycle with error handling.

        Returns:
            PipelineRun with results and metrics
        """
        self.metrics.start_time = datetime.now()
        start_time = time.time()

        logger.info(f"Starting learning cycle {self.run_id}")

        try:
            # Create pipeline run record
            pipeline_run = PipelineRun(
                run_id=self.run_id,
                started_at=self.metrics.start_time,
                status=ProcessingStatus.RUNNING
            )

            # Save initial run record
            if self.db_connector:
                await self.circuit_breaker.call(
                    self.db_connector.save_pipeline_run, pipeline_run
                )

            # Process new conversations
            conversation_results = await self.process_new_conversations()
            self.metrics.conversations_processed = len(conversation_results)

            # Extract and count learning items
            all_learning_items = []
            for result in conversation_results:
                all_learning_items.extend(result.learning_items)
                self.metrics.errors_encountered += len(result.errors)

            self.metrics.learning_items_extracted = len(all_learning_items)

            # Update vector database
            vectors_updated = await self.update_vector_database(all_learning_items)
            self.metrics.vectors_updated = vectors_updated

            # Calculate final metrics
            self.metrics.end_time = datetime.now()
            self.metrics.processing_time = time.time() - start_time

            # Update pipeline run record
            pipeline_run.completed_at = self.metrics.end_time
            pipeline_run.status = ProcessingStatus.COMPLETED
            pipeline_run.conversations_processed = self.metrics.conversations_processed
            pipeline_run.learning_items_extracted = self.metrics.learning_items_extracted
            pipeline_run.vectors_updated = self.metrics.vectors_updated
            pipeline_run.errors_encountered = self.metrics.errors_encountered
            pipeline_run.performance_metrics = self.metrics.to_dict()

            if self.db_connector:
                await self.circuit_breaker.call(
                    self.db_connector.update_pipeline_run, pipeline_run
                )

            logger.info(
                f"Learning cycle {self.run_id} completed successfully. "
                f"Processed {self.metrics.conversations_processed} conversations, "
                f"extracted {self.metrics.learning_items_extracted} items, "
                f"updated {self.metrics.vectors_updated} vectors"
            )

            return pipeline_run

        except Exception as e:
            self.metrics.end_time = datetime.now()
            self.metrics.processing_time = time.time() - start_time
            self.metrics.errors_encountered += 1

            logger.error(f"Learning cycle {self.run_id} failed: {e}")

            # Update pipeline run with failure
            if 'pipeline_run' in locals():
                pipeline_run.completed_at = self.metrics.end_time
                pipeline_run.status = ProcessingStatus.FAILED
                pipeline_run.error_message = str(e)
                pipeline_run.performance_metrics = self.metrics.to_dict()

                if self.db_connector:
                    try:
                        await self.db_connector.update_pipeline_run(pipeline_run)
                    except Exception as db_error:
                        logger.error(f"Failed to update failed pipeline run: {db_error}")

            raise

    async def process_new_conversations(self) -> List[ProcessingResult]:
        """
        Process new Claude conversations since last run.

        Returns:
            List of processing results
        """
        logger.info("Processing new conversations")

        try:
            # Get list of conversation files to process
            new_files = await self.claude_connector.get_new_conversations(
                since=self._get_last_run_time()
            )

            if not new_files:
                logger.info("No new conversations to process")
                return []

            logger.info(f"Found {len(new_files)} new/modified conversations")

            # Process files in batches
            results = []
            batch_size = self.config.pipeline.batch_size

            for i in range(0, len(new_files), batch_size):
                batch = new_files[i:i + batch_size]
                batch_results = await self._process_conversation_batch(batch)
                results.extend(batch_results)

                # Log progress
                processed_count = min(i + batch_size, len(new_files))
                logger.info(f"Processed {processed_count}/{len(new_files)} conversations")

            return results

        except Exception as e:
            logger.error(f"Error processing conversations: {e}")
            raise

    async def _process_conversation_batch(self, files: List[Path]) -> List[ProcessingResult]:
        """Process a batch of conversation files concurrently."""
        semaphore = asyncio.Semaphore(self.config.pipeline.max_concurrent_processors)

        async def process_file(file_path: Path) -> ProcessingResult:
            async with semaphore:
                try:
                    return await self.circuit_breaker.call(
                        self.conversation_processor.process_file, file_path
                    )
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    self.failed_files.add(str(file_path))
                    result = ProcessingResult()
                    result.add_error(f"Processing failed: {e}")
                    return result

        # Process files concurrently
        tasks = [process_file(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing exception: {result}")
                error_result = ProcessingResult()
                error_result.add_error(str(result))
                valid_results.append(error_result)
            else:
                valid_results.append(result)

        return valid_results

    async def update_vector_database(self, learning_items: List[LearningItem]) -> int:
        """
        Update Qdrant with new embeddings.

        Args:
            learning_items: Items to add to vector database

        Returns:
            Number of vectors updated
        """
        if not learning_items:
            logger.info("No learning items to update in vector database")
            return 0

        logger.info(f"Updating vector database with {len(learning_items)} items")

        try:
            vectors_updated = await self.circuit_breaker.call(
                self.vector_connector.add_learning_items, learning_items
            )

            # Save learning items to PostgreSQL
            if self.db_connector:
                await self.circuit_breaker.call(
                    self.db_connector.save_learning_items, learning_items
                )

            logger.info(f"Successfully updated {vectors_updated} vectors")
            return vectors_updated

        except Exception as e:
            logger.error(f"Failed to update vector database: {e}")
            raise

    def _get_last_run_time(self) -> Optional[datetime]:
        """Get timestamp of last successful pipeline run."""
        # This would query the database for the last successful run
        # For now, return None to process all files
        # TODO: Implement proper last run time tracking
        return None

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        circuit_metrics = self.circuit_breaker.get_metrics()

        return {
            'run_id': self.run_id,
            'status': 'running' if self.metrics.start_time and not self.metrics.end_time else 'completed',
            'metrics': self.metrics.to_dict(),
            'circuit_breaker': circuit_metrics,
            'processed_files': len(self.processed_files),
            'failed_files': len(self.failed_files),
            'config': {
                'batch_size': self.config.pipeline.batch_size,
                'max_concurrent_processors': self.config.pipeline.max_concurrent_processors
            }
        }

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up learning pipeline {self.run_id}")

        if self.db_connector:
            await self.db_connector.disconnect()

        if self.vector_connector:
            await self.vector_connector.disconnect()

        logger.info("Learning pipeline cleanup completed")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            'pipeline_id': self.run_id,
            'healthy': True,
            'components': {}
        }

        # Check database connection
        if self.db_connector:
            try:
                db_healthy = await self.circuit_breaker.call(
                    self.db_connector.health_check
                )
                health_status['components']['database'] = {'healthy': db_healthy}
            except Exception as e:
                health_status['components']['database'] = {
                    'healthy': False,
                    'error': str(e)
                }
                health_status['healthy'] = False

        # Check vector database connection
        if self.vector_connector:
            try:
                vector_healthy = await self.circuit_breaker.call(
                    self.vector_connector.health_check
                )
                health_status['components']['vector_database'] = {'healthy': vector_healthy}
            except Exception as e:
                health_status['components']['vector_database'] = {
                    'healthy': False,
                    'error': str(e)
                }
                health_status['healthy'] = False

        # Check Claude connector
        try:
            claude_healthy = await self.claude_connector.health_check()
            health_status['components']['claude_connector'] = {'healthy': claude_healthy}
        except Exception as e:
            health_status['components']['claude_connector'] = {
                'healthy': False,
                'error': str(e)
            }
            health_status['healthy'] = False

        # Add circuit breaker status
        health_status['components']['circuit_breaker'] = self.circuit_breaker.get_metrics()

        return health_status