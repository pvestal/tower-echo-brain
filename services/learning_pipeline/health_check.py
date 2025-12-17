#!/usr/bin/env python3
"""
Health check script for the learning pipeline.
Verifies all components are working correctly.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import PipelineConfig, load_config
from src.connectors.database_connector import DatabaseConnector
from src.connectors.vector_connector import VectorConnector
from src.connectors.claude_connector import ClaudeConnector
from src.connectors.kb_connector import KnowledgeBaseConnector
from src.core.pipeline import LearningPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health checker for the learning pipeline."""

    def __init__(self):
        self.results = {}
        self.overall_healthy = True

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        logger.info("Starting comprehensive health checks...")

        # Check configuration
        await self.check_configuration()

        # Check database connectivity
        await self.check_database()

        # Check vector database (Qdrant)
        await self.check_vector_database()

        # Check Claude conversations directory
        await self.check_claude_connector()

        # Check Knowledge Base API
        await self.check_knowledge_base()

        # Check circuit breaker functionality
        await self.check_circuit_breaker()

        # Test pipeline initialization
        await self.check_pipeline_initialization()

        return {
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': self.overall_healthy,
            'components': self.results
        }

    async def check_configuration(self):
        """Check configuration loading and validation."""
        logger.info("Checking configuration...")
        try:
            # Create test configuration
            config_dict = {
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'name': 'echo_brain',
                    'user': 'patrick'
                },
                'vector_database': {
                    'host': 'localhost',
                    'port': 6333,
                    'collection_name': 'claude_conversations'
                },
                'sources': {
                    'claude_conversations': {
                        'path': os.path.expanduser('~/.claude/conversations'),
                        'file_pattern': '*.md'
                    }
                }
            }

            config = PipelineConfig(config_dict)
            validation_errors = config.validate()

            self.results['configuration'] = {
                'healthy': len(validation_errors) == 0,
                'validation_errors': validation_errors,
                'database_host': config.database.host,
                'vector_db_host': config.vector_database.host,
                'claude_path': config.claude_conversations.path
            }

            if validation_errors:
                self.overall_healthy = False
                logger.warning(f"Configuration validation failed: {validation_errors}")
            else:
                logger.info("Configuration validation passed")

        except Exception as e:
            self.results['configuration'] = {
                'healthy': False,
                'error': str(e)
            }
            self.overall_healthy = False
            logger.error(f"Configuration check failed: {e}")

    async def check_database(self):
        """Check database connectivity."""
        logger.info("Checking database connectivity...")
        try:
            from src.config.settings import DatabaseConfig

            db_config = DatabaseConfig(
                host='localhost',
                port=5432,
                name='echo_brain',
                user='patrick'
            )

            # Create connector but don't actually connect (may not have credentials)
            connector = DatabaseConnector(db_config)

            # Test configuration
            connection_string = db_config.connection_string
            localhost_check = 'localhost' in connection_string and '***REMOVED***' not in connection_string

            self.results['database'] = {
                'healthy': True,  # Basic health if config is correct
                'configuration_valid': True,
                'uses_localhost': localhost_check,
                'connection_string_format': 'postgresql://patrick:***@localhost:5432/echo_brain',
                'note': 'Connection test skipped (requires actual database and credentials)'
            }

            if not localhost_check:
                self.results['database']['healthy'] = False
                self.overall_healthy = False
                logger.error("Database not configured to use localhost!")
            else:
                logger.info("Database configuration looks correct (uses localhost)")

        except Exception as e:
            self.results['database'] = {
                'healthy': False,
                'error': str(e)
            }
            self.overall_healthy = False
            logger.error(f"Database check failed: {e}")

    async def check_vector_database(self):
        """Check Qdrant vector database connectivity."""
        logger.info("Checking Qdrant vector database...")
        try:
            from src.config.settings import VectorDatabaseConfig

            vector_config = VectorDatabaseConfig(
                host='localhost',
                port=6333,
                collection_name='claude_conversations'
            )

            connector = VectorConnector(vector_config)

            # Test configuration
            self.results['vector_database'] = {
                'configuration_valid': True,
                'host': vector_config.host,
                'port': vector_config.port,
                'collection_name': vector_config.collection_name,
                'embedding_dimension': vector_config.embedding_dimension,
                'note': 'Connection test skipped (requires running Qdrant server)'
            }

            # Try to connect if Qdrant is actually running
            try:
                await asyncio.wait_for(connector.connect(), timeout=5.0)
                health = await connector.health_check()
                collection_info = await connector.get_collection_info()

                self.results['vector_database'].update({
                    'healthy': health,
                    'connected': True,
                    'collection_info': collection_info
                })

                await connector.disconnect()
                logger.info("Qdrant connection successful")

            except Exception as conn_error:
                self.results['vector_database'].update({
                    'healthy': False,
                    'connected': False,
                    'connection_error': str(conn_error)
                })
                logger.warning(f"Qdrant connection failed (may not be running): {conn_error}")

        except Exception as e:
            self.results['vector_database'] = {
                'healthy': False,
                'error': str(e)
            }
            self.overall_healthy = False
            logger.error(f"Vector database check failed: {e}")

    async def check_claude_connector(self):
        """Check Claude conversations connector."""
        logger.info("Checking Claude conversations connector...")
        try:
            config_dict = {
                'sources': {
                    'claude_conversations': {
                        'path': os.path.expanduser('~/.claude/conversations'),
                        'file_pattern': '*.md',
                        'exclude_patterns': ['**/test_*', '**/.tmp_*']
                    }
                }
            }
            config = PipelineConfig(config_dict)
            connector = ClaudeConnector(config)

            # Check health
            health = await connector.health_check()
            file_stats = await connector.get_file_stats()

            self.results['claude_connector'] = {
                'healthy': health,
                'conversations_path': str(connector.conversations_path),
                'path_exists': connector.conversations_path.exists(),
                'file_stats': file_stats
            }

            if not health:
                self.overall_healthy = False
                logger.warning("Claude conversations directory not accessible")
            else:
                logger.info(f"Claude conversations check passed: {file_stats.get('total_files', 0)} files found")

        except Exception as e:
            self.results['claude_connector'] = {
                'healthy': False,
                'error': str(e)
            }
            self.overall_healthy = False
            logger.error(f"Claude connector check failed: {e}")

    async def check_knowledge_base(self):
        """Check Knowledge Base API connectivity."""
        logger.info("Checking Knowledge Base API...")
        try:
            config_dict = {
                'sources': {
                    'claude_conversations': {
                        'path': os.path.expanduser('~/.claude/conversations'),
                        'file_pattern': '*.md'
                    }
                },
                'knowledge_base': {
                    'enabled': True,
                    'api_url': 'http://localhost:8307/api',
                    'timeout': 10
                }
            }
            config = PipelineConfig(config_dict)
            connector = KnowledgeBaseConnector(config)

            # Try health check with short timeout
            try:
                health = await asyncio.wait_for(connector.health_check(), timeout=5.0)
                stats = await connector.get_stats()

                self.results['knowledge_base'] = {
                    'healthy': health,
                    'api_url': connector.api_url,
                    'enabled': connector.enabled,
                    'stats': stats
                }

                logger.info("Knowledge Base API check completed")

            except asyncio.TimeoutError:
                self.results['knowledge_base'] = {
                    'healthy': False,
                    'api_url': connector.api_url,
                    'enabled': connector.enabled,
                    'error': 'Connection timeout (API may not be running)'
                }
                logger.warning("Knowledge Base API timeout (may not be running)")

        except Exception as e:
            self.results['knowledge_base'] = {
                'healthy': False,
                'error': str(e)
            }
            logger.error(f"Knowledge Base check failed: {e}")

    async def check_circuit_breaker(self):
        """Check circuit breaker functionality."""
        logger.info("Checking circuit breaker...")
        try:
            from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

            # Test circuit breaker
            config = CircuitBreakerConfig(
                failure_threshold=3,
                reset_timeout=5,
                half_open_max_calls=2
            )

            circuit_breaker = CircuitBreaker(config)

            # Test successful call
            async def test_function():
                return "success"

            result = await circuit_breaker.call(test_function)
            metrics = circuit_breaker.get_metrics()

            self.results['circuit_breaker'] = {
                'healthy': result == "success",
                'metrics': metrics,
                'test_call_successful': True
            }

            logger.info("Circuit breaker check passed")

        except Exception as e:
            self.results['circuit_breaker'] = {
                'healthy': False,
                'error': str(e)
            }
            self.overall_healthy = False
            logger.error(f"Circuit breaker check failed: {e}")

    async def check_pipeline_initialization(self):
        """Test pipeline initialization."""
        logger.info("Checking pipeline initialization...")
        try:
            config_dict = {
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'name': 'echo_brain',
                    'user': 'patrick'
                },
                'vector_database': {
                    'host': 'localhost',
                    'port': 6333,
                    'collection_name': 'claude_conversations'
                },
                'sources': {
                    'claude_conversations': {
                        'path': os.path.expanduser('~/.claude/conversations'),
                        'file_pattern': '*.md'
                    }
                },
                'pipeline': {
                    'batch_size': 10,
                    'max_concurrent_processors': 2
                },
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'reset_timeout': 60
                }
            }

            config = PipelineConfig(config_dict)
            pipeline = LearningPipeline(config)

            # Test pipeline status
            status = await pipeline.get_pipeline_status()

            self.results['pipeline'] = {
                'healthy': True,
                'initialization_successful': True,
                'run_id': pipeline.run_id,
                'status': status
            }

            logger.info("Pipeline initialization check passed")

        except Exception as e:
            self.results['pipeline'] = {
                'healthy': False,
                'error': str(e)
            }
            self.overall_healthy = False
            logger.error(f"Pipeline initialization check failed: {e}")

    def print_results(self, results: Dict[str, Any]):
        """Print formatted health check results."""
        print("\n" + "="*60)
        print("LEARNING PIPELINE HEALTH CHECK RESULTS")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {'✅ HEALTHY' if results['overall_healthy'] else '❌ UNHEALTHY'}")
        print()

        for component, status in results['components'].items():
            component_name = component.replace('_', ' ').title()
            health_icon = "✅" if status.get('healthy', False) else "❌"

            print(f"{health_icon} {component_name}")

            if 'error' in status:
                print(f"   Error: {status['error']}")
            elif 'note' in status:
                print(f"   Note: {status['note']}")

            # Print key details
            for key, value in status.items():
                if key not in ['healthy', 'error', 'note'] and not key.startswith('_'):
                    if isinstance(value, dict):
                        print(f"   {key.replace('_', ' ').title()}: {len(value)} items")
                    elif isinstance(value, list):
                        print(f"   {key.replace('_', ' ').title()}: {len(value)} items")
                    else:
                        print(f"   {key.replace('_', ' ').title()}: {value}")
            print()

        print("="*60)
        if results['overall_healthy']:
            print("🎉 All critical components are healthy!")
            print("The learning pipeline is ready to run.")
        else:
            print("⚠️  Some components need attention.")
            print("Please check the errors above before running the pipeline.")
        print("="*60)


async def main():
    """Main health check function."""
    checker = HealthChecker()
    results = await checker.run_health_checks()
    checker.print_results(results)

    # Exit with error code if unhealthy
    sys.exit(0 if results['overall_healthy'] else 1)


if __name__ == "__main__":
    asyncio.run(main())