"""
Test suite for Knowledge Base integration - comprehensive failing tests.
These tests will fail until KB integration is implemented.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json
import aiohttp
from typing import List, Dict, Any

from src.connectors.kb_connector import KnowledgeBaseConnector
from src.processors.kb_processor import KnowledgeBaseProcessor
from src.models.learning_item import LearningItem, LearningItemType, ProcessingResult
from src.config.settings import PipelineConfig


@pytest.fixture
def kb_config():
    """Configuration for Knowledge Base integration."""
    config_dict = {
        'knowledge_base': {
            'enabled': True,
            'api_url': 'http://localhost:8307/api',
            'timeout': 30,
            'batch_size': 20,
            'include_categories': ['technical', 'development', 'troubleshooting'],
            'exclude_categories': ['personal', 'draft'],
            'min_article_length': 200
        },
        'content_processing': {
            'extract_code_blocks': True,
            'extract_commands': True,
            'extract_insights': True,
            'extract_solutions': True,
            'min_insight_length': 100
        }
    }
    return PipelineConfig(config_dict)


@pytest.fixture
def mock_kb_articles():
    """Mock Knowledge Base articles for testing."""
    return [
        {
            'id': 1,
            'title': 'Database Connection Patterns',
            'content': """# Database Connection Best Practices

## Overview
This article covers best practices for database connections in Tower services.

## Key Principles
1. Use connection pooling for better performance
2. Always use localhost for development
3. Implement circuit breakers for resilience

## Code Examples
```python
async def create_connection_pool():
    return await asyncpg.create_pool(
        host='localhost',
        port=5432,
        user='patrick',
        min_size=5,
        max_size=20
    )
```

## Common Issues
- Connection timeouts in production
- Pool exhaustion under high load
- Credential management problems

## Solutions
1. Monitor connection usage
2. Implement retry logic
3. Use environment variables for credentials
""",
            'category': 'technical',
            'created_at': '2024-12-15T10:00:00Z',
            'updated_at': '2024-12-17T15:30:00Z',
            'tags': ['database', 'postgresql', 'connections', 'performance']
        },
        {
            'id': 2,
            'title': 'Vector Database Implementation',
            'content': """# Qdrant Vector Database Setup

## Architecture
Vector database for semantic search and memory storage.

## Configuration
- Host: localhost
- Port: 6333
- Collection: claude_conversations
- Embedding dimension: 384

## Implementation Notes
```bash
# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
```

## Performance Optimization
1. Batch operations for better throughput
2. Use appropriate distance metrics
3. Index configuration for your use case

## Troubleshooting
- Check Qdrant logs for connection issues
- Verify collection configuration
- Monitor memory usage
""",
            'category': 'technical',
            'created_at': '2024-12-16T14:20:00Z',
            'updated_at': '2024-12-17T09:15:00Z',
            'tags': ['vector-database', 'qdrant', 'embeddings', 'search']
        },
        {
            'id': 3,
            'title': 'System Monitoring Setup',
            'content': """# Tower System Monitoring

## Monitoring Stack
- Prometheus for metrics
- Grafana for visualization
- Alertmanager for notifications

## Key Metrics
1. Service health checks
2. Database connection counts
3. API response times
4. Error rates

## Alert Rules
```yaml
groups:
  - name: tower_alerts
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 5m
        annotations:
          summary: "Service {{ $labels.instance }} is down"
```

## Dashboard Configuration
Create dashboards for each service with:
- Response time percentiles
- Error rate tracking
- Resource utilization
""",
            'category': 'development',
            'created_at': '2024-12-10T08:30:00Z',
            'updated_at': '2024-12-12T16:45:00Z',
            'tags': ['monitoring', 'prometheus', 'grafana', 'alerts']
        },
        {
            'id': 4,
            'title': 'Personal Notes',
            'content': 'Just some personal thoughts that should be excluded.',
            'category': 'personal',
            'created_at': '2024-12-17T12:00:00Z',
            'updated_at': '2024-12-17T12:00:00Z',
            'tags': ['personal']
        },
        {
            'id': 5,
            'title': 'Short Article',
            'content': 'This is too short.',
            'category': 'technical',
            'created_at': '2024-12-17T11:00:00Z',
            'updated_at': '2024-12-17T11:00:00Z',
            'tags': ['short']
        }
    ]


@pytest.mark.asyncio
class TestKnowledgeBaseConnector:
    """Test suite for Knowledge Base connector."""

    async def test_kb_connector_initialization(self, kb_config):
        """Test connector initializes with proper configuration."""
        connector = KnowledgeBaseConnector(kb_config)

        assert connector.api_url == 'http://localhost:8307/api'
        assert connector.timeout == 30
        assert connector.batch_size == 20
        assert 'technical' in connector.include_categories
        assert 'personal' in connector.exclude_categories

    async def test_fetch_articles_from_api(self, kb_config, mock_kb_articles):
        """Test fetching articles from Knowledge Base API."""
        connector = KnowledgeBaseConnector(kb_config)

        # Mock aiohttp response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'articles': mock_kb_articles})
            mock_get.return_value.__aenter__.return_value = mock_response

            # Fetch articles
            articles = await connector.fetch_articles()

            assert len(articles) > 0
            assert articles[0]['title'] == 'Database Connection Patterns'
            assert articles[1]['title'] == 'Vector Database Implementation'

    async def test_filter_articles_by_category(self, kb_config, mock_kb_articles):
        """Test filtering articles by category."""
        connector = KnowledgeBaseConnector(kb_config)

        # Mock API response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'articles': mock_kb_articles})
            mock_get.return_value.__aenter__.return_value = mock_response

            articles = await connector.fetch_articles()
            filtered = await connector.filter_articles(articles)

            # Should exclude 'personal' category
            categories = [article['category'] for article in filtered]
            assert 'personal' not in categories
            assert 'technical' in categories
            assert 'development' in categories

    async def test_filter_articles_by_length(self, kb_config, mock_kb_articles):
        """Test filtering articles by minimum length."""
        connector = KnowledgeBaseConnector(kb_config)

        # Mock API response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'articles': mock_kb_articles})
            mock_get.return_value.__aenter__.return_value = mock_response

            articles = await connector.fetch_articles()
            filtered = await connector.filter_articles(articles)

            # Should exclude articles shorter than min_article_length (200)
            for article in filtered:
                assert len(article['content']) >= 200

            titles = [article['title'] for article in filtered]
            assert 'Short Article' not in titles  # Should be filtered out

    async def test_get_articles_since_date(self, kb_config, mock_kb_articles):
        """Test fetching articles modified since specific date."""
        connector = KnowledgeBaseConnector(kb_config)

        # Mock API response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'articles': mock_kb_articles})
            mock_get.return_value.__aenter__.return_value = mock_response

            # Get articles since yesterday
            since_date = datetime.now() - timedelta(days=1)
            articles = await connector.get_articles_since(since_date)

            # Should only include recently updated articles
            for article in articles:
                updated_at = datetime.fromisoformat(article['updated_at'].replace('Z', '+00:00'))
                assert updated_at.replace(tzinfo=None) > since_date

    async def test_health_check(self, kb_config):
        """Test Knowledge Base health check."""
        connector = KnowledgeBaseConnector(kb_config)

        # Mock successful health check
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'status': 'healthy'})
            mock_get.return_value.__aenter__.return_value = mock_response

            health = await connector.health_check()
            assert health is True

    async def test_health_check_failure(self, kb_config):
        """Test Knowledge Base health check failure."""
        connector = KnowledgeBaseConnector(kb_config)

        # Mock failed health check
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_get.return_value.__aenter__.return_value = mock_response

            health = await connector.health_check()
            assert health is False

    async def test_api_error_handling(self, kb_config):
        """Test handling API errors gracefully."""
        connector = KnowledgeBaseConnector(kb_config)

        # Mock API timeout
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = aiohttp.ClientTimeout()

            # Should handle timeout gracefully
            with pytest.raises(Exception):  # Should raise appropriate exception
                await connector.fetch_articles()

    async def test_concurrent_api_calls(self, kb_config, mock_kb_articles):
        """Test handling concurrent API calls."""
        connector = KnowledgeBaseConnector(kb_config)

        # Mock API response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'articles': mock_kb_articles})
            mock_get.return_value.__aenter__.return_value = mock_response

            # Make concurrent calls
            tasks = [
                connector.fetch_articles(),
                connector.health_check(),
                connector.get_articles_since(datetime.now() - timedelta(days=1))
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            for result in results:
                assert not isinstance(result, Exception)


@pytest.mark.asyncio
class TestKnowledgeBaseProcessor:
    """Test suite for Knowledge Base content processor."""

    async def test_processor_initialization(self, kb_config):
        """Test processor initializes with proper configuration."""
        processor = KnowledgeBaseProcessor(kb_config)

        assert processor.extract_code_blocks is True
        assert processor.extract_insights is True
        assert processor.min_insight_length == 100

    async def test_process_kb_article(self, kb_config, mock_kb_articles):
        """Test processing a Knowledge Base article."""
        processor = KnowledgeBaseProcessor(kb_config)

        # Process the database article
        article = mock_kb_articles[0]  # Database Connection Patterns
        result = await processor.process_article(article)

        assert isinstance(result, ProcessingResult)
        assert len(result.learning_items) > 0
        assert len(result.errors) == 0

        # Should extract different types of learning items
        item_types = [item.item_type for item in result.learning_items]
        assert LearningItemType.INSIGHT in item_types
        assert LearningItemType.CODE_EXAMPLE in item_types

        # Should include article metadata
        for item in result.learning_items:
            assert item.source_file == f"kb_article_{article['id']}"
            assert 'database' in [tag.lower() for tag in item.tags] if hasattr(item, 'tags') else True

    async def test_extract_code_blocks_from_article(self, kb_config, mock_kb_articles):
        """Test extracting code blocks from KB articles."""
        processor = KnowledgeBaseProcessor(kb_config)

        # Process vector database article (has Python and Bash code)
        article = mock_kb_articles[1]  # Vector Database Implementation
        result = await processor.process_article(article)

        # Should find code examples
        code_items = [item for item in result.learning_items if item.item_type == LearningItemType.CODE_EXAMPLE]
        assert len(code_items) > 0

        # Should extract both Python and Bash code
        code_content = " ".join(item.content for item in code_items)
        assert "QdrantClient" in code_content  # Python code
        assert "docker run" in code_content    # Bash code

    async def test_extract_insights_from_article(self, kb_config, mock_kb_articles):
        """Test extracting insights from KB articles."""
        processor = KnowledgeBaseProcessor(kb_config)

        article = mock_kb_articles[0]  # Database Connection Patterns
        result = await processor.process_article(article)

        # Should find insights
        insight_items = [item for item in result.learning_items if item.item_type == LearningItemType.INSIGHT]
        assert len(insight_items) > 0

        # Should contain key insights about database practices
        insight_content = " ".join(item.content for item in insight_items)
        assert any("connection pooling" in content.lower() for content in insight_content.split())

    async def test_extract_solutions_from_article(self, kb_config, mock_kb_articles):
        """Test extracting solutions from KB articles."""
        processor = KnowledgeBaseProcessor(kb_config)

        article = mock_kb_articles[0]  # Database Connection Patterns
        result = await processor.process_article(article)

        # Should find solutions
        solution_items = [item for item in result.learning_items if item.item_type == LearningItemType.SOLUTION]
        assert len(solution_items) > 0

        # Solutions should reference specific problems
        for item in solution_items:
            assert len(item.content) > 50  # Should be substantial

    async def test_article_metadata_extraction(self, kb_config, mock_kb_articles):
        """Test extracting metadata from articles."""
        processor = KnowledgeBaseProcessor(kb_config)

        article = mock_kb_articles[2]  # System Monitoring Setup
        result = await processor.process_article(article)

        # Should preserve article metadata in learning items
        for item in result.learning_items:
            assert item.source_file == f"kb_article_{article['id']}"
            assert hasattr(item, 'created_at') or hasattr(item, 'metadata')
            # Should include tags from article
            if hasattr(item, 'tags'):
                article_tags = article['tags']
                # At least some overlap between item tags and article tags
                assert any(tag in article_tags for tag in item.tags if item.tags)

    async def test_confidence_scoring_for_articles(self, kb_config, mock_kb_articles):
        """Test confidence scoring for KB article content."""
        processor = KnowledgeBaseProcessor(kb_config)

        article = mock_kb_articles[1]  # Vector Database Implementation
        result = await processor.process_article(article)

        # All items should have confidence scores
        for item in result.learning_items:
            assert hasattr(item, 'confidence_score')
            assert 0.0 <= item.confidence_score <= 1.0

        # Technical articles should generally have high confidence
        avg_confidence = sum(item.confidence_score for item in result.learning_items) / len(result.learning_items)
        assert avg_confidence > 0.6  # KB articles are typically high-quality

    async def test_batch_article_processing(self, kb_config, mock_kb_articles):
        """Test batch processing of multiple KB articles."""
        processor = KnowledgeBaseProcessor(kb_config)

        # Process multiple articles
        articles = mock_kb_articles[:3]  # First 3 articles
        results = await processor.process_articles_batch(articles)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert len(result.learning_items) > 0

        # Should extract items from all articles
        total_items = sum(len(result.learning_items) for result in results)
        assert total_items > 5  # Should have substantial content

    async def test_error_handling_malformed_article(self, kb_config):
        """Test error handling for malformed articles."""
        processor = KnowledgeBaseProcessor(kb_config)

        # Malformed article
        bad_article = {
            'id': 999,
            # Missing required fields
            'content': None,
            'category': 'technical'
        }

        result = await processor.process_article(bad_article)

        assert isinstance(result, ProcessingResult)
        assert len(result.errors) > 0
        # Should handle gracefully without crashing

    async def test_concurrent_article_processing(self, kb_config, mock_kb_articles):
        """Test concurrent processing of articles."""
        processor = KnowledgeBaseProcessor(kb_config)

        # Process articles concurrently
        tasks = [
            processor.process_article(article)
            for article in mock_kb_articles[:3]
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, ProcessingResult)

    async def test_article_deduplication(self, kb_config, mock_kb_articles):
        """Test handling duplicate or similar articles."""
        processor = KnowledgeBaseProcessor(kb_config)

        # Create duplicate article
        duplicate_article = mock_kb_articles[0].copy()
        duplicate_article['id'] = 999
        duplicate_article['title'] = 'Duplicate Database Article'

        result1 = await processor.process_article(mock_kb_articles[0])
        result2 = await processor.process_article(duplicate_article)

        # Should process both but potentially mark similarities
        assert len(result1.learning_items) > 0
        assert len(result2.learning_items) > 0

        # Could implement similarity detection in future