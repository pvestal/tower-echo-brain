"""
Knowledge Base connector for the learning pipeline.
Interfaces with WikiJS API to fetch and process KB articles.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import aiohttp
import json

from ..config.settings import PipelineConfig

logger = logging.getLogger(__name__)


class KnowledgeBaseConnector:
    """
    Connector for Knowledge Base (WikiJS) integration.

    Features:
    - Article fetching from WikiJS API
    - Category-based filtering
    - Modification time tracking
    - Batch processing support
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        kb_config = config.knowledge_base_config

        self.api_url = kb_config.get('api_url', 'http://localhost:8307/api')
        self.timeout = kb_config.get('timeout', 30)
        self.batch_size = kb_config.get('batch_size', 20)
        self.include_categories = kb_config.get('include_categories', [])
        self.exclude_categories = kb_config.get('exclude_categories', [])
        self.min_article_length = kb_config.get('min_article_length', 200)

        self.enabled = config.knowledge_base_enabled

    async def fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch all articles from Knowledge Base API."""
        if not self.enabled:
            logger.info("Knowledge Base integration disabled")
            return []

        try:
            logger.info(f"Fetching articles from Knowledge Base: {self.api_url}")

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Fetch articles list
                async with session.get(f"{self.api_url}/articles") as response:
                    if response.status != 200:
                        raise Exception(f"API returned status {response.status}")

                    data = await response.json()
                    articles = data.get('articles', [])

                logger.info(f"Fetched {len(articles)} articles from Knowledge Base")
                return articles

        except Exception as e:
            logger.error(f"Failed to fetch articles from Knowledge Base: {e}")
            raise

    async def filter_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter articles based on configuration criteria."""
        filtered_articles = []

        for article in articles:
            # Check category filters
            category = article.get('category', '').lower()

            if self.include_categories and category not in [c.lower() for c in self.include_categories]:
                continue

            if self.exclude_categories and category in [c.lower() for c in self.exclude_categories]:
                continue

            # Check minimum length
            content = article.get('content', '')
            if len(content) < self.min_article_length:
                continue

            filtered_articles.append(article)

        logger.info(f"Filtered to {len(filtered_articles)} articles after applying criteria")
        return filtered_articles

    async def get_articles_since(self, since_date: datetime) -> List[Dict[str, Any]]:
        """Get articles modified since specific date."""
        all_articles = await self.fetch_articles()
        filtered_articles = await self.filter_articles(all_articles)

        recent_articles = []
        for article in filtered_articles:
            updated_at_str = article.get('updated_at')
            if not updated_at_str:
                continue

            try:
                # Parse ISO format date
                updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                if updated_at.replace(tzinfo=None) > since_date:
                    recent_articles.append(article)
            except Exception as e:
                logger.warning(f"Could not parse date for article {article.get('id')}: {e}")
                # Include article if we can't parse the date
                recent_articles.append(article)

        logger.info(f"Found {len(recent_articles)} articles modified since {since_date}")
        return recent_articles

    async def get_article_content(self, article_id: int) -> Optional[Dict[str, Any]]:
        """Get full content for a specific article."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(f"{self.api_url}/articles/{article_id}") as response:
                    if response.status == 404:
                        logger.warning(f"Article {article_id} not found")
                        return None

                    if response.status != 200:
                        raise Exception(f"API returned status {response.status}")

                    article = await response.json()
                    return article

        except Exception as e:
            logger.error(f"Failed to get article {article_id}: {e}")
            raise

    async def health_check(self) -> bool:
        """Check Knowledge Base API health."""
        if not self.enabled:
            return True  # Not enabled, so "healthy"

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(f"{self.api_url}/health") as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Knowledge Base health check failed: {e}")
            return False

    async def get_categories(self) -> List[str]:
        """Get list of available categories."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(f"{self.api_url}/categories") as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    return data.get('categories', [])

        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get Knowledge Base statistics."""
        try:
            all_articles = await self.fetch_articles()
            filtered_articles = await self.filter_articles(all_articles)

            categories = {}
            total_content_length = 0

            for article in filtered_articles:
                category = article.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                total_content_length += len(article.get('content', ''))

            return {
                'enabled': self.enabled,
                'total_articles': len(all_articles),
                'filtered_articles': len(filtered_articles),
                'categories': categories,
                'total_content_length': total_content_length,
                'average_article_length': total_content_length // max(len(filtered_articles), 1),
                'api_url': self.api_url,
                'include_categories': self.include_categories,
                'exclude_categories': self.exclude_categories
            }

        except Exception as e:
            logger.error(f"Failed to get KB stats: {e}")
            return {
                'enabled': self.enabled,
                'error': str(e),
                'api_url': self.api_url
            }