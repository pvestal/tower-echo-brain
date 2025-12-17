"""
Knowledge Base article processor for extracting learning items.
Processes WikiJS articles to extract insights, code examples, and solutions.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config.settings import PipelineConfig
from ..models.learning_item import LearningItem, LearningItemType, ProcessingResult

logger = logging.getLogger(__name__)


class KnowledgeBaseProcessor:
    """
    Processes Knowledge Base articles to extract learning items.

    Features:
    - Structured content extraction
    - Code block parsing
    - Technical insight identification
    - Solution extraction
    - Article metadata preservation
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.extract_code_blocks = config.content_processing.extract_code_blocks
        self.extract_insights = config.content_processing.extract_insights
        self.extract_solutions = config.content_processing.extract_solutions
        self.min_insight_length = config.content_processing.min_insight_length
        self.code_block_languages = config.content_processing.code_block_languages

        # Compile regex patterns for performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for content extraction."""
        # Code blocks (triple backticks and indented blocks)
        self.code_block_pattern = re.compile(
            r'```(\w+)?\s*\n(.*?)\n```',
            re.DOTALL | re.MULTILINE
        )

        # Inline code
        self.inline_code_pattern = re.compile(r'`([^`\n]+)`')

        # Solution sections in KB articles
        self.solution_patterns = [
            re.compile(r'#{1,3}\s*Solutions?\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'#{1,3}\s*Implementation\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'#{1,3}\s*How to.*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'#{1,3}\s*Setup\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'#{1,3}\s*Configuration\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
        ]

        # Best practices and insights
        self.insight_patterns = [
            re.compile(r'#{1,3}\s*(?:Best Practices?|Key Points?)\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'#{1,3}\s*(?:Important|Critical|Note)\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'#{1,3}\s*(?:Tips?|Guidelines?)\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'#{1,3}\s*(?:Architecture|Design)\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
        ]

        # Troubleshooting sections
        self.troubleshooting_patterns = [
            re.compile(r'#{1,3}\s*(?:Troubleshooting|Common Issues?)\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'#{1,3}\s*(?:Problems?|Errors?)\s*\n(.*?)(?=\n#{1,3}|\n---|\n\*\*|$)', re.DOTALL | re.IGNORECASE),
        ]

    async def process_article(self, article: Dict[str, Any]) -> ProcessingResult:
        """Process a single Knowledge Base article."""
        result = ProcessingResult()

        try:
            article_id = article.get('id')
            title = article.get('title', 'Unknown Article')
            content = article.get('content', '')
            category = article.get('category', '')
            tags = article.get('tags', [])

            if not content:
                result.add_error(f"Article {article_id} has no content")
                return result

            logger.debug(f"Processing KB article: {title}")

            # Extract different types of learning items
            if self.extract_code_blocks:
                code_items = await self._extract_code_blocks(content, article)
                result.learning_items.extend(code_items)

            if self.extract_insights:
                insight_items = await self._extract_insights(content, article)
                result.learning_items.extend(insight_items)

            if self.extract_solutions:
                solution_items = await self._extract_solutions(content, article)
                result.learning_items.extend(solution_items)

            # Extract troubleshooting information
            troubleshooting_items = await self._extract_troubleshooting(content, article)
            result.learning_items.extend(troubleshooting_items)

            # Add article metadata to all items
            for item in result.learning_items:
                item.source_file = f"kb_article_{article_id}"
                item.category = category
                item.tags = tags.copy() if tags else []

                # Add article-specific metadata
                if not hasattr(item, 'metadata'):
                    item.metadata = {}
                item.metadata.update({
                    'article_id': article_id,
                    'article_title': title,
                    'article_category': category,
                    'article_tags': tags,
                    'created_at': article.get('created_at'),
                    'updated_at': article.get('updated_at')
                })

            logger.info(f"Extracted {len(result.learning_items)} learning items from article: {title}")

        except Exception as e:
            error_msg = f"Error processing article {article.get('id', 'unknown')}: {e}"
            logger.error(error_msg)
            result.add_error(error_msg)

        return result

    async def process_articles_batch(self, articles: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple articles concurrently."""
        logger.info(f"Processing batch of {len(articles)} KB articles")

        # Process articles concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

        async def process_single(article):
            async with semaphore:
                return await self.process_article(article)

        tasks = [process_single(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to ProcessingResults
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception processing article {articles[i].get('id')}: {result}")
                error_result = ProcessingResult()
                error_result.add_error(str(result))
                valid_results.append(error_result)
            else:
                valid_results.append(result)

        total_items = sum(len(result.learning_items) for result in valid_results)
        logger.info(f"Batch processing completed. Total items extracted: {total_items}")

        return valid_results

    async def _extract_code_blocks(self, content: str, article: Dict[str, Any]) -> List[LearningItem]:
        """Extract code blocks from article content."""
        items = []

        # Extract fenced code blocks
        for match in self.code_block_pattern.finditer(content):
            language = match.group(1) or 'text'
            code = match.group(2).strip()

            if not code or len(code) < 10:  # Skip very short code blocks
                continue

            # Filter by language if specified
            if self.code_block_languages and language.lower() not in [
                lang.lower() for lang in self.code_block_languages
            ]:
                continue

            # Determine title from surrounding context
            title = self._extract_title_from_context(content, match.start()) or f"{language.title()} Example"

            confidence_score = self._calculate_code_confidence(code, language, article)

            item = LearningItem(
                content=code,
                item_type=LearningItemType.CODE_EXAMPLE,
                title=title,
                confidence_score=confidence_score
            )

            # Add language metadata
            item.metadata = {
                'language': language,
                'line_count': len(code.split('\n')),
                'context_title': title,
                'extraction_source': 'kb_article'
            }

            items.append(item)

        logger.debug(f"Extracted {len(items)} code blocks from article {article.get('id')}")
        return items

    async def _extract_insights(self, content: str, article: Dict[str, Any]) -> List[LearningItem]:
        """Extract insights from article content."""
        items = []

        # Extract from structured insight sections
        for pattern in self.insight_patterns:
            for match in pattern.finditer(content):
                insight_content = match.group(1).strip()

                # Split into individual insights
                insights = self._split_into_insights(insight_content)

                for insight in insights:
                    if len(insight) < self.min_insight_length:
                        continue

                    confidence_score = self._calculate_insight_confidence(insight, article)

                    item = LearningItem(
                        content=insight,
                        item_type=LearningItemType.INSIGHT,
                        title=self._generate_insight_title(insight),
                        confidence_score=confidence_score
                    )

                    item.metadata = {
                        'source_section': 'structured_insight',
                        'length': len(insight),
                        'extraction_source': 'kb_article'
                    }

                    items.append(item)

        logger.debug(f"Extracted {len(items)} insights from article {article.get('id')}")
        return items

    async def _extract_solutions(self, content: str, article: Dict[str, Any]) -> List[LearningItem]:
        """Extract solutions from article content."""
        items = []

        # Extract from structured solution sections
        for pattern in self.solution_patterns:
            for match in pattern.finditer(content):
                solution_content = match.group(1).strip()

                # Split into individual solutions
                solutions = self._split_into_solutions(solution_content)

                for solution in solutions:
                    if len(solution) < 50:  # Skip very short solutions
                        continue

                    confidence_score = self._calculate_solution_confidence(solution, article)

                    item = LearningItem(
                        content=solution,
                        item_type=LearningItemType.SOLUTION,
                        title=self._generate_solution_title(solution),
                        confidence_score=confidence_score
                    )

                    item.metadata = {
                        'source_section': 'structured_solution',
                        'length': len(solution),
                        'extraction_source': 'kb_article'
                    }

                    items.append(item)

        logger.debug(f"Extracted {len(items)} solutions from article {article.get('id')}")
        return items

    async def _extract_troubleshooting(self, content: str, article: Dict[str, Any]) -> List[LearningItem]:
        """Extract troubleshooting information."""
        items = []

        for pattern in self.troubleshooting_patterns:
            for match in pattern.finditer(content):
                troubleshooting_content = match.group(1).strip()

                # Split into individual problems/solutions
                problems = self._split_into_solutions(troubleshooting_content)

                for problem in problems:
                    if len(problem) < 30:
                        continue

                    confidence_score = self._calculate_solution_confidence(problem, article)

                    item = LearningItem(
                        content=problem,
                        item_type=LearningItemType.SOLUTION,  # Troubleshooting is a type of solution
                        title=self._generate_troubleshooting_title(problem),
                        confidence_score=confidence_score
                    )

                    item.metadata = {
                        'source_section': 'troubleshooting',
                        'length': len(problem),
                        'extraction_source': 'kb_article'
                    }

                    items.append(item)

        logger.debug(f"Extracted {len(items)} troubleshooting items from article {article.get('id')}")
        return items

    def _extract_title_from_context(self, content: str, position: int) -> Optional[str]:
        """Extract title from surrounding context."""
        # Look for markdown headers above the position
        content_before = content[:position]
        lines = content_before.split('\n')

        # Search backwards for headers
        for line in reversed(lines[-10:]):  # Check last 10 lines
            line = line.strip()
            if line.startswith('#'):
                title = re.sub(r'^#+\s*', '', line)
                return title[:100]  # Limit title length

        return None

    def _split_into_insights(self, content: str) -> List[str]:
        """Split insight content into individual insights."""
        insights = []

        # Split by list markers or paragraphs
        lines = content.split('\n')
        current_insight = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_insight:
                    insight = '\n'.join(current_insight).strip()
                    if insight:
                        insights.append(insight)
                    current_insight = []
            elif re.match(r'^[-*•]\s+', line) or re.match(r'^\d+\.\s+', line):
                # New list item
                if current_insight:
                    insight = '\n'.join(current_insight).strip()
                    if insight:
                        insights.append(insight)
                current_insight = [line]
            else:
                current_insight.append(line)

        # Add final insight
        if current_insight:
            insight = '\n'.join(current_insight).strip()
            if insight:
                insights.append(insight)

        return insights

    def _split_into_solutions(self, content: str) -> List[str]:
        """Split solution content into individual solutions."""
        solutions = []

        # Split by numbered items or clear breaks
        parts = re.split(r'\n\s*(?:\d+\.|\*|-|#{1,3})\s+', content)

        for part in parts:
            part = part.strip()
            if part and len(part) > 20:  # Minimum solution length
                solutions.append(part)

        return solutions

    def _calculate_code_confidence(self, code: str, language: str, article: Dict[str, Any]) -> float:
        """Calculate confidence score for KB code blocks."""
        score = 0.8  # Higher base score for KB articles (curated content)

        # Language-specific patterns
        if language.lower() in ['python', 'javascript', 'bash', 'sql', 'yaml']:
            score += 0.1

        # Technical article categories
        category = article.get('category', '').lower()
        if category in ['technical', 'development', 'infrastructure']:
            score += 0.05

        # Length factor
        if len(code) > 100:
            score += 0.05

        return min(score, 1.0)

    def _calculate_insight_confidence(self, insight: str, article: Dict[str, Any]) -> float:
        """Calculate confidence score for KB insights."""
        score = 0.8  # Higher base score for KB articles

        # Technical categories increase confidence
        category = article.get('category', '').lower()
        if category in ['technical', 'development', 'troubleshooting']:
            score += 0.1

        # Article tags
        tags = article.get('tags', [])
        tech_tags = ['database', 'api', 'service', 'security', 'performance', 'monitoring']
        if any(tag.lower() in tech_tags for tag in tags):
            score += 0.05

        # Length factor
        if len(insight) > 200:
            score += 0.05

        return min(score, 1.0)

    def _calculate_solution_confidence(self, solution: str, article: Dict[str, Any]) -> float:
        """Calculate confidence score for KB solutions."""
        score = 0.9  # Very high base score for KB solutions

        # Category boost
        category = article.get('category', '').lower()
        if category in ['technical', 'troubleshooting', 'development']:
            score += 0.05

        return min(score, 1.0)

    def _generate_insight_title(self, insight: str) -> str:
        """Generate title for insight."""
        words = insight.split()[:8]
        title = ' '.join(words)
        if len(title) > 80:
            title = title[:77] + '...'
        return title

    def _generate_solution_title(self, solution: str) -> str:
        """Generate title for solution."""
        words = solution.split()[:6]
        title = ' '.join(words)
        if len(title) > 80:
            title = title[:77] + '...'
        return f"Solution: {title}"

    def _generate_troubleshooting_title(self, problem: str) -> str:
        """Generate title for troubleshooting item."""
        words = problem.split()[:6]
        title = ' '.join(words)
        if len(title) > 80:
            title = title[:77] + '...'
        return f"Troubleshoot: {title}"