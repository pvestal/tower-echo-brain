"""
Conversation content processor for extracting learning items.
Parses Claude conversation files and extracts insights, code, solutions, and commands.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config.settings import PipelineConfig
from ..models.learning_item import LearningItem, LearningItemType, ProcessingResult

logger = logging.getLogger(__name__)


class ConversationProcessor:
    """
    Processes Claude conversation files to extract learning items.

    Features:
    - Code block extraction
    - Insight identification
    - Solution extraction
    - Command extraction
    - Confidence scoring
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.extract_code_blocks = config.content_processing.extract_code_blocks
        self.extract_commands = config.content_processing.extract_commands
        self.extract_insights = config.content_processing.extract_insights
        self.extract_solutions = config.content_processing.extract_solutions
        self.min_insight_length = config.content_processing.min_insight_length
        self.code_block_languages = config.content_processing.code_block_languages

        # Compile regex patterns for performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for content extraction."""
        # Code blocks (triple backticks)
        self.code_block_pattern = re.compile(
            r'```(\w+)?\s*\n(.*?)\n```',
            re.DOTALL | re.MULTILINE
        )

        # Inline code (single backticks)
        self.inline_code_pattern = re.compile(r'`([^`\n]+)`')

        # Command patterns
        self.command_patterns = [
            re.compile(r'^[$#]\s+(.+)', re.MULTILINE),  # Shell commands
            re.compile(r'^\s*(?:sudo\s+)?(?:apt|yum|pip|npm|docker)\s+.+', re.MULTILINE),  # Package managers
            re.compile(r'^\s*(?:git|curl|wget|ssh)\s+.+', re.MULTILINE),  # Common commands
        ]

        # Solution markers
        self.solution_patterns = [
            re.compile(r'## Solutions?\s*\n(.*?)(?=\n##|\n#|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'## Fix(?:es)?\s*\n(.*?)(?=\n##|\n#|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'## Implementation\s*\n(.*?)(?=\n##|\n#|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'To fix.*?:\s*\n(.*?)(?=\n##|\n#|$)', re.DOTALL | re.IGNORECASE),
        ]

        # Insight patterns
        self.insight_patterns = [
            re.compile(r'## (?:Key )?Insights?\s*\n(.*?)(?=\n##|\n#|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'## (?:Technical )?Insights?\s*\n(.*?)(?=\n##|\n#|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'## Best Practices?\s*\n(.*?)(?=\n##|\n#|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'## (?:Important|Critical)\s*\n(.*?)(?=\n##|\n#|$)', re.DOTALL | re.IGNORECASE),
        ]

    async def process_file(self, file_path: Path) -> ProcessingResult:
        """Process a single conversation file."""
        result = ProcessingResult()

        try:
            if not file_path.exists():
                result.add_error(f"File not found: {file_path}")
                return result

            # Read file content
            raw_content = await self._read_file_async(file_path)
            if not raw_content:
                result.add_error(f"Empty or unreadable file: {file_path}")
                return result

            # Extract text content (handles both JSON and markdown formats)
            content = self._extract_text_from_json_conversation(raw_content)
            if not content:
                result.add_error(f"No extractable content found in: {file_path}")
                return result

            logger.debug(f"Processing conversation file: {file_path} (extracted {len(content)} chars)")

            # Extract different types of learning items
            if self.extract_code_blocks:
                code_items = await self._extract_code_blocks(content, file_path)
                result.learning_items.extend(code_items)

            if self.extract_insights:
                insight_items = await self._extract_insights(content, file_path)
                result.learning_items.extend(insight_items)

            if self.extract_solutions:
                solution_items = await self._extract_solutions(content, file_path)
                result.learning_items.extend(solution_items)

            if self.extract_commands:
                command_items = await self._extract_commands(content, file_path)
                result.learning_items.extend(command_items)

            # Add metadata to all items
            for item in result.learning_items:
                item.source_file = str(file_path)
                if not hasattr(item, 'created_at'):
                    item.created_at = datetime.fromtimestamp(file_path.stat().st_mtime)

            logger.info(f"Extracted {len(result.learning_items)} learning items from {file_path}")

        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            logger.error(error_msg)
            result.add_error(error_msg)

        return result

    async def _read_file_async(self, file_path: Path) -> str:
        """Read file content asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, file_path.read_text, 'utf-8')

    def _extract_text_from_json_conversation(self, content: str) -> str:
        """Extract relevant text content from JSON conversation format."""
        try:
            data = json.loads(content)

            # Extract text from various possible fields in the JSON
            text_parts = []

            # Common conversation fields that might contain meaningful content
            text_fields = [
                'task', 'result', 'content', 'text', 'message', 'response',
                'question', 'answer', 'description', 'summary', 'notes',
                'conversation', 'dialogue', 'transcript'
            ]

            # Extract from top-level fields
            for field in text_fields:
                if field in data and isinstance(data[field], str) and len(data[field]) > 20:
                    text_parts.append(f"## {field.title()}\n{data[field]}")

            # Extract from nested objects (like messages in conversation arrays)
            if isinstance(data.get('messages'), list):
                for i, msg in enumerate(data['messages']):
                    if isinstance(msg, dict):
                        for field in ['content', 'text', 'message']:
                            if field in msg and isinstance(msg[field], str) and len(msg[field]) > 20:
                                role = msg.get('role', msg.get('type', f'message_{i}'))
                                text_parts.append(f"## {role.title()}\n{msg[field]}")

            # If we found text content, join it with newlines
            if text_parts:
                extracted_text = "\n\n".join(text_parts)
                logger.debug(f"Extracted {len(extracted_text)} characters from JSON conversation")
                return extracted_text
            else:
                logger.debug("No extractable text found in JSON conversation")
                return ""

        except json.JSONDecodeError:
            # Not JSON, return original content for markdown processing
            return content
        except Exception as e:
            logger.warning(f"Error extracting text from JSON conversation: {e}")
            return content

    async def _extract_code_blocks(self, content: str, file_path: Path) -> List[LearningItem]:
        """Extract code blocks from content."""
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
            title = self._extract_title_from_context(content, match.start())

            confidence_score = self._calculate_code_confidence(code, language)

            item = LearningItem(
                content=code,
                item_type=LearningItemType.CODE_EXAMPLE,
                title=title or f"{language.title()} Code Example",
                confidence_score=confidence_score
            )

            # Add language metadata
            item.metadata = {
                'language': language,
                'line_count': len(code.split('\n')),
                'context_title': title
            }

            items.append(item)

        logger.debug(f"Extracted {len(items)} code blocks from {file_path}")
        return items

    async def _extract_insights(self, content: str, file_path: Path) -> List[LearningItem]:
        """Extract insights from content."""
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

                    confidence_score = self._calculate_insight_confidence(insight)

                    item = LearningItem(
                        content=insight,
                        item_type=LearningItemType.INSIGHT,
                        title=self._generate_insight_title(insight),
                        confidence_score=confidence_score
                    )

                    item.metadata = {
                        'source_section': 'structured_insight',
                        'length': len(insight)
                    }

                    items.append(item)

        # Extract general insights from conversation
        general_insights = await self._extract_general_insights(content)
        items.extend(general_insights)

        logger.debug(f"Extracted {len(items)} insights from {file_path}")
        return items

    async def _extract_solutions(self, content: str, file_path: Path) -> List[LearningItem]:
        """Extract solutions from content."""
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

                    confidence_score = self._calculate_solution_confidence(solution)

                    item = LearningItem(
                        content=solution,
                        item_type=LearningItemType.SOLUTION,
                        title=self._generate_solution_title(solution),
                        confidence_score=confidence_score
                    )

                    item.metadata = {
                        'source_section': 'structured_solution',
                        'length': len(solution)
                    }

                    items.append(item)

        logger.debug(f"Extracted {len(items)} solutions from {file_path}")
        return items

    async def _extract_commands(self, content: str, file_path: Path) -> List[LearningItem]:
        """Extract commands from content."""
        items = []

        # Extract shell commands and tool invocations
        for pattern in self.command_patterns:
            for match in pattern.finditer(content):
                command = match.group(0).strip()

                if len(command) < 5:  # Skip very short commands
                    continue

                # Clean up command (remove prompt characters)
                clean_command = re.sub(r'^[$#]\s*', '', command)

                confidence_score = self._calculate_command_confidence(clean_command)

                item = LearningItem(
                    content=clean_command,
                    item_type=LearningItemType.COMMAND,
                    title=self._generate_command_title(clean_command),
                    confidence_score=confidence_score
                )

                item.metadata = {
                    'command_type': self._classify_command(clean_command),
                    'original_format': command
                }

                items.append(item)

        logger.debug(f"Extracted {len(items)} commands from {file_path}")
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

        # Split by list markers or line breaks
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
        parts = re.split(r'\n\s*(?:\d+\.|\*|-)\s+', content)

        for part in parts:
            part = part.strip()
            if part and len(part) > 20:  # Minimum solution length
                solutions.append(part)

        return solutions

    async def _extract_general_insights(self, content: str) -> List[LearningItem]:
        """Extract general insights from conversation content."""
        insights = []

        # Look for sentences that contain insight keywords
        insight_keywords = [
            'important to', 'key point', 'note that', 'remember that',
            'best practice', 'avoid', 'ensure that', 'make sure',
            'critical', 'essential', 'crucial', 'significant'
        ]

        sentences = re.split(r'[.!?]+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < self.min_insight_length:
                continue

            # Check for insight keywords
            if any(keyword in sentence.lower() for keyword in insight_keywords):
                confidence_score = self._calculate_insight_confidence(sentence)

                if confidence_score > 0.5:  # Only high-confidence general insights
                    item = LearningItem(
                        content=sentence,
                        item_type=LearningItemType.INSIGHT,
                        title=self._generate_insight_title(sentence),
                        confidence_score=confidence_score
                    )

                    item.metadata = {
                        'source_section': 'general_conversation',
                        'extraction_method': 'keyword_detection'
                    }

                    insights.append(item)

        return insights

    def _calculate_code_confidence(self, code: str, language: str) -> float:
        """Calculate confidence score for code blocks."""
        score = 0.7  # Base score for code blocks

        # Language-specific patterns increase confidence
        if language.lower() in ['python', 'javascript', 'bash', 'sql']:
            score += 0.1

        # Presence of common programming constructs
        if re.search(r'\b(def|class|function|async|await|import|from)\b', code):
            score += 0.1

        # Length factor
        if len(code) > 100:
            score += 0.05
        if len(code) > 500:
            score += 0.05

        return min(score, 1.0)

    def _calculate_insight_confidence(self, insight: str) -> float:
        """Calculate confidence score for insights."""
        score = 0.6  # Base score for insights

        # Technical terms increase confidence
        tech_terms = [
            'database', 'api', 'service', 'client', 'server', 'authentication',
            'configuration', 'optimization', 'performance', 'security',
            'deployment', 'integration', 'monitoring', 'logging'
        ]

        if any(term in insight.lower() for term in tech_terms):
            score += 0.2

        # Specific recommendations increase confidence
        recommendation_terms = ['should', 'must', 'recommended', 'best practice', 'avoid', 'ensure']
        if any(term in insight.lower() for term in recommendation_terms):
            score += 0.1

        # Length factor
        if len(insight) > 200:
            score += 0.05

        return min(score, 1.0)

    def _calculate_solution_confidence(self, solution: str) -> float:
        """Calculate confidence score for solutions."""
        score = 0.8  # Base score for solutions (higher than insights)

        # Action words increase confidence
        action_words = ['fix', 'solve', 'implement', 'configure', 'install', 'update', 'modify']
        if any(word in solution.lower() for word in action_words):
            score += 0.1

        # Step-by-step format
        if re.search(r'\d+\.\s+', solution):
            score += 0.05

        return min(score, 1.0)

    def _calculate_command_confidence(self, command: str) -> float:
        """Calculate confidence score for commands."""
        score = 0.9  # High base score for commands

        # Well-known commands
        known_commands = ['git', 'docker', 'npm', 'pip', 'apt', 'curl', 'wget', 'ssh']
        if any(command.startswith(cmd) for cmd in known_commands):
            score += 0.05

        return min(score, 1.0)

    def _generate_insight_title(self, insight: str) -> str:
        """Generate title for insight."""
        # Use first few words, limited to reasonable length
        words = insight.split()[:8]
        title = ' '.join(words)

        if len(title) > 80:
            title = title[:77] + '...'

        return title

    def _generate_solution_title(self, solution: str) -> str:
        """Generate title for solution."""
        # Look for action at the beginning
        words = solution.split()[:6]
        title = ' '.join(words)

        if len(title) > 80:
            title = title[:77] + '...'

        return f"Solution: {title}"

    def _generate_command_title(self, command: str) -> str:
        """Generate title for command."""
        # Use the command itself, limited length
        if len(command) > 80:
            return command[:77] + '...'
        return command

    def _classify_command(self, command: str) -> str:
        """Classify command by type."""
        if command.startswith('git'):
            return 'git'
        elif command.startswith('docker'):
            return 'docker'
        elif any(command.startswith(pkg) for pkg in ['pip', 'npm', 'apt', 'yum']):
            return 'package_manager'
        elif command.startswith(('curl', 'wget')):
            return 'http_client'
        elif command.startswith('ssh'):
            return 'remote_access'
        else:
            return 'general'