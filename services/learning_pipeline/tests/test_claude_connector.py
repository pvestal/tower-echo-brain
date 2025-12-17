"""
Test suite for ClaudeConnector - comprehensive failing tests for ~/.claude/conversations/ processing.
These tests will fail until ClaudeConnector is implemented.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from pathlib import Path
import os
import tempfile
import shutil

from src.connectors.claude_connector import ClaudeConnector
from src.processors.conversation_processor import ConversationProcessor
from src.models.learning_item import LearningItem, LearningItemType, ProcessingResult
from src.config.settings import PipelineConfig, SourceConfig


@pytest.fixture
def claude_config():
    """Configuration for Claude conversations processing."""
    config_dict = {
        'sources': {
            'claude_conversations': {
                'path': os.path.expanduser('~/.claude/conversations'),
                'file_pattern': '*.md',
                'watch_for_changes': True,
                'exclude_patterns': ['**/test_*', '**/.tmp_*'],
                'max_file_age_days': 365
            }
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
def temp_claude_dir():
    """Create temporary Claude conversations directory for testing."""
    temp_dir = tempfile.mkdtemp()
    claude_dir = Path(temp_dir) / ".claude" / "conversations"
    claude_dir.mkdir(parents=True)

    # Create sample conversation files
    sample_conversations = {
        "2024-12-17-database-optimization.md": """# Database Optimization Discussion

User asked about optimizing PostgreSQL connections.

## Key Insights
- Use connection pooling for better performance
- Localhost connections are faster than remote
- Circuit breaker pattern prevents cascade failures

## Code Examples
```python
async def create_connection_pool():
    return await asyncpg.create_pool(
        host='localhost',
        port=5432,
        user='patrick',
        password=password,
        min_size=5,
        max_size=20
    )
```

## Solutions Implemented
1. Database connector with localhost configuration
2. Circuit breaker for resilience
3. Async connection management
""",

        "2024-12-17-vector-search.md": """# Vector Search Implementation

Discussion about implementing Qdrant vector database integration.

## Technical Insights
- Qdrant provides better performance than alternatives
- 384-dimension embeddings work well for conversational data
- Batch processing improves throughput

## Command Examples
```bash
curl -X PUT "http://localhost:6333/collections/claude_conversations" \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 384, "distance": "Cosine"}}'
```

## Architecture Decisions
- Use localhost Qdrant instance for development
- Implement fallback embedding generation
- Add collection management automation
""",

        "2024-12-16-learning-pipeline.md": """# Learning Pipeline Development

Building a comprehensive learning system for Claude conversations.

## System Requirements
- Process ~/.claude/conversations/ automatically
- Extract insights, code, and solutions
- Store in PostgreSQL and Qdrant
- Implement circuit breaker pattern

## Implementation Notes
The pipeline should:
1. Scan for new/modified conversation files
2. Process content to extract learning items
3. Generate embeddings for vector search
4. Store structured data in PostgreSQL
5. Handle errors gracefully with circuit breakers

## Testing Strategy
Test-driven development with comprehensive coverage:
- Database connectivity tests
- Vector operations tests
- Conversation processing tests
- Error handling and resilience tests
""",

        "old-conversation.md": """# Old Conversation

This is an older conversation that might be filtered out based on date.

Just some basic content here for testing file age filtering.
""",

        "test_file_exclude.md": """# Test File to Exclude

This file should be excluded based on pattern matching.
"""
    }

    # Write sample files
    for filename, content in sample_conversations.items():
        file_path = claude_dir / filename
        file_path.write_text(content)

    yield claude_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
class TestClaudeConnector:
    """Test suite for Claude conversations connector."""

    async def test_claude_connector_initialization(self, claude_config):
        """Test connector initializes with proper configuration."""
        connector = ClaudeConnector(claude_config)

        assert connector.conversations_path == Path(os.path.expanduser('~/.claude/conversations'))
        assert connector.file_pattern == "*.md"
        assert connector.exclude_patterns == ['**/test_*', '**/.tmp_*']
        assert connector.max_file_age_days == 365

    async def test_discover_conversation_files(self, claude_config, temp_claude_dir):
        """Test discovering conversation files in ~/.claude/conversations/."""
        # Override path to use temp directory
        claude_config.claude_conversations.path = str(temp_claude_dir)
        connector = ClaudeConnector(claude_config)

        # Should find conversation files
        files = await connector.discover_conversation_files()

        assert len(files) > 0
        file_names = [f.name for f in files]
        assert "2024-12-17-database-optimization.md" in file_names
        assert "2024-12-17-vector-search.md" in file_names
        assert "2024-12-16-learning-pipeline.md" in file_names

        # Should exclude test files
        assert "test_file_exclude.md" not in file_names

    async def test_get_new_conversations_since_date(self, claude_config, temp_claude_dir):
        """Test filtering conversations by modification date."""
        claude_config.claude_conversations.path = str(temp_claude_dir)
        connector = ClaudeConnector(claude_config)

        # Set cutoff to yesterday
        cutoff_date = datetime.now() - timedelta(days=1)

        # Should find files modified after cutoff
        new_files = await connector.get_new_conversations(since=cutoff_date)
        assert len(new_files) > 0

        # All returned files should be newer than cutoff
        for file_path in new_files:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            assert mtime > cutoff_date

    async def test_get_new_conversations_all_files(self, claude_config, temp_claude_dir):
        """Test getting all conversation files when no date filter."""
        claude_config.claude_conversations.path = str(temp_claude_dir)
        connector = ClaudeConnector(claude_config)

        # Should find all non-excluded files
        all_files = await connector.get_new_conversations(since=None)
        assert len(all_files) >= 3  # At least the 3 main test files

        file_names = [f.name for f in all_files]
        assert "test_file_exclude.md" not in file_names  # Should be excluded

    async def test_file_pattern_matching(self, claude_config, temp_claude_dir):
        """Test file pattern matching works correctly."""
        claude_config.claude_conversations.path = str(temp_claude_dir)

        # Create non-markdown file that should be excluded
        (temp_claude_dir / "not_markdown.txt").write_text("This should be excluded")
        (temp_claude_dir / "another.py").write_text("# Python file")

        connector = ClaudeConnector(claude_config)
        files = await connector.discover_conversation_files()

        # Should only include .md files
        for file_path in files:
            assert file_path.suffix == ".md"

    async def test_exclude_patterns(self, claude_config, temp_claude_dir):
        """Test exclude patterns work correctly."""
        claude_config.claude_conversations.path = str(temp_claude_dir)

        # Create files that should be excluded
        excluded_files = [
            "test_conversation.md",  # Matches **/test_*
            ".tmp_working.md",       # Matches **/.tmp_*
            "subdir/test_file.md"    # Matches **/test_*
        ]

        for file_path_str in excluded_files:
            file_path = temp_claude_dir / file_path_str
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("Excluded content")

        connector = ClaudeConnector(claude_config)
        files = await connector.discover_conversation_files()

        # Should exclude pattern-matched files
        file_names = [f.name for f in files]
        for excluded in excluded_files:
            assert Path(excluded).name not in file_names

    async def test_health_check(self, claude_config):
        """Test connector health check functionality."""
        connector = ClaudeConnector(claude_config)

        # Should pass if conversations directory exists
        if Path(os.path.expanduser('~/.claude/conversations')).exists():
            health = await connector.health_check()
            assert health is True
        else:
            # Should fail if directory doesn't exist
            health = await connector.health_check()
            assert health is False

    async def test_health_check_with_temp_dir(self, claude_config, temp_claude_dir):
        """Test health check with valid temporary directory."""
        claude_config.claude_conversations.path = str(temp_claude_dir)
        connector = ClaudeConnector(claude_config)

        health = await connector.health_check()
        assert health is True

    async def test_file_modification_tracking(self, claude_config, temp_claude_dir):
        """Test tracking file modifications correctly."""
        claude_config.claude_conversations.path = str(temp_claude_dir)
        connector = ClaudeConnector(claude_config)

        # Get initial files
        initial_files = await connector.get_new_conversations()
        initial_count = len(initial_files)

        # Modify an existing file
        existing_file = temp_claude_dir / "2024-12-17-database-optimization.md"
        current_content = existing_file.read_text()
        existing_file.write_text(current_content + "\n\n## New section added")

        # Get files modified after initial scan
        cutoff_time = datetime.now() - timedelta(seconds=10)
        new_files = await connector.get_new_conversations(since=cutoff_time)

        # Should include the modified file
        modified_names = [f.name for f in new_files]
        assert "2024-12-17-database-optimization.md" in modified_names

    async def test_large_directory_handling(self, claude_config, temp_claude_dir):
        """Test handling directories with many conversation files."""
        claude_config.claude_conversations.path = str(temp_claude_dir)

        # Create many conversation files
        for i in range(50):
            file_path = temp_claude_dir / f"conversation_{i:03d}.md"
            file_path.write_text(f"# Conversation {i}\n\nSome content for conversation {i}")

        connector = ClaudeConnector(claude_config)
        files = await connector.discover_conversation_files()

        # Should handle many files efficiently
        assert len(files) >= 50

        # Should be able to filter by date
        recent_files = await connector.get_new_conversations(since=datetime.now() - timedelta(hours=1))
        assert len(recent_files) >= 50

    async def test_concurrent_file_discovery(self, claude_config, temp_claude_dir):
        """Test concurrent file discovery operations."""
        claude_config.claude_conversations.path = str(temp_claude_dir)
        connector = ClaudeConnector(claude_config)

        # Perform multiple concurrent discovery operations
        tasks = [
            connector.discover_conversation_files(),
            connector.get_new_conversations(),
            connector.get_new_conversations(since=datetime.now() - timedelta(days=1))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, list)
            assert len(result) > 0

    async def test_error_handling_invalid_directory(self, claude_config):
        """Test error handling when conversations directory is invalid."""
        # Set invalid path
        claude_config.claude_conversations.path = "/nonexistent/path"
        connector = ClaudeConnector(claude_config)

        # Should handle gracefully
        files = await connector.discover_conversation_files()
        assert files == []

        health = await connector.health_check()
        assert health is False


@pytest.mark.asyncio
class TestConversationProcessor:
    """Test suite for conversation content processing."""

    async def test_processor_initialization(self, claude_config):
        """Test processor initializes with proper configuration."""
        processor = ConversationProcessor(claude_config)

        assert processor.extract_code_blocks is True
        assert processor.extract_commands is True
        assert processor.extract_insights is True
        assert processor.extract_solutions is True
        assert processor.min_insight_length == 100

    async def test_process_conversation_file(self, claude_config, temp_claude_dir):
        """Test processing a complete conversation file."""
        processor = ConversationProcessor(claude_config)

        # Process the database optimization conversation
        file_path = temp_claude_dir / "2024-12-17-database-optimization.md"
        result = await processor.process_file(file_path)

        assert isinstance(result, ProcessingResult)
        assert len(result.learning_items) > 0
        assert len(result.errors) == 0

        # Should extract different types of learning items
        item_types = [item.item_type for item in result.learning_items]
        assert LearningItemType.INSIGHT in item_types
        assert LearningItemType.CODE_EXAMPLE in item_types
        assert LearningItemType.SOLUTION in item_types

    async def test_extract_code_blocks(self, claude_config, temp_claude_dir):
        """Test extraction of code blocks from conversations."""
        processor = ConversationProcessor(claude_config)
        file_path = temp_claude_dir / "2024-12-17-database-optimization.md"

        result = await processor.process_file(file_path)

        # Should find the Python code block
        code_items = [item for item in result.learning_items if item.item_type == LearningItemType.CODE_EXAMPLE]
        assert len(code_items) > 0

        # Should contain the connection pool code
        code_content = " ".join(item.content for item in code_items)
        assert "asyncpg.create_pool" in code_content
        assert "localhost" in code_content

    async def test_extract_insights(self, claude_config, temp_claude_dir):
        """Test extraction of insights from conversations."""
        processor = ConversationProcessor(claude_config)
        file_path = temp_claude_dir / "2024-12-17-database-optimization.md"

        result = await processor.process_file(file_path)

        # Should find insights
        insight_items = [item for item in result.learning_items if item.item_type == LearningItemType.INSIGHT]
        assert len(insight_items) > 0

        # Should contain key insights about connections
        insight_content = " ".join(item.content for item in insight_items)
        assert any("connection pooling" in content.lower() for content in insight_content.split())

    async def test_extract_solutions(self, claude_config, temp_claude_dir):
        """Test extraction of solutions from conversations."""
        processor = ConversationProcessor(claude_config)
        file_path = temp_claude_dir / "2024-12-16-learning-pipeline.md"

        result = await processor.process_file(file_path)

        # Should find solutions
        solution_items = [item for item in result.learning_items if item.item_type == LearningItemType.SOLUTION]
        assert len(solution_items) > 0

    async def test_confidence_scoring(self, claude_config, temp_claude_dir):
        """Test confidence scoring for extracted items."""
        processor = ConversationProcessor(claude_config)
        file_path = temp_claude_dir / "2024-12-17-vector-search.md"

        result = await processor.process_file(file_path)

        # All items should have confidence scores
        for item in result.learning_items:
            assert hasattr(item, 'confidence_score')
            assert 0.0 <= item.confidence_score <= 1.0

        # Technical content should have higher confidence
        tech_items = [item for item in result.learning_items if "Qdrant" in item.content]
        if tech_items:
            assert any(item.confidence_score > 0.7 for item in tech_items)

    async def test_error_handling_invalid_file(self, claude_config):
        """Test error handling for invalid or missing files."""
        processor = ConversationProcessor(claude_config)

        # Try to process non-existent file
        result = await processor.process_file(Path("/nonexistent/file.md"))

        assert isinstance(result, ProcessingResult)
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower() or "no such file" in result.errors[0].lower()

    async def test_concurrent_processing(self, claude_config, temp_claude_dir):
        """Test concurrent processing of multiple files."""
        processor = ConversationProcessor(claude_config)

        files = [
            temp_claude_dir / "2024-12-17-database-optimization.md",
            temp_claude_dir / "2024-12-17-vector-search.md",
            temp_claude_dir / "2024-12-16-learning-pipeline.md"
        ]

        # Process files concurrently
        tasks = [processor.process_file(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, ProcessingResult)
            assert len(result.learning_items) > 0