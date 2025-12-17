"""
Claude conversations connector for the learning pipeline.
Handles discovery and processing of ~/.claude/conversations/ files.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import fnmatch
import os

from ..config.settings import PipelineConfig

logger = logging.getLogger(__name__)


class ClaudeConnector:
    """
    Connector for Claude conversation files.

    Features:
    - Automatic file discovery
    - Pattern-based filtering
    - Modification time tracking
    - Exclude pattern support
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.conversations_path = Path(config.claude_conversations.path).expanduser()
        self.file_pattern = config.claude_conversations.file_pattern
        self.exclude_patterns = config.claude_conversations.exclude_patterns or []
        self.max_file_age_days = config.claude_conversations.max_file_age_days

    async def discover_conversation_files(self) -> List[Path]:
        """Discover all conversation files matching patterns."""
        try:
            if not self.conversations_path.exists():
                logger.warning(f"Conversations directory not found: {self.conversations_path}")
                return []

            logger.info(f"Scanning for conversation files in: {self.conversations_path}")

            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()
            files = await loop.run_in_executor(None, self._scan_directory)

            logger.info(f"Found {len(files)} conversation files")
            return files

        except Exception as e:
            logger.error(f"Failed to discover conversation files: {e}")
            return []

    def _scan_directory(self) -> List[Path]:
        """Synchronous directory scanning."""
        files = []

        try:
            # Walk through directory recursively
            for root, dirs, filenames in os.walk(self.conversations_path):
                root_path = Path(root)

                for filename in filenames:
                    file_path = root_path / filename

                    # Check file pattern
                    if not fnmatch.fnmatch(filename, self.file_pattern):
                        continue

                    # Check exclude patterns
                    relative_path = file_path.relative_to(self.conversations_path)
                    if self._should_exclude(relative_path):
                        continue

                    # Check file age if specified
                    if self.max_file_age_days > 0:
                        if not self._is_file_recent_enough(file_path):
                            continue

                    files.append(file_path)

            return files

        except Exception as e:
            logger.error(f"Error scanning directory: {e}")
            return []

    def _should_exclude(self, relative_path: Path) -> bool:
        """Check if file should be excluded based on patterns."""
        path_str = str(relative_path)

        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True

            # Also check individual path components
            for part in relative_path.parts:
                if fnmatch.fnmatch(part, pattern.replace('**/', '')):
                    return True

        return False

    def _is_file_recent_enough(self, file_path: Path) -> bool:
        """Check if file is recent enough based on max age."""
        try:
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_days = (datetime.now() - file_mtime).days
            return age_days <= self.max_file_age_days
        except Exception:
            # If we can't get the timestamp, include the file
            return True

    async def get_new_conversations(self, since: Optional[datetime] = None) -> List[Path]:
        """Get conversation files modified since specific time."""
        try:
            all_files = await self.discover_conversation_files()

            if since is None:
                return all_files

            # Filter by modification time
            new_files = []
            for file_path in all_files:
                try:
                    # Get file modification time
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                    # Convert to timezone-aware if since is timezone-aware
                    if since.tzinfo is not None and mtime.tzinfo is None:
                        mtime = mtime.replace(tzinfo=timezone.utc)
                    elif since.tzinfo is None and mtime.tzinfo is not None:
                        since = since.replace(tzinfo=timezone.utc)

                    if mtime > since:
                        new_files.append(file_path)

                except Exception as e:
                    logger.warning(f"Could not get modification time for {file_path}: {e}")
                    # Include file if we can't get timestamp
                    new_files.append(file_path)

            logger.info(f"Found {len(new_files)} files modified since {since}")
            return new_files

        except Exception as e:
            logger.error(f"Failed to get new conversations: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if conversations directory is accessible."""
        try:
            if not self.conversations_path.exists():
                logger.warning(f"Conversations directory does not exist: {self.conversations_path}")
                return False

            if not self.conversations_path.is_dir():
                logger.error(f"Conversations path is not a directory: {self.conversations_path}")
                return False

            # Try to list directory contents
            try:
                list(self.conversations_path.iterdir())
                return True
            except PermissionError:
                logger.error(f"No permission to read conversations directory: {self.conversations_path}")
                return False

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_file_stats(self) -> dict:
        """Get statistics about conversation files."""
        try:
            all_files = await self.discover_conversation_files()

            total_files = len(all_files)
            total_size = 0
            oldest_file = None
            newest_file = None

            for file_path in all_files:
                try:
                    stat = file_path.stat()
                    total_size += stat.st_size

                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    if oldest_file is None or mtime < oldest_file:
                        oldest_file = mtime
                    if newest_file is None or mtime > newest_file:
                        newest_file = mtime

                except Exception as e:
                    logger.warning(f"Could not get stats for {file_path}: {e}")

            return {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'oldest_file': oldest_file.isoformat() if oldest_file else None,
                'newest_file': newest_file.isoformat() if newest_file else None,
                'conversations_path': str(self.conversations_path),
                'file_pattern': self.file_pattern,
                'exclude_patterns': self.exclude_patterns
            }

        except Exception as e:
            logger.error(f"Failed to get file stats: {e}")
            return {
                'error': str(e),
                'conversations_path': str(self.conversations_path)
            }