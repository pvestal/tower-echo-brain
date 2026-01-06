"""
Conversation and metadata models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class ConversationMetadata:
    """Metadata for a conversation file."""
    file_path: Path
    file_size: int
    created_at: datetime
    modified_at: datetime
    title: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'file_path': str(self.file_path),
            'file_size': self.file_size,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'title': self.title,
            'tags': self.tags,
            'category': self.category,
            'language': self.language
        }


@dataclass
class Conversation:
    """Represents a complete conversation."""
    content: str
    metadata: ConversationMetadata
    processed_at: Optional[datetime] = None
    processing_status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'content': self.content,
            'metadata': self.metadata.to_dict(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'processing_status': self.processing_status
        }