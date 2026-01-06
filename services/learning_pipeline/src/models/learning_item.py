"""
Learning item data models and types.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import hashlib


class LearningItemType(Enum):
    """Types of learning items that can be extracted."""
    INSIGHT = "insight"                 # Key insights and understanding
    CODE_EXAMPLE = "code_example"       # Code snippets and examples
    SOLUTION = "solution"               # Problem solutions
    ERROR_FIX = "error_fix"            # Bug fixes and troubleshooting
    BEST_PRACTICE = "best_practice"     # Best practices and patterns
    COMMAND = "command"                 # CLI commands and usage
    CONFIGURATION = "configuration"     # Config files and settings
    WORKFLOW = "workflow"               # Processes and workflows


@dataclass
class LearningItem:
    """
    Represents a single learning item extracted from content.

    This is the core unit of learning that gets processed into
    vector embeddings and stored in the knowledge system.
    """

    # Core identification
    content: str                        # The actual learning content
    item_type: LearningItemType        # Type of learning item
    title: Optional[str] = None         # Human-readable title

    # Source information
    source_conversation_id: Optional[str] = None  # Parent conversation
    source_file_path: Optional[Path] = None       # Source file
    source_type: str = "unknown"                  # 'claude', 'kb_article', etc.

    # Content metadata
    categories: List[str] = field(default_factory=list)  # Categorization tags
    tags: List[str] = field(default_factory=list)        # Additional tags
    importance_score: float = 0.0                        # ML-generated importance (0-1)
    confidence_score: float = 0.0                        # Extraction confidence (0-1)

    # Technical metadata
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    content_hash: Optional[str] = None                       # Content hash for deduplication

    # Timestamps
    extracted_at: datetime = field(default_factory=datetime.now)
    source_modified_at: Optional[datetime] = None

    # Vector database references
    vector_embedding_id: Optional[str] = None  # Qdrant vector ID
    embedding_model: Optional[str] = None      # Model used for embedding

    def __post_init__(self):
        """Calculate content hash if not provided."""
        if self.content_hash is None:
            self.content_hash = self._calculate_content_hash()

    def _calculate_content_hash(self) -> str:
        """Calculate SHA-256 hash of content for deduplication."""
        content_bytes = self.content.encode('utf-8')
        return hashlib.sha256(content_bytes).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'content': self.content,
            'item_type': self.item_type.value,
            'title': self.title,
            'source_conversation_id': self.source_conversation_id,
            'source_file_path': str(self.source_file_path) if self.source_file_path else None,
            'source_type': self.source_type,
            'categories': self.categories,
            'tags': self.tags,
            'importance_score': self.importance_score,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata,
            'content_hash': self.content_hash,
            'extracted_at': self.extracted_at.isoformat(),
            'source_modified_at': self.source_modified_at.isoformat() if self.source_modified_at else None,
            'vector_embedding_id': self.vector_embedding_id,
            'embedding_model': self.embedding_model
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningItem':
        """Create LearningItem from dictionary."""
        # Convert string dates back to datetime objects
        extracted_at = datetime.fromisoformat(data['extracted_at'])
        source_modified_at = None
        if data.get('source_modified_at'):
            source_modified_at = datetime.fromisoformat(data['source_modified_at'])

        # Convert file path string back to Path
        source_file_path = None
        if data.get('source_file_path'):
            source_file_path = Path(data['source_file_path'])

        return cls(
            content=data['content'],
            item_type=LearningItemType(data['item_type']),
            title=data.get('title'),
            source_conversation_id=data.get('source_conversation_id'),
            source_file_path=source_file_path,
            source_type=data.get('source_type', 'unknown'),
            categories=data.get('categories', []),
            tags=data.get('tags', []),
            importance_score=data.get('importance_score', 0.0),
            confidence_score=data.get('confidence_score', 0.0),
            metadata=data.get('metadata', {}),
            content_hash=data.get('content_hash'),
            extracted_at=extracted_at,
            source_modified_at=source_modified_at,
            vector_embedding_id=data.get('vector_embedding_id'),
            embedding_model=data.get('embedding_model')
        )

    def get_qdrant_payload(self) -> Dict[str, Any]:
        """Get payload for Qdrant vector database."""
        return {
            'conversation_id': self.source_conversation_id,
            'item_type': self.item_type.value,
            'title': self.title or '',
            'source_type': self.source_type,
            'categories': self.categories,
            'tags': self.tags,
            'importance_score': self.importance_score,
            'confidence_score': self.confidence_score,
            'extracted_at': self.extracted_at.isoformat(),
            'content_preview': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'content_hash': self.content_hash
        }

    def is_duplicate(self, other: 'LearningItem') -> bool:
        """Check if this item is a duplicate of another."""
        return self.content_hash == other.content_hash

    def merge_metadata(self, other_metadata: Dict[str, Any]) -> None:
        """Merge additional metadata into this item."""
        self.metadata.update(other_metadata)

    def add_category(self, category: str) -> None:
        """Add a category if not already present."""
        if category not in self.categories:
            self.categories.append(category)

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)

    def set_vector_reference(self, vector_id: str, model_name: str) -> None:
        """Set vector database reference information."""
        self.vector_embedding_id = vector_id
        self.embedding_model = model_name


@dataclass
class ProcessingResult:
    """Result of processing content into learning items."""

    learning_items: List[LearningItem] = field(default_factory=list)
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True if processing completed without errors."""
        return len(self.errors) == 0

    @property
    def item_count(self) -> int:
        """Number of learning items extracted."""
        return len(self.learning_items)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'learning_items': [item.to_dict() for item in self.learning_items],
            'processing_time': self.processing_time,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'success': self.success,
            'item_count': self.item_count
        }