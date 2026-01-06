"""
Pipeline state and run tracking models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class ProcessingStatus(Enum):
    """Pipeline run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineRun:
    """Represents a single pipeline execution run."""
    run_id: str
    started_at: datetime
    status: ProcessingStatus
    completed_at: Optional[datetime] = None
    conversations_processed: int = 0
    articles_processed: int = 0
    learning_items_extracted: int = 0
    vectors_updated: int = 0
    errors_encountered: int = 0
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_successful(self) -> bool:
        """Check if run completed successfully."""
        return self.status == ProcessingStatus.COMPLETED and self.errors_encountered == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'run_id': self.run_id,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'conversations_processed': self.conversations_processed,
            'articles_processed': self.articles_processed,
            'learning_items_extracted': self.learning_items_extracted,
            'vectors_updated': self.vectors_updated,
            'errors_encountered': self.errors_encountered,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics,
            'duration_seconds': self.duration_seconds
        }


@dataclass
class PipelineState:
    """Overall pipeline state tracking."""
    current_run: Optional[PipelineRun] = None
    last_successful_run: Optional[PipelineRun] = None
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    last_error: Optional[str] = None
    average_processing_time: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100

    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return (self.current_run is not None and
                self.current_run.status == ProcessingStatus.RUNNING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'current_run': self.current_run.to_dict() if self.current_run else None,
            'last_successful_run': self.last_successful_run.to_dict() if self.last_successful_run else None,
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'success_rate': self.success_rate,
            'last_error': self.last_error,
            'average_processing_time': self.average_processing_time,
            'is_running': self.is_running
        }