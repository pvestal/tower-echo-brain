#!/usr/bin/env python3
"""
Epistemic Models for Echo Brain - Epistemological Tracking System
Compatible with existing consciousness architecture
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import hashlib
import json

class EpistemicKnowledgeType(Enum):
    """Types of knowledge that Echo can have about information"""
    OBSERVED = "OBSERVED"          # Direct sensory input or system logs
    REMEMBERED = "REMEMBERED"      # Retrieved from long-term memory
    INFERRED = "INFERRED"         # Logical deduction from known facts
    UNCERTAIN = "UNCERTAIN"       # High uncertainty, requires verification
    ASSUMED = "ASSUMED"           # Working assumption, not verified
    CONTRADICTED = "CONTRADICTED" # Previously believed but now contradicted

class ContradictionSeverity(Enum):
    """Severity levels for epistemic contradictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResolutionStatus(Enum):
    """Status of contradiction resolution"""
    UNRESOLVED = "unresolved"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ACCEPTED_UNCERTAINTY = "accepted_uncertainty"

class EvidenceType(Enum):
    """Types of evidence supporting epistemic claims"""
    SOURCE = "source"
    CORROBORATION = "corroboration"
    LOGICAL_SUPPORT = "logical_support"
    EMPIRICAL = "empirical"
    AUTHORITY = "authority"
    CONSENSUS = "consensus"

@dataclass
class EpistemicStatus:
    """
    Core epistemic status tracking for every piece of information
    Compatible with existing consciousness architecture
    """
    # Foreign key references to existing memory systems
    interaction_id: Optional[int] = None
    memory_id: Optional[int] = None
    qdrant_memory_id: Optional[str] = None

    # Core epistemic tracking
    knowledge_type: EpistemicKnowledgeType = EpistemicKnowledgeType.UNCERTAIN
    confidence: float = 0.5  # 0.0 to 1.0
    source: str = ""  # Where this information came from
    source_reliability: float = 0.5  # 0.0 to 1.0

    # Temporal tracking
    first_learned: datetime = field(default_factory=datetime.utcnow)
    last_verified: datetime = field(default_factory=datetime.utcnow)
    verification_count: int = 0

    # Content tracking
    content_hash: Optional[str] = None  # SHA256 of content for change detection

    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Database ID
    id: Optional[int] = None

    def __post_init__(self):
        """Validation and setup after initialization"""
        # Ensure confidence is within bounds
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.source_reliability = max(0.0, min(1.0, self.source_reliability))

        # Ensure at least one memory reference exists
        if not any([self.interaction_id, self.memory_id, self.qdrant_memory_id]):
            raise ValueError("At least one memory reference must be provided")

    def generate_content_hash(self, content: str) -> str:
        """Generate SHA256 hash of content for change tracking"""
        content_str = str(content)
        self.content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        return self.content_hash

    def update_confidence(self, new_confidence: float, reason: str = "") -> None:
        """Update confidence with validation and tracking"""
        old_confidence = self.confidence
        self.confidence = max(0.0, min(1.0, new_confidence))
        self.updated_at = datetime.utcnow()

        if reason:
            if not self.notes:
                self.notes = ""
            self.notes += f"\n[{self.updated_at}] Confidence updated from {old_confidence} to {self.confidence}: {reason}"

    def mark_verified(self, verification_method: str = "") -> None:
        """Mark as verified and update tracking"""
        self.last_verified = datetime.utcnow()
        self.verification_count += 1
        self.updated_at = self.last_verified

        if verification_method:
            if not self.notes:
                self.notes = ""
            self.notes += f"\n[{self.last_verified}] Verified via: {verification_method}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "interaction_id": self.interaction_id,
            "memory_id": self.memory_id,
            "qdrant_memory_id": self.qdrant_memory_id,
            "knowledge_type": self.knowledge_type.value,
            "confidence": self.confidence,
            "source": self.source,
            "source_reliability": self.source_reliability,
            "first_learned": self.first_learned,
            "last_verified": self.last_verified,
            "verification_count": self.verification_count,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

@dataclass
class ConfidenceCalibration:
    """Tracks how well Echo calibrates its confidence predictions"""
    # Prediction details
    prediction: str
    predicted_confidence: float  # 0.0 to 1.0
    prediction_type: str  # 'factual', 'outcome', 'recommendation'

    # Actual outcome
    actual_outcome: Optional[bool] = None
    actual_confidence: Optional[float] = None
    outcome_verified: bool = False
    verification_method: Optional[str] = None

    # Context
    conversation_id: Optional[str] = None
    model_used: Optional[str] = None
    domain: Optional[str] = None

    # Temporal tracking
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)
    verification_timestamp: Optional[datetime] = None

    # Calibration metrics (calculated)
    brier_score: Optional[float] = None
    log_score: Optional[float] = None

    # Reference to epistemic status
    epistemic_status_id: Optional[int] = None

    # Database ID
    id: Optional[int] = None

    def __post_init__(self):
        """Validation after initialization"""
        self.predicted_confidence = max(0.0, min(1.0, self.predicted_confidence))
        if self.actual_confidence is not None:
            self.actual_confidence = max(0.0, min(1.0, self.actual_confidence))

    def calculate_brier_score(self) -> Optional[float]:
        """Calculate Brier score if outcome is verified"""
        if self.actual_outcome is None:
            return None

        actual_value = 1.0 if self.actual_outcome else 0.0
        self.brier_score = (self.predicted_confidence - actual_value) ** 2
        return self.brier_score

    def calculate_log_score(self) -> Optional[float]:
        """Calculate logarithmic scoring rule"""
        if self.actual_outcome is None:
            return None

        # Avoid log(0) by clamping confidence
        clamped_confidence = max(0.001, min(0.999, self.predicted_confidence))

        if self.actual_outcome:
            self.log_score = -1 * math.log(clamped_confidence)
        else:
            self.log_score = -1 * math.log(1 - clamped_confidence)

        return self.log_score

    def verify_outcome(self, outcome: bool, method: str = "", confidence: Optional[float] = None) -> None:
        """Verify the actual outcome and calculate metrics"""
        self.actual_outcome = outcome
        self.outcome_verified = True
        self.verification_timestamp = datetime.utcnow()
        self.verification_method = method

        if confidence is not None:
            self.actual_confidence = max(0.0, min(1.0, confidence))

        # Calculate metrics
        self.calculate_brier_score()
        self.calculate_log_score()

@dataclass
class EpistemicContradiction:
    """Represents a contradiction between two pieces of information"""
    # The contradicting epistemic statuses
    primary_epistemic_id: int
    conflicting_epistemic_id: int

    # Contradiction details
    contradiction_type: str  # 'factual', 'temporal', 'logical'
    severity: ContradictionSeverity = ContradictionSeverity.MEDIUM
    description: str = ""

    # Resolution tracking
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    resolution_method: Optional[str] = None
    resolution_notes: Optional[str] = None

    # Resolution outcome
    preferred_epistemic_id: Optional[int] = None
    confidence_in_resolution: Optional[float] = None

    # Temporal tracking
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    # Metadata
    detected_by: str = "system"  # 'system', 'user', 'external'
    domain: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Database ID
    id: Optional[int] = None

    def resolve(self,
                preferred_id: int,
                method: str,
                confidence: float = 0.8,
                notes: str = "") -> None:
        """Mark contradiction as resolved"""
        self.preferred_epistemic_id = preferred_id
        self.resolution_method = method
        self.confidence_in_resolution = max(0.0, min(1.0, confidence))
        self.resolution_status = ResolutionStatus.RESOLVED
        self.resolved_at = datetime.utcnow()

        if notes:
            self.resolution_notes = notes

@dataclass
class EpistemicEvidence:
    """Evidence supporting or refuting epistemic claims"""
    epistemic_status_id: int

    # Evidence details
    evidence_type: EvidenceType
    evidence_description: str
    evidence_strength: float = 0.5  # 0.0 to 1.0

    # Source information
    evidence_source: Optional[str] = None
    source_credibility: float = 0.5  # 0.0 to 1.0

    # Cross-references
    supporting_epistemic_id: Optional[int] = None
    supporting_interaction_id: Optional[int] = None

    # Temporal tracking
    recorded_at: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    # Database ID
    id: Optional[int] = None

    def __post_init__(self):
        """Validation after initialization"""
        self.evidence_strength = max(0.0, min(1.0, self.evidence_strength))
        self.source_credibility = max(0.0, min(1.0, self.source_credibility))

@dataclass
class EpistemicUncertainty:
    """Detailed uncertainty quantification"""
    epistemic_status_id: int

    # Types of uncertainty
    aleatoric_uncertainty: float = 0.0  # Inherent randomness
    epistemic_uncertainty: float = 0.0  # Knowledge uncertainty
    model_uncertainty: float = 0.0      # Model/method uncertainty

    # Uncertainty sources
    uncertainty_sources: List[str] = field(default_factory=list)

    # Confidence intervals
    confidence_lower_bound: Optional[float] = None
    confidence_upper_bound: Optional[float] = None
    confidence_interval_level: float = 0.95

    # Temporal tracking
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    calculation_method: Optional[str] = None

    # Metadata
    notes: Optional[str] = None

    # Database ID
    id: Optional[int] = None

    def total_uncertainty(self) -> float:
        """Calculate total uncertainty as combination of all types"""
        # Simple additive model - could be more sophisticated
        return min(1.0, self.aleatoric_uncertainty + self.epistemic_uncertainty + self.model_uncertainty)

# Pydantic models for API serialization/validation

class EpistemicStatusAPI(BaseModel):
    """Pydantic model for API operations"""
    interaction_id: Optional[int] = None
    memory_id: Optional[int] = None
    qdrant_memory_id: Optional[str] = None
    knowledge_type: EpistemicKnowledgeType = EpistemicKnowledgeType.UNCERTAIN
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    source: str
    source_reliability: float = Field(0.5, ge=0.0, le=1.0)
    tags: List[str] = []
    notes: Optional[str] = None

    @validator('confidence', 'source_reliability')
    def validate_probability(cls, v):
        return max(0.0, min(1.0, v))

    class Config:
        use_enum_values = True

class ConfidenceCalibrationAPI(BaseModel):
    """Pydantic model for confidence calibration API"""
    prediction: str
    predicted_confidence: float = Field(..., ge=0.0, le=1.0)
    prediction_type: str
    conversation_id: Optional[str] = None
    model_used: Optional[str] = None
    domain: Optional[str] = None

    class Config:
        use_enum_values = True

class ContradictionAPI(BaseModel):
    """Pydantic model for contradiction API"""
    primary_epistemic_id: int
    conflicting_epistemic_id: int
    contradiction_type: str
    severity: ContradictionSeverity = ContradictionSeverity.MEDIUM
    description: str
    domain: Optional[str] = None
    tags: List[str] = []

    class Config:
        use_enum_values = True

# Utility functions for integration

def create_epistemic_status_from_interaction(
    interaction_id: int,
    query: str,
    response: str,
    confidence: float = 0.5,
    source: str = "conversation"
) -> EpistemicStatus:
    """Create epistemic status from conversation interaction"""

    # Infer knowledge type based on response content
    knowledge_type = infer_knowledge_type_from_response(response)

    epistemic_status = EpistemicStatus(
        interaction_id=interaction_id,
        knowledge_type=knowledge_type,
        confidence=confidence,
        source=source,
        tags=extract_tags_from_content(query + " " + response)
    )

    # Generate content hash
    epistemic_status.generate_content_hash(response)

    return epistemic_status

def infer_knowledge_type_from_response(response: str) -> EpistemicKnowledgeType:
    """Infer knowledge type from response content"""
    response_lower = response.lower()

    # Check for uncertainty indicators
    uncertainty_indicators = ['might', 'maybe', 'possibly', 'uncertain', 'unclear', 'not sure']
    if any(indicator in response_lower for indicator in uncertainty_indicators):
        return EpistemicKnowledgeType.UNCERTAIN

    # Check for memory retrieval indicators
    memory_indicators = ['remember', 'recall', 'previously', 'earlier', 'database', 'memory']
    if any(indicator in response_lower for indicator in memory_indicators):
        return EpistemicKnowledgeType.REMEMBERED

    # Check for inference indicators
    inference_indicators = ['therefore', 'thus', 'implies', 'suggests', 'infer', 'conclude']
    if any(indicator in response_lower for indicator in inference_indicators):
        return EpistemicKnowledgeType.INFERRED

    # Check for observation indicators
    observation_indicators = ['see', 'observe', 'found', 'detected', 'measured', 'log', 'status']
    if any(indicator in response_lower for indicator in observation_indicators):
        return EpistemicKnowledgeType.OBSERVED

    # Default to uncertain for safety
    return EpistemicKnowledgeType.UNCERTAIN

def extract_tags_from_content(content: str) -> List[str]:
    """Extract relevant tags from content for categorization"""
    import re

    tags = []
    content_lower = content.lower()

    # Domain tags
    if any(word in content_lower for word in ['anime', 'character', 'generation']):
        tags.append('anime')
    if any(word in content_lower for word in ['music', 'song', 'audio']):
        tags.append('music')
    if any(word in content_lower for word in ['code', 'programming', 'function']):
        tags.append('technical')
    if any(word in content_lower for word in ['personal', 'family', 'preference']):
        tags.append('personal')

    # Epistemic tags
    if any(word in content_lower for word in ['fact', 'true', 'false', 'verify']):
        tags.append('factual')
    if any(word in content_lower for word in ['opinion', 'think', 'believe']):
        tags.append('subjective')

    return list(set(tags))  # Remove duplicates

# For backward compatibility with existing consciousness architecture
def enhance_thought_with_epistemic_status(thought, epistemic_status: EpistemicStatus):
    """Enhance existing Thought object with epistemic information"""
    if hasattr(thought, 'epistemic_status'):
        thought.epistemic_status = epistemic_status
    else:
        # For thoughts without epistemic support, add as attribute
        thought.__dict__['epistemic_status'] = epistemic_status

    return thought

import math  # Add this import at the top with other imports