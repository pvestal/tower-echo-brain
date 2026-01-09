#!/usr/bin/env python3
"""
Meta-Cognitive Models for Echo Brain - "Thinking About Thinking"
Real-time reasoning quality monitoring and cognitive bias detection
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import json
import uuid

class CognitiveBiasType(Enum):
    """Types of cognitive biases that can be detected"""
    OVERCONFIDENCE_BIAS = "overconfidence_bias"
    CONFIRMATION_BIAS = "confirmation_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    ANCHORING_BIAS = "anchoring_bias"
    DUNNING_KRUGER = "dunning_kruger"
    REPRESENTATIVE_HEURISTIC = "representative_heuristic"
    HINDSIGHT_BIAS = "hindsight_bias"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    CONJUNCTION_FALLACY = "conjunction_fallacy"
    SURVIVORSHIP_BIAS = "survivorship_bias"

class ReasoningIssueType(Enum):
    """Types of reasoning quality issues"""
    LOGICAL_CONTRADICTION = "logical_contradiction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CIRCULAR_REASONING = "circular_reasoning"
    FALSE_DICHOTOMY = "false_dichotomy"
    HASTY_GENERALIZATION = "hasty_generalization"
    AD_HOC_REASONING = "ad_hoc_reasoning"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    CORRELATION_CAUSATION = "correlation_causation"
    MISSING_UNCERTAINTY = "missing_uncertainty"
    INAPPROPRIATE_CONFIDENCE = "inappropriate_confidence"

class QualityAssessment(Enum):
    """Overall quality assessment recommendations"""
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    REQUIRE_VERIFICATION = "REQUIRE_VERIFICATION"

class ConfidenceCalibration(Enum):
    """Assessment of confidence appropriateness"""
    APPROPRIATE = "appropriate"
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"
    MISCALIBRATED = "miscalibrated"

@dataclass
class BiasDetection:
    """
    Detection of a specific cognitive bias in reasoning
    """
    bias_type: CognitiveBiasType
    severity: float  # 0.0 to 1.0
    confidence: float  # How confident we are in this detection
    evidence: List[str]  # Supporting evidence for bias detection
    affected_reasoning: List[str]  # Which parts of reasoning show this bias
    bias_description: str  # Human-readable description
    mitigation_suggestion: str  # How to address this bias
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bias_type": self.bias_type.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "affected_reasoning": self.affected_reasoning,
            "bias_description": self.bias_description,
            "mitigation_suggestion": self.mitigation_suggestion,
            "detected_at": self.detected_at.isoformat()
        }

@dataclass
class ReasoningIssue:
    """
    Specific reasoning quality issue detected
    """
    issue_type: ReasoningIssueType
    severity: float  # 0.0 to 1.0
    affected_claims: List[str]  # Which claims are affected
    explanation: str  # What the issue is
    suggestion: str  # How to improve
    evidence_location: List[str]  # Where in the reasoning this occurs
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity,
            "affected_claims": self.affected_claims,
            "explanation": self.explanation,
            "suggestion": self.suggestion,
            "evidence_location": self.evidence_location,
            "detected_at": self.detected_at.isoformat()
        }

@dataclass
class SourceQualityReport:
    """
    Assessment of source quality and citations
    """
    total_sources: int
    reliable_sources: int
    questionable_sources: int
    missing_sources: int
    source_diversity_score: float  # 0.0 to 1.0
    citation_quality_score: float  # 0.0 to 1.0
    source_details: List[Dict[str, Any]]
    recommendations: List[str]

@dataclass
class UncertaintyAssessment:
    """
    Assessment of how appropriately uncertainty is expressed
    """
    should_express_uncertainty: bool
    current_confidence: float
    recommended_confidence: float
    uncertainty_appropriateness: float  # 0.0 to 1.0 (1.0 = perfectly appropriate)
    missing_caveats: List[str]
    recommended_qualifiers: List[str]
    reasoning: str

@dataclass
class LogicalValidation:
    """
    Results of logical consistency checking
    """
    is_logically_consistent: bool
    contradictions_found: int
    consistency_score: float  # 0.0 to 1.0
    logical_issues: List[Dict[str, Any]]
    valid_reasoning_chains: List[str]
    invalid_reasoning_chains: List[str]
    recommendations: List[str]

@dataclass
class QualityReport:
    """
    Comprehensive meta-cognitive quality assessment
    """
    # Unique identifier
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Reference to the task/response being analyzed
    task_id: Optional[str] = None
    interaction_id: Optional[int] = None
    response_text: str = ""

    # Core quality metrics
    quality_score: float = 0.0  # 0.0 to 1.0 overall reasoning quality

    # Detected issues and biases
    issues_detected: List[ReasoningIssue] = field(default_factory=list)
    bias_flags: List[BiasDetection] = field(default_factory=list)

    # Specific assessments
    confidence_assessment: ConfidenceCalibration = ConfidenceCalibration.APPROPRIATE
    uncertainty_assessment: Optional[UncertaintyAssessment] = None
    logical_validation: Optional[LogicalValidation] = None
    source_quality: Optional[SourceQualityReport] = None

    # Overall recommendation
    recommendation: QualityAssessment = QualityAssessment.ACCEPT
    improvement_suggestions: List[str] = field(default_factory=list)

    # Meta-information
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    analysis_duration_ms: Optional[int] = None
    analyzer_version: str = "1.0.0"

    # Detailed breakdowns
    reasoning_chain_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence_calibration_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            "report_id": self.report_id,
            "task_id": self.task_id,
            "interaction_id": self.interaction_id,
            "response_text": self.response_text,
            "quality_score": self.quality_score,
            "issues_detected": [issue.to_dict() for issue in self.issues_detected],
            "bias_flags": [bias.to_dict() for bias in self.bias_flags],
            "confidence_assessment": self.confidence_assessment.value,
            "recommendation": self.recommendation.value,
            "improvement_suggestions": self.improvement_suggestions,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "analysis_duration_ms": self.analysis_duration_ms,
            "analyzer_version": self.analyzer_version,
            "reasoning_chain_analysis": self.reasoning_chain_analysis,
            "confidence_calibration_details": self.confidence_calibration_details
        }

# Pydantic models for API interaction

class BiasDetectionAPI(BaseModel):
    """API model for bias detection results"""
    bias_type: str
    severity: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[str]
    affected_reasoning: List[str]
    bias_description: str
    mitigation_suggestion: str

class ReasoningIssueAPI(BaseModel):
    """API model for reasoning issues"""
    issue_type: str
    severity: float = Field(..., ge=0.0, le=1.0)
    affected_claims: List[str]
    explanation: str
    suggestion: str
    evidence_location: List[str]

class QualityReportAPI(BaseModel):
    """API model for quality reports"""
    task_id: Optional[str]
    interaction_id: Optional[int]
    response_text: str
    quality_score: float = Field(..., ge=0.0, le=1.0)
    issues_detected: List[ReasoningIssueAPI]
    bias_flags: List[BiasDetectionAPI]
    confidence_assessment: str
    recommendation: str
    improvement_suggestions: List[str]
    analysis_duration_ms: Optional[int]

class MetaCognitiveAnalysisRequest(BaseModel):
    """Request for meta-cognitive analysis"""
    response_text: str
    task_context: Optional[Dict[str, Any]] = {}
    reasoning_chain: Optional[List[str]] = []
    confidence_level: Optional[float] = None
    sources_cited: Optional[List[str]] = []
    interaction_id: Optional[int] = None

class MetaCognitiveMetrics(BaseModel):
    """Real-time metrics for meta-cognitive monitoring"""
    total_analyses: int
    avg_quality_score: float
    bias_detection_rate: float
    most_common_biases: List[str]
    quality_trend: str  # "improving", "stable", "declining"
    confidence_calibration_accuracy: float
    recent_issues: List[str]

# Task model for async processing
@dataclass
class Task:
    """Simplified task model for meta-cognitive monitoring"""
    task_id: str
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AgentResponse:
    """Simplified agent response model"""
    response_text: str
    confidence: Optional[float] = None
    reasoning_chain: List[str] = field(default_factory=list)
    sources_cited: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)