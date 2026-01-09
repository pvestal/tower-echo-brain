#!/usr/bin/env python3
"""
Orchestration Models - Properly typed models for dynamic orchestration
Replaces generic Dict usage with strongly typed Pydantic models
"""

from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ServiceType(str, Enum):
    """Available service types Echo can orchestrate"""
    KNOWLEDGE_BASE = "knowledge_base"
    ANIME_PRODUCTION = "anime_production"
    MUSIC_PRODUCTION = "music_production"
    APPLE_MUSIC = "apple_music"
    COMFYUI = "comfyui"
    GIT_ANALYSIS = "git_analysis"
    FILE_SYSTEM = "file_system"
    COMMAND_EXECUTION = "command_execution"
    VOICE_GENERATION = "voice_generation"
    SEMANTIC_SEARCH = "semantic_search"
    VISUAL_QC = "visual_qc"

class ServiceCapability(BaseModel):
    """Describes a service capability"""
    service_type: ServiceType
    url: Optional[str] = None
    endpoints: Dict[str, str] = Field(default_factory=dict)
    description: str
    requires_auth: bool = False
    timeout_seconds: int = 30

class QueryUnderstanding(BaseModel):
    """LLM's understanding of what the query needs"""
    understanding: str = Field(..., description="Brief explanation of user intent")
    required_services: List[ServiceType] = Field(..., description="Services needed")
    parallel_possible: bool = Field(True, description="Can services run in parallel")
    sequence: Optional[List[ServiceType]] = Field(None, description="If sequential, the order")
    parameters: Dict[ServiceType, Dict[str, Any]] = Field(default_factory=dict)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    complexity_score: float = Field(0.5, ge=0.0, le=1.0)

class ServiceResult(BaseModel):
    """Result from a service call"""
    service_type: ServiceType
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OrchestrationContext(BaseModel):
    """Context maintained across orchestration"""
    conversation_id: str
    user_id: str = "default"
    previous_queries: List[str] = Field(default_factory=list)
    previous_responses: List[str] = Field(default_factory=list)
    active_services: Set[ServiceType] = Field(default_factory=set)
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class OrchestrationRequest(BaseModel):
    """Request for orchestration"""
    query: str
    conversation_id: Optional[str] = None
    user_id: str = "default"
    context: Optional[OrchestrationContext] = None
    model_preference: str = "auto"
    timeout: int = 60

class OrchestrationResponse(BaseModel):
    """Response from orchestration"""
    response: str
    understanding: QueryUnderstanding
    services_used: List[ServiceType]
    service_results: Dict[ServiceType, ServiceResult]
    processing_time: float
    conversation_id: str
    model_used: str = "orchestrator"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Backward compatibility with existing QueryResponse
    @property
    def intent(self) -> Optional[str]:
        """Map to legacy intent for backward compatibility"""
        if self.services_used:
            return self.services_used[0].value
        return "orchestrated"

    @property
    def escalation_path(self) -> List[str]:
        """Map to legacy escalation path"""
        return [s.value for s in self.services_used]

class ServiceRegistration(BaseModel):
    """Registration of a service with the orchestrator"""
    service_type: ServiceType
    capability: ServiceCapability
    health_check_endpoint: str
    version: str
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True