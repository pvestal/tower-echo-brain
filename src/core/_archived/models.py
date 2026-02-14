"""
Shared Data Models
Prevents circular dependencies by keeping models separate from business logic
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# API Response Models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str = "4.0.0"
    service: Optional[str] = None

class MemoryItem(BaseModel):
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class AgentRequest(BaseModel):
    task: str
    agent_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class Conversation(BaseModel):
    id: str
    user_id: str = "default"
    messages: List[ConversationMessage]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# Enum Types
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ModelType(str, Enum):
    LLM = "llm"
    DIFFUSION = "diffusion"
    EMBEDDING = "embedding"
    VAE = "vae"
    CHECKPOINT = "checkpoint"
    LORA = "lora"

# Echo Brain specific models
class EchoMemory(BaseModel):
    """Echo Brain memory structure"""
    id: str
    content: str
    source: str = "echo_brain"
    type: str = "memory"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    score: Optional[float] = None

class MoltbookAgent(BaseModel):
    """Moltbook agent profile"""
    id: str
    name: str
    personality: Dict[str, Any]
    capabilities: List[str]
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.now)

class SystemHealth(BaseModel):
    """System health status"""
    service: str
    status: str
    uptime: Optional[int] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    active_connections: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Request/Response models for API endpoints
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query: str
    timestamp: datetime = Field(default_factory=datetime.now)

class MCPRequest(BaseModel):
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)