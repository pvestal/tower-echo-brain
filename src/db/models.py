#!/usr/bin/env python3
"""
Database models and Pydantic schemas for Echo Brain system
"""

from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict] = {}
    intelligence_level: Optional[str] = "auto"
    user_id: Optional[str] = "default"  # For conversation tracking
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    model_used: str
    intelligence_level: str
    processing_time: float
    escalation_path: List[str]
    requires_clarification: bool = False
    clarifying_questions: List[str] = []
    conversation_id: str
    intent: Optional[str] = None
    confidence: float = 0.0

class ExecuteRequest(BaseModel):
    command: str
    context: Optional[Dict] = {}
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "default"
    safe_mode: bool = True

class ExecuteResponse(BaseModel):
    command: str
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int
    processing_time: float
    conversation_id: str
    safety_checks: Dict[str, bool]

class TestRequest(BaseModel):
    target: str
    test_type: Optional[str] = "universal"
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "default"

class VoiceNotificationRequest(BaseModel):
    message: str
    character: Optional[str] = "yukiko"
    user_id: Optional[str] = "default"
    priority: Optional[str] = "normal"

class VoiceStatusRequest(BaseModel):
    user_id: Optional[str] = "default"

# Database Connection Models
class DatabaseConfig:
    def __init__(self):
        self.host = "localhost"
        self.database = "tower_consolidated"
        self.user = "patrick"
        self.password = None  # No password for local connections
        self.port = 5432

    def get_connection_string(self) -> str:
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"