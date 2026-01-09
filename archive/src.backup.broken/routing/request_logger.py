from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class EvidenceType(Enum):
    PERFORMANCE = "performance"
    SECURITY = "security" 
    QUALITY = "quality"
    USER_FEEDBACK = "user_feedback"

class DecisionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass
class Evidence:
    type: EvidenceType
    data: Dict[str, Any]
    timestamp: datetime
    source: str

@dataclass 
class DirectorEvaluation:
    director_name: str
    score: float
    reasoning: str
    evidence: List[Evidence]
    
@dataclass
class TaskDecision:
    task_id: str
    description: str
    status: DecisionStatus
    evaluations: List[DirectorEvaluation]
    final_score: float
    timestamp: datetime

@dataclass
class DecisionPoint:
    id: str
    task_id: str
    timestamp: datetime
    context: Dict[str, Any]

class RequestLogger:
    def __init__(self):
        self.requests = []
        
    def log_request(self, request):
        self.requests.append(request)
        
    def get_requests(self):
        return self.requests

