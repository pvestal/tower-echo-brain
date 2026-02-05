"""
Pydantic schemas for the intelligence layer
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# Core types
class SymbolType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    CONSTANT = "constant"


class ServiceType(str, Enum):
    SYSTEMD = "systemd"
    DOCKER = "docker"
    PROCESS = "process"
    API = "api"


class ActionType(str, Enum):
    SHELL = "shell"
    FILE_MODIFY = "file_modify"
    API_CALL = "api_call"
    SERVICE_MANAGE = "service_manage"
    DB_QUERY = "db_query"


class SafetyLevel(str, Enum):
    SAFE = "safe"
    NEEDS_CONFIRM = "needs_confirm"
    DANGEROUS = "dangerous"


class ProcedureCategory(str, Enum):
    DIAGNOSTIC = "diagnostic"
    REMEDIATION = "remediation"
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"


# Code Intelligence
class CodeLocation(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    column_start: Optional[int] = None
    column_end: Optional[int] = None


class CodeSymbol(BaseModel):
    id: Optional[int] = None
    file_id: int
    name: str
    symbol_type: SymbolType
    line_start: int
    line_end: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_id: Optional[int] = None


class CodeFile(BaseModel):
    id: Optional[int] = None
    path: str
    content_hash: str
    last_indexed: datetime
    line_count: int
    language: str = "python"


class CodeDependency(BaseModel):
    id: Optional[int] = None
    from_file_id: int
    to_module: str
    import_names: List[str] = []
    is_relative: bool = False


class APIEndpoint(BaseModel):
    id: Optional[int] = None
    file_id: int
    http_method: str
    path_pattern: str
    function_name: str
    parameters: Dict[str, Any] = {}


class DependencyGraph(BaseModel):
    file_path: str
    dependencies: List[str]
    dependents: List[str]


class CodeIssue(BaseModel):
    severity: str  # error, warning, info
    message: str
    location: CodeLocation
    suggestion: Optional[str] = None


# System Model
class Service(BaseModel):
    id: Optional[int] = None
    name: str
    service_type: ServiceType
    port: Optional[int] = None
    status: str
    config_path: Optional[str] = None
    last_checked: datetime
    metadata: Dict[str, Any] = {}


class ServiceStatus(BaseModel):
    name: str
    status: str  # running, stopped, failed, unknown
    port: Optional[int] = None
    uptime: Optional[str] = None
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    recent_errors: List[str] = []


class ServiceDependency(BaseModel):
    id: Optional[int] = None
    service_id: int
    depends_on: str
    dependency_type: str  # requires, wants, network


class ServiceHealth(BaseModel):
    id: Optional[int] = None
    service_id: int
    checked_at: datetime
    status: str
    response_time_ms: Optional[int] = None
    memory_mb: Optional[float] = None
    error_count: int = 0
    details: Dict[str, Any] = {}


class NetworkMap(BaseModel):
    services: List[Service]
    connections: List[Dict[str, Any]]


class Schema(BaseModel):
    database: str
    tables: List[Dict[str, Any]]


# Procedures and Actions
class Step(BaseModel):
    action: ActionType
    target: str
    command: str
    description: str
    safety_level: SafetyLevel = SafetyLevel.SAFE
    timeout_seconds: int = 30
    expected_output: Optional[str] = None
    on_failure: Optional[str] = None  # continue, abort, retry


class Procedure(BaseModel):
    id: Optional[int] = None
    name: str
    description: str
    trigger_patterns: List[str] = []
    category: ProcedureCategory
    steps: List[Step]
    created_at: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


class ProcedureExecution(BaseModel):
    id: Optional[int] = None
    procedure_id: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: Optional[bool] = None
    context: Dict[str, Any] = {}
    result: Dict[str, Any] = {}
    error_message: Optional[str] = None


class ActionPlan(BaseModel):
    intent: str
    steps: List[Step]
    estimated_time: int  # seconds
    requires_confirmation: bool = False


# Execution Results
class ShellResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: int
    success: bool


class ModifyResult(BaseModel):
    file_path: str
    backup_path: str
    changes_applied: int
    success: bool
    validation_passed: bool = True
    error: Optional[str] = None


class APIResult(BaseModel):
    url: str
    method: str
    status_code: int
    response_data: Dict[str, Any]
    response_time_ms: int
    success: bool


class ServiceResult(BaseModel):
    service_name: str
    action: str
    success: bool
    new_status: str
    message: Optional[str] = None


class QueryResult(BaseModel):
    query: str
    rows_affected: int
    data: List[Dict[str, Any]] = []
    execution_time_ms: int
    success: bool


# Action Log
class ActionLog(BaseModel):
    id: Optional[int] = None
    timestamp: datetime
    action_type: ActionType
    target: str
    command: str
    result: Dict[str, Any]
    success: bool
    user_confirmed: bool = False
    execution_time_ms: int


# Reasoning and Queries
class QueryType(str, Enum):
    SELF_INTROSPECTION = "self_introspection"
    SYSTEM_QUERY = "system_query"
    CODE_QUERY = "code_query"
    ACTION_REQUEST = "action_request"
    GENERAL_KNOWLEDGE = "general_knowledge"


class QueryRequest(BaseModel):
    query: str
    allow_actions: bool = False
    context: Dict[str, Any] = {}


class ActionRequest(BaseModel):
    action: str
    parameters: Dict[str, Any] = {}
    confirm_dangerous: bool = False


class DiagnoseRequest(BaseModel):
    issue: str
    service: Optional[str] = None
    context: Dict[str, Any] = {}


class Response(BaseModel):
    query: str
    query_type: QueryType
    response: str
    actions_taken: List[Dict[str, Any]] = []
    confidence: float = 1.0
    sources: List[str] = []
    execution_time_ms: int


class Diagnosis(BaseModel):
    issue: str
    findings: List[str]
    root_cause: Optional[str] = None
    recommendations: List[str]
    severity: str  # low, medium, high, critical
    estimated_fix_time: Optional[int] = None  # minutes