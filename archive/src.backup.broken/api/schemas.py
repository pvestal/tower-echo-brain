#!/usr/bin/env python3
"""
Enhanced Pydantic schemas for Echo Brain API with comprehensive validation
Provides request/response models with security-focused input validation
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import re

# Constants for validation
MAX_QUERY_LENGTH = 10000
MAX_COMMAND_LENGTH = 1000
MAX_CONTEXT_SIZE = 50000
MAX_USERNAME_LENGTH = 50
MIN_PASSWORD_LENGTH = 8
MAX_FILENAME_LENGTH = 255

class UserRole(str, Enum):
    """User role enumeration"""
    USER = "user"
    ADMIN = "admin"
    PATRICK = "patrick"
    SYSTEM = "system"

class IntelligenceLevel(str, Enum):
    """Intelligence level enumeration"""
    AUTO = "auto"
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"

class RequestType(str, Enum):
    """Request type enumeration"""
    CONVERSATION = "conversation"
    SYSTEM_COMMAND = "system_command"
    COLLABORATION = "collaboration"
    ANALYSIS = "analysis"
    GENERATION = "generation"

class ModelOperationType(str, Enum):
    """Model operation type enumeration"""
    PULL = "pull"
    REMOVE = "remove"
    UPDATE = "update"
    LIST = "list"

# Base validation mixins
class SanitizedStringMixin:
    """Mixin for string sanitization"""

    @validator('*', pre=True)
    def sanitize_strings(cls, v):
        """Basic string sanitization"""
        if isinstance(v, str):
            # Remove null bytes and control characters
            v = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', v)
            # Strip leading/trailing whitespace
            v = v.strip()
        return v

# Authentication schemas
class LoginRequest(BaseModel, SanitizedStringMixin):
    """User login request"""
    username: str = Field(..., min_length=1, max_length=MAX_USERNAME_LENGTH, description="Username")
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, description="Password")

    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError('Username can only contain alphanumeric characters, dots, hyphens, and underscores')
        return v.lower()

class TokenRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Refresh token")

class TokenResponse(BaseModel):
    """Token response"""
    access_token: str = Field(..., description="Access token")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Expiration time in seconds")

class UserInfo(BaseModel):
    """User information"""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    role: UserRole = Field(UserRole.USER, description="User role")
    email: Optional[str] = Field(None, description="Email address")
    created_at: datetime = Field(..., description="Account creation time")
    last_login: Optional[datetime] = Field(None, description="Last login time")

# Query and conversation schemas
class QueryRequest(BaseModel, SanitizedStringMixin):
    """Echo Brain query request with enhanced validation"""
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH, description="User query")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    intelligence_level: IntelligenceLevel = Field(IntelligenceLevel.AUTO, description="Intelligence level")
    user_id: Optional[str] = Field(None, max_length=MAX_USERNAME_LENGTH, description="User ID")
    conversation_id: Optional[str] = Field(None, max_length=100, description="Conversation ID")
    request_type: RequestType = Field(RequestType.CONVERSATION, description="Request type")
    model_preference: Optional[str] = Field(None, max_length=100, description="Preferred model")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Response temperature")

    @validator('context')
    def validate_context_size(cls, v):
        if v and len(str(v)) > MAX_CONTEXT_SIZE:
            raise ValueError(f'Context too large, maximum {MAX_CONTEXT_SIZE} characters')
        return v

    @validator('query')
    def validate_query_content(cls, v):
        # Check for potential injection attempts
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Query contains potentially dangerous content')
        return v

class QueryResponse(BaseModel):
    """Echo Brain query response"""
    response: str = Field(..., description="Generated response")
    model_used: str = Field(..., description="Model used for generation")
    intelligence_level: str = Field(..., description="Intelligence level used")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    escalation_path: List[str] = Field(default_factory=list, description="Escalation path taken")
    requires_clarification: bool = Field(False, description="Whether clarification is needed")
    clarifying_questions: List[str] = Field(default_factory=list, description="Clarifying questions")
    conversation_id: str = Field(..., description="Conversation ID")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Response confidence")

    # Debug information
    debug_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Debug information")
    reasoning: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Reasoning information")
    memory_accessed: Optional[List[Dict]] = Field(default_factory=list, description="Memory accessed")
    processing_breakdown: Optional[Dict[str, float]] = Field(default_factory=dict, description="Processing time breakdown")

# Command execution schemas
class ExecuteRequest(BaseModel, SanitizedStringMixin):
    """Command execution request with security validation"""
    command: str = Field(..., min_length=1, max_length=MAX_COMMAND_LENGTH, description="Command to execute")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution context")
    conversation_id: Optional[str] = Field(None, max_length=100, description="Conversation ID")
    user_id: Optional[str] = Field(None, max_length=MAX_USERNAME_LENGTH, description="User ID")
    safe_mode: bool = Field(True, description="Enable safe mode execution")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Execution timeout in seconds")
    working_directory: Optional[str] = Field(None, max_length=500, description="Working directory")

    @validator('command')
    def validate_command_safety(cls, v):
        """Validate command for safety"""
        # Block dangerous commands in safe mode
        dangerous_commands = [
            'rm -rf', 'rmdir /s', 'format', 'del /f', 'dd if=',
            'mkfs', 'fdisk', ':(){:|:&};:', 'chmod 777', '>(', 'curl | sh',
            'wget | sh', 'shutdown', 'reboot', 'halt', 'poweroff'
        ]

        v_lower = v.lower()
        for dangerous in dangerous_commands:
            if dangerous in v_lower:
                raise ValueError(f'Command contains dangerous operation: {dangerous}')

        return v

    @validator('working_directory')
    def validate_working_directory(cls, v):
        if v:
            # Prevent path traversal
            if '..' in v or v.startswith('/'):
                if not v.startswith('/tmp/') and not v.startswith('/home/patrick/'):
                    raise ValueError('Working directory must be within allowed paths')
        return v

class ExecuteResponse(BaseModel):
    """Command execution response"""
    command: str = Field(..., description="Executed command")
    success: bool = Field(..., description="Execution success")
    output: str = Field(..., description="Command output")
    error: Optional[str] = Field(None, description="Error message")
    exit_code: int = Field(..., description="Exit code")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    safe_mode: bool = Field(..., description="Was executed in safe mode")

# Model management schemas
class ModelRequest(BaseModel, SanitizedStringMixin):
    """Model management request"""
    model_name: str = Field(..., min_length=1, max_length=100, description="Model name")
    operation: ModelOperationType = Field(..., description="Operation type")
    force: bool = Field(False, description="Force operation")

    @validator('model_name')
    def validate_model_name(cls, v):
        # Only allow alphanumeric, hyphens, underscores, colons, and dots
        if not re.match(r'^[a-zA-Z0-9\-_:.]+$', v):
            raise ValueError('Model name contains invalid characters')
        return v

class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    size: Optional[str] = Field(None, description="Model size")
    parameter_count: Optional[str] = Field(None, description="Parameter count")
    quantization: Optional[str] = Field(None, description="Quantization level")
    family: Optional[str] = Field(None, description="Model family")
    modified_at: Optional[datetime] = Field(None, description="Last modified")
    available: bool = Field(True, description="Model availability")

class ModelResponse(BaseModel):
    """Model operation response"""
    success: bool = Field(..., description="Operation success")
    message: str = Field(..., description="Response message")
    model_info: Optional[ModelInfo] = Field(None, description="Model information")
    request_id: Optional[str] = Field(None, description="Request ID for async operations")

# File analysis schemas
class FileAnalysisRequest(BaseModel, SanitizedStringMixin):
    """File analysis request"""
    file_path: str = Field(..., min_length=1, max_length=MAX_FILENAME_LENGTH, description="File path")
    analysis_type: Literal["code", "text", "security", "performance"] = Field("code", description="Analysis type")
    include_suggestions: bool = Field(True, description="Include improvement suggestions")
    max_lines: Optional[int] = Field(1000, ge=1, le=10000, description="Maximum lines to analyze")

    @validator('file_path')
    def validate_file_path(cls, v):
        # Prevent path traversal
        if '..' in v or v.startswith('/'):
            allowed_paths = ['/tmp/', '/home/patrick/', '/opt/tower-']
            if not any(v.startswith(path) for path in allowed_paths):
                raise ValueError('File path must be within allowed directories')
        return v

class FileAnalysisResponse(BaseModel):
    """File analysis response"""
    file_path: str = Field(..., description="Analyzed file path")
    analysis_type: str = Field(..., description="Analysis type")
    summary: str = Field(..., description="Analysis summary")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Found issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Analysis metrics")
    processing_time: float = Field(..., ge=0.0, description="Analysis time")

# Health and status schemas
class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., ge=0.0, description="Uptime in seconds")
    components: Dict[str, str] = Field(default_factory=dict, description="Component statuses")

class RateLimitInfo(BaseModel):
    """Rate limit information"""
    limit: int = Field(..., ge=0, description="Rate limit")
    remaining: int = Field(..., ge=0, description="Remaining requests")
    reset_time: int = Field(..., description="Reset timestamp")
    tier: str = Field(..., description="User tier")

# Error schemas
class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field with error")
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ValidationError]] = Field(None, description="Validation errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID")

# Git operation schemas
class GitStatusResponse(BaseModel):
    """Git status response"""
    branch: str = Field(..., description="Current branch")
    staged_changes: List[str] = Field(default_factory=list, description="Staged changes")
    unstaged_changes: List[str] = Field(default_factory=list, description="Unstaged changes")
    untracked_files: List[str] = Field(default_factory=list, description="Untracked files")
    behind_remote: int = Field(0, description="Commits behind remote")
    ahead_remote: int = Field(0, description="Commits ahead of remote")

class GitCommitRequest(BaseModel, SanitizedStringMixin):
    """Git commit request"""
    message: str = Field(..., min_length=1, max_length=500, description="Commit message")
    files: Optional[List[str]] = Field(None, description="Specific files to commit")
    amend: bool = Field(False, description="Amend previous commit")

class GitCommitResponse(BaseModel):
    """Git commit response"""
    success: bool = Field(..., description="Commit success")
    commit_hash: Optional[str] = Field(None, description="Commit hash")
    message: str = Field(..., description="Response message")
    files_committed: List[str] = Field(default_factory=list, description="Files committed")

# System metrics schemas
class SystemMetrics(BaseModel):
    """System metrics"""
    cpu_usage: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0.0, le=100.0, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0.0, le=100.0, description="Disk usage percentage")
    gpu_usage: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU usage percentage")
    load_average: List[float] = Field(default_factory=list, description="Load average")
    uptime: float = Field(..., ge=0.0, description="System uptime in seconds")
    active_connections: int = Field(..., ge=0, description="Active connections")

# Configuration schemas
class ConfigurationUpdate(BaseModel, SanitizedStringMixin):
    """Configuration update request"""
    key: str = Field(..., min_length=1, max_length=100, description="Configuration key")
    value: Union[str, int, float, bool, Dict, List] = Field(..., description="Configuration value")
    description: Optional[str] = Field(None, max_length=500, description="Configuration description")

    @validator('key')
    def validate_key(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError('Configuration key can only contain alphanumeric characters, dots, hyphens, and underscores')
        return v

class ConfigurationResponse(BaseModel):
    """Configuration response"""
    key: str = Field(..., description="Configuration key")
    value: Union[str, int, float, bool, Dict, List] = Field(..., description="Configuration value")
    description: Optional[str] = Field(None, description="Configuration description")
    last_modified: datetime = Field(..., description="Last modified timestamp")
    modified_by: str = Field(..., description="Modified by user")

# Batch operation schemas
class BatchRequest(BaseModel):
    """Batch operation request"""
    operations: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="List of operations")
    parallel: bool = Field(False, description="Execute operations in parallel")
    stop_on_error: bool = Field(True, description="Stop on first error")

class BatchResponse(BaseModel):
    """Batch operation response"""
    total_operations: int = Field(..., ge=0, description="Total operations")
    successful_operations: int = Field(..., ge=0, description="Successful operations")
    failed_operations: int = Field(..., ge=0, description="Failed operations")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Operation results")
    execution_time: float = Field(..., ge=0.0, description="Total execution time")

# Export all schemas for easy importing
__all__ = [
    'UserRole', 'IntelligenceLevel', 'RequestType', 'ModelOperationType',
    'LoginRequest', 'TokenRequest', 'TokenResponse', 'UserInfo',
    'QueryRequest', 'QueryResponse', 'ExecuteRequest', 'ExecuteResponse',
    'ModelRequest', 'ModelInfo', 'ModelResponse',
    'FileAnalysisRequest', 'FileAnalysisResponse',
    'HealthResponse', 'RateLimitInfo', 'ValidationError', 'ErrorResponse',
    'GitStatusResponse', 'GitCommitRequest', 'GitCommitResponse',
    'SystemMetrics', 'ConfigurationUpdate', 'ConfigurationResponse',
    'BatchRequest', 'BatchResponse'
]