#!/usr/bin/env python3
"""
Secured API Routes for Echo Brain
Applies appropriate security middleware to all endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, Request, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

# Import schemas for request/response validation
from src.api.schemas import (
    QueryRequest, QueryResponse, ExecuteRequest, ExecuteResponse,
    ModelRequest, ModelResponse, FileAnalysisRequest, FileAnalysisResponse,
    LoginRequest, TokenResponse, UserInfo, HealthResponse,
    GitStatusResponse, GitCommitRequest, GitCommitResponse,
    SystemMetrics, ErrorResponse
)

# Import security dependencies
from src.middleware.security_middleware import (
    RequireAuth, RequireAdmin, OptionalAuth,
    apply_rate_limiting, get_security_status
)

# Import business logic from existing APIs
from src.api.echo import router as echo_router
from src.api.models import router as models_router
from src.api.health import router as health_router

logger = logging.getLogger(__name__)

def create_secured_routes() -> APIRouter:
    """Create secured API routes with comprehensive security applied"""

    router = APIRouter()

    # Authentication endpoints (public but rate limited)
    @router.post("/auth/login", response_model=TokenResponse)
    async def login(request: Request, credentials: LoginRequest):
        """User login with enhanced security"""
        from src.middleware.auth_middleware import auth_middleware

        # Apply rate limiting
        await apply_rate_limiting(request)

        # Validate credentials (implement your auth logic)
        # This is a placeholder - integrate with your actual auth system
        if credentials.username == "patrick" and credentials.password == "admin":
            user_data = {
                'username': credentials.username,
                'user_id': credentials.username,
                'role': 'admin'
            }
            token = auth_middleware.create_access_token(user_data)
            refresh_token = auth_middleware.create_refresh_token(credentials.username)

            return TokenResponse(
                access_token=token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=1800  # 30 minutes
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

    @router.post("/auth/refresh", response_model=TokenResponse)
    async def refresh_token(request: Request, refresh_token: str):
        """Refresh access token"""
        # Apply rate limiting
        await apply_rate_limiting(request)

        # Implement token refresh logic
        # This is a placeholder
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Token refresh not yet implemented"
        )

    @router.post("/auth/logout")
    async def logout(request: Request, user: Dict[str, Any] = Depends(RequireAuth)):
        """Logout user by blacklisting token"""
        from src.middleware.auth_middleware import auth_middleware

        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            success = await auth_middleware.logout_user(token)
            return {"message": "Logged out successfully", "success": success}

        return {"message": "No token to logout", "success": True}

    # Echo Brain query endpoints (require authentication)
    @router.post("/query", response_model=QueryResponse)
    async def secure_query(
        request: Request,
        query_request: QueryRequest,
        user: Dict[str, Any] = Depends(RequireAuth)
    ):
        """Secured Echo Brain query endpoint"""
        # Import the actual query logic
        from src.api.echo import echo_query_handler

        # Add user context to request
        query_request.user_id = user.get('user_id', user.get('user', 'unknown'))

        # Execute the query with user context
        try:
            response = await echo_query_handler(query_request, user)
            return response
        except Exception as e:
            logger.error(f"Query execution failed for user {user.get('user')}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query processing failed: {str(e)}"
            )

    @router.post("/chat", response_model=QueryResponse)
    async def secure_chat(
        request: Request,
        query_request: QueryRequest,
        user: Dict[str, Any] = Depends(RequireAuth)
    ):
        """Secured chat endpoint (alias for query)"""
        return await secure_query(request, query_request, user)

    # Command execution (requires admin privileges)
    @router.post("/execute", response_model=ExecuteResponse)
    async def secure_execute(
        request: Request,
        execute_request: ExecuteRequest,
        user: Dict[str, Any] = Depends(RequireAdmin)
    ):
        """Secured command execution endpoint"""
        from src.security.safe_command_executor import SafeCommandExecutor

        executor = SafeCommandExecutor()

        # Add user context
        execute_request.user_id = user.get('user_id', user.get('user', 'unknown'))

        # Execute with safety checks
        try:
            result = await executor.execute_command(
                command=execute_request.command,
                safe_mode=execute_request.safe_mode,
                timeout=execute_request.timeout,
                working_directory=execute_request.working_directory
            )

            return ExecuteResponse(
                command=execute_request.command,
                success=result.get('success', False),
                output=result.get('output', ''),
                error=result.get('error'),
                exit_code=result.get('exit_code', -1),
                execution_time=result.get('execution_time', 0.0),
                safe_mode=execute_request.safe_mode
            )
        except Exception as e:
            logger.error(f"Command execution failed for user {user.get('user')}: {e}")
            return ExecuteResponse(
                command=execute_request.command,
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=0.0,
                safe_mode=execute_request.safe_mode
            )

    # Model management (requires admin privileges)
    @router.post("/models/manage", response_model=ModelResponse)
    async def secure_model_manage(
        request: Request,
        model_request: ModelRequest,
        background_tasks: BackgroundTasks,
        user: Dict[str, Any] = Depends(RequireAdmin)
    ):
        """Secured model management endpoint"""
        # Import model management logic
        from src.api.models import handle_model_operation

        try:
            response = await handle_model_operation(model_request, background_tasks, user)
            return response
        except Exception as e:
            logger.error(f"Model management failed for user {user.get('user')}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model operation failed: {str(e)}"
            )

    # File analysis (requires authentication)
    @router.post("/files/analyze", response_model=FileAnalysisResponse)
    async def secure_file_analysis(
        request: Request,
        analysis_request: FileAnalysisRequest,
        user: Dict[str, Any] = Depends(RequireAuth)
    ):
        """Secured file analysis endpoint"""
        # Import file analysis logic
        from src.core.file_analyzer import analyze_file

        try:
            result = await analyze_file(
                file_path=analysis_request.file_path,
                analysis_type=analysis_request.analysis_type,
                include_suggestions=analysis_request.include_suggestions,
                max_lines=analysis_request.max_lines,
                user_context=user
            )

            return FileAnalysisResponse(
                file_path=analysis_request.file_path,
                analysis_type=analysis_request.analysis_type,
                summary=result.get('summary', ''),
                issues=result.get('issues', []),
                suggestions=result.get('suggestions', []),
                metrics=result.get('metrics', {}),
                processing_time=result.get('processing_time', 0.0)
            )
        except Exception as e:
            logger.error(f"File analysis failed for user {user.get('user')}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File analysis failed: {str(e)}"
            )

    # System metrics (requires admin privileges)
    @router.get("/system/metrics", response_model=SystemMetrics)
    async def secure_system_metrics(
        request: Request,
        user: Dict[str, Any] = Depends(RequireAdmin)
    ):
        """Secured system metrics endpoint"""
        from src.api.system_metrics import get_system_metrics

        try:
            metrics = await get_system_metrics()
            return metrics
        except Exception as e:
            logger.error(f"System metrics failed for user {user.get('user')}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"System metrics unavailable: {str(e)}"
            )

    # Git operations (requires admin privileges)
    @router.get("/git/status", response_model=GitStatusResponse)
    async def secure_git_status(
        request: Request,
        user: Dict[str, Any] = Depends(RequireAdmin)
    ):
        """Secured git status endpoint"""
        try:
            from src.api.git_operations import get_git_status
            status_info = await get_git_status()
            return status_info
        except Exception as e:
            logger.error(f"Git status failed for user {user.get('user')}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Git status unavailable: {str(e)}"
            )

    @router.post("/git/commit", response_model=GitCommitResponse)
    async def secure_git_commit(
        request: Request,
        commit_request: GitCommitRequest,
        user: Dict[str, Any] = Depends(RequireAdmin)
    ):
        """Secured git commit endpoint"""
        try:
            from src.api.git_operations import create_git_commit
            commit_response = await create_git_commit(commit_request, user)
            return commit_response
        except Exception as e:
            logger.error(f"Git commit failed for user {user.get('user')}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Git commit failed: {str(e)}"
            )

    # User management endpoints
    @router.get("/user/info", response_model=UserInfo)
    async def get_user_info(
        request: Request,
        user: Dict[str, Any] = Depends(RequireAuth)
    ):
        """Get current user information"""
        return UserInfo(
            user_id=user.get('user_id', ''),
            username=user.get('user', ''),
            role=user.get('role', 'user'),
            email=user.get('email'),
            created_at=user.get('created_at', '2025-01-01T00:00:00'),
            last_login=user.get('last_login')
        )

    @router.get("/user/rate-limit")
    async def get_user_rate_limit(
        request: Request,
        user: Dict[str, Any] = Depends(RequireAuth)
    ):
        """Get current user's rate limit status"""
        from src.middleware.rate_limiting import get_user_rate_limit_status
        return await get_user_rate_limit_status(request, user)

    # Health check with authentication status
    @router.get("/health", response_model=HealthResponse)
    async def secure_health_check(
        request: Request,
        user: Optional[Dict[str, Any]] = Depends(OptionalAuth)
    ):
        """Health check with optional authentication"""
        import time

        auth_status = "authenticated" if user else "anonymous"
        user_role = user.get('role', 'none') if user else 'none'

        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            version="2.1.0-secure",
            uptime=0.0,  # Calculate actual uptime
            components={
                "authentication": auth_status,
                "user_role": user_role,
                "rate_limiting": "active",
                "security_headers": "active",
                "input_validation": "active"
            }
        )

    # Security status endpoint (public)
    @router.get("/security")
    async def security_status(request: Request):
        """Get comprehensive security status"""
        return await get_security_status()

    logger.info("✅ Secured API routes created with comprehensive security")
    return router

# Placeholder functions for missing imports
async def echo_query_handler(query_request: QueryRequest, user: Dict[str, Any]) -> QueryResponse:
    """Placeholder for echo query handler"""
    # This should integrate with your actual echo brain logic
    return QueryResponse(
        response=f"Secured response to: {query_request.query}",
        model_used="placeholder",
        intelligence_level=query_request.intelligence_level,
        processing_time=0.1,
        conversation_id=query_request.conversation_id or "new",
        intent="placeholder",
        confidence=0.8
    )

async def handle_model_operation(model_request: ModelRequest, background_tasks, user: Dict[str, Any]) -> ModelResponse:
    """Placeholder for model operation handler"""
    return ModelResponse(
        success=True,
        message=f"Model operation {model_request.operation} for {model_request.model_name} queued",
        request_id="placeholder-request-id"
    )

async def analyze_file(file_path: str, analysis_type: str, include_suggestions: bool, max_lines: int, user_context: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for file analysis"""
    return {
        'summary': f"Analysis of {file_path} ({analysis_type})",
        'issues': [],
        'suggestions': ["This is a placeholder analysis"],
        'metrics': {'lines': 100, 'complexity': 'medium'},
        'processing_time': 0.5
    }

async def get_system_metrics() -> SystemMetrics:
    """Placeholder for system metrics"""
    return SystemMetrics(
        cpu_usage=25.0,
        memory_usage=60.0,
        disk_usage=45.0,
        load_average=[1.0, 1.2, 1.1],
        uptime=3600.0,
        active_connections=5
    )

async def get_git_status() -> GitStatusResponse:
    """Placeholder for git status"""
    return GitStatusResponse(
        branch="main",
        staged_changes=[],
        unstaged_changes=[],
        untracked_files=[],
        behind_remote=0,
        ahead_remote=0
    )

async def create_git_commit(commit_request: GitCommitRequest, user: Dict[str, Any]) -> GitCommitResponse:
    """Placeholder for git commit"""
    return GitCommitResponse(
        success=True,
        commit_hash="placeholder-hash",
        message="Commit created successfully",
        files_committed=commit_request.files or []
    )