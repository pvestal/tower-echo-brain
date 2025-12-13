"""
FastAPI Dependency Injection for Echo Brain
Provides lazy initialization of dependencies to avoid circular imports
"""

from typing import Optional, Any
from fastapi import Depends, Request, HTTPException
import logging

logger = logging.getLogger(__name__)

# Global instances (initialized by main.py at startup)
_service_registry = None
_request_logger = None
_model_manager = None

def set_dependencies(service_registry, request_logger):
    """Called by main.py after creating dependencies"""
    global _service_registry, _request_logger
    _service_registry = service_registry
    _request_logger = request_logger
    logger.info("✅ Dependencies registered for injection")

def get_service_registry():
    """Dependency provider for ServiceRegistry"""
    if _service_registry is None:
        # Fallback: create new instance if not initialized
        from routing.service_registry import ServiceRegistry
        return ServiceRegistry()
    return _service_registry

def get_request_logger():
    """Dependency provider for RequestLogger"""
    if _request_logger is None:
        # Fallback: create new instance if not initialized
        from routing.request_logger import RequestLogger
        return RequestLogger()
    return _request_logger

def get_model_manager_instance(
    service_registry=Depends(get_service_registry),
    request_logger=Depends(get_request_logger)
):
    """Dependency provider for ModelManager with lazy initialization"""
    global _model_manager

    if _model_manager is None:
        from model_manager import get_model_manager
        try:
            _model_manager = get_model_manager(service_registry, request_logger)
            logger.info("✅ Model manager initialized via dependency injection")
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            # Return None and let endpoints handle fallback
            return None

    return _model_manager

# Convenience function for direct ollama operations
async def execute_ollama_command(command: list):
    """Execute ollama command directly as fallback"""
    import asyncio
    import subprocess

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
            "returncode": process.returncode
        }
    except Exception as e:
        logger.error(f"Ollama command failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# User context dependencies
def get_current_user(request: Request) -> str:
    """Get current username from request state (set by middleware)"""
    if hasattr(request.state, 'username'):
        return request.state.username
    return "anonymous"

def get_user_context(request: Request) -> Optional[Any]:
    """Get user context from request state (set by middleware)"""
    if hasattr(request.state, 'user_context'):
        return request.state.user_context
    return None

def get_user_recognition(request: Request) -> dict:
    """Get user recognition from request state (set by middleware)"""
    if hasattr(request.state, 'user_recognition'):
        return request.state.user_recognition
    return {"access_level": "none", "recognized": False}

def require_permission(permission: str):
    """Dependency to require specific permission"""
    def permission_checker(request: Request):
        if not hasattr(request.state, 'user_context'):
            raise HTTPException(status_code=403, detail="No user context available")

        user_context = request.state.user_context
        if not user_context:
            raise HTTPException(status_code=403, detail="User not authenticated")

        # Special case for creator permission
        if permission == "creator":
            if request.state.username != "patrick":
                raise HTTPException(status_code=403, detail="Creator access required")
        elif not user_context.permissions.get(permission, False):
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")

        return True

    return permission_checker

def get_user_manager(request: Request) -> Optional[Any]:
    """Get user manager from request state"""
    if hasattr(request.state, 'user_manager'):
        return request.state.user_manager
    return None

def get_echo_identity(request: Request) -> Optional[Any]:
    """Get echo identity from request state"""
    if hasattr(request.state, 'echo_identity'):
        return request.state.echo_identity
    return None