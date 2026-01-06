# Security module for Echo Brain authentication and safe command execution

from fastapi import HTTPException, status
from fastapi.security import HTTPBearer as FastAPIHTTPBearer
from fastapi.security import HTTPAuthorizationCredentials

class HTTPBearer(FastAPIHTTPBearer):
    """Simple HTTP Bearer authentication"""
    def __init__(self):
        super().__init__(auto_error=True)

# Import safe command executor
from .safe_command_executor import SafeCommandExecutor, safe_command_executor

# Export for compatibility
__all__ = ["HTTPBearer", "HTTPAuthorizationCredentials", "SafeCommandExecutor", "safe_command_executor"]
