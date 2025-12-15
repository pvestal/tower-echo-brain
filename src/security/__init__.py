# Security module for Echo Brain authentication

from fastapi import HTTPException, status
from fastapi.security import HTTPBearer as FastAPIHTTPBearer
from fastapi.security import HTTPAuthorizationCredentials

class HTTPBearer(FastAPIHTTPBearer):
    """Simple HTTP Bearer authentication"""
    def __init__(self):
        super().__init__(auto_error=True)

# Export for compatibility
__all__ = ["HTTPBearer", "HTTPAuthorizationCredentials"]
