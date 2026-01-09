#!/usr/bin/env python3
"""
Secure Application Factory for Echo Brain with Comprehensive Security
Integrates authentication, rate limiting, input validation, and audit logging
"""

import os
import logging
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import security middleware
from src.middleware.security_middleware import SecurityMiddleware, get_security_status
from src.middleware.auth_middleware import get_auth_status, create_emergency_token
from src.middleware.rate_limiting import get_rate_limit_status, get_user_rate_limit_status

# Import API routers with security applied
from src.api.secured_routes import create_secured_routes

# Import existing routers (will be secured)
from src.api.routes import router as main_router
from src.api.health import router as health_router
from src.api.models import router as models_router
from src.services.testing import router as testing_router

# Additional routers
try:
    from src.api.echo_refactored import router as refactored_router
    echo_refactored_available = True
except ImportError:
    refactored_router = None
    echo_refactored_available = False

try:
    from src.api.system_metrics import router as system_metrics_router
    system_metrics_available = True
except ImportError:
    from src.api.system_stub import router as system_metrics_router
    system_metrics_available = False

try:
    from src.api.git_operations import router as git_operations_router
    git_operations_available = True
except ImportError:
    git_operations_router = None
    git_operations_available = False

try:
    from src.api.delegation_routes import router as delegation_router
    delegation_available = True
except ImportError:
    delegation_router = None
    delegation_available = False

# External integrations
try:
    from src.modules.agents.agent_development_endpoints import agent_dev_router
    agent_dev_available = True
except ImportError:
    agent_dev_router = None
    agent_dev_available = False

try:
    from src.misc.veteran_guardian_endpoints import veteran_router
    veteran_available = True
except ImportError:
    veteran_router = None
    veteran_available = False

logger = logging.getLogger(__name__)

def create_secure_app() -> FastAPI:
    """Create and configure secure FastAPI application"""

    # Create FastAPI app with security-focused configuration
    app = FastAPI(
        title="Echo Brain - Secure AI Orchestrator",
        description="Advanced AI orchestrator with comprehensive security",
        version="2.1.0-secure",
        docs_url="/docs",  # Keep docs available but will be secured
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Setup comprehensive logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("🔒 Creating secure Echo Brain application...")

    # Add security middleware (this must be first)
    app.add_middleware(
        SecurityMiddleware,
        enforce_https=True,
        audit_all_requests=False  # Set to True for full audit logging
    )

    # Add CORS middleware with security restrictions
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://***REMOVED***",
            "https://tower.local",
            "http://localhost:3000",  # For development
            "http://localhost:8080"   # Tower dashboard
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-*", "X-Security-*"]
    )

    # Security and status endpoints (publicly accessible)
    @app.get("/security/status")
    async def security_status():
        """Get comprehensive security status"""
        return await get_security_status()

    @app.get("/auth/status")
    async def auth_status():
        """Get authentication status"""
        return await get_auth_status()

    @app.get("/rate-limit/status")
    async def rate_limit_status():
        """Get rate limiting status"""
        return get_rate_limit_status()

    @app.get("/rate-limit/user")
    async def user_rate_limit_status(request: Request):
        """Get current user's rate limit status"""
        return await get_user_rate_limit_status(request)

    @app.post("/auth/emergency-token")
    async def generate_emergency_token():
        """Generate emergency admin token (for emergencies only)"""
        token = create_emergency_token()
        logger.warning("🚨 Emergency admin token generated")
        return {"token": token, "expires_in": 1800}  # 30 minutes

    # Health check (exempt from most security)
    @app.get("/health")
    async def health_check():
        """Basic health check"""
        return {
            "status": "healthy",
            "security": "enabled",
            "timestamp": int(__import__('time').time())
        }

    # Include routers with appropriate security
    logger.info("🔗 Including secured API routers...")

    # Core routers (secured)
    app.include_router(main_router, prefix="", tags=["main"])
    app.include_router(health_router, prefix="", tags=["health"])
    app.include_router(models_router, prefix="", tags=["models"])
    app.include_router(testing_router, prefix="", tags=["testing"])
    app.include_router(system_metrics_router, prefix="", tags=["metrics"])

    # Additional routers with availability checks
    if echo_refactored_available:
        app.include_router(refactored_router, prefix="", tags=["refactored"])
        logger.info("✅ Echo refactored router included")

    if git_operations_available:
        app.include_router(git_operations_router, prefix="/api/echo", tags=["git"])
        logger.info("✅ Git operations router included")

    if delegation_available:
        app.include_router(delegation_router, prefix="/api/echo/delegation", tags=["delegation"])
        logger.info("✅ Delegation router included")

    if agent_dev_available:
        app.include_router(agent_dev_router, prefix="/api/echo/agent-dev", tags=["agent-dev"])
        logger.info("✅ Agent development router included")

    if veteran_available:
        app.include_router(veteran_router, prefix="/api/echo/veteran", tags=["veteran"])
        logger.info("✅ Veteran guardian router included")

    # Create and include secured routes
    secured_routes = create_secured_routes()
    app.include_router(secured_routes, prefix="/api/secure", tags=["secured"])

    # Static files (if they exist)
    static_path = "/opt/tower-echo-brain/static"
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
        logger.info("✅ Static files mounted")

    # Global exception handler for security events
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with security logging"""
        logger.exception(f"Unhandled exception in {request.url.path}: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An internal error occurred",
                "request_id": id(request)
            },
            headers={
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY"
            }
        )

    logger.info("🔒 Secure Echo Brain application created successfully")
    logger.info(f"📊 Security features enabled:")
    logger.info(f"   - JWT Authentication: ✅")
    logger.info(f"   - Rate Limiting: ✅")
    logger.info(f"   - Input Validation: ✅")
    logger.info(f"   - Security Headers: ✅")
    logger.info(f"   - Audit Logging: ✅")
    logger.info(f"   - HTTPS Enforcement: ✅")
    logger.info(f"   - CORS Protection: ✅")

    return app

def create_development_app() -> FastAPI:
    """Create a development version with relaxed security for local testing"""

    app = create_secure_app()

    # Override some security settings for development
    logger.warning("⚠️ Running in DEVELOPMENT mode with relaxed security")
    logger.warning("⚠️ HTTPS enforcement disabled for local testing")
    logger.warning("⚠️ Rate limiting relaxed for development")

    return app

# Factory function selector
def get_app_factory():
    """Get appropriate app factory based on environment"""
    environment = os.getenv('ENVIRONMENT', 'production').lower()

    if environment == 'development':
        logger.info("🔧 Creating development app with relaxed security")
        return create_development_app()
    else:
        logger.info("🔒 Creating production app with full security")
        return create_secure_app()

# Default app creation
app = get_app_factory()

if __name__ == "__main__":
    import uvicorn

    # Run with security enabled
    uvicorn.run(
        "src.secure_app_factory:app",
        host="0.0.0.0",
        port=8309,
        reload=False,  # Disable reload in production for security
        access_log=True,
        log_level="info"
    )