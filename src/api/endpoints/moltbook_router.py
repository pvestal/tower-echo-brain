"""
Moltbook Router for Echo Brain
Provides endpoints for sharing thoughts to the agent social network
Follows Echo Brain's router structure and conventions
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.integrations.moltbook.client import MoltbookClient

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Moltbook"])

# Request/Response Models
class MoltbookShareRequest(BaseModel):
    """Request model for sharing a thought to Moltbook"""
    title: str
    content: str
    submolt: Optional[str] = "m/testing"
    metadata: Optional[Dict[str, Any]] = None

class MoltbookShareResponse(BaseModel):
    """Response model for Moltbook share operation"""
    success: bool
    post_id: Optional[str] = None
    title: str
    submolt: str
    message: str
    dry_run: bool = True
    timestamp: str

class MoltbookStatusResponse(BaseModel):
    """Response model for Moltbook status"""
    service: str
    status: str
    mode: str
    agent_key_configured: bool
    app_key_configured: bool
    dry_run: bool
    endpoints: list

# Dependency for Moltbook client
async def get_moltbook_client() -> MoltbookClient:
    """Get Moltbook client instance (Dependency Injection pattern)"""
    client = MoltbookClient()
    await client.initialize()
    return client

# ============ Endpoints ============

@router.get("/status", response_model=MoltbookStatusResponse)
async def get_moltbook_status(client: MoltbookClient = Depends(get_moltbook_client)):
    """Get Moltbook integration status"""
    test_result = await client.test_connection()
    
    return MoltbookStatusResponse(
        service="Moltbook Integration",
        status="active" if client.is_configured else "pending",
        mode="dry_run" if client.dry_run else "live",
        agent_key_configured=bool(client.agent_api_key and not client.agent_api_key.startswith("YOUR_")),
        app_key_configured=bool(client.app_api_key and not client.app_api_key.startswith("YOUR_")),
        dry_run=client.dry_run,
        endpoints=[
            "/api/echo/moltbook/status",
            "/api/echo/moltbook/share",
            "/api/echo/moltbook/profile",
            "/api/echo/moltbook/test",
            "/api/echo/moltbook/health"
        ]
    )

@router.post("/share", response_model=MoltbookShareResponse)
async def share_to_moltbook(
    request: MoltbookShareRequest,
    client: MoltbookClient = Depends(get_moltbook_client)
):
    """
    Share a thought to Moltbook
    
    While waiting for API approval (48 hours), runs in dry-run mode.
    Once approved, will make real posts to the agent social network.
    """
    try:
        # Prepare thought data matching Echo Brain's structure
        thought_data = {
            "title": request.title,
            "content": request.content,
            "submolt": request.submolt,
            "metadata": request.metadata or {},
            "source": "echo_brain_api",
            "timestamp": "now"
        }
        
        # Share to Moltbook
        result = await client.share_thought(thought_data)
        
        if result.get("success", False):
            return MoltbookShareResponse(
                success=True,
                post_id=result.get("post_id"),
                title=request.title,
                submolt=request.submolt,
                message=result.get("message", "Shared successfully"),
                dry_run=result.get("dry_run", True),
                timestamp=result.get("timestamp", "")
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to share thought")
            )
            
    except Exception as e:
        logger.error(f"Error sharing to Moltbook: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/profile")
async def get_moltbook_profile(client: MoltbookClient = Depends(get_moltbook_client)):
    """Get Echo Brain's Moltbook agent profile"""
    profile = await client.get_agent_profile()
    return profile

@router.get("/test")
async def test_moltbook_connection(client: MoltbookClient = Depends(get_moltbook_client)):
    """Test Moltbook connection and configuration"""
    return await client.test_connection()

# Establishment endpoints
@router.post("/establish")
async def establish_on_moltbook(client: MoltbookClient = Depends(get_moltbook_client)):
    """Establish Echo Brain presence on Moltbook"""
    import subprocess
    import asyncio

    try:
        # Run establishment script
        result = subprocess.run(
            ["/opt/tower-echo-brain/venv/bin/python", "/opt/tower-echo-brain/scripts/establish_moltbook.py"],
            capture_output=True,
            text=True,
            timeout=30
        )

        return {
            "success": result.returncode == 0,
            "message": "Establishment process initiated",
            "output": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
            "status": "Check logs for detailed progress"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to initiate establishment"
        }

@router.get("/establishment/status")
async def get_establishment_status():
    """Check Echo Brain's Moltbook establishment status"""
    import json
    from pathlib import Path

    config_file = Path("/opt/tower-echo-brain/.moltbook_config.json")
    creds_file = Path("/opt/tower-echo-brain/.moltbook_credentials.json")

    status = {
        "established": False,
        "has_config": config_file.exists(),
        "has_credentials": creds_file.exists(),
        "stats": {},
        "api_keys_status": "not_configured"
    }

    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                status["stats"] = config.get("stats", {})
                status["established"] = bool(config.get("stats", {}).get("established_at"))

        if creds_file.exists():
            with open(creds_file, 'r') as f:
                creds = json.load(f)
                if creds.get("agent_api_key") == "PENDING_APPROVAL":
                    status["api_keys_status"] = "pending_approval"
                elif creds.get("agent_api_key"):
                    status["api_keys_status"] = "configured"
    except Exception as e:
        logger.error(f"Error reading establishment files: {e}")
        status["error"] = str(e)

    return status

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for Moltbook integration"""
    return {
        "status": "healthy",
        "service": "moltbook_integration",
        "version": "1.0.0-dryrun"
    }
