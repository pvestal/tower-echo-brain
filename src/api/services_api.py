#!/usr/bin/env python3
"""
Unified external services status endpoint for Echo Brain
Shows connection status for all integrated services (Google, Apple Music, Plaid)
"""
from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["services"])


@router.get("/services")
async def get_services_status():
    """Get connection status for all external services"""
    from src.integrations.tower_auth_bridge import tower_auth

    result = {
        "timestamp": datetime.now().isoformat(),
        "tower_auth": {"connected": False},
        "google": {"connected": False, "scopes": []},
        "apple_music": {"connected": False, "has_user_token": False},
        "plaid": {"connected": False, "accounts_linked": 0},
    }

    # Check tower-auth connectivity
    try:
        auth_ok = await tower_auth.initialize()
        result["tower_auth"]["connected"] = auth_ok
    except Exception as e:
        logger.error(f"Tower auth status check failed: {e}")

    # Check Google
    try:
        token = await tower_auth.get_valid_token('google')
        if token:
            result["google"]["connected"] = True
            result["google"]["scopes"] = ["calendar", "gmail", "photos", "drive"]
            if 'google' in tower_auth.cached_tokens:
                token_info = tower_auth.cached_tokens['google']
                expires_at = token_info.get('expires_at')
                if expires_at:
                    result["google"]["token_expires_at"] = str(expires_at)
    except Exception as e:
        logger.error(f"Google status check failed: {e}")

    # Check Apple Music
    try:
        apple_status = await tower_auth.get_apple_music_status()
        result["apple_music"]["connected"] = True  # tower-auth reachable
        result["apple_music"]["has_user_token"] = apple_status.get("authorized", False)
        if apple_status.get("authorized_at"):
            result["apple_music"]["authorized_at"] = apple_status["authorized_at"]
    except Exception as e:
        logger.error(f"Apple Music status check failed: {e}")

    # Check Plaid
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{tower_auth.auth_service_url}/api/auth/plaid/status")
            if response.status_code == 200:
                plaid_data = response.json()
                result["plaid"]["connected"] = plaid_data.get("linked", False)
                result["plaid"]["configured"] = plaid_data.get("configured", False)
                if plaid_data.get("linked_at"):
                    result["plaid"]["linked_at"] = plaid_data["linked_at"]
    except Exception as e:
        logger.error(f"Plaid status check failed: {e}")

    return result
