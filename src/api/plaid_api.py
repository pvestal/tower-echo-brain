#!/usr/bin/env python3
"""
Plaid Financial API endpoints for Echo Brain
Proxies requests through tower-auth bridge for Plaid integration
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/finance", tags=["plaid"])


@router.get("/status")
async def plaid_status():
    """Check Plaid connection status"""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        async with __import__('httpx').AsyncClient() as client:
            response = await client.get(f"{tower_auth.auth_service_url}/api/auth/plaid/status")
            if response.status_code == 200:
                return response.json()
        return {"configured": False, "linked": False, "accounts_count": 0}
    except Exception as e:
        logger.error(f"Error checking Plaid status: {e}")
        return {"configured": False, "linked": False, "error": str(e)}


@router.get("/accounts")
async def get_accounts():
    """Get linked financial accounts"""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        result = await tower_auth.get_plaid_accounts()
        if "error" in result and "accounts" not in result:
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Plaid accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transactions")
async def get_transactions(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get recent transactions"""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        result = await tower_auth.get_plaid_transactions(start_date, end_date)
        if "error" in result and "transactions" not in result:
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Plaid transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balances")
async def get_balances():
    """Get account balances"""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        result = await tower_auth.get_plaid_balances()
        if "error" in result and "balances" not in result:
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Plaid balances: {e}")
        raise HTTPException(status_code=500, detail=str(e))
