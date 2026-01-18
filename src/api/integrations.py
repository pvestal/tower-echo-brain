"""
Integrations API - Manage service connections.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import asyncpg
import json

router = APIRouter(prefix="/api/integrations", tags=["integrations"])

DATABASE_URL = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/tower_consolidated"

@router.get("")
async def list_integrations():
    """List all integrations and their status."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, provider, display_name, status, scopes,
                       last_sync_at, connected_at, error_message
                FROM integrations ORDER BY display_name
            """)
        return {"integrations": [dict(r) for r in rows]}
    finally:
        await pool.close()


@router.get("/{provider}")
async def get_integration(provider: str):
    """Get details for a specific integration."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM integrations WHERE provider = $1", provider
            )
            if not row:
                raise HTTPException(status_code=404, detail="Integration not found")
        return {"integration": dict(row)}
    finally:
        await pool.close()


@router.post("/{provider}/connect")
async def initiate_connect(provider: str):
    """Start OAuth flow for a provider."""
    # This would return an OAuth URL
    # Implementation depends on the specific provider
    return {
        "message": f"OAuth flow for {provider} not yet implemented",
        "provider": provider
    }


@router.delete("/{provider}")
async def disconnect(provider: str):
    """Disconnect an integration."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE integrations SET
                    status = 'disconnected',
                    connected_at = NULL,
                    scopes = NULL
                WHERE provider = $1
            """, provider)
        return {"success": True}
    finally:
        await pool.close()


@router.post("/{provider}/sync")
async def trigger_sync(provider: str):
    """Manually trigger a sync for an integration."""
    # Would trigger background sync job
    return {
        "message": f"Sync triggered for {provider}",
        "job_id": "not-implemented"
    }