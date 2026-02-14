"""
Integrations API - Manage service connections.
"""
import os
from fastapi import APIRouter, HTTPException
import asyncpg

router = APIRouter(tags=["integrations"])

DATABASE_URL = f"postgresql://patrick:{os.getenv('DB_PASSWORD', '')}@localhost/echo_brain"


async def _connect():
    return await asyncpg.connect(DATABASE_URL)


@router.get("")
async def list_integrations():
    """List all integrations and their status."""
    conn = await _connect()
    try:
        rows = await conn.fetch("""
            SELECT id, provider, display_name, status, scopes,
                   last_sync_at, connected_at, error_message
            FROM integrations ORDER BY display_name
        """)
        return {"integrations": [dict(r) for r in rows]}
    finally:
        await conn.close()


@router.get("/{provider}")
async def get_integration(provider: str):
    """Get details for a specific integration."""
    conn = await _connect()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM integrations WHERE provider = $1", provider
        )
        if not row:
            raise HTTPException(status_code=404, detail="Integration not found")
        return {"integration": dict(row)}
    finally:
        await conn.close()


@router.post("/{provider}/connect")
async def initiate_connect(provider: str):
    """Start OAuth flow for a provider."""
    return {
        "message": f"OAuth flow for {provider} not yet implemented",
        "provider": provider
    }


@router.delete("/{provider}")
async def disconnect(provider: str):
    """Disconnect an integration."""
    conn = await _connect()
    try:
        await conn.execute("""
            UPDATE integrations SET
                status = 'disconnected',
                connected_at = NULL,
                scopes = NULL
            WHERE provider = $1
        """, provider)
        return {"success": True}
    finally:
        await conn.close()


@router.post("/{provider}/sync")
async def trigger_sync(provider: str):
    """Manually trigger a sync for an integration."""
    return {
        "message": f"Sync triggered for {provider}",
        "job_id": "not-implemented"
    }
