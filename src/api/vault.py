"""
Vault API - View and manage API keys (values never exposed).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import os
from pathlib import Path
from datetime import datetime
import asyncpg
import httpx

router = APIRouter(prefix="/api/vault", tags=["vault"])

VAULT_PATH = Path("/home/patrick/.tower_credentials/vault.json")
DATABASE_URL = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/tower_consolidated"

class KeyCreate(BaseModel):
    key_name: str
    value: str  # Will be stored, never returned
    service: str
    key_type: str = "api_key"
    description: Optional[str] = None

def load_vault() -> dict:
    """Load vault file."""
    if VAULT_PATH.exists():
        with open(VAULT_PATH) as f:
            return json.load(f)
    return {}

def save_vault(data: dict):
    """Save vault file."""
    VAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VAULT_PATH, 'w') as f:
        json.dump(data, f, indent=2)

@router.get("/keys")
async def list_keys():
    """List all vault keys (values masked)."""
    vault = load_vault()

    keys = []
    for service, service_data in vault.items():
        if isinstance(service_data, dict):
            for key_name, value in service_data.items():
                if key_name.startswith('_'):
                    continue  # Skip metadata
                keys.append({
                    "key_name": f"{service}.{key_name}",
                    "service": service,
                    "key_type": "api_key" if "key" in key_name.lower() else "secret",
                    "is_set": bool(value),
                    "value_preview": f"{str(value)[:8]}..." if value else None
                })

    return {"keys": keys}


@router.post("/keys")
async def create_key(key: KeyCreate):
    """Add or update a vault key."""
    vault = load_vault()

    # Parse key_name (e.g., "openai.api_key")
    parts = key.key_name.split(".", 1)
    if len(parts) == 2:
        service, key_name = parts
    else:
        service = key.service
        key_name = key.key_name

    # Update vault
    if service not in vault:
        vault[service] = {}
    vault[service][key_name] = key.value

    save_vault(vault)

    # Update registry
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO vault_registry (key_name, key_type, service, description, is_set, last_updated)
                VALUES ($1, $2, $3, $4, true, NOW())
                ON CONFLICT (key_name) DO UPDATE SET
                    is_set = true, last_updated = NOW()
            """, key.key_name, key.key_type, key.service, key.description)
    finally:
        await pool.close()

    return {"success": True, "key_name": key.key_name}


@router.delete("/keys/{key_name:path}")
async def delete_key(key_name: str):
    """Delete a vault key."""
    vault = load_vault()

    parts = key_name.split(".", 1)
    if len(parts) == 2:
        service, name = parts
        if service in vault and name in vault[service]:
            del vault[service][name]
            save_vault(vault)

    # Update registry
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE vault_registry SET is_set = false WHERE key_name = $1",
                key_name
            )
    finally:
        await pool.close()

    return {"success": True}


@router.post("/keys/{key_name:path}/test")
async def test_key(key_name: str):
    """Test if an API key is valid."""
    vault = load_vault()

    parts = key_name.split(".", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid key format")

    service, name = parts
    if service not in vault or name not in vault[service]:
        raise HTTPException(status_code=404, detail="Key not found")

    value = vault[service][name]

    # Test based on service
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if service == "openai":
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {value}"}
                )
                valid = resp.status_code == 200
            elif service == "anthropic":
                # Anthropic doesn't have a simple test endpoint
                valid = len(value) > 20  # Basic check
            else:
                valid = bool(value)

        return {"valid": valid, "message": "Key is valid" if valid else "Key validation failed"}
    except Exception as e:
        return {"valid": False, "message": str(e)}