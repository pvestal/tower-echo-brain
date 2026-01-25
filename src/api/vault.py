"""
Vault API - Secure secrets management via HashiCorp Vault
Keys are NEVER stored in files or exposed in responses
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import hvac
import asyncpg
import httpx
from datetime import datetime

router = APIRouter(tags=["vault"])

DATABASE_URL = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"

def get_vault_client() -> hvac.Client:
    """Get authenticated Vault client"""
    vault_addr = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")

    # Check environment variable first
    token = os.getenv("VAULT_TOKEN")

    # Fall back to token file
    if not token:
        token_path = os.path.expanduser("~/.vault-token")
        if os.path.exists(token_path):
            with open(token_path) as f:
                token = f.read().strip()
        else:
            raise HTTPException(500, "Vault token not found in environment or file")

    client = hvac.Client(url=vault_addr, token=token)
    if not client.is_authenticated():
        raise HTTPException(500, "Vault authentication failed")

    return client

class KeyInfo(BaseModel):
    name: str
    has_value: bool
    fields: List[str] = []
    service: str = ""
    key_type: str = "api_key"

class KeyCreate(BaseModel):
    key_name: str
    value: Optional[str] = None
    fields: Optional[Dict[str, str]] = None
    service: str
    key_type: str = "api_key"
    description: Optional[str] = None

@router.get("/keys")
async def list_keys() -> Dict:
    """List all API keys (values never exposed)"""
    client = get_vault_client()

    keys = []
    try:
        # List all secrets under api_keys
        secret_list = client.secrets.kv.v2.list_secrets(
            path="api_keys",
            mount_point="secret"
        )

        for key_name in secret_list["data"]["keys"]:
            # Get metadata only (not the actual secret values)
            try:
                secret = client.secrets.kv.v2.read_secret_version(
                    path=f"api_keys/{key_name.rstrip('/')}",
                    mount_point="secret"
                )
                fields = list(secret["data"]["data"].keys())

                # Determine key type
                key_type = "api_key" if "key" in key_name.lower() or "api" in key_name.lower() else "secret"

                keys.append({
                    "key_name": key_name.rstrip('/'),
                    "service": key_name.rstrip('/'),  # Service name is the key name in our structure
                    "has_value": True,
                    "fields": fields,
                    "key_type": key_type,
                    "is_set": True,
                    "value_preview": "***VAULT***"  # Never expose actual values
                })
            except Exception as e:
                keys.append({
                    "key_name": key_name.rstrip('/'),
                    "service": key_name.rstrip('/'),
                    "has_value": False,
                    "fields": [],
                    "key_type": "unknown",
                    "is_set": False,
                    "value_preview": None
                })

    except Exception as e:
        if "no secrets" not in str(e).lower():
            raise HTTPException(500, f"Vault error: {e}")

    return {"keys": keys}

@router.post("/keys")
async def create_key(key: KeyCreate):
    """Store a new API key in Vault"""
    client = get_vault_client()

    # Parse the key_name to determine path
    parts = key.key_name.split(".", 1)
    if len(parts) == 2:
        service, field_name = parts
        path = f"api_keys/{service}"

        # Read existing secret if it exists
        try:
            existing = client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point="secret"
            )
            data = existing["data"]["data"]
        except:
            data = {}

        # Add the new field
        data[field_name] = key.value
    else:
        # Simple key name, store under service
        path = f"api_keys/{key.service}"

        if key.fields:
            data = key.fields
        elif key.value:
            data = {"key": key.value, "api_key": key.value}  # Store in both formats for compatibility
        else:
            raise HTTPException(400, "Must provide value or fields")

    try:
        client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=data,
            mount_point="secret"
        )

        # Update database registry
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

        return {"success": True, "key_name": key.key_name, "status": "created"}
    except Exception as e:
        raise HTTPException(500, f"Failed to store key: {e}")

@router.delete("/keys/{key_name:path}")
async def delete_key(key_name: str):
    """Delete an API key from Vault"""
    client = get_vault_client()

    parts = key_name.split(".", 1)
    if len(parts) == 2:
        service, field_name = parts
        path = f"api_keys/{service}"

        try:
            # Read existing secret
            existing = client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point="secret"
            )
            data = existing["data"]["data"]

            # Remove the field
            if field_name in data:
                del data[field_name]

                # Update the secret
                if data:  # If there are still other fields
                    client.secrets.kv.v2.create_or_update_secret(
                        path=path,
                        secret=data,
                        mount_point="secret"
                    )
                else:  # If no fields left, delete the entire secret
                    client.secrets.kv.v2.delete_metadata_and_all_versions(
                        path=path,
                        mount_point="secret"
                    )
        except:
            pass  # Key doesn't exist
    else:
        # Delete entire service key
        try:
            client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=f"api_keys/{key_name}",
                mount_point="secret"
            )
        except:
            pass

    # Update database registry
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE vault_registry SET is_set = false WHERE key_name = $1",
                key_name
            )
    finally:
        await pool.close()

    return {"success": True, "status": "deleted", "key_name": key_name}

@router.post("/keys/{key_name:path}/test")
async def test_key(key_name: str):
    """Test if an API key is valid"""
    client = get_vault_client()

    # Parse key name
    parts = key_name.split(".", 1)
    if len(parts) == 2:
        service, field_name = parts
    else:
        service = key_name
        field_name = "key"  # Default field name

    try:
        # Read from Vault
        secret = client.secrets.kv.v2.read_secret_version(
            path=f"api_keys/{service}",
            mount_point="secret"
        )
        data = secret["data"]["data"]

        # Get the value
        if field_name in data:
            value = data[field_name]
        elif "api_key" in data:
            value = data["api_key"]
        elif "key" in data:
            value = data["key"]
        else:
            raise HTTPException(404, f"Field {field_name} not found in {service}")

        # Test based on service
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            if service == "openai":
                resp = await http_client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {value}"}
                )
                valid = resp.status_code == 200
            elif service == "anthropic":
                # Basic check for Anthropic key format
                valid = value.startswith("sk-") and len(value) > 40
            else:
                # For other services, just check if value exists
                valid = bool(value)

        return {"valid": valid, "message": "Key is valid" if valid else "Key validation failed"}
    except HTTPException:
        raise
    except Exception as e:
        return {"valid": False, "message": str(e)}

@router.get("/health")
async def vault_health():
    """Check Vault connection status"""
    try:
        client = get_vault_client()
        health = client.sys.read_health_status(method="GET")

        # Count total secrets
        try:
            secret_list = client.secrets.kv.v2.list_secrets(
                path="api_keys",
                mount_point="secret"
            )
            secret_count = len(secret_list["data"]["keys"])
        except:
            secret_count = 0

        return {
            "status": "healthy",
            "sealed": health.get("sealed", False),
            "initialized": health.get("initialized", True),
            "secret_count": secret_count,
            "vault_addr": os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "vault_addr": os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")}