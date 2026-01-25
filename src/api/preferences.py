"""
Preferences API - User preferences CRUD.
Categories: music, anime, communication, appearance, etc.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, List
from uuid import UUID, uuid4
from datetime import datetime
import asyncpg
import json

router = APIRouter(tags=["preferences"])

DATABASE_URL = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"

class PreferenceCreate(BaseModel):
    category: str
    key: str
    value: Any  # Can be string, list, dict
    metadata: dict = {}

class PreferenceUpdate(BaseModel):
    value: Any
    metadata: Optional[dict] = None

@router.get("")
async def list_preferences(category: Optional[str] = None):
    """List all preferences, optionally by category."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            if category:
                rows = await conn.fetch("""
                    SELECT id, category, preference_key as key, preference_value as value,
                           confidence, source, last_updated
                    FROM user_preferences WHERE category = $1 ORDER BY preference_key
                """, category)
            else:
                rows = await conn.fetch("""
                    SELECT id, category, preference_key as key, preference_value as value,
                           confidence, source, last_updated
                    FROM user_preferences ORDER BY category, preference_key
                """)

            # Get unique categories
            categories = await conn.fetch(
                "SELECT DISTINCT category FROM user_preferences ORDER BY category"
            )

        return {
            "preferences": [dict(r) for r in rows],
            "categories": [r["category"] for r in categories]
        }
    finally:
        await pool.close()


@router.get("/{category}")
async def get_category(category: str):
    """Get all preferences in a category."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, preference_key as key, preference_value as value,
                       confidence, source, last_updated
                FROM user_preferences WHERE category = $1 ORDER BY preference_key
            """, category)
        return {"category": category, "items": [dict(r) for r in rows]}
    finally:
        await pool.close()


@router.post("")
async def create_preference(pref: PreferenceCreate):
    """Create a new preference."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            # Store as text (existing schema)
            value_str = json.dumps(pref.value) if not isinstance(pref.value, str) else pref.value
            await conn.execute("""
                INSERT INTO user_preferences (category, preference_key, preference_value, source, confidence)
                VALUES ($1, $2, $3, 'api', 1.0)
                ON CONFLICT (category, preference_key) DO UPDATE SET
                    preference_value = $3, last_updated = NOW()
            """, pref.category, pref.key, value_str)

        return {"success": True, "preference": pref.dict()}
    finally:
        await pool.close()


@router.put("/{pref_id}")
async def update_preference(pref_id: int, update: PreferenceUpdate):
    """Update a preference."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE id = $1", pref_id
            )
            if not row:
                raise HTTPException(status_code=404, detail="Preference not found")

            value_str = json.dumps(update.value) if not isinstance(update.value, str) else update.value
            await conn.execute("""
                UPDATE user_preferences SET preference_value = $1, last_updated = NOW()
                WHERE id = $2
            """, value_str, pref_id)

        return {"success": True}
    finally:
        await pool.close()


@router.delete("/{pref_id}")
async def delete_preference(pref_id: int):
    """Delete a preference."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM user_preferences WHERE id = $1", pref_id
            )
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Preference not found")
        return {"success": True}
    finally:
        await pool.close()