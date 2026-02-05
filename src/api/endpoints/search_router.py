"""
Search Router with proper async and connection pooling
"""
import logging
from typing import Optional
import asyncpg
from asyncpg.pool import Pool

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["search"])

# Connection pool (initialized on startup)
_pool: Optional[Pool] = None

async def get_pool() -> Pool:
    """Get or create the connection pool"""
    global _pool
    if not _pool:
        _pool = await asyncpg.create_pool(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE",
            min_size=2,
            max_size=10
        )
    return _pool

@router.on_event("startup")
async def startup():
    """Initialize connection pool on startup"""
    await get_pool()

@router.on_event("shutdown")
async def shutdown():
    """Close connection pool on shutdown"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

@router.get("/search")
async def search_conversations(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results")
):
    """
    Search conversations with proper async and parameterized queries
    """
    try:
        pool = await get_pool()

        async with pool.acquire() as conn:
            # Parameterized query - safe from SQL injection
            rows = await conn.fetch("""
                SELECT
                    conversation_id,
                    role,
                    content,
                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', $1)
                    ) as rank
                FROM claude_conversations
                WHERE
                    to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                    OR content ILIKE '%' || $2 || '%'
                ORDER BY rank DESC NULLS LAST
                LIMIT $3
            """, q, q, limit)

            results = [
                {
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"][:500] if row["content"] else "",
                    "rank": float(row["rank"]) if row["rank"] else 0.0
                }
                for row in rows
            ]

            return {
                "query": q,
                "count": len(results),
                "results": results
            }

    except asyncpg.PostgresError as e:
        logger.error(f"Database error in search: {e}")
        raise HTTPException(status_code=500, detail="Database search failed")
    except Exception as e:
        logger.error(f"Unexpected error in search: {e}")
        raise HTTPException(status_code=500, detail="Search operation failed")