"""Solutions API - Search past solutions"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import DictCursor
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

DB_CONFIG = {
    'host': 'localhost',
    'database': 'tower_consolidated',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE'
}

@router.get("/api/echo/solutions/search")
async def search_solutions(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Max results")
):
    """Search past solutions"""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT problem_description, solution_applied, files_modified,
                           created_at, verified_working
                    FROM past_solutions
                    WHERE problem_description ILIKE %s
                       OR solution_applied ILIKE %s
                    ORDER BY verified_working DESC, created_at DESC
                    LIMIT %s
                """, (f'%{q}%', f'%{q}%', limit))

                results = [dict(row) for row in cur.fetchall()]

                return {
                    "query": q,
                    "count": len(results),
                    "solutions": results
                }
    except Exception as e:
        logger.error(f"Solutions search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/echo/solutions/stats")
async def get_solutions_stats():
    """Get solutions database statistics"""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_solutions,
                        COUNT(*) FILTER (WHERE verified_working = TRUE) as verified_solutions,
                        COUNT(DISTINCT files_modified) as unique_files_touched
                    FROM past_solutions
                """)

                stats = dict(cur.fetchone())
                return stats
    except Exception as e:
        logger.error(f"Solutions stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))