"""Codebase API - Search and analyze codebase"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import DictCursor
import logging
import sys
import os

# Add project root to path for imports
sys.path.insert(0, '/opt/tower-echo-brain')

try:
    from ..indexers.codebase_indexer import CodebaseIndexer
except ImportError:
    # Fallback import path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "codebase_indexer",
        "/opt/tower-echo-brain/src/indexers/codebase_indexer.py"
    )
    codebase_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codebase_module)
    CodebaseIndexer = codebase_module.CodebaseIndexer

logger = logging.getLogger(__name__)
router = APIRouter()

DB_CONFIG = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"))
}

@router.get("/api/echo/codebase/search")
async def search_codebase(
    q: str = Query("", description="Search query"),
    limit: int = Query(10, description="Max results"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type (function, class, etc.)")
):
    """Search codebase entities"""
    try:
        # Return empty result if no query
        if not q.strip():
            return {"results": [], "total": 0, "message": "No search query provided"}

        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                query_parts = []
                params = []

                # Base search conditions
                query_parts.append("""
                    (entity_name ILIKE %s OR content ILIKE %s)
                """)
                search_term = f'%{q}%'
                params.extend([search_term, search_term])

                # Optional entity type filter
                if entity_type:
                    query_parts.append("entity_type = %s")
                    params.append(entity_type)

                # Final query
                where_clause = " AND ".join(query_parts)
                params.append(limit)

                cur.execute(f"""
                    SELECT entity_type, entity_name, file_path, line_number, content
                    FROM codebase_index
                    WHERE {where_clause}
                    ORDER BY entity_name
                    LIMIT %s
                """, params)

                results = [dict(row) for row in cur.fetchall()]

                return {
                    "query": q,
                    "entity_type": entity_type,
                    "count": len(results),
                    "entities": results
                }
    except Exception as e:
        logger.error(f"Codebase search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/echo/codebase/stats")
async def get_codebase_stats():
    """Get codebase statistics"""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_entities,
                        COUNT(DISTINCT entity_type) as entity_types,
                        COUNT(DISTINCT file_path) as unique_files,
                        COUNT(*) FILTER (WHERE entity_type = 'function') as functions,
                        COUNT(*) FILTER (WHERE entity_type = 'class') as classes,
                        COUNT(*) FILTER (WHERE entity_type = 'async_function') as async_functions
                    FROM codebase_index
                """)

                stats = dict(cur.fetchone())

                # Get top files by entity count
                cur.execute("""
                    SELECT file_path, COUNT(*) as entity_count
                    FROM codebase_index
                    GROUP BY file_path
                    ORDER BY entity_count DESC
                    LIMIT 5
                """)

                top_files = [dict(row) for row in cur.fetchall()]
                stats["top_files"] = top_files

                return stats
    except Exception as e:
        logger.error(f"Codebase stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/echo/codebase/reindex")
async def reindex_codebase():
    """Reindex the codebase (admin operation)"""
    try:
        indexer = CodebaseIndexer()

        # Index the current project
        count = indexer.index_directory('/opt/tower-echo-brain/src')

        # Save to database
        saved_count = indexer.save_to_database()

        return {
            "indexed_entities": count,
            "saved_entities": saved_count,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Codebase reindexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/echo/codebase/file/{file_path:path}")
async def get_file_entities(file_path: str):
    """Get all entities in a specific file"""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT entity_type, entity_name, line_number, content
                    FROM codebase_index
                    WHERE file_path LIKE %s
                    ORDER BY line_number
                """, (f'%{file_path}%',))

                results = [dict(row) for row in cur.fetchall()]

                return {
                    "file_path": file_path,
                    "entity_count": len(results),
                    "entities": results
                }
    except Exception as e:
        logger.error(f"File entities search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))