"""
Knowledge API - CRUD for facts Echo Brain knows about the user.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime
import asyncpg
import httpx
import sys

sys.path.insert(0, '/opt/tower-echo-brain')
from src.services.embedding_service import create_embedding_service

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

DATABASE_URL = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/tower_consolidated"
QDRANT_URL = "http://localhost:6333"

# Models
class FactCreate(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9

class FactUpdate(BaseModel):
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None

class Fact(BaseModel):
    id: UUID
    subject: str
    predicate: str
    object: str
    confidence: float
    created_at: datetime
    source: Optional[str] = None

# Endpoints
@router.get("/facts")
async def list_facts(
    subject: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    offset: int = 0
):
    """List all facts, optionally filtered by subject or search."""
    pool = await asyncpg.create_pool(DATABASE_URL)

    try:
        async with pool.acquire() as conn:
            if subject:
                rows = await conn.fetch("""
                    SELECT id, subject, predicate, object, confidence, created_at,
                           source_document_id as source
                    FROM facts
                    WHERE subject ILIKE $1
                    ORDER BY subject, predicate
                    LIMIT $2 OFFSET $3
                """, f"%{subject}%", limit, offset)
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM facts WHERE subject ILIKE $1",
                    f"%{subject}%"
                )
            elif search:
                # Use Qdrant for semantic search
                embedding_service = await create_embedding_service()
                query_vector = await embedding_service.embed_single(search)
                await embedding_service.close()

                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{QDRANT_URL}/collections/facts/points/search",
                        json={"vector": query_vector, "limit": limit, "with_payload": True}
                    )
                    results = resp.json().get("result", [])

                # Get full facts from PostgreSQL
                fact_ids = [r["payload"].get("source_point_id") or str(r["id"]) for r in results]
                if fact_ids:
                    rows = await conn.fetch("""
                        SELECT id, subject, predicate, object, confidence, created_at,
                               source_document_id as source
                        FROM facts WHERE id::text = ANY($1)
                    """, fact_ids)
                else:
                    rows = []
                total = len(rows)
            else:
                rows = await conn.fetch("""
                    SELECT id, subject, predicate, object, confidence, created_at,
                           source_document_id as source
                    FROM facts
                    ORDER BY subject, predicate
                    LIMIT $1 OFFSET $2
                """, limit, offset)
                total = await conn.fetchval("SELECT COUNT(*) FROM facts")

        return {
            "facts": [dict(r) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset
        }
    finally:
        await pool.close()


@router.get("/facts/{fact_id}")
async def get_fact(fact_id: UUID):
    """Get a single fact by ID."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, subject, predicate, object, confidence, created_at,
                       source_document_id as source
                FROM facts WHERE id = $1
            """, fact_id)
            if not row:
                raise HTTPException(status_code=404, detail="Fact not found")
            return {"fact": dict(row)}
    finally:
        await pool.close()


@router.post("/facts")
async def create_fact(fact: FactCreate):
    """Create a new fact."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    embedding_service = await create_embedding_service()

    try:
        # Check duplicate
        async with pool.acquire() as conn:
            exists = await conn.fetchval("""
                SELECT 1 FROM facts
                WHERE subject = $1 AND predicate = $2 AND object = $3
            """, fact.subject, fact.predicate, fact.object)
            if exists:
                raise HTTPException(status_code=409, detail="Fact already exists")

        # Create embedding
        fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
        embedding = await embedding_service.embed_single(fact_text)

        # Store in PostgreSQL
        fact_id = str(uuid4())
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO facts (id, subject, predicate, object, confidence, qdrant_point_id)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, fact_id, fact.subject, fact.predicate, fact.object, fact.confidence, fact_id)

        # Store in Qdrant
        async with httpx.AsyncClient() as client:
            await client.put(
                f"{QDRANT_URL}/collections/facts/points",
                json={"points": [{
                    "id": fact_id,
                    "vector": embedding,
                    "payload": {
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "fact_text": fact_text,
                        "source_collection": "manual",
                        "confidence": fact.confidence
                    }
                }]}
            )

        return {"id": fact_id, "fact": fact.dict()}
    finally:
        await pool.close()
        await embedding_service.close()


@router.put("/facts/{fact_id}")
async def update_fact(fact_id: UUID, update: FactUpdate):
    """Update an existing fact."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    embedding_service = await create_embedding_service()

    try:
        async with pool.acquire() as conn:
            # Get existing
            row = await conn.fetchrow("SELECT * FROM facts WHERE id = $1", fact_id)
            if not row:
                raise HTTPException(status_code=404, detail="Fact not found")

            # Merge updates
            subject = update.subject or row['subject']
            predicate = update.predicate or row['predicate']
            obj = update.object or row['object']

            # Update PostgreSQL
            await conn.execute("""
                UPDATE facts SET subject=$1, predicate=$2, object=$3, updated_at=NOW()
                WHERE id = $4
            """, subject, predicate, obj, fact_id)

        # Update Qdrant
        fact_text = f"{subject} {predicate} {obj}"
        embedding = await embedding_service.embed_single(fact_text)

        async with httpx.AsyncClient() as client:
            await client.put(
                f"{QDRANT_URL}/collections/facts/points",
                json={"points": [{
                    "id": str(fact_id),
                    "vector": embedding,
                    "payload": {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "fact_text": fact_text,
                        "source_collection": "manual"
                    }
                }]}
            )

        return {"fact": {"id": str(fact_id), "subject": subject, "predicate": predicate, "object": obj}}
    finally:
        await pool.close()
        await embedding_service.close()


@router.delete("/facts/{fact_id}")
async def delete_fact(fact_id: UUID):
    """Delete a fact."""
    pool = await asyncpg.create_pool(DATABASE_URL)

    try:
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM facts WHERE id = $1", fact_id)
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Fact not found")

        # Delete from Qdrant
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{QDRANT_URL}/collections/facts/points/delete",
                json={"points": [str(fact_id)]}
            )

        return {"success": True}
    finally:
        await pool.close()


@router.get("/subjects")
async def list_subjects():
    """Get all unique subjects for autocomplete."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT subject, COUNT(*) as count
                FROM facts
                GROUP BY subject
                ORDER BY count DESC
            """)
            return {"subjects": [{"name": r["subject"], "count": r["count"]} for r in rows]}
    finally:
        await pool.close()


@router.get("/about/{subject}")
async def facts_about(subject: str):
    """Get all facts about a specific subject."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, subject, predicate, object, confidence, created_at
                FROM facts
                WHERE subject ILIKE $1
                ORDER BY predicate
            """, f"%{subject}%")
            return {
                "subject": subject,
                "facts": [dict(r) for r in rows],
                "count": len(rows)
            }
    finally:
        await pool.close()