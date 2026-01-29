#!/usr/bin/env python3
"""
Backfill Ingestion Tracking

This script populates the ingestion_tracking table from existing Qdrant vectors.
Run this once to bootstrap the system, then the normal ingestion pipeline takes over.

Usage:
    python backfill_tracking.py [--postgres-dsn DSN] [--qdrant-host HOST] [--qdrant-port PORT]
"""

import argparse
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import ScrollRequest


async def backfill_from_qdrant(
    postgres_dsn: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "echo_brain",
    batch_size: int = 100
):
    """
    Backfill ingestion_tracking from existing Qdrant vectors.
    
    This creates tracking records for all vectors that exist in Qdrant
    but aren't yet tracked in PostgreSQL.
    """
    print("=" * 60)
    print("BACKFILL INGESTION TRACKING FROM QDRANT")
    print("=" * 60)
    
    # Connect to PostgreSQL
    pool = await asyncpg.create_pool(postgres_dsn, min_size=2, max_size=10)
    
    # Connect to Qdrant (sync client is fine for scrolling)
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    try:
        # Get collection info
        collection_info = qdrant.get_collection(collection_name)
        total_vectors = collection_info.points_count
        print(f"Collection: {collection_name}")
        print(f"Total vectors: {total_vectors}")
        
        # Get existing tracked vector IDs
        async with pool.acquire() as conn:
            existing = await conn.fetch(
                "SELECT vector_id FROM ingestion_tracking WHERE vector_id IS NOT NULL"
            )
            existing_ids = {str(row["vector_id"]) for row in existing}
        
        print(f"Already tracked: {len(existing_ids)}")
        print(f"Expected new: ~{total_vectors - len(existing_ids)}")
        print()
        
        # Scroll through all vectors
        offset = None
        processed = 0
        inserted = 0
        skipped = 0
        errors = 0
        
        while True:
            # Scroll batch
            scroll_result = qdrant.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False  # We don't need vectors, just metadata
            )
            
            points, next_offset = scroll_result
            
            if not points:
                break
            
            # Process batch
            records_to_insert = []
            content_to_insert = []
            
            for point in points:
                processed += 1
                point_id = str(point.id)
                
                # Skip if already tracked
                if point_id in existing_ids:
                    skipped += 1
                    continue
                
                payload = point.payload or {}
                
                # Extract metadata
                source_path = payload.get("source_path", f"unknown/{point_id}")
                source_type = payload.get("source_type", "document")
                content = payload.get("content", "")
                domain = payload.get("domain")
                created_at = payload.get("created_at")
                
                if created_at:
                    try:
                        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except:
                        created_at = datetime.now(timezone.utc)
                else:
                    created_at = datetime.now(timezone.utc)
                
                # Generate hash from content
                content_hash = hashlib.sha256(
                    (source_path + content[:1000]).encode()
                ).hexdigest()
                
                # Create tracking record
                tracking_id = UUID(point_id) if is_valid_uuid(point_id) else None
                
                records_to_insert.append({
                    "source_type": source_type,
                    "source_path": source_path,
                    "source_hash": content_hash,
                    "vector_id": point_id,
                    "vectorized_at": created_at,
                    "domain": domain,
                    "token_count": len(content.split()) if content else 0,
                    "created_at": created_at
                })
                
                # Store content for fact extraction
                if content:
                    content_to_insert.append({
                        "source_hash": content_hash,
                        "content": content,
                        "content_hash": hashlib.sha256(content.encode()).hexdigest()
                    })
            
            # Batch insert
            if records_to_insert:
                async with pool.acquire() as conn:
                    for record in records_to_insert:
                        try:
                            # Insert tracking record
                            tracking_id = await conn.fetchval("""
                                INSERT INTO ingestion_tracking (
                                    source_type, source_path, source_hash,
                                    vector_id, vectorized_at, domain,
                                    token_count, created_at
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                                ON CONFLICT (source_hash) DO UPDATE SET
                                    vector_id = EXCLUDED.vector_id,
                                    vectorized_at = EXCLUDED.vectorized_at,
                                    updated_at = NOW()
                                RETURNING id
                            """,
                                record["source_type"],
                                record["source_path"],
                                record["source_hash"],
                                record["vector_id"],
                                record["vectorized_at"],
                                record["domain"],
                                record["token_count"],
                                record["created_at"]
                            )
                            
                            # Insert content if we have it
                            for content_record in content_to_insert:
                                if content_record["source_hash"] == record["source_hash"]:
                                    await conn.execute("""
                                        INSERT INTO vector_content (
                                            tracking_id, content, content_hash
                                        ) VALUES ($1, $2, $3)
                                        ON CONFLICT (tracking_id, chunk_index) DO NOTHING
                                    """,
                                        tracking_id,
                                        content_record["content"],
                                        content_record["content_hash"]
                                    )
                                    break
                            
                            inserted += 1
                            
                        except Exception as e:
                            print(f"Error inserting {record['source_path']}: {e}")
                            errors += 1
            
            # Progress update
            pct = (processed / total_vectors) * 100 if total_vectors > 0 else 0
            print(f"[{processed}/{total_vectors}] ({pct:.1f}%) - Inserted: {inserted}, Skipped: {skipped}, Errors: {errors}")
            
            # Move to next batch
            offset = next_offset
            if offset is None:
                break
        
        # Final summary
        print()
        print("=" * 60)
        print("BACKFILL COMPLETE")
        print("=" * 60)
        print(f"Total processed: {processed}")
        print(f"New records inserted: {inserted}")
        print(f"Already tracked (skipped): {skipped}")
        print(f"Errors: {errors}")
        
        # Get final stats
        async with pool.acquire() as conn:
            total_tracked = await conn.fetchval(
                "SELECT COUNT(*) FROM ingestion_tracking"
            )
            pending_extraction = await conn.fetchval(
                "SELECT COUNT(*) FROM ingestion_tracking WHERE fact_extracted = FALSE"
            )
        
        print()
        print(f"Total tracked records: {total_tracked}")
        print(f"Pending fact extraction: {pending_extraction}")
        
        if pending_extraction > 0:
            print()
            print("Next step: Run fact extraction with:")
            print("  python -m src.ingestion.fact_extractor")
        
    finally:
        await pool.close()


def is_valid_uuid(val: str) -> bool:
    """Check if string is valid UUID."""
    try:
        UUID(val)
        return True
    except (ValueError, TypeError):
        return False


async def verify_schema(postgres_dsn: str) -> bool:
    """Verify required tables exist."""
    pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=2)
    
    try:
        async with pool.acquire() as conn:
            # Check for required tables
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name IN ('ingestion_tracking', 'vector_content', 'facts')
            """)
            
            table_names = {row["table_name"] for row in tables}
            required = {"ingestion_tracking", "vector_content", "facts"}
            
            missing = required - table_names
            
            if missing:
                print(f"ERROR: Missing tables: {missing}")
                print("Run the migration first:")
                print("  psql -d your_database -f migrations/001_create_tables.sql")
                return False
            
            return True
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill ingestion tracking from Qdrant")
    parser.add_argument(
        "--postgres-dsn",
        default="postgresql://localhost/echo_brain",
        help="PostgreSQL connection string"
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant host"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port"
    )
    parser.add_argument(
        "--collection",
        default="echo_brain",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for scrolling"
    )
    
    args = parser.parse_args()
    
    async def run():
        # Verify schema first
        if not await verify_schema(args.postgres_dsn):
            return
        
        # Run backfill
        await backfill_from_qdrant(
            postgres_dsn=args.postgres_dsn,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection_name=args.collection,
            batch_size=args.batch_size
        )
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
