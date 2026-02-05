#!/usr/bin/env python3
"""
Extract facts from vectors in Qdrant
Learn structured knowledge from the ingested content
"""

import asyncio
import httpx
import asyncpg
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
DB_CONFIG = {
    "host": "localhost",
    "database": "echo_brain",
    "user": "patrick",
    "password": os.getenv("DB_PASSWORD", "")
}

async def extract_facts_from_vector(vector_data: dict) -> list:
    """Extract facts from a single vector using LLM"""
    content = vector_data.get("content", "")
    source = vector_data.get("source", "unknown")
    vec_type = vector_data.get("type", "unknown")

    if not content or len(content) < 50:
        return []

    # Craft prompt based on content type
    if vec_type == "code":
        prompt = f"""Extract key facts from this code snippet. Focus on:
- What functions/classes are defined
- What they do
- Key dependencies or imports
- Important configuration values

Code from {source}:
{content[:1500]}

Return facts as JSON array with 'fact' and 'category' keys. Be concise."""

    elif vec_type == "documentation":
        prompt = f"""Extract key facts from this documentation:
{content[:1500]}

Return facts as JSON array with 'fact' and 'category' keys."""

    else:
        prompt = f"""Extract key facts from this content:
{content[:1000]}

Return facts as JSON array with 'fact' and 'category' keys."""

    try:
        # Call Ollama
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": "mistral:7b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.3
                }
            )

            if response.status_code != 200:
                return []

            result = response.json()
            response_text = result.get("message", {}).get("content", "")

            # Parse JSON response
            try:
                facts_data = json.loads(response_text)
                if isinstance(facts_data, list):
                    return facts_data
                elif isinstance(facts_data, dict) and "facts" in facts_data:
                    return facts_data["facts"]
            except json.JSONDecodeError:
                # Try to extract facts manually
                facts = []
                for line in response_text.split('\n'):
                    if '-' in line or '•' in line:
                        fact_text = line.strip('- •').strip()
                        if len(fact_text) > 10:
                            facts.append({
                                "fact": fact_text,
                                "category": vec_type
                            })
                return facts[:5]  # Limit to 5 facts per vector

    except Exception as e:
        logger.error(f"Error extracting facts: {e}")
        return []

    return []

async def process_vectors():
    """Process all vectors and extract facts"""
    conn = await asyncpg.connect(**DB_CONFIG)

    try:
        # Get vectors from Qdrant
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get total count
            resp = await client.get(f"{QDRANT_URL}/collections/echo_memory")
            total = resp.json()["result"]["points_count"]
            logger.info(f"Processing {total} vectors for fact extraction...")

            facts_extracted = 0
            vectors_processed = 0
            offset = None

            while vectors_processed < total:
                # Get batch of vectors
                payload = {"limit": 10, "with_payload": True}
                if offset:
                    payload["offset"] = offset

                resp = await client.post(
                    f"{QDRANT_URL}/collections/echo_memory/points/scroll",
                    json=payload
                )
                batch = resp.json()

                if not batch["result"]["points"]:
                    break

                for point in batch["result"]["points"]:
                    vector_id = point["id"]
                    payload_data = point["payload"]

                    # Skip if already processed
                    existing = await conn.fetchval(
                        "SELECT COUNT(*) FROM facts WHERE source_vector_id = $1",
                        str(vector_id)
                    )

                    if existing > 0:
                        logger.info(f"  Skipping {vector_id} - already processed")
                        vectors_processed += 1
                        continue

                    # Extract facts
                    facts = await extract_facts_from_vector(payload_data)

                    if facts:
                        # Store facts in database (using triple format)
                        for fact_data in facts:
                            fact_text = fact_data.get("fact", "")
                            category = fact_data.get("category", "general")

                            if fact_text:
                                # Parse fact into subject-predicate-object
                                parts = fact_text.split(" ", 2)
                                if len(parts) >= 3:
                                    subject = parts[0]
                                    predicate = parts[1]
                                    obj = " ".join(parts[2:])
                                else:
                                    subject = "Echo Brain"
                                    predicate = "has fact"
                                    obj = fact_text

                                await conn.execute(
                                    """INSERT INTO facts
                                       (subject, predicate, object, fact_text, category, source, confidence, source_vector_id, created_at)
                                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                                       ON CONFLICT (subject, predicate, object) DO NOTHING""",
                                    subject[:200],
                                    predicate[:200],
                                    obj[:500],
                                    fact_text[:500],
                                    category,
                                    payload_data.get("source", "unknown")[:200],
                                    0.8,
                                    str(vector_id),
                                    datetime.now()
                                )
                                facts_extracted += 1

                        logger.info(f"  Extracted {len(facts)} facts from {payload_data.get('type')} vector")

                    vectors_processed += 1

                    # Progress update
                    if vectors_processed % 10 == 0:
                        pct = (vectors_processed / total) * 100
                        logger.info(f"Progress: {vectors_processed}/{total} ({pct:.1f}%), Facts: {facts_extracted}")

                if batch["result"].get("next_page_offset"):
                    offset = batch["result"]["next_page_offset"]
                else:
                    break

            logger.info(f"\n✅ Extraction complete!")
            logger.info(f"  Vectors processed: {vectors_processed}")
            logger.info(f"  Facts extracted: {facts_extracted}")

            # Get total facts in DB
            total_facts = await conn.fetchval("SELECT COUNT(*) FROM facts")
            logger.info(f"  Total facts in database: {total_facts}")

    finally:
        await conn.close()

async def main():
    logger.info("=== FACT EXTRACTION ===")
    logger.info(f"Started at {datetime.now()}")

    # Create facts table if needed
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id SERIAL PRIMARY KEY,
                fact TEXT UNIQUE NOT NULL,
                category VARCHAR(100),
                source VARCHAR(200),
                confidence FLOAT DEFAULT 0.5,
                source_vector_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INT DEFAULT 0
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
            CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_facts_source_vector ON facts(source_vector_id);
        """)
        logger.info("Facts table ready")
    finally:
        await conn.close()

    # Process vectors
    await process_vectors()

    logger.info(f"\nCompleted at {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())