#!/usr/bin/env python3
"""
Fact Extraction Worker - Autonomous learning component
Extracts structured facts from unprocessed vectors in Qdrant
"""

import json
import logging
import httpx
import asyncpg
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4
import asyncio
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactExtractionWorker:
    """Extracts facts from vectors and builds knowledge graph"""

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain")
        self.qdrant_url = "http://localhost:6333"
        self.ollama_url = "http://localhost:11434"
        self.collection = "echo_memory"
        self.batch_size = 50

    async def register_with_autonomous_core(self):
        """Register as an autonomous goal"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8309/api/autonomous/goals",
                    json={
                        "name": "continuous_fact_extraction",
                        "description": "Extract facts from unprocessed vectors",
                        "safety_level": "auto",
                        "schedule": "*/30 * * * *",  # Every 30 minutes
                        "enabled": True
                    }
                )
                logger.info(f"Registered with autonomous core: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to register: {e}")

    async def get_unprocessed_vectors(self, conn) -> List[Dict]:
        """Get vectors that haven't been processed for fact extraction"""
        try:
            # Get all vector IDs from Qdrant
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.qdrant_url}/collections/{self.collection}/points/scroll",
                    json={"limit": 1000, "with_payload": True, "with_vector": False}
                )
                vectors = response.json().get("result", {}).get("points", [])

            # Check which ones are processed
            processed_ids = await conn.fetch(
                "SELECT vector_id FROM extraction_coverage WHERE source_collection = $1",
                self.collection
            )
            processed_set = {row['vector_id'] for row in processed_ids}

            # Return unprocessed vectors
            unprocessed = [v for v in vectors if v['id'] not in processed_set]
            logger.info(f"Found {len(unprocessed)} unprocessed vectors out of {len(vectors)}")
            return unprocessed[:self.batch_size]

        except Exception as e:
            logger.error(f"Error getting unprocessed vectors: {e}")
            return []

    async def extract_facts_from_text(self, text: str) -> List[Dict]:
        """Use Ollama to extract facts from text"""
        prompt = f"""Extract all factual statements from this text. Return ONLY a JSON array of objects with keys: "subject", "predicate", "object", "confidence" (0.0-1.0).

Focus on:
- Technical specifications (hardware, software, versions, ports)
- Personal facts (preferences, ownership, locations)
- System configurations (services, databases, integrations)
- Project details (names, status, technologies used)

Skip vague or speculative content. Only extract verifiable facts.

Text: {text[:2000]}

Response (JSON array only):"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "gemma2:9b",
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.3
                    }
                )

                result = response.json()
                response_text = result.get("response", "[]")

                # Try to parse JSON response
                try:
                    facts = json.loads(response_text)
                    if isinstance(facts, list):
                        return facts
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        try:
                            facts = json.loads(json_match.group())
                            return facts
                        except:
                            pass

                return []

        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []

    async def store_facts(self, conn, facts: List[Dict], source_vector_id: str):
        """Store extracted facts in database and as new vectors"""
        stored_count = 0

        for fact in facts:
            if not all(k in fact for k in ['subject', 'predicate', 'object']):
                continue

            try:
                # Store in facts table
                fact_id = await conn.fetchval(
                    """INSERT INTO facts (subject, predicate, object, confidence, source)
                       VALUES ($1, $2, $3, $4, $5)
                       ON CONFLICT DO NOTHING
                       RETURNING id""",
                    fact['subject'],
                    fact['predicate'],
                    fact['object'],
                    fact.get('confidence', 0.8),
                    f"extracted_from_{source_vector_id}"
                )

                if fact_id:
                    # Also store as a vector for better retrieval
                    fact_text = f"{fact['subject']} {fact['predicate']} {fact['object']}"

                    async with httpx.AsyncClient() as client:
                        # Get embedding
                        embed_response = await client.post(
                            f"{self.ollama_url}/api/embeddings",
                            json={"model": "mxbai-embed-large", "prompt": fact_text}
                        )
                        embedding = embed_response.json()["embedding"]

                        # Store in Qdrant
                        await client.put(
                            f"{self.qdrant_url}/collections/{self.collection}/points",
                            json={
                                "points": [{
                                    "id": str(uuid4()),
                                    "vector": embedding,
                                    "payload": {
                                        "text": fact_text,
                                        "source": "extracted_fact",
                                        "fact_id": fact_id,
                                        "subject": fact['subject'],
                                        "predicate": fact['predicate'],
                                        "object": fact['object'],
                                        "confidence": fact.get('confidence', 0.8),
                                        "extracted_at": datetime.now().isoformat()
                                    }
                                }]
                            }
                        )
                    stored_count += 1

            except Exception as e:
                logger.error(f"Error storing fact: {e}")
                continue

        return stored_count

    async def process_batch(self, conn, vectors: List[Dict]):
        """Process a batch of vectors for fact extraction"""
        for vector in vectors:
            vector_id = vector['id']
            payload = vector.get('payload', {})
            text = payload.get('text', '')

            if not text:
                continue

            try:
                logger.info(f"Processing vector {vector_id}")

                # Extract facts
                facts = await self.extract_facts_from_text(text)

                # Store facts
                stored_count = await self.store_facts(conn, facts, vector_id)

                # Mark as processed
                await conn.execute(
                    """INSERT INTO extraction_coverage
                       (source_collection, vector_id, facts_found)
                       VALUES ($1, $2, $3)
                       ON CONFLICT (source_collection, vector_id)
                       DO UPDATE SET processed_at = NOW(), facts_found = $3""",
                    self.collection, vector_id, stored_count
                )

                logger.info(f"Extracted {stored_count} facts from vector {vector_id}")

            except Exception as e:
                logger.error(f"Error processing vector {vector_id}: {e}")
                # Mark as processed with error
                await conn.execute(
                    """INSERT INTO extraction_coverage
                       (source_collection, vector_id, error)
                       VALUES ($1, $2, $3)
                       ON CONFLICT (source_collection, vector_id)
                       DO UPDATE SET processed_at = NOW(), error = $3""",
                    self.collection, vector_id, str(e)
                )

    async def run_cycle(self):
        """Run one extraction cycle"""
        try:
            conn = await asyncpg.connect(self.db_url)

            # Get unprocessed vectors
            unprocessed = await self.get_unprocessed_vectors(conn)

            if unprocessed:
                logger.info(f"Starting fact extraction for {len(unprocessed)} vectors")
                await self.process_batch(conn, unprocessed)

                # Get stats
                stats = await conn.fetchrow(
                    """SELECT COUNT(*) as processed,
                              SUM(facts_found) as total_facts
                       FROM extraction_coverage
                       WHERE source_collection = $1""",
                    self.collection
                )
                logger.info(f"Stats: {stats['processed']} vectors processed, {stats['total_facts']} facts extracted")
            else:
                logger.info("No unprocessed vectors found")

            await conn.close()

        except Exception as e:
            logger.error(f"Error in extraction cycle: {e}")

    async def start(self):
        """Start the worker"""
        await self.register_with_autonomous_core()

        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(1800)  # Wait 30 minutes
            except KeyboardInterrupt:
                logger.info("Stopping fact extraction worker")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    worker = FactExtractionWorker()
    asyncio.run(worker.start())