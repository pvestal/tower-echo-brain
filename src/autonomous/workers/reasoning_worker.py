"""
Reasoning Worker - Echo Brain Phase 2c
Processes newly ingested content through CLASSIFY → EXTRACT → CONNECT → REASON → ACT.

This is Echo Brain's "System 2" — slow, deliberate thinking about new information.
It watches for new domain_ingestion_log entries that haven't been reasoned about yet,
extracts structured facts, finds connections to existing knowledge, detects conflicts,
and generates notifications when action is needed.
"""
import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import asyncpg
import httpx


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "echo_memory")
DB_URL = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")

# Which LLM to use for reasoning (must be available in Ollama)
REASONING_MODEL = os.getenv("REASONING_MODEL", "mistral:7b")

# How many new items to process per cycle (avoid overloading Ollama)
BATCH_SIZE = 20


class ReasoningWorker:
    """
    Watches for newly ingested content and processes it through the reasoning pipeline.

    Pipeline stages:
    1. CLASSIFY: Determine content type and importance
    2. EXTRACT: Pull structured facts from content
    3. CONNECT: Search existing knowledge for relationships
    4. REASON: Synthesize implications, detect conflicts
    5. ACT: Store facts, create connections, generate notifications
    """

    def __init__(self):
        self.stats = {
            "items_processed": 0,
            "facts_extracted": 0,
            "connections_found": 0,
            "conflicts_detected": 0,
            "notifications_created": 0,
            "errors": 0,
        }

    async def run_cycle(self):
        """Main worker cycle — process unprocessed ingestion items."""
        start_time = time.time()
        print(f"[ReasoningWorker] Starting cycle at {datetime.now().isoformat()}")
        self.stats = {k: 0 for k in self.stats}

        conn = await asyncpg.connect(DB_URL)
        try:
            # Find content that was ingested but not yet reasoned about
            new_items = await conn.fetch("""
                SELECT id, source_path, category, chunk_count, ingested_at
                FROM domain_ingestion_log
                WHERE reasoned_at IS NULL
                  AND status = 'completed'
                ORDER BY ingested_at ASC
                LIMIT $1
            """, BATCH_SIZE)

            if not new_items:
                print("[ReasoningWorker] No new items to process")
                return

            print(f"[ReasoningWorker] Processing {len(new_items)} new items")

            for item in new_items:
                try:
                    await self._process_item(conn, item)
                    # Mark as reasoned
                    await conn.execute(
                        "UPDATE domain_ingestion_log SET reasoned_at = NOW() WHERE id = $1",
                        item["id"]
                    )
                    self.stats["items_processed"] += 1
                except Exception as e:
                    print(f"[ReasoningWorker] Error processing {item['source_path']}: {e}")
                    self.stats["errors"] += 1

            # Log the reasoning cycle
            duration_ms = int((time.time() - start_time) * 1000)
            await conn.execute("""
                INSERT INTO reasoning_log
                    (trigger_source, trigger_category, input_summary,
                     facts_extracted, connections_found, conflicts_detected,
                     actions_proposed, duration_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                "scheduled_cycle",
                "mixed",
                f"Processed {self.stats['items_processed']} items from domain_ingestion_log",
                self.stats["facts_extracted"],
                self.stats["connections_found"],
                self.stats["conflicts_detected"],
                json.dumps([]),
                duration_ms,
            )

            print(f"[ReasoningWorker] Cycle complete in {duration_ms}ms: {json.dumps(self.stats)}")

        except Exception as e:
            print(f"[ReasoningWorker] ERROR in cycle: {e}")
            self.stats["errors"] += 1
        finally:
            await conn.close()

    async def _process_item(self, conn, item):
        """Process a single ingested item through the full reasoning pipeline."""
        source_path = item["source_path"]
        category = item["category"]

        # 1. Retrieve the content from Qdrant
        content = await self._get_content(source_path)
        if not content:
            return

        # 2. CLASSIFY + EXTRACT: Ask LLM to extract structured facts
        facts = await self._extract_facts(content, category, source_path)

        # 3. CONNECT: For each fact, search for related existing facts
        connections = []
        for fact in facts:
            related = await self._find_connections(conn, fact)
            connections.extend(related)

        # 4. REASON: If we found connections, ask LLM about conflicts/implications
        conflicts = []
        if connections:
            conflicts = await self._detect_conflicts(facts, connections)

        # 5. ACT: Store facts, connections, and create notifications
        await self._store_results(conn, facts, connections, conflicts, source_path, category)

    # ============================================================
    # Stage 1+2: Retrieve and Extract
    # ============================================================

    async def _get_content(self, source_path: str) -> Optional[str]:
        """Retrieve ingested content from Qdrant by source path."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                    json={
                        "filter": {
                            "must": [{"key": "source", "match": {"value": source_path}}]
                        },
                        "limit": 5,
                        "with_payload": True,
                    }
                )
                points = resp.json().get("result", {}).get("points", [])
                if not points:
                    return None
                # Combine all chunks for this source
                texts = [p["payload"].get("text", "") for p in points]
                return "\n---\n".join(texts)
        except Exception as e:
            print(f"[ReasoningWorker] Error retrieving {source_path}: {e}")
            return None

    async def _extract_facts(self, content: str, category: str, source_path: str) -> List[dict]:
        """Use LLM to extract structured facts from content."""
        # Truncate content to fit context window
        content_truncated = content[:4000]

        prompt = f"""You are a knowledge extraction system. Extract concrete, specific facts from this content.

Content category: {category}
Source: {source_path}

Content:
{content_truncated}

Extract facts as a JSON array. Each fact should have:
- "text": The specific fact (one clear sentence)
- "type": One of: technical, config, preference, decision, relationship, workflow, model_info
- "confidence": "high" if explicitly stated, "medium" if implied, "low" if uncertain
- "entities": Array of important names/identifiers mentioned (model names, file paths, port numbers, people, etc.)

Rules:
- Only extract facts that would be useful for answering future questions
- Be specific: "Port 8328 runs the anime API" not "there is an API"
- Include configuration values: model names, paths, settings, parameters
- Include relationships: what connects to what, what depends on what
- Skip boilerplate, imports, and obvious code structure facts
- Maximum 10 facts per content block

Output ONLY the JSON array, no other text."""

        response = await self._llm_query(prompt)
        return self._parse_json_array(response)

    # ============================================================
    # Stage 3: Connect
    # ============================================================

    async def _find_connections(self, conn, fact: dict) -> List[dict]:
        """Search existing knowledge_facts for related information."""
        connections = []
        fact_text = fact.get("text", "")
        if not fact_text:
            return connections

        # Search existing facts by entity overlap
        entities = fact.get("entities", [])
        if entities:
            for entity in entities[:3]:  # Limit entity searches
                existing = await conn.fetch("""
                    SELECT id, fact_text, fact_type, entities, category
                    FROM knowledge_facts
                    WHERE valid_until IS NULL
                      AND (fact_text ILIKE $1 OR entities::text ILIKE $1)
                    LIMIT 5
                """, f"%{entity}%")

                for row in existing:
                    connections.append({
                        "new_fact": fact,
                        "existing_fact_id": str(row["id"]),
                        "existing_fact_text": row["fact_text"],
                        "existing_category": row["category"],
                        "matched_entity": entity,
                    })

        self.stats["connections_found"] += len(connections)
        return connections

    # ============================================================
    # Stage 4: Reason about conflicts
    # ============================================================

    async def _detect_conflicts(self, new_facts: list, connections: list) -> List[dict]:
        """Ask LLM to identify conflicts between new and existing facts."""
        if not connections:
            return []

        # Build a summary of new vs existing
        comparison = []
        for conn_item in connections[:10]:
            comparison.append({
                "new": conn_item["new_fact"].get("text", ""),
                "existing": conn_item["existing_fact_text"],
                "entity": conn_item["matched_entity"],
            })

        prompt = f"""You are a conflict detection system. Compare these NEW facts against EXISTING facts.

{json.dumps(comparison, indent=2)}

For each pair, determine:
1. Do they AGREE (same information, no conflict)?
2. Do they CONFLICT (contradictory information)?
3. Does the new fact UPDATE/SUPERSEDE the old one?
4. Are they simply RELATED but different topics?

Output a JSON array of conflicts/updates ONLY. Skip pairs that simply agree or are unrelated.
Each item should have:
- "new_fact": the new fact text
- "existing_fact": the existing fact text
- "relationship": "conflicts" or "supersedes" or "updates"
- "explanation": Brief reason why

If no conflicts or updates, output an empty array: []

Output ONLY the JSON array."""

        response = await self._llm_query(prompt)
        conflicts = self._parse_json_array(response)
        self.stats["conflicts_detected"] += len(conflicts)
        return conflicts

    # ============================================================
    # Stage 5: Store results and act
    # ============================================================

    async def _store_results(self, conn, facts: list, connections: list,
                             conflicts: list, source_path: str, category: str):
        """Store extracted facts, connections, and create notifications."""

        # Store each fact
        fact_ids = {}
        for fact in facts:
            try:
                fact_id = await conn.fetchval("""
                    INSERT INTO knowledge_facts
                        (fact_text, fact_type, category, confidence, entities,
                         source_path, reasoning_context)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """,
                    fact.get("text", ""),
                    fact.get("type", "technical"),
                    category,
                    fact.get("confidence", "medium"),
                    json.dumps(fact.get("entities", [])),
                    source_path,
                    f"Extracted by reasoning worker from {source_path}",
                )
                fact_ids[fact.get("text", "")] = fact_id
                self.stats["facts_extracted"] += 1
            except Exception as e:
                print(f"[ReasoningWorker] Error storing fact: {e}")

        # Store connections
        for conn_item in connections:
            new_fact_text = conn_item["new_fact"].get("text", "")
            new_fact_id = fact_ids.get(new_fact_text)
            existing_id = conn_item.get("existing_fact_id")
            if new_fact_id and existing_id:
                try:
                    from uuid import UUID
                    await conn.execute("""
                        INSERT INTO knowledge_connections
                            (fact_a, fact_b, relationship, reasoning)
                        VALUES ($1, $2, 'related_to', $3)
                        ON CONFLICT DO NOTHING
                    """, new_fact_id, UUID(existing_id),
                        f"Matched on entity: {conn_item.get('matched_entity', '?')}")
                except Exception:
                    pass

        # Handle conflicts — create notifications for important ones
        for conflict in conflicts:
            rel = conflict.get("relationship", "")
            if rel in ("conflicts", "supersedes"):
                try:
                    await conn.execute("""
                        INSERT INTO notifications
                            (title, body, priority, source, category)
                        VALUES ($1, $2, $3, $4, $5)
                    """,
                        f"Knowledge {rel}: {conflict.get('explanation', '')[:100]}",
                        f"New: {conflict.get('new_fact', '')}\n\nExisting: {conflict.get('existing_fact', '')}\n\nExplanation: {conflict.get('explanation', '')}",
                        "high" if rel == "conflicts" else "normal",
                        source_path,
                        category,
                    )
                    self.stats["notifications_created"] += 1
                except Exception as e:
                    print(f"[ReasoningWorker] Error creating notification: {e}")

    # ============================================================
    # LLM and utility methods
    # ============================================================

    async def _llm_query(self, prompt: str) -> str:
        """Query Ollama for reasoning."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": REASONING_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 2000},
                    }
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "")
                print(f"[ReasoningWorker] LLM error: {resp.status_code}")
                return ""
        except Exception as e:
            print(f"[ReasoningWorker] LLM query failed: {e}")
            return ""

    def _parse_json_array(self, text: str) -> list:
        """Safely parse a JSON array from LLM output."""
        if not text:
            return []
        # Try to find JSON array in the response
        text = text.strip()
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Find the array
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1:
            return []
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []