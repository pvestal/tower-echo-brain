"""
MCP Service - Connects to Qdrant database for memory operations
"""
import json
import os
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self):
        self._db_pool = None
        try:
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(url="http://localhost:6333")
            self.collection_name = "echo_memory"

            # Get vector count to verify connection
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            self.vector_count = collection_info.points_count
            logger.info(f"✅ MCP Service initialized with {self.vector_count:,} vectors")

            # Initialize Ollama for embeddings (matching unified memory system)
            self.ollama_url = "http://localhost:11434"
            self.embedding_model = "nomic-embed-text"
            self.embeddings_available = True

        except Exception as e:
            logger.error(f"Failed to initialize MCP service: {e}")
            self.qdrant_client = None
            self.vector_count = 0
            self.embeddings_available = False

    def get_vector_count(self) -> int:
        """Get actual vector count from Qdrant"""
        if not self.qdrant_client:
            return 0
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Failed to get vector count: {e}")
            return 0

    async def _get_db_pool(self):
        """Lazy-init asyncpg pool for PostgreSQL dual-writes."""
        if self._db_pool is None:
            import asyncpg
            self._db_pool = await asyncpg.create_pool(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', '5432')),
                user=os.getenv('DB_USER', 'patrick'),
                password=os.getenv('DB_PASSWORD', ''),
                database=os.getenv('DB_NAME', 'echo_brain'),
                min_size=1,
                max_size=5,
            )
        return self._db_pool

    async def search_memory(self, query: str, limit: int = 10, after: str = "", before: str = "") -> List[Dict[str, Any]]:
        """Memory search using domain-aware ParallelRetriever"""
        try:
            from src.context_assembly.retriever import ParallelRetriever

            # Use the ParallelRetriever which handles domain classification
            # and searches appropriate collections (including story_bible for anime)
            retriever = ParallelRetriever()
            await retriever.initialize()

            # Get domain-classified results (with optional temporal filtering)
            retrieve_kwargs = {"max_results": limit}
            if after:
                retrieve_kwargs["after"] = after
            if before:
                retrieve_kwargs["before"] = before
            retrieval_result = await retriever.retrieve(query, **retrieve_kwargs)

            # Convert to MCP format expected by calling code
            results = []
            for source in retrieval_result.get("sources", []):
                result_entry = {
                    "id": source.get("metadata", {}).get("source_id", ""),
                    "score": source.get("score", 0),
                    "content": source.get("content", ""),
                    "source": source.get("source", "unknown"),
                    "type": source.get("type", "memory"),
                    "payload": source.get("metadata", {}),
                }
                # Include confidence from payload if available
                meta = source.get("metadata", {})
                if "confidence" in meta:
                    result_entry["confidence"] = meta["confidence"]
                results.append(result_entry)

            await retriever.shutdown()

            logger.info(f"Domain-aware search for '{query}' classified as {retrieval_result.get('domain')} "
                       f"returned {len(results)} results from collections {retrieval_result.get('allowed_collections')}")

            # Return enriched dict with metadata when available
            query_type = retrieval_result.get("query_type")
            search_weights = retrieval_result.get("search_weights")
            if query_type or search_weights:
                return {
                    "results": results,
                    "query_type": query_type,
                    "search_weights": search_weights,
                    "domain": retrieval_result.get("domain"),
                }
            return results

        except Exception as e:
            logger.error(f"ParallelRetriever search failed: {e}")

            # Fallback to old direct search
            if not self.qdrant_client:
                return []

            try:
                # Generate embedding using Ollama (matching unified memory system)
                import httpx
                try:
                    with httpx.Client(timeout=120) as client:
                        response = client.post(
                            f"{self.ollama_url}/api/embed",
                            json={"model": self.embedding_model, "input": query}
                        )
                        if response.status_code == 200:
                            resp_data = response.json()
                            query_embedding = (resp_data.get("embeddings") or [[]])[0] or resp_data.get("embedding", [])
                        else:
                            # Fallback: create random embedding for testing
                            import random
                            query_embedding = [random.random() for _ in range(768)]
                            logger.warning(f"Ollama embedding failed with status {response.status_code}, using random")
                except Exception as e:
                    # Fallback: create random embedding for testing
                    import random
                    query_embedding = [random.random() for _ in range(768)]
                    logger.warning(f"Ollama embedding failed: {e}, using random")

                # Search in Qdrant using HTTP API directly
                import httpx
                with httpx.Client(timeout=10) as client:
                    response = client.post(
                        f"http://localhost:6333/collections/{self.collection_name}/points/search",
                        json={
                            "vector": query_embedding,
                            "limit": limit,
                            "with_payload": True
                        }
                    )
                    if response.status_code == 200:
                        search_result = response.json().get("result", [])
                    else:
                        logger.error(f"Qdrant search failed with status {response.status_code}")
                        search_result = []

                # Format results (from HTTP API response)
                results = []
                for point in search_result:
                    payload = point.get("payload", {})
                    results.append({
                        "id": str(point.get("id", "")),
                        "score": float(point.get("score", 0)),
                        "content": payload.get("content", payload.get("text", "")),
                        "source": payload.get("source", "echo_memory"),
                        "type": payload.get("type", "memory"),
                        "payload": payload
                    })

                logger.info(f"Fallback search for '{query}' returned {len(results)} results")
                return results

            except Exception as e2:
                logger.error(f"Fallback search failed: {e2}")
                return []

    async def store_fact(self, subject: str, predicate: str, object_: str, confidence: float = 1.0) -> dict | str:
        """Store a fact in vector database. Returns point_id string on success, or error dict on failure."""
        if not self.qdrant_client:
            return {"fact_id": "", "stored": False, "error": "Qdrant client not initialized"}

        try:
            # Create text representation
            text = f"{subject} {predicate} {object_}"

            # Get embedding via Ollama
            import httpx
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        f"{self.ollama_url}/api/embed",
                        json={"model": self.embedding_model, "input": text}
                    )
                    resp_data = resp.json()
                    embedding = (resp_data.get("embeddings") or [[]])[0] or resp_data.get("embedding", [])
                    if not embedding:
                        logger.warning("Empty embedding from Ollama for store_fact")
                        return {"fact_id": "", "stored": False, "error": "Ollama returned empty embedding"}
            except Exception as embed_err:
                logger.error(f"Failed to embed fact: {embed_err}")
                return {"fact_id": "", "stored": False, "error": f"Embedding failed: {embed_err}"}

            # Store in Qdrant
            from qdrant_client.models import PointStruct
            import uuid

            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "type": "fact",
                    "content": text,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "ingested_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 0,
                }
            )

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            # Dual-write to PostgreSQL facts table (non-fatal on failure)
            try:
                pool = await self._get_db_pool()
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO facts (subject, predicate, object, confidence, qdrant_point_id, source, created_at)
                        VALUES ($1, $2, $3, $4, $5, 'mcp_store_fact', NOW())
                        ON CONFLICT (subject, predicate, object) DO UPDATE
                        SET confidence = EXCLUDED.confidence,
                            qdrant_point_id = EXCLUDED.qdrant_point_id,
                            updated_at = NOW()
                    """, subject, predicate, object_, confidence, point_id)
                logger.info(f"Dual-write to PostgreSQL facts: {text}")
            except Exception as pg_err:
                logger.warning(f"PostgreSQL dual-write failed (non-fatal): {pg_err}")

            logger.info(f"Stored fact: {text}")
            return point_id

        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            return {"fact_id": "", "stored": False, "error": f"Storage failed: {e}"}

    async def get_facts(self, topic: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get facts from PostgreSQL facts table via unified knowledge layer"""
        from src.core.unified_knowledge import get_unified_knowledge

        knowledge = get_unified_knowledge()

        try:
            # Get facts from the unified layer
            if topic:
                facts = await knowledge.search_facts(topic, limit)
            else:
                # Get all recent facts
                facts = await knowledge.search_facts("", limit)

            # Convert to expected format
            result = []
            for fact in facts:
                result.append({
                    "content": fact.content,
                    "confidence": fact.confidence,
                    "type": fact.source_type,
                    "metadata": fact.metadata
                })

            logger.info(f"Retrieved {len(result)} facts from unified knowledge layer")
            return result

        except Exception as e:
            logger.error(f"Failed to get facts: {e}")
            return []

    async def store_memory(self, content: str, type_: str = "memory", metadata: Optional[Dict] = None) -> dict | str:
        """Store free-form text memory in Qdrant (not a structured SPO triple).
        Returns the point ID string on success, or error dict on failure."""
        if not self.qdrant_client:
            return {"memory_id": "", "stored": False, "error": "Qdrant client not initialized"}

        try:
            # Generate embedding
            import httpx
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        f"{self.ollama_url}/api/embed",
                        json={"model": self.embedding_model, "input": content}
                    )
                    resp_data = resp.json()
                    embedding = (resp_data.get("embeddings") or [[]])[0] or resp_data.get("embedding", [])
                    if not embedding:
                        logger.warning("Empty embedding from Ollama for store_memory")
                        return {"memory_id": "", "stored": False, "error": "Ollama returned empty embedding"}
            except Exception as embed_err:
                logger.error(f"Failed to embed memory: {embed_err}")
                return {"memory_id": "", "stored": False, "error": f"Embedding failed: {embed_err}"}

            from qdrant_client.models import PointStruct
            import uuid

            point_id = str(uuid.uuid4())
            payload = {
                "type": type_,
                "content": content,
                "text": content,
                "source": "mcp_store_memory",
                "timestamp": datetime.now().isoformat(),
                "ingested_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0,
            }
            if metadata:
                payload.update(metadata)

            point = PointStruct(id=point_id, vector=embedding, payload=payload)
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Stored memory ({type_}): {content[:80]}...")
            return point_id

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return {"memory_id": "", "stored": False, "error": f"Storage failed: {e}"}

    # ── New MCP action tools ───────────────────────────────────────────

    async def send_notification(self, message: str, title: str | None = None,
                                channels: list[str] | None = None,
                                priority: str = "normal") -> dict:
        """Send a notification via NotificationService and log to DB."""
        from src.services.notification_service import (
            get_notification_service, NotificationType, NotificationChannel,
        )

        priority_map = {"low": 1, "normal": 3, "high": 4, "urgent": 5}
        ntfy_priority = priority_map.get(priority, 3)

        channel_map = {
            "telegram": NotificationChannel.TELEGRAM,
            "ntfy": NotificationChannel.NTFY,
            "email": NotificationChannel.EMAIL,
        }
        channel_list = []
        for ch in (channels or ["telegram"]):
            mapped = channel_map.get(ch)
            if mapped:
                channel_list.append(mapped)
        if not channel_list:
            channel_list = [NotificationChannel.TELEGRAM]

        service = await get_notification_service()
        if not service:
            return {"sent": False, "error": "Notification service unavailable"}

        results = await service.send_notification(
            message=message,
            title=title,
            notification_type=NotificationType.INFO,
            channels=channel_list,
            priority=ntfy_priority,
        )

        # Log to notifications table (non-fatal)
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO notifications (title, body, priority, source, category, status, metadata, delivered_at)
                       VALUES ($1, $2, $3, 'mcp_send_notification', 'notification', $4, $5, $6)""",
                    title or "MCP Notification",
                    message,
                    priority,
                    "delivered" if any(results.values()) else "failed",
                    json.dumps({"channels": list(results.keys()), "results": results}),
                    datetime.now() if any(results.values()) else None,
                )
        except Exception as db_err:
            logger.warning(f"Failed to log notification to DB (non-fatal): {db_err}")

        return {
            "sent": any(results.values()),
            "channels": results,
            "message": message[:100],
        }

    async def check_services(self) -> dict:
        """Check all Tower services and return structured status."""
        from src.services.health_service import HealthService

        health_service = HealthService()
        system_health = await health_service.check_all()

        services = []
        for svc in system_health.services:
            services.append({
                "name": svc.name,
                "status": svc.status,
                "latency_ms": svc.latency_ms,
                "error": svc.error,
            })

        return {
            "overall_status": system_health.overall_status,
            "uptime_seconds": round(system_health.uptime_seconds),
            "services": services,
            "resources": {
                "cpu_percent": system_health.resources.get("cpu_percent", 0),
                "memory_percent": system_health.resources.get("memory_percent", 0),
                "disk_percent": system_health.resources.get("disk_percent", 0),
            },
        }

    async def schedule_reminder(self, message: str, remind_at: str,
                                title: str | None = None,
                                channel: str = "telegram") -> dict:
        """Schedule a reminder for a future time."""
        try:
            scheduled_time = datetime.fromisoformat(remind_at)
        except ValueError:
            return {"scheduled": False, "error": f"Invalid datetime format: {remind_at}"}

        if scheduled_time <= datetime.now():
            return {"scheduled": False, "error": "remind_at must be in the future"}

        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """INSERT INTO notifications
                       (title, body, priority, source, category, status, metadata, scheduled_for)
                       VALUES ($1, $2, 'normal', 'mcp_reminder', 'reminder', 'pending', $3, $4)
                       RETURNING id""",
                    title or "Reminder",
                    message,
                    json.dumps({"channel": channel}),
                    scheduled_time,
                )
            return {
                "scheduled": True,
                "reminder_id": str(row["id"]),
                "remind_at": scheduled_time.isoformat(),
                "channel": channel,
            }
        except Exception as e:
            logger.error(f"Failed to schedule reminder: {e}")
            return {"scheduled": False, "error": str(e)}

    async def trigger_generation(self, character_slug: str, count: int = 1,
                                 prompt_override: str | None = None) -> dict:
        """Trigger image generation via Anime Studio API."""
        import httpx

        count = max(1, min(count, 5))
        results = []
        errors = []

        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(count):
                try:
                    body: dict = {"generation_type": "image"}
                    if prompt_override:
                        body["prompt_override"] = prompt_override
                    resp = await client.post(
                        f"http://localhost:8401/api/visual/generate/{character_slug}",
                        json=body,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    results.append({
                        "prompt_id": data.get("prompt_id", data.get("id", "unknown")),
                        "status": data.get("status", "queued"),
                    })
                except httpx.ConnectError:
                    errors.append("Anime Studio not reachable at localhost:8401")
                    break
                except httpx.HTTPStatusError as e:
                    errors.append(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
                except Exception as e:
                    errors.append(str(e))

        return {
            "generated": len(results),
            "character_slug": character_slug,
            "results": results,
            "errors": errors,
        }

    async def telegram_bot_status(self) -> dict:
        """Get Telegram bot listener status."""
        try:
            from src.main import telegram_bot
            if telegram_bot:
                return telegram_bot.get_status()
            return {"running": False, "error": "Bot not initialized"}
        except Exception as e:
            return {"running": False, "error": str(e)}

    async def web_fetch(self, url: str, max_length: int = 5000) -> dict:
        """Fetch a URL and return its text content."""
        import httpx

        max_length = max(100, min(max_length, 50000))

        try:
            async with httpx.AsyncClient(
                timeout=30, follow_redirects=True,
                headers={"User-Agent": "EchoBrain/1.0 (Tower System)"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            text = resp.text

            # Strip HTML if needed
            if "html" in content_type.lower():
                text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', text, flags=re.IGNORECASE)
                text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text, flags=re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()

            truncated = len(text) > max_length
            text = text[:max_length]

            return {
                "url": str(resp.url),
                "content": text,
                "length": len(text),
                "truncated": truncated,
                "content_type": content_type.split(";")[0].strip(),
            }
        except httpx.ConnectError:
            return {"url": url, "error": f"Connection failed: {url}"}
        except httpx.HTTPStatusError as e:
            return {"url": url, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            return {"url": url, "error": str(e)}


# Global instance
mcp_service = MCPService()