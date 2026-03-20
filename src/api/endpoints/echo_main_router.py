"""
Echo Brain Main API Router
Consolidated ALL Echo Brain endpoints - NO DUPLICATION
All endpoints under /api/echo/*
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging
from datetime import datetime
import psutil
import os
import httpx
from src.model_config import get_model

logger = logging.getLogger(__name__)

# Pydantic models for request validation
class AskRequest(BaseModel):
    question: str
    allow_actions: bool = False
    session_id: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    allow_actions: bool = False
    session_id: Optional[str] = None

class SessionSaveRequest(BaseModel):
    session_id: str
    summary: str
    topics: List[str] = []
    key_decisions: List[str] = []
    files_modified: List[str] = []
    project_path: Optional[str] = None

# Main router - will be mounted at /api/echo
router = APIRouter(tags=["echo-brain"])

# ============= PRIMARY STATUS (Only ONE health endpoint!) =============
@router.get("/health")
async def echo_health():
    """Main Echo Brain health & status - THE ONLY HEALTH ENDPOINT"""
    from src.main import REQUEST_METRICS

    # Get service health
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # Check subsystems
    checks = {
        "database": await _check_database(),
        "vector_db": await _check_qdrant(),
        "llm": await _check_ollama()
    }

    # Calculate metrics
    total_requests = REQUEST_METRICS["total_requests"]
    avg_response_ms = 0
    if REQUEST_METRICS["response_times"]:
        avg_response_ms = sum(REQUEST_METRICS["response_times"]) / len(REQUEST_METRICS["response_times"])

    error_rate = 0
    if total_requests > 0:
        total_errors = REQUEST_METRICS["errors_4xx"] + REQUEST_METRICS["errors_5xx"]
        error_rate = (total_errors / total_requests) * 100

    # Get vector count
    vector_count = 0
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:6333/collections/echo_memory")
            vector_count = response.json().get("result", {}).get("points_count", 0)
    except:
        pass

    return {
        "status": "healthy" if all(checks.values()) else "degraded",
        "version": "0.4.0",
        "timestamp": datetime.now().isoformat(),
        "services": checks,
        "metrics": {
            "requests": total_requests,
            "memory_mb": round(memory_usage_mb, 1),
            "cpu_percent": cpu_percent,
            "vectors": vector_count,
            "avg_response_ms": round(avg_response_ms, 2),
            "error_rate": f"{error_rate:.2f}%"
        }
    }

@router.get("/status")
async def echo_status():
    """Echo Brain status endpoint - alias for health for frontend compatibility"""
    return await echo_health()

# ============= SYSTEM INFO (Only unique system endpoints) =============
@router.get("/system/resources")
async def system_resources():
    """System resource usage"""
    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count()
        },
        "memory": {
            "percent": psutil.virtual_memory().percent,
            "used_gb": psutil.virtual_memory().used / 1024**3,
            "total_gb": psutil.virtual_memory().total / 1024**3
        },
        "disk": {
            "percent": psutil.disk_usage('/').percent,
            "used_gb": psutil.disk_usage('/').used / 1024**3,
            "total_gb": psutil.disk_usage('/').total / 1024**3
        },
        "gpu": await _get_gpu_info()
    }

@router.get("/system/logs")
async def system_logs(lines: int = 100, level: Optional[str] = None, service: Optional[str] = None):
    """Get system logs with structured activity history"""
    import subprocess
    import json
    from src.main import REQUEST_METRICS

    # Filter journald logs by service if requested
    unit = "tower-echo-brain"
    if service:
        service_map = {"postgres": "postgresql", "ollama": "ollama", "qdrant": "qdrant", "comfyui": "tower-comfyui"}
        unit = service_map.get(service.lower(), "tower-echo-brain")

    cmd = ["sudo", "journalctl", "-u", unit, "-n", str(lines), "--no-pager", "-o", "json"]
    if level:
        priority_map = {"DEBUG": "7", "INFO": "6", "WARNING": "4", "ERROR": "3"}
        if level.upper() in priority_map:
            cmd.extend(["-p", priority_map[level.upper()]])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        logs = []
        for line in result.stdout.strip().split('\n') if result.stdout else []:
            if line:
                try:
                    log_entry = json.loads(line)
                    priority = log_entry.get("PRIORITY", "6")
                    level_map = {"0": "EMERG", "1": "ALERT", "2": "CRIT", "3": "ERROR", "4": "WARNING", "5": "NOTICE", "6": "INFO", "7": "DEBUG"}

                    # Parse timestamp from microseconds to ISO format
                    timestamp_us = int(log_entry.get("__REALTIME_TIMESTAMP", 0))
                    timestamp = datetime.fromtimestamp(timestamp_us / 1000000).isoformat() if timestamp_us else datetime.now().isoformat()

                    logs.append({
                        "timestamp": timestamp,
                        "message": log_entry.get("MESSAGE", ""),
                        "level": level_map.get(str(priority), "INFO"),
                        "service": unit
                    })
                except:
                    continue

        # Structured activity: recent API requests from middleware
        recent_requests = list(REQUEST_METRICS.get("recent_requests", []))

        return {
            "logs": logs,
            "count": len(logs),
            "recent_requests": recent_requests[-50:],
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/system/dashboard")
async def dashboard_metrics():
    """Dashboard summary with key metrics"""
    from src.main import REQUEST_METRICS, ERROR_LOG

    avg_time = sum(REQUEST_METRICS["response_times"]) / len(REQUEST_METRICS["response_times"]) if REQUEST_METRICS["response_times"] else 0
    total = REQUEST_METRICS["total_requests"] or 1
    errors = REQUEST_METRICS["errors_4xx"] + REQUEST_METRICS["errors_5xx"]

    health_score = max(0, 100 - (errors/total * 100) * 10 - min(avg_time / 100, 30))

    return {
        "health_score": round(health_score, 1),
        "requests": total,
        "errors": errors,
        "avg_ms": round(avg_time, 2),
        "top_endpoints": dict(sorted(
            REQUEST_METRICS["requests_by_endpoint"].items(),
            key=lambda x: x[1], reverse=True
        )[:5]),
        "recent_errors": list(ERROR_LOG)[-3:]
    }

# ============= MEMORY OPERATIONS =============
@router.get("/memory")
async def memory_info():
    """Basic memory endpoint for compatibility"""
    return {
        "status": "operational",
        "endpoints": [
            "/api/echo/memory/status",
            "/api/echo/memory/search",
            "/api/echo/memory/ingest"
        ],
        "description": "Echo Brain memory system"
    }

@router.get("/memory/status")
async def memory_status():
    """Memory system status with real statistics"""
    import asyncpg
    from pathlib import Path

    # Count conversation files
    conv_files = len(list(Path("/home/patrick/.claude/projects").rglob("*.jsonl")))

    # Get database stats
    conversations_count = 0
    embeddings_count = 0
    try:
        import os
        password = os.environ.get('PGPASSWORD')
        if not password:
            raise HTTPException(status_code=500, detail="Database password not configured")

        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="patrick",
            password=password,
            database="echo_brain"
        )

        # Count conversations
        conv_result = await conn.fetchval("SELECT COUNT(*) FROM conversations")
        conversations_count = conv_result if conv_result else 0

        # Count embeddings in Qdrant
        async with httpx.AsyncClient(timeout=5) as client:
            qdrant_response = await client.get("http://localhost:6333/collections/echo_memory")
            if qdrant_response.status_code == 200:
                qdrant_data = qdrant_response.json()
                embeddings_count = qdrant_data.get("result", {}).get("points_count", 0)

        await conn.close()
    except Exception as e:
        logger.warning(f"Failed to get database stats: {e}")

    return {
        "config": {
            "collection": "echo_memory",
            "embedding_model": "nomic-embed-text"
        },
        "is_running": False,  # Ingestion not currently running
        "last_ingestion": None,
        "conversation_files": conv_files,
        "conversations_processed": conversations_count,
        "embeddings_created": embeddings_count
    }

@router.post("/memory/search")
async def memory_search(request: Dict[str, Any]):
    """Search memory vectors"""
    from src.integrations.mcp_service import mcp_service

    query = request.get("query", "")
    limit = request.get("limit", 10)

    results = await mcp_service.search_memory(query, limit)
    return results

@router.post("/memory/ingest")
async def memory_ingest(request: Dict[str, Any]):
    """Ingest new memory (currently disabled)"""
    return {
        "status": "disabled",
        "message": "Memory ingestion is not currently running"
    }

# ============= SESSION PERSISTENCE =============
@router.post("/session/save")
async def save_session(request: SessionSaveRequest):
    """Save a session summary for pre-compaction persistence"""
    import asyncpg

    try:
        password = os.environ.get('PGPASSWORD', os.environ.get('DB_PASSWORD', ''))
        conn = await asyncpg.connect(
            host='localhost', database='echo_brain',
            user='patrick', password=password, timeout=5
        )

        # Generate embedding for the summary
        embedding = None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": request.summary}
                )
                if resp.status_code == 200:
                    embedding = resp.json().get("embedding", [])
                    if not embedding:
                        embedding = None
        except Exception as e:
            logger.warning(f"Embedding generation failed for session: {e}")

        # Insert into PostgreSQL
        row_id = await conn.fetchval("""
            INSERT INTO session_summaries (session_id, summary, topics, key_decisions, files_modified, project_path, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        """, request.session_id, request.summary, request.topics,
            request.key_decisions, request.files_modified, request.project_path,
            str(embedding) if embedding else None)

        await conn.close()

        # Also store in Qdrant for semantic search
        if embedding:
            try:
                import uuid
                point_id = str(uuid.uuid4())
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.put(
                        "http://localhost:6333/collections/echo_memory/points",
                        json={
                            "points": [{
                                "id": point_id,
                                "vector": embedding,
                                "payload": {
                                    "content": request.summary,
                                    "type": "session_summary",
                                    "session_id": request.session_id,
                                    "topics": request.topics,
                                    "project_path": request.project_path,
                                    "ingested_at": datetime.now().isoformat(),
                                }
                            }]
                        }
                    )
            except Exception as e:
                logger.warning(f"Qdrant session storage failed: {e}")

        return {"id": str(row_id), "stored": True}

    except Exception as e:
        logger.error(f"Session save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/recent")
async def get_recent_sessions(limit: int = 5):
    """Get recent session summaries"""
    import asyncpg

    try:
        password = os.environ.get('PGPASSWORD', os.environ.get('DB_PASSWORD', ''))
        conn = await asyncpg.connect(
            host='localhost', database='echo_brain',
            user='patrick', password=password, timeout=5
        )

        rows = await conn.fetch("""
            SELECT id, session_id, summary, topics, key_decisions, files_modified, project_path, created_at
            FROM session_summaries
            ORDER BY created_at DESC
            LIMIT $1
        """, limit)

        await conn.close()

        return {
            "sessions": [
                {
                    "id": str(row["id"]),
                    "session_id": row["session_id"],
                    "summary": row["summary"][:200],
                    "topics": row["topics"],
                    "project_path": row["project_path"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                }
                for row in rows
            ],
            "count": len(rows)
        }

    except Exception as e:
        logger.error(f"Recent sessions retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve a session summary by session_id"""
    import asyncpg

    try:
        password = os.environ.get('PGPASSWORD', os.environ.get('DB_PASSWORD', ''))
        conn = await asyncpg.connect(
            host='localhost', database='echo_brain',
            user='patrick', password=password, timeout=5
        )

        row = await conn.fetchrow("""
            SELECT id, session_id, summary, topics, key_decisions, files_modified, project_path, created_at
            FROM session_summaries
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT 1
        """, session_id)

        await conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return {
            "id": str(row["id"]),
            "session_id": row["session_id"],
            "summary": row["summary"],
            "topics": row["topics"],
            "key_decisions": row["key_decisions"],
            "files_modified": row["files_modified"],
            "project_path": row["project_path"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= INTELLIGENCE =============
@router.get("/intelligence/map")
async def knowledge_map():
    """Knowledge domain mapping — real counts from Qdrant by vector type"""
    from src.integrations.mcp_service import mcp_service

    # Domain groupings: map vector types to human-readable domains
    domain_types = {
        "Conversations": ["conversation"],
        "Codebase": ["code", "domain_code"],
        "Knowledge & Facts": ["fact", "knowledge_fact", "kb_article", "documentation"],
        "Media": ["photo", "video", "generation"],
        "Personal": ["email", "calendar_event", "memory"],
        "Infrastructure": ["domain_record", "domain_git", "schema"],
    }

    domains = {}
    counted_total = 0

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            for domain_name, type_list in domain_types.items():
                domain_count = 0
                for vtype in type_list:
                    resp = await client.post(
                        "http://localhost:6333/collections/echo_memory/points/count",
                        json={"filter": {"must": [{"key": "type", "match": {"value": vtype}}]}}
                    )
                    if resp.status_code == 200:
                        domain_count += resp.json().get("result", {}).get("count", 0)
                counted_total += domain_count
                if domain_count > 0:
                    domains[domain_name] = {"vectors": domain_count, "types": type_list}
    except Exception as e:
        logger.error(f"Knowledge map failed: {e}")

    total = mcp_service.get_vector_count()

    # Add uncategorized if counts don't add up
    uncategorized = total - counted_total
    if uncategorized > 0:
        domains["Other"] = {"vectors": uncategorized, "types": []}

    return {
        "domains": domains,
        "total_vectors": total,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/intelligence/think")
async def think(request: Dict[str, Any]):
    """Process through intelligence layer with multi-stage reasoning"""
    query = request.get("query", "")
    depth = request.get("depth", 2)  # How many reasoning stages

    # Stage 1: Gather context from all sources
    from src.integrations.mcp_service import mcp_service

    # Get memory context — search_memory returns list OR dict with "results" key
    memory_results = await mcp_service.search_memory(query, limit=5)
    if isinstance(memory_results, dict):
        memory_results = memory_results.get("results", [])
    memory_context = []
    if memory_results:
        for mem in memory_results[:5]:
            memory_context.append(mem.get("content", "")[:300])

    # Get facts if available
    key_terms = query.split()[:3]  # First 3 words as topics
    facts = []
    for term in key_terms:
        try:
            fact_results = await mcp_service.get_facts(term, limit=5)
            if fact_results:
                facts.extend(fact_results[:2])
        except:
            pass

    # Stage 2: Initial analysis with context
    analysis_prompt = f"""Analyze this query using available context:
Query: {query}

Memory Context:
{chr(10).join(memory_context[:3])}

Known Facts:
{chr(10).join([f"- {f.get('content', str(f)) if isinstance(f, dict) else str(f)}" for f in facts[:5]])}

Provide a structured analysis:
1. What is being asked
2. Relevant context identified
3. Key insights from memory
4. Reasoning approach"""

    async with httpx.AsyncClient(timeout=30) as client:
        analysis_response = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": get_model("analysis"), "prompt": analysis_prompt, "stream": False,
                  "keep_alive": "5m", "options": {"num_gpu": 0}}
        )
        analysis = analysis_response.json().get("response", "")

        # Stage 3: Deep reasoning if depth > 1
        if depth > 1:
            reasoning_prompt = f"""Based on this analysis:
{analysis}

Original Query: {query}

Now provide deep reasoning:
1. Synthesize all information
2. Identify patterns or connections
3. Generate insights beyond the obvious
4. Provide a comprehensive answer"""

            final_response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": get_model("reasoning"), "prompt": reasoning_prompt, "stream": False,
                      "keep_alive": "5m", "options": {"num_gpu": 0}}
            )
            final_answer = final_response.json().get("response", "")
        else:
            final_answer = analysis

    return {
        "response": final_answer,
        "model": get_model("general"),
        "reasoning_stages": depth,
        "context_sources": {
            "memory_items": len(memory_context),
            "facts": len(facts)
        },
        "analysis": analysis if depth > 1 else None
    }

# ============= CONVERSATIONS =============
@router.post("/conversations/search")
async def search_conversations(request: Dict[str, Any]):
    """Search conversation history"""
    query = request.get("query", "")
    limit = request.get("limit", 10)

    results = []
    total_count = 0
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host='localhost',
            database='echo_brain',
            user='patrick',
            password=os.getenv('DB_PASSWORD', ''),
            timeout=5
        )

        # Get total count
        total_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM claude_conversations
            WHERE content ILIKE $1
        """, f'%{query}%')

        # Get paginated results
        rows = await conn.fetch("""
            SELECT conversation_id, role, content, created_at
            FROM claude_conversations
            WHERE content ILIKE $1
            ORDER BY created_at DESC
            LIMIT $2
        """, f'%{query}%', limit)

        logger.info(f"Conversation search for '{query}': {len(rows)} of {total_count} total")
        results = [dict(row) for row in rows]
        await conn.close()
    except Exception as e:
        logger.error(f"Conversation search failed: {e}")

    return {"results": results, "count": len(results), "total": total_count}

# ============= Q&A =============
@router.post("/ask")
async def ask(request: AskRequest):
    """Main Q&A endpoint - NOW USES INTELLIGENCE LAYER"""
    question = request.question
    allow_actions = request.allow_actions

    # Route through the new intelligence layer
    from src.intelligence.reasoner import get_reasoning_engine
    reasoner = get_reasoning_engine()

    try:
        result = await reasoner.process(
            query=question,
            allow_actions=allow_actions,
            session_id=request.session_id
        )

        return {
            "answer": result.response,
            "model": result.model_used or "intelligence-layer",
            "agent": result.agent_name or "unknown",
            "query_type": result.query_type.value,
            "confidence": result.confidence,
            "sources": result.sources,
            "actions_taken": result.actions_taken,
            "execution_time_ms": result.execution_time_ms,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Intelligence layer failed: {e}")
        # Fallback to basic response
        return {
            "answer": f"I encountered an error processing your question: {str(e)}",
            "model": "error-fallback",
            "query_type": "error",
            "confidence": 0.0,
            "sources": [],
            "actions_taken": [],
            "execution_time_ms": 0,
            "timestamp": datetime.now().isoformat()
        }

@router.post("/query")
async def query(request: QueryRequest):
    """Query endpoint - compatible with ask but uses 'query' field"""
    from src.intelligence.reasoner import get_reasoning_engine
    reasoner = get_reasoning_engine()

    try:
        result = await reasoner.process(
            query=request.query,
            allow_actions=request.allow_actions,
            session_id=request.session_id
        )

        return {
            "response": result.response,
            "answer": result.response,  # Include both for compatibility
            "model_used": result.model_used or "intelligence-layer",
            "agent": result.agent_name or "unknown",
            "query_type": result.query_type.value,
            "confidence": result.confidence,
            "sources": result.sources,
            "execution_time_ms": result.execution_time_ms,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return {
            "response": f"I encountered an error processing your query: {str(e)}",
            "answer": f"I encountered an error processing your query: {str(e)}",
            "model_used": "error-fallback",
            "query_type": "error",
            "confidence": 0.0,
            "sources": [],
            "execution_time_ms": 0,
            "timestamp": datetime.now().isoformat()
        }

@router.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Streaming version of /ask — returns SSE events with retrieval progress + LLM tokens"""
    from src.intelligence.reasoner import get_reasoning_engine
    reasoner = get_reasoning_engine()

    async def generate():
        async for event in reasoner.process_stream(
            query=request.question,
            allow_actions=request.allow_actions,
            session_id=request.session_id
        ):
            yield event

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming version of /query — returns SSE events with retrieval progress + LLM tokens"""
    from src.intelligence.reasoner import get_reasoning_engine
    reasoner = get_reasoning_engine()

    async def generate():
        async for event in reasoner.process_stream(
            query=request.query,
            allow_actions=request.allow_actions,
            session_id=request.session_id
        ):
            yield event

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/search")
async def search(q: str, limit: int = 10):
    """General search across all data"""
    # Combine memory and conversation search
    from src.integrations.mcp_service import mcp_service

    memory_results = await mcp_service.search_memory(q, limit//2)

    # Also search conversations
    conv_results = await search_conversations({"query": q, "limit": limit//2})

    mem_count = len(memory_results) if isinstance(memory_results, list) else len(memory_results.get("results", []))

    return {
        "memory": memory_results,
        "conversations": conv_results,
        "total": mem_count + conv_results["count"]
    }

# ============= BRAIN VISUALIZATION =============
@router.get("/brain")
async def brain_activity():
    """Brain activity visualization"""
    from src.integrations.mcp_service import mcp_service

    vector_count = mcp_service.get_vector_count()  # Not async
    process = psutil.Process(os.getpid())

    return {
        "activity": {
            "state": "active",
            "vectors": vector_count,
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "intensity": min(psutil.cpu_percent() / 100, 1.0),
            "regions": {
                "memory": {"active": vector_count > 0, "neurons": vector_count},
                "reasoning": {"active": True, "neurons": 1000},
                "language": {"active": True, "neurons": 500}
            }
        }
    }

# ============= MODELS =============
@router.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            models = response.json().get("models", [])
            return {
                "models": [{"name": m["name"], "size_gb": m.get("size", 0) / 1e9} for m in models],
                "count": len(models)
            }
    except:
        return {"models": [], "error": "Ollama not available"}

# ============= AGENT REGISTRY =============
@router.get("/agents")
async def list_agents():
    """List all loaded agent definitions from the registry"""
    try:
        from src.core.agent_registry import get_agent_registry
        registry = get_agent_registry()
        agents = registry.get_all()
        return {
            "agents": [
                {
                    "name": a.name,
                    "model": a.model,
                    "fallback_model": a.fallback_model,
                    "intents": a.intents,
                    "token_budget_model": a.token_budget_model,
                    "options": a.options,
                    "source_file": a.source_file,
                    "system_prompt_preview": a.system_prompt[:200] + "..." if len(a.system_prompt) > 200 else a.system_prompt,
                }
                for a in agents
            ],
            "count": len(agents),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        return {"agents": [], "count": 0, "error": str(e)}


@router.post("/agents/reload")
async def reload_agents():
    """Force hot-reload of all agent definitions"""
    try:
        from src.core.agent_registry import get_agent_registry
        registry = get_agent_registry()
        registry.force_reload()
        agents = registry.get_all()
        return {
            "status": "reloaded",
            "agents": [{"name": a.name, "model": a.model, "intents": a.intents} for a in agents],
            "count": len(agents),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reload agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= MCP PROTOCOL =============
@router.post("/mcp")
async def mcp_protocol(request: dict):
    """MCP protocol handler for Claude integration"""
    from src.integrations.mcp_service import mcp_service

    method = request.get("method", "")

    if method == "tools/list":
        return {
            "tools": [
                {"name": "search_memory", "description": "Search Echo Brain memories"},
                {"name": "get_facts", "description": "Get structured facts"},
                {"name": "store_fact", "description": "Store new fact"},
                {"name": "manage_ollama", "description": "Manage Ollama models: list, pull, delete, refresh, show running"},
                {"name": "search_photos", "description": "Search personal photos and videos by semantic query, with optional filters for media_type, year, category, person"}
            ]
        }

    params = request.get("params", {})
    tool_name = params.get("name")
    args = params.get("arguments", {})

    if tool_name == "search_memory":
        return await mcp_service.search_memory(args.get("query", ""), args.get("limit", 10))
    elif tool_name == "get_facts":
        return await mcp_service.get_facts(args.get("topic"), args.get("limit", 100))
    elif tool_name == "manage_ollama":
        # Delegate to the main app's handler
        from src.main import _handle_ollama_mcp
        return await _handle_ollama_mcp(args)
    elif tool_name == "search_photos":
        from src.services.photo_dedup_service import PhotoDedupService
        svc = PhotoDedupService()
        results = await svc.search_media(
            query=args.get("query", ""),
            media_type=args.get("media_type"),
            year=args.get("year"),
            category=args.get("category"),
            person=args.get("person"),
            limit=args.get("limit", 10),
        )
        return {"results": results, "count": len(results)}

    return {"error": f"Unknown tool: {tool_name}"}

# ============= HELPERS =============
async def _check_database():
    try:
        import asyncpg
        import os
        password = os.environ.get('PGPASSWORD')
        if not password:
            return False

        conn = await asyncpg.connect(
            host='localhost',
            database='echo_brain',
            user='patrick',
            password=password,
            timeout=2
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except:
        return False

async def _check_qdrant():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:6333/collections", timeout=2)
            return r.status_code == 200
    except:
        return False

async def _check_ollama():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:11434/api/tags", timeout=2)
            return r.status_code == 200
    except:
        return False

async def _get_gpu_info():
    import subprocess
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "name": parts[0],
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2])
            }
    except:
        pass
    return None
# ============= VECTOR HEALTH =============
@router.get("/vector-health")
async def vector_health():
    """Live vector pipeline health — per-type counts, missing types, required field coverage."""
    from src.monitoring.vector_health import get_vector_health
    return await get_vector_health()


@router.post("/vector-health/contract")
async def run_vector_contract():
    """Run the full vector health contract test on demand."""
    from src.monitoring.vector_health import get_vector_health_contract
    contract = get_vector_health_contract()
    return await contract.run_all()


# ============= SELF-DIAGNOSTIC =============
@router.get("/diagnostic")
async def run_self_diagnostic():
    """
    Run comprehensive self-diagnostic.
    Tests knowledge sources, integrations, codebase, and provides recommendations.
    """
    from src.core.self_diagnostic import run_diagnostic
    return await run_diagnostic()

@router.get("/diagnostic/quick")
async def quick_diagnostic():
    """Quick health check with key metrics"""
    from src.core.self_diagnostic import get_diagnostic
    diagnostic = get_diagnostic()

    # Run subset of checks
    await diagnostic._diagnose_knowledge_sources()
    await diagnostic._diagnose_unified_layer()

    return {
        "health_score": diagnostic._calculate_health_score(),
        "summary": diagnostic._generate_summary(),
        "quick_stats": await diagnostic._get_quick_stats()
    }


@router.get("/diagnostic/deep")
async def deep_diagnostic():
    """
    Deep self-diagnostic that:
    - Tests if Echo Brain can answer questions about itself
    - Verifies context actually improves answers
    - Identifies knowledge gaps
    - Analyzes fact quality and contradictions
    - Checks code quality
    - Provides actionable fixes
    """
    from src.core.deep_diagnostic import run_deep_diagnostic
    return await run_deep_diagnostic()
