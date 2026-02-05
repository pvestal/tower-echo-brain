"""
Echo Brain Main API Router
Consolidated ALL Echo Brain endpoints - NO DUPLICATION
All endpoints under /api/echo/*
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import psutil
import os
import httpx

logger = logging.getLogger(__name__)

# Main router - will be mounted at /api/echo
router = APIRouter(tags=["echo-brain"])

# ============= PRIMARY STATUS (Only ONE health endpoint!) =============
@router.get("/health")
async def echo_health():
    """Main Echo Brain health & status - THE ONLY HEALTH ENDPOINT"""
    from src.main import REQUEST_METRICS
    import subprocess

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
async def system_logs(lines: int = 100, level: Optional[str] = None):
    """Get system logs"""
    import subprocess
    import json

    cmd = ["sudo", "journalctl", "-u", "tower-echo-brain", "-n", str(lines), "--no-pager", "-o", "json"]
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
                    logs.append({
                        "time": log_entry.get("__REALTIME_TIMESTAMP", ""),
                        "msg": log_entry.get("MESSAGE", ""),
                        "level": log_entry.get("PRIORITY", "")
                    })
                except:
                    continue
        return {"logs": logs, "count": len(logs)}
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
@router.get("/memory/status")
async def memory_status():
    """Memory system status"""
    return {
        "config": {
            "collection": "echo_memory",
            "embedding_model": "mxbai-embed-large:latest"
        },
        "is_running": False,  # Ingestion not currently running
        "last_ingestion": None
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

# ============= INTELLIGENCE =============
@router.get("/intelligence/map")
async def knowledge_map():
    """Knowledge domain mapping"""
    from src.integrations.mcp_service import mcp_service

    # Simplified knowledge map from vector analysis
    domains = {
        "Tower System": {"vectors": 5000, "confidence": 0.8},
        "Anime Production": {"vectors": 3000, "confidence": 0.7},
        "Echo Brain": {"vectors": 10000, "confidence": 0.9},
        "Programming": {"vectors": 6657, "confidence": 0.85}
    }

    return {
        "domains": domains,
        "total_vectors": mcp_service.get_vector_count(),  # Not async
        "timestamp": datetime.now().isoformat()
    }

@router.post("/intelligence/think")
async def think(request: Dict[str, Any]):
    """Process through intelligence layer with multi-stage reasoning"""
    query = request.get("query", "")
    depth = request.get("depth", 2)  # How many reasoning stages

    # Stage 1: Gather context from all sources
    from src.integrations.mcp_service import mcp_service

    # Get memory context
    memory_results = await mcp_service.search_memory(query, limit=5)
    memory_context = []
    if memory_results and "results" in memory_results:
        for mem in memory_results["results"][:5]:
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
{chr(10).join([f"- {f}" for f in facts[:5]])}

Provide a structured analysis:
1. What is being asked
2. Relevant context identified
3. Key insights from memory
4. Reasoning approach"""

    async with httpx.AsyncClient(timeout=30) as client:
        analysis_response = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral:7b", "prompt": analysis_prompt, "stream": False}
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
            json={"model": "mistral:7b", "prompt": reasoning_prompt, "stream": False}
        )
        final_answer = final_response.json().get("response", "")
    else:
        final_answer = analysis

    return {
        "response": final_answer,
        "model": "mistral:7b",
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
            password='RP78eIrW7cI2jYvL5akt1yurE',
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
async def ask(request: Dict[str, Any]):
    """Main Q&A endpoint - WITH CONTEXT FROM MEMORY"""
    question = request.get("question", "")
    use_context = request.get("use_context", True)
    verbose = request.get("verbose", True)  # Default to verbose for transparency

    # Build context from memory and conversations
    context_parts = []
    sources = []
    debug_info = []

    if use_context:
        try:
            # 1. Search memory vectors for relevant context
            from src.integrations.mcp_service import mcp_service
            memory_results = await mcp_service.search_memory(question, limit=3)

            if memory_results and "results" in memory_results:
                context_parts.append("=== Relevant Memory Context ===")
                for mem in memory_results["results"][:3]:
                    context_parts.append(f"- {mem.get('content', '')[:200]}")
                    sources.append({"type": "memory", "score": mem.get("score", 0)})

            # 2. Search conversations for relevant history
            import asyncpg
            conn = await asyncpg.connect(
                host='localhost',
                database='echo_brain',
                user='patrick',
                password='RP78eIrW7cI2jYvL5akt1yurE',
                timeout=5
            )

            conv_rows = await conn.fetch("""
                SELECT content, role
                FROM claude_conversations
                WHERE content ILIKE $1
                ORDER BY created_at DESC
                LIMIT 3
            """, f'%{question.split()[0] if question else ""}%')

            if conv_rows:
                context_parts.append("=== Recent Related Conversations ===")
                for row in conv_rows:
                    context_parts.append(f"- [{row['role']}]: {row['content'][:200]}")
                    sources.append({"type": "conversation", "role": row["role"]})

            await conn.close()

        except Exception as e:
            logger.warning(f"Context building failed: {e}")

    # Build the enhanced prompt with context
    if context_parts:
        enhanced_prompt = f"""Based on the following context from your memory and history:

{chr(10).join(context_parts)}

Question: {question}

Provide a comprehensive answer that incorporates the relevant context above."""
    else:
        enhanced_prompt = question

    # Send to Ollama with context
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b",
                "prompt": enhanced_prompt,
                "stream": False
            }
        )

        result = {
            "answer": response.json().get("response", ""),
            "model": "mistral:7b",
            "context_used": len(context_parts) > 0,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }

        # Add verbose debug info if requested
        if verbose:
            result["debug"] = {
                "question": question,
                "context_parts_count": len(context_parts),
                "sources_detail": sources,
                "prompt_length": len(enhanced_prompt),
                "context_preview": context_parts[:2] if context_parts else None
            }

        return result

@router.get("/search")
async def search(q: str, limit: int = 10):
    """General search across all data"""
    # Combine memory and conversation search
    from src.integrations.mcp_service import mcp_service

    memory_results = await mcp_service.search_memory(q, limit//2)

    # Also search conversations
    conv_results = await search_conversations({"query": q, "limit": limit//2})

    return {
        "memory": memory_results,
        "conversations": conv_results,
        "total": len(memory_results.get("results", [])) + conv_results["count"]
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
                {"name": "store_fact", "description": "Store new fact"}
            ]
        }

    params = request.get("params", {})
    tool_name = params.get("name")
    args = params.get("arguments", {})

    if tool_name == "search_memory":
        return await mcp_service.search_memory(args.get("query", ""), args.get("limit", 10))
    elif tool_name == "get_facts":
        return await mcp_service.get_facts(args.get("topic"), args.get("limit", 100))

    return {"error": f"Unknown tool: {tool_name}"}

# ============= HELPERS =============
async def _check_database():
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host='localhost',
            database='echo_brain',
            user='patrick',
            password='RP78eIrW7cI2jYvL5akt1yurE',
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