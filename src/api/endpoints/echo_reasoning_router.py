"""
Echo Brain Reasoning Router - Transparent, multi-stage reasoning with progress tracking
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import logging
import json
import asyncio
from datetime import datetime
import httpx
import asyncpg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reasoning", tags=["reasoning"])

@router.post("/analyze")
async def analyze_with_progress(request: Dict[str, Any]):
    """
    Multi-stage analysis with full transparency and progress tracking
    Returns step-by-step reasoning process
    """
    query = request.get("query", "")
    max_depth = request.get("max_depth", 3)

    async def generate_analysis():
        """Generator that yields progress updates as JSON lines"""

        # Stage 1: Understanding the query
        yield json.dumps({
            "stage": 1,
            "name": "Understanding Query",
            "status": "starting",
            "timestamp": datetime.now().isoformat()
        }) + "\n"

        # Parse query for key concepts
        key_terms = query.lower().split()
        yield json.dumps({
            "stage": 1,
            "name": "Understanding Query",
            "status": "complete",
            "result": {
                "query": query,
                "key_terms": key_terms[:5],
                "query_type": "question" if "?" in query else "statement"
            }
        }) + "\n"

        # Stage 2: Memory Search
        yield json.dumps({
            "stage": 2,
            "name": "Searching Memory Vectors",
            "status": "starting",
            "details": "Searching 24,657 memory vectors..."
        }) + "\n"

        try:
            from src.integrations.mcp_service import mcp_service
            memory_results = await mcp_service.search_memory(query, limit=5)

            memory_context = []
            if memory_results and "results" in memory_results:
                for mem in memory_results["results"]:
                    memory_context.append({
                        "content": mem.get("content", "")[:200],
                        "score": mem.get("score", 0),
                        "source": mem.get("source", "unknown")
                    })

            yield json.dumps({
                "stage": 2,
                "name": "Searching Memory Vectors",
                "status": "complete",
                "result": {
                    "found": len(memory_context),
                    "top_scores": [m["score"] for m in memory_context[:3]],
                    "preview": memory_context[0]["content"][:100] if memory_context else None
                }
            }) + "\n"
        except Exception as e:
            yield json.dumps({
                "stage": 2,
                "status": "error",
                "error": str(e)
            }) + "\n"
            memory_context = []

        # Stage 3: Conversation History Search
        yield json.dumps({
            "stage": 3,
            "name": "Searching Conversation History",
            "status": "starting",
            "details": "Searching thousands of conversations..."
        }) + "\n"

        conversation_context = []
        try:
            conn = await asyncpg.connect(
                host='localhost',
                database='echo_brain',
                user='patrick',
                password='RP78eIrW7cI2jYvL5akt1yurE',
                timeout=5
            )

            # Get count first
            total_count = await conn.fetchval(
                "SELECT COUNT(*) FROM claude_conversations WHERE content ILIKE $1",
                f'%{key_terms[0] if key_terms else query}%'
            )

            # Get samples
            rows = await conn.fetch("""
                SELECT conversation_id, role, content, created_at
                FROM claude_conversations
                WHERE content ILIKE $1
                ORDER BY created_at DESC
                LIMIT 5
            """, f'%{key_terms[0] if key_terms else query}%')

            for row in rows:
                conversation_context.append({
                    "role": row["role"],
                    "content": row["content"][:200],
                    "timestamp": row["created_at"].isoformat() if row["created_at"] else None
                })

            await conn.close()

            yield json.dumps({
                "stage": 3,
                "name": "Searching Conversation History",
                "status": "complete",
                "result": {
                    "total_matching": total_count,
                    "retrieved": len(conversation_context),
                    "roles": [c["role"] for c in conversation_context]
                }
            }) + "\n"
        except Exception as e:
            yield json.dumps({
                "stage": 3,
                "status": "error",
                "error": str(e)
            }) + "\n"

        # Stage 4: Initial Analysis
        yield json.dumps({
            "stage": 4,
            "name": "Initial Analysis",
            "status": "starting",
            "details": "Analyzing context with Mistral 7B..."
        }) + "\n"

        analysis_prompt = f"""Analyze this query with the available context:

Query: {query}

Memory Context ({len(memory_context)} items found):
{chr(10).join([f"- [Score: {m['score']:.2f}] {m['content']}" for m in memory_context[:3]])}

Conversation History ({len(conversation_context)} recent items):
{chr(10).join([f"- [{c['role']}]: {c['content']}" for c in conversation_context[:3]])}

Provide analysis:
1. Core question/intent
2. Relevant patterns in context
3. Key insights
4. Connections identified"""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "mistral:7b", "prompt": analysis_prompt, "stream": False}
                )
                initial_analysis = response.json().get("response", "")

                yield json.dumps({
                    "stage": 4,
                    "name": "Initial Analysis",
                    "status": "complete",
                    "result": {
                        "analysis_preview": initial_analysis[:300],
                        "model": "mistral:7b"
                    }
                }) + "\n"
        except Exception as e:
            yield json.dumps({
                "stage": 4,
                "status": "error",
                "error": str(e)
            }) + "\n"
            initial_analysis = "Analysis failed"

        # Stage 5: Deep Reasoning
        if max_depth > 1:
            yield json.dumps({
                "stage": 5,
                "name": "Deep Reasoning",
                "status": "starting",
                "details": "Synthesizing comprehensive answer..."
            }) + "\n"

            synthesis_prompt = f"""Based on the analysis:
{initial_analysis}

Original Query: {query}

Context Summary:
- Memory vectors searched: {len(memory_context)} relevant items
- Conversation history: {total_count if 'total_count' in locals() else 0} total matches
- Time range: Recent to historical

Provide comprehensive reasoning:
1. Synthesize all information
2. Identify non-obvious connections
3. Generate actionable insights
4. Final answer with confidence level"""

            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "mistral:7b", "prompt": synthesis_prompt, "stream": False}
                    )
                    final_answer = response.json().get("response", "")

                    yield json.dumps({
                        "stage": 5,
                        "name": "Deep Reasoning",
                        "status": "complete",
                        "result": {
                            "final_answer": final_answer,
                            "confidence": "high" if len(memory_context) > 3 else "moderate"
                        }
                    }) + "\n"
            except Exception as e:
                yield json.dumps({
                    "stage": 5,
                    "status": "error",
                    "error": str(e)
                }) + "\n"
                final_answer = initial_analysis
        else:
            final_answer = initial_analysis

        # Final Summary
        yield json.dumps({
            "stage": "final",
            "name": "Complete",
            "status": "success",
            "summary": {
                "query": query,
                "memory_items_used": len(memory_context),
                "conversations_found": total_count if 'total_count' in locals() else 0,
                "reasoning_depth": max_depth,
                "final_answer_preview": final_answer[:500]
            },
            "timestamp": datetime.now().isoformat()
        }) + "\n"

    return StreamingResponse(
        generate_analysis(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/debug/{request_id}")
async def get_reasoning_debug(request_id: str):
    """Get debug information for a specific reasoning request"""
    # This would retrieve stored debug info
    return {
        "request_id": request_id,
        "message": "Debug info would be retrieved from cache/storage"
    }