#!/usr/bin/env python3
"""
Echo Brain MCP HTTP Server

Provides memory and fact access through the Model Context Protocol via HTTP.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

# Don't use nest_asyncio with uvicorn - it causes conflicts
# import nest_asyncio
# nest_asyncio.apply()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Vector database and async imports
from qdrant_client import QdrantClient
import httpx
import asyncpg

# FastAPI for HTTP server
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("echo-brain-mcp-http")

# FastAPI app
app = FastAPI(title="Echo Brain MCP Server", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
qdrant_client = None
db_pool = None
ollama_url = "http://localhost:11434"
embedding_model = "mxbai-embed-large:latest"  # Using 1024-dim model
vector_collection = "echo_memory"  # Standard collection with 1024 dimensions
fallback_collection = None  # No fallback needed

# Database configuration
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'echo_brain',
    'user': 'patrick',
    'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
}


class MCPRequest(BaseModel):
    method: str
    params: Optional[Dict] = {}


class MCPToolCall(BaseModel):
    name: str
    arguments: Dict


# Initialize connections on startup
@app.on_event("startup")
async def startup():
    """Initialize connections to Qdrant and PostgreSQL."""
    global qdrant_client, db_pool, vector_collection

    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)

        # Test Qdrant connection and determine which collection to use
        try:
            collections = qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if vector_collection in collection_names:
                logger.info(f"Using vector collection: {vector_collection}")
            else:
                logger.error(f"Vector collection '{vector_collection}' not found. Please ensure it exists.")

        except Exception as e:
            logger.error(f"Error checking Qdrant collections: {e}")

        # Initialize PostgreSQL connection pool
        db_pool = await asyncpg.create_pool(
            **db_config,
            min_size=2,
            max_size=10,
            command_timeout=30
        )

        logger.info("Echo Brain MCP HTTP Server initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Echo Brain MCP HTTP Server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources."""
    if db_pool:
        await db_pool.close()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "echo-brain-mcp", "version": "1.0.0"}


@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    """Handle MCP protocol requests."""

    if request.method == "tools/list":
        return {
            "tools": [
                {
                    "name": "search_memory",
                    "description": "Search Echo Brain memory using vector similarity",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for memory retrieval"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_facts",
                    "description": "Get facts related to a specific topic from the knowledge base",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to search for related facts"
                            }
                        },
                        "required": ["topic"]
                    }
                },
                {
                    "name": "store_fact",
                    "description": "Store a new fact as a subject-predicate-object triple",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "The subject of the fact"
                            },
                            "predicate": {
                                "type": "string",
                                "description": "The predicate/relationship of the fact"
                            },
                            "object": {
                                "type": "string",
                                "description": "The object/value of the fact"
                            }
                        },
                        "required": ["subject", "predicate", "object"]
                    }
                },
                {
                    "name": "get_recent_context",
                    "description": "Get recent conversation summaries and context",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "hours": {
                                "type": "integer",
                                "description": "Number of hours to look back (default: 24)",
                                "default": 24
                            }
                        }
                    }
                }
            ]
        }

    elif request.method == "tools/call":
        tool_call = MCPToolCall(**request.params)
        return await handle_tool_call(tool_call)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")


async def handle_tool_call(tool_call: MCPToolCall):
    """Handle tool execution."""

    if tool_call.name == "search_memory":
        return await search_memory(tool_call.arguments)
    elif tool_call.name == "get_facts":
        return await get_facts(tool_call.arguments)
    elif tool_call.name == "store_fact":
        return await store_fact(tool_call.arguments)
    elif tool_call.name == "get_recent_context":
        return await get_recent_context(tool_call.arguments)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_call.name}")


async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding using Ollama with mxbai-embed-large model (1024D)"""
    try:
        # Truncate text if too long
        if len(text) > 1500:
            text = text[:1500]

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ollama_url}/api/embeddings",  # Use OLD API that actually works!
                json={
                    "model": embedding_model,
                    "prompt": text  # Use "prompt" not "input"
                }
            )

        if response.status_code == 200:
            data = response.json()
            return data.get("embedding")  # Use "embedding" not "embeddings[0]"
        else:
            logger.error(f"Embedding generation failed: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


async def search_memory(arguments: dict) -> dict:
    """Search Echo Brain memory using vector similarity."""
    try:
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)

        if not query:
            return {"error": "query parameter is required"}

        # Generate embedding for query
        query_embedding = await generate_embedding(query)

        if not query_embedding:
            return {"results": []}

        # Search Qdrant using query_points (correct method)
        search_results = qdrant_client.query_points(
            collection_name=vector_collection,
            query=query_embedding,
            limit=limit
        ).points

        results = []
        for result in search_results:
            results.append({
                'score': result.score,
                'text': result.payload.get('text', ''),
                'metadata': result.payload
            })

        response_text = f"Found {len(results)} relevant memories for '{query}':\n\n"

        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            text = result.get('text', '')
            metadata = result.get('metadata', {})

            response_text += f"**Result {i}** (Score: {score:.3f})\n"
            response_text += f"{text}\n"

            if metadata:
                source = metadata.get('source_file', metadata.get('original_cache_id', 'Unknown'))
                response_text += f"*Source: {source}*\n"

            response_text += "\n---\n\n"

        return {"content": [{"type": "text", "text": response_text}]}

    except Exception as e:
        logger.error(f"Error in search_memory: {e}")
        return {"error": f"Error searching memory: {str(e)}"}


async def get_facts(arguments: dict) -> dict:
    """Get facts related to a specific topic."""
    try:
        topic = arguments.get("topic", "")

        if not topic:
            return {"error": "topic parameter is required"}

        async with db_pool.acquire() as conn:
            # Search for facts where subject, predicate, or object contains the topic
            query = """
                SELECT subject, predicate, object, confidence, created_at, updated_at
                FROM facts
                WHERE LOWER(subject) LIKE LOWER($1)
                   OR LOWER(predicate) LIKE LOWER($1)
                   OR LOWER(object) LIKE LOWER($1)
                ORDER BY confidence DESC, created_at DESC
                LIMIT 20
            """

            rows = await conn.fetch(query, f"%{topic}%")

            facts = []
            for row in rows:
                facts.append({
                    'subject': row['subject'],
                    'predicate': row['predicate'],
                    'object': row['object'],
                    'confidence': float(row['confidence']) if row['confidence'] else 0.0,
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None
                })

        if not facts:
            return {"content": [{"type": "text", "text": f"No facts found related to topic: '{topic}'"}]}

        # Format facts as subject-predicate-object triples
        response_text = f"Found {len(facts)} facts related to '{topic}':\n\n"

        for fact in facts:
            subject = fact['subject']
            predicate = fact['predicate']
            obj = fact['object']
            confidence = fact.get('confidence', 0)
            created_at = fact.get('created_at', 'Unknown')

            response_text += f"• **{subject}** {predicate} **{obj}**\n"
            response_text += f"  *Confidence: {confidence:.2f} | Added: {created_at}*\n\n"

        return {"content": [{"type": "text", "text": response_text}]}

    except Exception as e:
        logger.error(f"Error in get_facts: {e}")
        return {"error": f"Error retrieving facts: {str(e)}"}


async def store_fact(arguments: dict) -> dict:
    """Store a new fact in the knowledge base."""
    try:
        subject = arguments.get("subject", "").strip()
        predicate = arguments.get("predicate", "").strip()
        obj = arguments.get("object", "").strip()

        if not all([subject, predicate, obj]):
            return {"error": "subject, predicate, and object parameters are all required"}

        async with db_pool.acquire() as conn:
            # Insert or update fact
            query = """
                INSERT INTO facts (subject, predicate, object, confidence, created_at, updated_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                ON CONFLICT (subject, predicate, object)
                DO UPDATE SET
                    confidence = GREATEST(facts.confidence, EXCLUDED.confidence),
                    updated_at = NOW()
            """

            await conn.execute(query, subject, predicate, obj, 0.8)  # Default confidence

        return {"content": [{"type": "text", "text": f"✅ Successfully stored fact: '{subject} {predicate} {obj}'"}]}

    except Exception as e:
        logger.error(f"Error in store_fact: {e}")
        return {"error": f"Error storing fact: {str(e)}"}


async def get_recent_context(arguments: dict) -> dict:
    """Get recent conversation context from the last N hours."""
    try:
        hours = arguments.get("hours", 24)

        async with db_pool.acquire() as conn:
            # Get recent conversations and learning items
            since_time = datetime.now() - timedelta(hours=hours)

            # Try to get from echo_conversations first
            query = """
                SELECT
                    created_at as timestamp,
                    'conversation' as source,
                    COALESCE(
                        CASE
                            WHEN context IS NOT NULL THEN context::text
                            ELSE 'Recent conversation'
                        END,
                        'No context available'
                    ) as content
                FROM echo_conversations
                WHERE created_at >= $1
                ORDER BY created_at DESC
                LIMIT 10
            """

            rows = await conn.fetch(query, since_time)

            context = []
            for row in rows:
                context.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'source': row['source'],
                    'content': row['content'][:500] + '...' if len(row['content']) > 500 else row['content']
                })

        if not context:
            return {"content": [{"type": "text", "text": f"No recent context found in the last {hours} hours"}]}

        response_text = f"Recent context from the last {hours} hours:\n\n"

        for item in context:
            timestamp = item.get('timestamp', 'Unknown')
            content = item.get('content', '')
            source = item.get('source', 'Unknown')

            response_text += f"**{timestamp}** ({source})\n"
            response_text += f"{content}\n\n---\n\n"

        return {"content": [{"type": "text", "text": response_text}]}

    except Exception as e:
        logger.error(f"Error in get_recent_context: {e}")
        return {"error": f"Error retrieving recent context: {str(e)}"}


if __name__ == "__main__":
    # Run the HTTP server
    uvicorn.run(app, host="0.0.0.0", port=8312, log_level="info")