#!/usr/bin/env python3
"""
Echo Brain MCP Server - stdio transport for Claude Code
This server communicates via stdin/stdout using JSON-RPC protocol.
"""

import sys
import json
import asyncio
import logging
from typing import Any
import psycopg2
from qdrant_client import QdrantClient
import httpx

# Configure logging to stderr (stdout is for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger('echo-brain-mcp')

# Database config - now using dedicated echo_brain database
DB_CONFIG = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE'
}

QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
OLLAMA_URL = 'http://localhost:11434'
EMBEDDING_MODEL = 'mxbai-embed-large'


class EchoBrainMCP:
    def __init__(self):
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.db_conn = None

    def get_db(self):
        if self.db_conn is None or self.db_conn.closed:
            self.db_conn = psycopg2.connect(**DB_CONFIG)
        return self.db_conn

    async def get_embedding(self, text: str) -> list:
        """Get embedding from Ollama."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": text},
                timeout=30.0
            )
            data = response.json()
            return data.get('embeddings', [[]])[0]

    async def search_memory(self, query: str, limit: int = 5) -> list:
        """Search vector memory for relevant context."""
        try:
            embedding = await self.get_embedding(query)
            if not embedding:
                return [{"error": "Failed to generate embedding"}]

            from qdrant_client.models import PointStruct, VectorParams, Distance
            results = self.qdrant.query_points(
                collection_name="echo_memory",
                query=embedding,
                limit=limit
            ).points

            return [{
                "score": hit.score,
                "content": hit.payload.get("content", "")[:500],
                "source": hit.payload.get("source", "unknown"),
                "type": hit.payload.get("type", "unknown")
            } for hit in results]

        except Exception as e:
            logger.error(f"search_memory error: {e}")
            return [{"error": str(e)}]

    def get_facts(self, topic: str, limit: int = 20) -> list:
        """Get structured facts about a topic."""
        try:
            conn = self.get_db()
            cur = conn.cursor()

            cur.execute("""
                SELECT subject, predicate, object, confidence, source_conversation_id
                FROM facts
                WHERE subject ILIKE %s OR object ILIKE %s OR predicate ILIKE %s
                ORDER BY confidence DESC
                LIMIT %s
            """, (f'%{topic}%', f'%{topic}%', f'%{topic}%', limit))

            rows = cur.fetchall()
            return [{
                "subject": r[0],
                "predicate": r[1],
                "object": r[2],
                "confidence": r[3],
                "source": str(r[4]) if r[4] else None
            } for r in rows]

        except Exception as e:
            logger.error(f"get_facts error: {e}")
            return [{"error": str(e)}]

    def store_fact(self, subject: str, predicate: str, obj: str, source: str = "user") -> dict:
        """Store a new fact."""
        try:
            conn = self.get_db()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO facts (subject, predicate, object, confidence)
                VALUES (%s, %s, %s, 1.0)
                ON CONFLICT DO NOTHING
                RETURNING id
            """, (subject, predicate, obj))

            conn.commit()
            result = cur.fetchone()

            return {"success": True, "id": str(result[0]) if result else None}

        except Exception as e:
            logger.error(f"store_fact error: {e}")
            return {"success": False, "error": str(e)}

    def get_anime_context(self, query: str) -> dict:
        """Get context from anime_production database via foreign tables."""
        try:
            conn = self.get_db()
            cur = conn.cursor()

            # Search projects
            cur.execute("""
                SELECT name, description
                FROM anime.projects
                WHERE name ILIKE %s OR description ILIKE %s
                LIMIT 3
            """, (f'%{query}%', f'%{query}%'))
            projects = cur.fetchall()

            # Search characters (foreign table has limited columns)
            cur.execute("""
                SELECT name, description
                FROM anime.characters
                WHERE name ILIKE %s OR description ILIKE %s
                LIMIT 5
            """, (f'%{query}%', f'%{query}%'))
            characters = cur.fetchall()

            return {
                "projects": [{"name": p[0], "description": p[1]} for p in projects],
                "characters": [{"name": c[0], "description": c[1]} for c in characters]
            }

        except Exception as e:
            logger.error(f"get_anime_context error: {e}")
            return {"error": str(e)}

    def list_tools(self) -> list:
        """Return available tools."""
        return [
            {
                "name": "search_memory",
                "description": "Search Echo Brain's vector memory for relevant context about any topic",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 5, "description": "Max results"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_facts",
                "description": "Get structured facts about a topic (people, projects, preferences)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to search facts about"},
                        "limit": {"type": "integer", "default": 20}
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "store_fact",
                "description": "Store a new fact in Echo Brain's knowledge base",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                        "source": {"type": "string", "default": "user"}
                    },
                    "required": ["subject", "predicate", "object"]
                }
            },
            {
                "name": "get_anime_context",
                "description": "Get context about anime projects, characters, and storylines",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for anime content"}
                    },
                    "required": ["query"]
                }
            }
        ]


async def handle_request(mcp: EchoBrainMCP, request: dict) -> dict:
    """Handle incoming MCP request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "echo-brain", "version": "1.0.0"}
                }
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": mcp.list_tools()}
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})

            if tool_name == "search_memory":
                result = await mcp.search_memory(args.get("query", ""), args.get("limit", 5))
            elif tool_name == "get_facts":
                result = mcp.get_facts(args.get("topic", ""), args.get("limit", 20))
            elif tool_name == "store_fact":
                result = mcp.store_fact(
                    args.get("subject", ""),
                    args.get("predicate", ""),
                    args.get("object", ""),
                    args.get("source", "user")
                )
            elif tool_name == "get_anime_context":
                result = mcp.get_anime_context(args.get("query", ""))
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                }
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"}
            }

    except Exception as e:
        logger.error(f"Request error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32603, "message": str(e)}
        }


async def main():
    """Main loop - read from stdin, write to stdout."""
    mcp = EchoBrainMCP()
    logger.info("Echo Brain MCP server started (stdio)")

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line = await reader.readline()
            if not line:
                break

            line = line.decode('utf-8').strip()
            if not line:
                continue

            request = json.loads(line)
            response = await handle_request(mcp, request)

            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Main loop error: {e}")


if __name__ == "__main__":
    asyncio.run(main())