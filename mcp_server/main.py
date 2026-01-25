#!/usr/bin/env python3
"""
Echo Brain MCP Server

Provides memory and fact access through the Model Context Protocol (MCP).
Connects to Qdrant vector database and PostgreSQL for comprehensive knowledge retrieval.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

# Handle nested event loops only when needed
import nest_asyncio

# Vector database and async imports
from qdrant_client import QdrantClient
import httpx
import asyncpg

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    GetPromptRequest,
    GetPromptResult,
    PromptMessage,
    LoggingLevel
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("echo-brain-mcp")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class EchoBrainMCPServer:
    """Echo Brain MCP Server for memory and fact access."""

    def __init__(self):
        self.server = Server("echo-brain")
        self.qdrant_client = None
        self.db_pool = None
        self.ollama_url = "http://localhost:11434"
        self.embedding_model = "mxbai-embed-large:latest"  # Using 1024-dim model
        self.vector_collection = "echo_memory"  # Standard collection with 1024 dimensions
        self.fallback_collection = None  # No fallback needed

        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }

        self._setup_tools()
        self._setup_health_check()

    def _setup_tools(self):
        """Setup MCP tools."""

        # Tool 1: Search Memory
        @self.server.call_tool()
        async def search_memory(arguments: dict) -> list[TextContent]:
            """Search Echo Brain memory using vector similarity."""
            try:
                query = arguments.get("query", "")
                limit = arguments.get("limit", 5)

                if not query:
                    return [TextContent(
                        type="text",
                        text="Error: query parameter is required"
                    )]

                results = await self._search_vector_memory(query, limit)

                if not results:
                    return [TextContent(
                        type="text",
                        text=f"No memories found for query: '{query}'"
                    )]

                # Format results
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

                return [TextContent(type="text", text=response_text)]

            except Exception as e:
                logger.error(f"Error in search_memory: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error searching memory: {str(e)}"
                )]

        # Tool 2: Get Facts
        @self.server.call_tool()
        async def get_facts(arguments: dict) -> list[TextContent]:
            """Get facts related to a specific topic."""
            try:
                topic = arguments.get("topic", "")

                if not topic:
                    return [TextContent(
                        type="text",
                        text="Error: topic parameter is required"
                    )]

                facts = await self._get_facts_from_db(topic)

                if not facts:
                    return [TextContent(
                        type="text",
                        text=f"No facts found related to topic: '{topic}'"
                    )]

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

                return [TextContent(type="text", text=response_text)]

            except Exception as e:
                logger.error(f"Error in get_facts: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error retrieving facts: {str(e)}"
                )]

        # Tool 3: Store Fact
        @self.server.call_tool()
        async def store_fact(arguments: dict) -> list[TextContent]:
            """Store a new fact in the knowledge base."""
            try:
                subject = arguments.get("subject", "").strip()
                predicate = arguments.get("predicate", "").strip()
                obj = arguments.get("object", "").strip()

                if not all([subject, predicate, obj]):
                    return [TextContent(
                        type="text",
                        text="Error: subject, predicate, and object parameters are all required"
                    )]

                success = await self._store_fact_in_db(subject, predicate, obj)

                if success:
                    return [TextContent(
                        type="text",
                        text=f"✅ Successfully stored fact: '{subject} {predicate} {obj}'"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text="❌ Failed to store fact"
                    )]

            except Exception as e:
                logger.error(f"Error in store_fact: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error storing fact: {str(e)}"
                )]

        # Tool 4: Get Recent Context
        @self.server.call_tool()
        async def get_recent_context(arguments: dict) -> list[TextContent]:
            """Get recent conversation context from the last N hours."""
            try:
                hours = arguments.get("hours", 24)

                context = await self._get_recent_context_from_db(hours)

                if not context:
                    return [TextContent(
                        type="text",
                        text=f"No recent context found in the last {hours} hours"
                    )]

                response_text = f"Recent context from the last {hours} hours:\n\n"

                for item in context:
                    timestamp = item.get('timestamp', 'Unknown')
                    content = item.get('content', '')
                    source = item.get('source', 'Unknown')

                    response_text += f"**{timestamp}** ({source})\n"
                    response_text += f"{content}\n\n---\n\n"

                return [TextContent(type="text", text=response_text)]

            except Exception as e:
                logger.error(f"Error in get_recent_context: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error retrieving recent context: {str(e)}"
                )]

    def _setup_health_check(self):
        """Setup health check endpoint."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_memory",
                    description="Search Echo Brain memory using vector similarity",
                    inputSchema={
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
                ),
                Tool(
                    name="get_facts",
                    description="Get facts related to a specific topic from the knowledge base",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to search for related facts"
                            }
                        },
                        "required": ["topic"]
                    }
                ),
                Tool(
                    name="store_fact",
                    description="Store a new fact as a subject-predicate-object triple",
                    inputSchema={
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
                ),
                Tool(
                    name="get_recent_context",
                    description="Get recent conversation summaries and context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hours": {
                                "type": "integer",
                                "description": "Number of hours to look back (default: 24)",
                                "default": 24
                            }
                        }
                    }
                )
            ]

    async def initialize(self):
        """Initialize connections to Qdrant and PostgreSQL."""
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)

            # Test Qdrant connection and determine which collection to use
            try:
                collections = self.qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]

                if self.vector_collection in collection_names:
                    logger.info(f"Using vector collection: {self.vector_collection}")
                else:
                    logger.error(f"Vector collection '{self.vector_collection}' not found. Please ensure it exists.")

            except Exception as e:
                logger.error(f"Error checking Qdrant collections: {e}")

            # Initialize PostgreSQL connection pool
            self.db_pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                command_timeout=30
            )

            logger.info("Echo Brain MCP Server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Echo Brain MCP Server: {e}")
            raise

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using Ollama."""
        try:
            # Truncate text if too long
            if len(text) > 1500:
                text = text[:1500]

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get('embedding')
                else:
                    logger.error(f"Embedding generation failed: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    async def _search_vector_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search vector memory using query embedding."""
        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)

            if not query_embedding:
                return []

            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.vector_collection,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )

            results = []
            for result in search_results:
                results.append({
                    'score': result.score,
                    'text': result.payload.get('text', ''),
                    'metadata': result.payload
                })

            return results

        except Exception as e:
            logger.error(f"Error searching vector memory: {e}")
            return []

    async def _get_facts_from_db(self, topic: str) -> List[Dict[str, Any]]:
        """Get facts related to a topic from PostgreSQL."""
        try:
            async with self.db_pool.acquire() as conn:
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
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None
                    })

                return facts

        except Exception as e:
            logger.error(f"Error getting facts from database: {e}")
            return []

    async def _store_fact_in_db(self, subject: str, predicate: str, obj: str) -> bool:
        """Store a fact in the PostgreSQL facts table."""
        try:
            async with self.db_pool.acquire() as conn:
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
                return True

        except Exception as e:
            logger.error(f"Error storing fact in database: {e}")
            return False

    async def _get_recent_context_from_db(self, hours: int) -> List[Dict[str, Any]]:
        """Get recent conversation context from database."""
        try:
            async with self.db_pool.acquire() as conn:
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

                # If no conversations found, try learning_items
                if not context:
                    query = """
                        SELECT
                            created_at as timestamp,
                            'learning_item' as source,
                            COALESCE(content, 'Learning item') as content
                        FROM learning_items
                        WHERE created_at >= $1
                        ORDER BY created_at DESC
                        LIMIT 10
                    """

                    rows = await conn.fetch(query, since_time)

                    for row in rows:
                        context.append({
                            'timestamp': row['timestamp'].isoformat(),
                            'source': row['source'],
                            'content': row['content'][:500] + '...' if len(row['content']) > 500 else row['content']
                        })

                return context

        except Exception as e:
            logger.error(f"Error getting recent context from database: {e}")
            return []

    async def cleanup(self):
        """Clean up resources."""
        if self.db_pool:
            await self.db_pool.close()


async def main():
    """Main entry point for the MCP server."""
    # Initialize the server
    echo_server = EchoBrainMCPServer()
    await echo_server.initialize()

    try:
        # Run the stdio server with proper exception handling
        async with stdio_server() as streams:
            try:
                await echo_server.server.run(
                    streams[0], streams[1], InitializationOptions(
                        server_name="echo-brain",
                        server_version="1.0.0",
                        capabilities=echo_server.server.get_capabilities(
                            notification_options={},
                            experimental_capabilities={}
                        )
                    )
                )
            except Exception as e:
                logger.error(f"Server run error: {e}")
                # Try to handle gracefully
                pass
    except Exception as e:
        logger.error(f"stdio_server error: {e}")
    finally:
        try:
            await echo_server.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


if __name__ == "__main__":
    # Set up logging for development
    logging.basicConfig(level=logging.INFO)

    try:
        # Remove nest_asyncio for systemd service - it conflicts
        if hasattr(nest_asyncio, 'apply'):
            try:
                loop = asyncio.get_running_loop()
                logger.info("Existing event loop detected, skipping nest_asyncio")
            except RuntimeError:
                # No running loop, safe to apply nest_asyncio
                nest_asyncio.apply()
                logger.info("Applied nest_asyncio for nested loop support")

        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)