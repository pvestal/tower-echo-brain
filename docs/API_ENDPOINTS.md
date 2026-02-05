# Echo Brain API Documentation

## Overview
Echo Brain provides a unified AI knowledge system combining vectors, facts, and conversations.

- **Base URL**: `http://localhost:8309`
- **API Prefix**: `/api/echo/` (all endpoints)
- **Port**: 8309
- **Frontend**: `/echo-brain/` (via nginx proxy)

## Core Endpoints

### GET /health
Service health check (outside `/api/echo/` for monitoring compatibility).
```json
{
  "status": "healthy",
  "service": "echo-brain",
  "database_user": "patrick",
  "timestamp": "2026-02-05T02:34:06.848655"
}
```

### GET /docs
Interactive OpenAPI documentation (Swagger UI).

### POST /mcp
Model Context Protocol integration for Claude.
- Methods: `search_memory`, `get_facts`, `store_fact`

## Knowledge & Q&A

### POST /api/echo/ask
Main Q&A endpoint using unified knowledge layer.
```json
Request:
{
  "question": "What port does Echo Brain run on?",
  "use_context": true,
  "verbose": true
}

Response:
{
  "answer": "Echo Brain runs on port 8309",
  "model": "mistral:7b",
  "context_used": true,
  "sources": [
    {
      "type": "fact",
      "content": "Echo Brain runs on port 8309",
      "confidence": 1.0
    }
  ],
  "debug": {
    "search_terms": ["port", "echo", "brain"],
    "steps": ["Found 5 facts, 3 vectors, 3 conversations"],
    "total_sources": 11,
    "prompt_length": 2245
  },
  "timestamp": "2026-02-05T02:34:11"
}
```

### POST /api/echo/intelligence/think
Multi-stage reasoning with memory and facts.
```json
Request:
{
  "query": "Explain Echo Brain architecture",
  "depth": 2  // Number of reasoning stages
}

Response:
{
  "response": "Detailed analysis...",
  "model": "mistral:7b",
  "reasoning_stages": 2,
  "context_sources": {
    "memory_items": 5,
    "facts": 2
  },
  "analysis": "Initial analysis..."
}
```

## Memory Operations

### POST /api/echo/memory/search
Search Qdrant vector memory (24,657 vectors).
```json
Request:
{
  "query": "search term",
  "limit": 10
}

Response:
{
  "results": [
    {
      "id": "uuid",
      "score": 0.85,
      "content": "memory content",
      "source": "echo_memory",
      "type": "memory"
    }
  ],
  "count": 10,
  "collection": "echo_memory"
}
```

### GET /api/echo/memory/stats
Memory system statistics.
```json
{
  "vector_count": 24657,
  "collection": "echo_memory",
  "embedding_model": "mxbai-embed-large:latest",
  "dimensions": 1024
}
```

## Facts & Knowledge

### GET /api/echo/facts
Retrieve facts from PostgreSQL (6,129 facts).
```json
Query params: ?topic=echo%20brain&limit=10

Response:
[
  {
    "subject": "Echo Brain",
    "predicate": "runs on port",
    "object": "8309",
    "confidence": 1.0,
    "timestamp": "2026-02-05T02:34:11"
  }
]
```

### POST /api/echo/facts/add
Add a new fact to the knowledge base.
```json
Request:
{
  "subject": "Echo Brain",
  "predicate": "version",
  "object": "0.4.0",
  "confidence": 1.0
}
```

## Conversations

### POST /api/echo/conversations/search
Search conversation history (13,630 messages).
```json
Request:
{
  "query": "search terms",
  "limit": 10
}

Response:
{
  "results": [
    {
      "conversation_id": "uuid",
      "role": "user",
      "content": "message content",
      "created_at": "2026-02-05T02:34:11"
    }
  ],
  "total_found": 10
}
```

## System & Monitoring

### GET /api/echo/metrics
System metrics and request statistics.
```json
{
  "total_requests": 1234,
  "errors_4xx": 5,
  "errors_5xx": 2,
  "average_response_time_ms": 45.2,
  "requests_per_minute": 12.5,
  "uptime_seconds": 3600,
  "active_requests": 2
}
```

### GET /api/echo/logs
Service logs with filtering.
```json
Query params: ?lines=100&level=ERROR

Response:
{
  "logs": [
    {
      "timestamp": "2026-02-05T02:34:11",
      "level": "ERROR",
      "message": "Error details",
      "request_id": "abc123"
    }
  ],
  "count": 100
}
```

### GET /api/echo/status
Comprehensive status of all subsystems.
```json
{
  "database": {
    "connected": true,
    "facts": 6129,
    "conversations": 13630
  },
  "vectors": {
    "connected": true,
    "count": 24657,
    "collection": "echo_memory"
  },
  "llm": {
    "available": true,
    "models": ["mistral:7b", "deepseek-r1:8b"],
    "embedding_model": "mxbai-embed-large:latest"
  },
  "unified_knowledge": {
    "core_facts": 22,
    "total_sources": 44408
  }
}
```

## Unified Knowledge Layer

The UnifiedKnowledgeLayer is the core innovation in Echo Brain 0.4.0, combining:

1. **PostgreSQL Facts** (6,129): Structured subject-predicate-object facts
2. **Qdrant Vectors** (24,657): Semantic embeddings from conversations/code
3. **Conversations** (13,630): Historical Claude conversation messages
4. **Core Facts** (22): Hardcoded essential facts always available

All `/ask` and `/intelligence/think` endpoints use this unified layer to provide comprehensive, contextual responses with transparent source attribution.

## Authentication
Currently no authentication required (localhost only).

## Rate Limiting
No rate limits currently enforced.

## WebSocket Support
Not currently implemented.

## Frontend Access
Vue3 dashboard available at: `http://localhost/echo-brain/`
- Real-time monitoring
- Interactive Q&A interface
- Source visualization
- Debug information display