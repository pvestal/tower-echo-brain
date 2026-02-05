# Echo Brain API Endpoints

## Overview
All Echo Brain endpoints are served under the `/api/echo/` prefix.
The service runs on port 8309.

Base URL: `http://localhost:8309/api/echo`

## Health & Status

### GET /api/echo/health
Main health and status endpoint for Echo Brain.
- Returns service health, subsystem status, and key metrics
- Response includes database, vector_db, and LLM status
- Metrics: requests, memory, CPU, vectors, response times

## System Information

### GET /api/echo/system/resources
System resource usage information.
- CPU usage and core count
- Memory usage (percent, used, total)
- Disk usage (percent, used, total)
- GPU information (if available)

### GET /api/echo/system/logs
Retrieve system logs from journalctl.
- Query params: `lines` (default: 100), `level` (DEBUG/INFO/WARNING/ERROR)
- Returns parsed journal entries with timestamp, message, and level

### GET /api/echo/system/dashboard
Dashboard metrics summary.
- Health score calculation
- Request statistics
- Top endpoints by usage
- Recent errors

## Memory Operations

### GET /api/echo/memory/status
Memory system configuration and status.
- Collection name and embedding model
- Ingestion status

### POST /api/echo/memory/search
Search memory vectors.
```json
{
  "query": "search term",
  "limit": 10
}
```

### POST /api/echo/memory/ingest
Ingest new memory (currently disabled).

## Intelligence

### GET /api/echo/intelligence/map
Knowledge domain mapping.
- Domain statistics with vector counts
- Confidence scores per domain
- Total vector count

### POST /api/echo/intelligence/think
Process query through intelligence layer.
```json
{
  "query": "your question"
}
```

## Conversations

### POST /api/echo/conversations/search
Search conversation history.
```json
{
  "query": "search term",
  "limit": 10
}
```

## Q&A

### POST /api/echo/ask
Main question-answering endpoint.
```json
{
  "question": "your question"
}
```

### GET /api/echo/search
General search across all data.
- Query param: `q` (search query), `limit` (max results)
- Returns combined memory and conversation results

## Brain Visualization

### GET /api/echo/brain
Brain activity visualization data.
- Current activity state
- Vector count and memory usage
- CPU usage and intensity
- Active brain regions with neuron counts

## Models

### GET /api/echo/models
List available Ollama models.
- Model names and sizes
- Total count

## MCP Protocol

### POST /api/echo/mcp
MCP protocol handler for Claude integration.
```json
{
  "method": "tools/call",
  "params": {
    "name": "search_memory",
    "arguments": {
      "query": "search term",
      "limit": 10
    }
  }
}
```

Available tools:
- `search_memory`: Search Echo Brain memories
- `get_facts`: Get structured facts
- `store_fact`: Store new fact

## Additional Services

The following specialized routers may be available:
- `/api/echo/self-test`: Self-test diagnostics
- `/api/echo/memory`: Extended memory operations
- `/api/echo/intelligence`: Deep intelligence operations
- `/api/echo/reasoning`: LLM reasoning endpoints
- `/api/echo/search`: Direct PostgreSQL search

## Frontend

The Vue3 frontend is served at `/echo-brain/` and connects to these API endpoints.

## Authentication

Currently no authentication required (local service only).

## Response Format

All endpoints return JSON responses with appropriate HTTP status codes:
- 200: Success
- 400: Bad request
- 500: Internal server error

Error responses include:
```json
{
  "error": "error message",
  "request_id": "unique-id"
}
```

## Rate Limiting

No rate limiting currently implemented (local service).

## Monitoring

All requests are logged with:
- Unique request IDs
- Response times
- Error tracking
- Endpoint metrics

Access logs via systemd: `journalctl -u tower-echo-brain`