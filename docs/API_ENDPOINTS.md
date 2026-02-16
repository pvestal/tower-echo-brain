# Echo Brain API Endpoints

Version: 0.6.1
Last Updated: 2026-02-16
Total: **90+ endpoints** across 85+ paths

---

## Quick Reference

- **Base URL:** `http://localhost:8309`
- **OpenAPI Docs:** `http://localhost:8309/docs`
- **Frontend:** `http://localhost:8311/echo-brain/` (or via nginx proxy at port 80)

---

## Health & System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check (status, uptime) |
| GET | `/api` | API root — lists available endpoints |
| GET | `/api/v1/health` | Legacy v1 health check |
| GET | `/api/workers/status` | All autonomous workers: running state, error counts, last run times |
| GET | `/api/echo/health/detailed` | Comprehensive health: workers, knowledge counts, quality metrics, open issues |
| GET | `/api/echo/system/logs` | Recent service logs |

---

## Reasoning & Q&A

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/echo/ask` | Primary Q&A endpoint — classifies query, retrieves context, reasons via LLM |
| POST | `/api/echo/reasoning/analyze` | Multi-step reasoning with progress tracking |
| GET | `/api/echo/reasoning/debug/{request_id}` | Debug info for a specific reasoning request |

---

## Memory & Search

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/echo/memory/search` | Semantic vector search across echo_memory (317,222 vectors, 768D) |
| GET | `/api/echo/memory/status` | Vector store connection status and collection info |
| GET | `/api/echo/memory/health` | Memory subsystem health |
| POST | `/api/echo/memory/ingest` | Trigger manual ingestion |
| GET | `/api/echo/search` | Search conversations |
| POST | `/api/echo/search/domain` | Domain-specific search |

---

## Knowledge & Facts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/knowledge/facts` | List extracted facts (2,558 knowledge triples) |
| GET | `/api/echo/knowledge/stats` | Knowledge base statistics |

---

## Knowledge Graph (v0.6.0+)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/graph/related/{entity}` | BFS traversal — find entities related to a given entity |
| GET | `/api/echo/graph/path` | Shortest path between two entities |
| GET | `/api/echo/graph/neighborhood/{entity}` | Ego subgraph around an entity |
| GET | `/api/echo/graph/stats` | Graph statistics (node count, edge count, density) |
| POST | `/api/echo/graph/refresh` | Trigger incremental graph reload |

---

## Voice Service

All voice endpoints support Whisper large-v3 STT and Piper TTS.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/echo/voice/transcribe` | Speech-to-text (file upload or base64 audio) |
| POST | `/api/echo/voice/synthesize` | Text-to-speech (returns WAV audio) |
| POST | `/api/echo/voice/synthesize/base64` | Text-to-speech (returns base64 JSON) |
| POST | `/api/echo/voice/chat` | Full voice pipeline: audio in → STT → reason → TTS → audio out |
| GET | `/api/echo/voice/status` | Voice service health (STT loaded, TTS loaded, VAD available) |
| GET | `/api/echo/voice/voices` | Available Piper voice models |
| **WS** | `/api/echo/voice/ws` | **Real-time bidirectional voice streaming** — VAD-gated recording, live transcription, streamed TTS response |

---

## Intelligence Layer

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/echo/intelligence/query` | Intelligent query with full context assembly |
| GET | `/api/echo/intelligence/status` | Intelligence subsystem status |
| POST | `/api/echo/intelligence/diagnose` | Diagnose a system issue |
| POST | `/api/echo/intelligence/execute` | Execute an action |
| POST | `/api/echo/intelligence/maintenance` | Run maintenance operations |

### Code Intelligence

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/echo/intelligence/code/index` | Index a codebase path |
| GET | `/api/echo/intelligence/code/search` | Search indexed code |
| GET | `/api/echo/intelligence/code/dependencies/{path}` | Get dependency tree for a file |
| GET | `/api/echo/intelligence/explain/{path}` | Explain a code file |

### System Intelligence

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/intelligence/system/services` | List all Tower services |
| GET | `/api/echo/intelligence/system/topology` | Service topology map |
| POST | `/api/echo/intelligence/system/scan` | Scan system state |
| GET | `/api/echo/intelligence/service/{name}` | Get info for a specific service |

### Procedures

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/intelligence/procedures` | List runbook procedures |
| POST | `/api/echo/intelligence/procedures/{name}/execute` | Execute a named procedure |

---

## Autonomous System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/autonomous/status` | Autonomous system status (running, paused, goals, tasks) |
| POST | `/api/autonomous/start` | Start autonomous operations |
| POST | `/api/autonomous/stop` | Stop autonomous operations |
| POST | `/api/autonomous/pause` | Pause (soft stop) |
| POST | `/api/autonomous/resume` | Resume from pause |
| POST | `/api/autonomous/kill` | Emergency kill switch — stops all autonomous activity |
| POST | `/api/autonomous/cycle` | Manually trigger one worker cycle |
| POST | `/api/autonomous/repair` | Run self-repair routines |
| GET | `/api/autonomous/diagnosis` | Run system diagnosis |

### Goals

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/autonomous/goals` | List all goals |
| POST | `/api/autonomous/goals` | Create a new goal |
| PATCH | `/api/autonomous/goals/{goal_id}` | Update a goal |
| DELETE | `/api/autonomous/goals/{goal_id}` | Cancel a goal |

### Tasks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/autonomous/tasks` | List autonomous tasks |

### Approvals (REVIEW-gated actions)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/autonomous/approvals` | Pending approvals queue |
| POST | `/api/autonomous/approvals/{task_id}/approve` | Approve a REVIEW-gated task |
| POST | `/api/autonomous/approvals/{task_id}/reject` | Reject a REVIEW-gated task |

### Notifications

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/autonomous/notifications` | List notifications |
| GET | `/api/autonomous/notifications/count` | Unread count |
| POST | `/api/autonomous/notifications/{notification_id}/read` | Mark as read |
| POST | `/api/autonomous/notifications/mark-all-read` | Mark all as read |

### Audit

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/autonomous/audit` | Audit log of all autonomous actions |

---

## Improvement Proposals

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/proposals` | List improvement proposals (filter by status) |
| GET | `/api/echo/proposals/{proposal_id}` | Full proposal detail (current code, proposed code, reasoning) |
| POST | `/api/echo/proposals/{proposal_id}/approve` | Approve a proposal |
| POST | `/api/echo/proposals/{proposal_id}/reject` | Reject a proposal |

---

## Self-Test & Diagnostics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/self-test/run` | Run full self-test suite |
| GET | `/api/echo/self-test/quick` | Quick health check |

### Contract Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/diagnostics/contracts` | Current contract test status |
| POST | `/api/echo/diagnostics/contracts/run` | Trigger a contract test run |
| GET | `/api/echo/diagnostics/contracts/history` | Historical contract results |
| GET | `/api/echo/diagnostics/contracts/issues` | Open contract issues |
| GET | `/api/echo/diagnostics/contracts/{contract_id}` | History for a specific contract |
| GET | `/api/echo/diagnostics/test` | Test the contract monitor itself |

---

## Notifications (Echo)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/notifications` | List Echo notifications |
| POST | `/api/echo/notifications/{notification_id}/read` | Mark notification as read |

---

## Ingestion

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/echo/ingestion/status` | Ingestion pipeline status |
| GET | `/api/v1/ingestion/status` | Legacy v1 ingestion status |

---

## Pipeline (Legacy)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pipeline/health` | Pipeline health |
| GET | `/api/pipeline/query` | Pipeline query (GET) |
| POST | `/api/pipeline/query` | Pipeline query (POST) |

---

## Legacy v1 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/ingestion/status` | Ingestion status |
| GET | `/api/v1/memories` | List memories |
| POST | `/api/v1/memories` | Create memory |
| POST | `/api/v1/query` | Query |

---

## MCP (Model Context Protocol)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/mcp` | MCP tool handler — exposes `search_memory`, `get_facts`, `store_fact`, `explore_graph`, `manage_ollama` tools for Claude Code |
| GET | `/mcp/health` | MCP server health |

---

## Authentication

No authentication required (service binds to localhost only, accessed via nginx reverse proxy on the local network).

## WebSocket

Voice WebSocket at `/api/echo/voice/ws` provides real-time bidirectional streaming:
- Client sends binary audio frames
- Server responds with JSON messages (transcript, response text, timing) and binary audio (TTS)
- VAD-gated: server detects speech boundaries automatically
- Includes STT timing, reasoning timing, and TTS timing in responses
