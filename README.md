# Echo Brain

**Personal AI Assistant System — Self-Hosted Knowledge Layer & Agent Orchestrator**

Version: **0.6.1** (Retrieval Quality Overhaul + Garbage Cleanup)
Last Updated: 2026-02-16

---

## What Is Echo Brain?

Echo Brain is Patrick's self-hosted personal AI system running on the Tower server. It's not a chatbot — it's a knowledge layer that ingests personal data from multiple sources, builds a structured understanding through fact extraction and knowledge graphs, and provides intelligent responses grounded in that personal context.

The long-term vision is a system that:
- **Knows** Patrick's projects, preferences, history, and context across all data sources
- **Thinks** about what it knows — detects patterns, contradictions, and gaps
- **Improves** itself — finds its own bugs, proposes fixes, validates its own output
- **Acts** autonomously within safety boundaries — orchestrating content pipelines, monitoring systems, and proactive assistance

## Current Capabilities

| Capability | Status | Details |
|-----------|--------|---------|
| Vector search over personal data | ✅ Working | 317,222 vectors in Qdrant (768D nomic-embed-text) |
| Hybrid search (vector + text) | ✅ Working | Adaptive weighting, OR-semantics text search, garbage filtering |
| Fact extraction from vectors | ✅ Running | 2,558 structured facts in PostgreSQL |
| Conversation ingestion | ✅ Running | Watches `~/.claude/projects/`, 10 min cycles, readability filter |
| Knowledge graph building | ✅ Running | Connections between facts, conflict detection (daily) |
| Domain knowledge ingestion | ✅ Running | Anime production, ComfyUI workflows, models (60 min) |
| Reasoning pipeline | ✅ Working | CLASSIFY → RETRIEVE → REASON → RESPOND |
| File system monitoring | ✅ Running | Watches models, workflows, outputs (10 min) |
| Self-awareness (own code) | ✅ Running | Codebase indexing (6h), log monitoring (15 min), self-testing (60 min) |
| Self-improvement proposals | ✅ Running | Detect issues → reason → propose fix (2h, REVIEW gated) |
| Self-diagnostic system | ✅ Running | Comprehensive health checks, contract monitoring (5 min) |
| Voice interface (STT + TTS) | ✅ Working | Whisper large-v3 (CUDA) + Piper TTS, WebSocket streaming |
| Web dashboard (Vue 3) | ✅ Working | Voice panel, system view, memory search, endpoint tester |
| MCP server | ✅ Working | Model Context Protocol at `/mcp` for Claude integration |
| Intelligence layer (ask/query) | ✅ Working | `/api/echo/ask` with fact-prioritized context and strengthened prompts |
| Contract test suite | ✅ Running | Pact V4 consumer contracts + pytest smoke tests |
| Temporal decay + confidence | ✅ Running | Logarithmic decay, usage-validated exemptions, daily worker |
| Semantic deduplication | ✅ Running | Inline (0.97) + background (0.98) with metadata merge |
| Knowledge graph traversal | ✅ Working | NetworkX DiGraph, BFS, shortest path, MCP integration |
| HMLR fact verification | ✅ Running | FactScrubber (2h) + Governor (12h) conflict resolution |
| Home Assistant integration | -- Planned | Phase 4+ |

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      TOWER SERVER                                │
│  AMD Ryzen 9 24-core | 96GB DDR6 | RTX 3060 12GB | RX 9070 XT  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │               ECHO BRAIN (port 8309)                       │   │
│  │                                                            │   │
│  │  FastAPI Application                                       │   │
│  │  ├── /health                    (basic health)             │   │
│  │  ├── /mcp                       (Model Context Protocol)   │   │
│  │  ├── /api/echo/ask              (intelligence queries)     │   │
│  │  ├── /api/echo/memory/*         (search, ingest, status)   │   │
│  │  ├── /api/echo/intelligence/*   (reasoning, code, system)  │   │
│  │  ├── /api/echo/knowledge/*      (facts, stats)             │   │
│  │  ├── /api/echo/voice/*          (STT, TTS, WebSocket)      │   │
│  │  ├── /api/echo/reasoning/*      (analyze, debug)           │   │
│  │  ├── /api/echo/self-test/*      (quick, full)              │   │
│  │  ├── /api/echo/diagnostics/*    (contracts, issues)        │   │
│  │  ├── /api/echo/notifications    (notification queue)       │   │
│  │  ├── /api/echo/proposals/*      (improvement proposals)    │   │
│  │  ├── /api/autonomous/*          (goal/task management)     │   │
│  │  ├── /api/workers/status        (worker monitoring)        │   │
│  │  └── /api/pipeline/*            (pipeline health/query)    │   │
│  │                                                            │   │
│  │  Worker Scheduler (12 workers)                             │   │
│  │  ├── contract_monitor       (5 min)                        │   │
│  │  ├── conversation_watcher   (10 min)                       │   │
│  │  ├── file_watcher           (10 min)                       │   │
│  │  ├── log_monitor            (15 min)                       │   │
│  │  ├── fact_extraction        (30 min)                       │   │
│  │  ├── reasoning_worker       (30 min)                       │   │
│  │  ├── domain_ingestor        (60 min)                       │   │
│  │  ├── self_test_runner       (60 min)                       │   │
│  │  ├── improvement_engine     (2 hours)                      │   │
│  │  ├── codebase_indexer       (6 hours)                      │   │
│  │  ├── schema_indexer         (daily)                        │   │
│  │  └── knowledge_graph        (daily)                        │   │
│  │                                                            │   │
│  │  Voice Service                                             │   │
│  │  ├── STT: Whisper large-v3 (CUDA, float16)                │   │
│  │  ├── TTS: Piper en_US-lessac-medium (22050 Hz)            │   │
│  │  ├── VAD: Silero voice activity detection                  │   │
│  │  └── WebSocket: Real-time bidirectional streaming          │   │
│  │                                                            │   │
│  │  Intelligence Layer                                        │   │
│  │  ├── Query Classification (regex + embedding)              │   │
│  │  ├── Hybrid Retrieval (vector + text, OR-semantics)        │   │
│  │  ├── Context Compiler (fact-prioritized, budget-managed)   │   │
│  │  ├── LLM Reasoning (Ollama, model-per-intent)             │   │
│  │  └── Response Synthesis                                    │   │
│  │                                                            │   │
│  │  Frontend (Vue 3 + TypeScript, port 8311)                  │   │
│  │  ├── Voice Panel (record, playback, mute, debug)           │   │
│  │  ├── System View (diagnostics, workers, self-test)         │   │
│  │  ├── Memory Search (vector search UI)                      │   │
│  │  ├── Ask View (intelligence queries)                       │   │
│  │  └── Endpoint Tester (API explorer)                        │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐     │
│  │  PostgreSQL    │  │   Qdrant      │  │     Ollama       │     │
│  │  (echo_brain)  │  │  (6333)       │  │    (11434)       │     │
│  │  - knowledge   │  │  echo_memory  │  │  mistral:7b      │     │
│  │    facts       │  │  768D vecs    │  │  nomic-embed-text│     │
│  │  - autonomous  │  │  317,222 pts  │  │  gemma2:9b       │     │
│  │  - self-aware  │  │  text+keyword │  │  deepseek-r1:8b  │     │
│  │  - proposals   │  │  indexes      │  │  deepseek-coder  │     │
│  └───────────────┘  └───────────────┘  └──────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for detailed component documentation.
See [ROADMAP.md](./docs/ROADMAP.md) for the full development plan.
See [CHANGELOG.md](./CHANGELOG.md) for version history.

## Voice Interface

Echo Brain includes a full voice pipeline: speak to it, get spoken responses back.

| Component | Technology | Details |
|-----------|-----------|---------|
| STT | Whisper large-v3 | CUDA-accelerated, float16, on AMD RX 9070 XT |
| TTS | Piper (en_US-lessac-medium) | Local ONNX model, 22050 Hz output |
| VAD | Silero | Voice activity detection for utterance segmentation |
| Streaming | WebSocket | Real-time bidirectional at `/api/echo/voice/ws` |

**REST endpoints:**
- `POST /api/echo/voice/transcribe` — Upload audio, get text
- `POST /api/echo/voice/synthesize` — Send text, get WAV audio
- `POST /api/echo/voice/chat` — Full loop: audio in -> transcribe -> reason -> TTS -> audio out
- `GET /api/echo/voice/status` — Voice service health
- `GET /api/echo/voice/voices` — List available TTS voices

**WebSocket protocol** (`/api/echo/voice/ws`):
- Client sends `audio_chunk` (base64 PCM) then `audio_end`
- Server responds with `transcript`, then `response` (text + base64 audio)
- Status updates: `listening` / `processing` / `speaking`

## Self-Diagnostic System

Echo Brain monitors its own health across multiple dimensions:

- **Contract Monitor** — Tests critical API endpoints every 5 minutes, tracks response times and status codes
- **Log Monitor** — Reads own service logs every 15 minutes, classifies errors, creates issues
- **Self-Test Runner** — Validates query responses every 60 minutes, detects regressions
- **Improvement Engine** — Reasons about detected issues, proposes code fixes (REVIEW gated, never auto-applies)

**Diagnostic endpoints:**
- `GET /api/echo/diagnostics/test` — Run basic diagnostic test
- `GET /api/echo/diagnostics/contracts` — Current contract status
- `GET /api/echo/diagnostics/contracts/history` — Historical trend data
- `POST /api/echo/diagnostics/contracts/run` — Trigger manual test
- `GET /api/echo/self-test/quick` — Quick self-test
- `GET /api/echo/self-test/run` — Full self-test suite

## Quick Reference

```bash
# Service management
sudo systemctl status tower-echo-brain
sudo systemctl restart tower-echo-brain
sudo journalctl -u tower-echo-brain -f

# Health checks
curl -s http://localhost:8309/health | python3 -m json.tool
curl -s http://localhost:8309/api/workers/status | python3 -m json.tool
curl -s http://localhost:8309/api/echo/health/detailed | python3 -m json.tool

# Ask Echo Brain a question
curl -s -X POST http://localhost:8309/api/echo/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What hardware does Tower have?"}'

# Search memory vectors
curl -s -X POST http://localhost:8309/api/echo/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "LoRA training settings", "limit": 5}'

# MCP search (for Claude integration)
curl -s -X POST http://localhost:8309/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "search_memory", "arguments": {"query": "test", "limit": 5}}}'

# Voice service status
curl -s http://localhost:8309/api/echo/voice/status | python3 -m json.tool

# Knowledge facts and graph stats
curl -s http://localhost:8309/api/echo/knowledge/facts?limit=10 | python3 -m json.tool
curl -s http://localhost:8309/api/echo/knowledge/stats | python3 -m json.tool

# Worker status
curl -s http://localhost:8309/api/workers/status | python3 -m json.tool

# Check improvement proposals
curl -s http://localhost:8309/api/echo/proposals | python3 -m json.tool
```

## Testing

```bash
# Smoke tests (pytest) — tests live service endpoints
cd /opt/tower-echo-brain && python3 -m pytest tests/echo_brain_smoke_test.py -v

# Contract tests (vitest) — Pact V4 consumer contracts
cd /opt/tower-echo-brain/contract-tests/consumer && npx vitest run --reporter=verbose
```

See [tests/TEST_INVENTORY.md](./tests/TEST_INVENTORY.md) for the full test file inventory.

## Directory Structure

```
/opt/tower-echo-brain/
├── src/
│   ├── main.py                     # FastAPI app entry point
│   ├── api/
│   │   ├── endpoints/              # Route handlers
│   │   ├── autonomous.py           # Autonomous system API
│   │   └── voice.py                # Voice REST + WebSocket API
│   ├── intelligence/
│   │   └── reasoner.py             # Query processing pipeline
│   ├── context_assembly/
│   │   ├── retriever.py            # Vector search + context building
│   │   └── classifier.py           # Query intent classification
│   ├── core/
│   │   └── self_diagnostic.py      # Comprehensive system diagnostics
│   ├── autonomous/
│   │   ├── core.py                 # Autonomous orchestration
│   │   ├── goals.py                # Goal management
│   │   ├── safety.py               # Safety level enforcement
│   │   ├── audit.py                # Audit logging
│   │   ├── worker_scheduler.py     # Worker lifecycle management
│   │   └── workers/                # 12 autonomous workers
│   ├── services/
│   │   ├── embedding_service.py    # Ollama embedding integration
│   │   ├── voice_service.py        # Whisper STT + Piper TTS
│   │   └── voice_memory.py         # Voice conversation persistence
│   └── pipeline/
│       └── context_layer.py        # Context retrieval pipeline
├── frontend/                       # Vue 3 + TypeScript dashboard
│   └── src/
│       ├── api/echoApi.ts          # API client (all endpoint bindings)
│       ├── views/
│       │   ├── VoicePanel.vue      # Full voice interface (mute, debug, WS)
│       │   ├── VoiceSimple.vue     # Minimal voice interface
│       │   ├── VoiceTest.vue       # Voice endpoint tester
│       │   ├── SystemView.vue      # System diagnostics dashboard
│       │   ├── AskView.vue         # Intelligence query interface
│       │   ├── MemoryView.vue      # Vector search UI
│       │   └── EndpointsView.vue   # API endpoint explorer
│       └── router/index.ts         # Vue Router config
├── contract-tests/                 # Pact V4 consumer contract tests
│   └── consumer/
│       ├── tests/echo-brain.consumer.spec.ts
│       └── src/api/echo-brain-client.ts
├── tests/
│   ├── echo_brain_smoke_test.py    # Pytest smoke suite (31 tests)
│   ├── test_auth_integration.py    # Auth integration tests
│   ├── TEST_INVENTORY.md           # Test file documentation
│   └── tmp_archive/                # Archived stale/duplicate tests
├── docs/
│   ├── ARCHITECTURE.md             # System architecture deep dive
│   ├── ROADMAP.md                  # Development phases and milestones
│   └── SELF_DIAGNOSTIC.md          # Diagnostic system documentation
├── config/
│   └── self_tests.json             # Extensible self-test definitions
├── models/voice/piper/             # Piper TTS model files
├── CHANGELOG.md
├── MIGRATION_COMPLETE.md
└── README.md                       # This file
```

## Configuration

| Variable | Value | Location |
|----------|-------|----------|
| DATABASE_URL | `postgresql://patrick:...@localhost/echo_brain` | systemd service env |
| QDRANT_URL | `http://localhost:6333` | Hardcoded in workers |
| OLLAMA_URL | `http://localhost:11434` | Hardcoded in workers |
| EMBEDDING_MODEL | `nomic-embed-text` (768D) | workers + retriever |
| EMBEDDING_DIM | `768` | echo_memory Qdrant collection |
| REASONING_MODEL | `mistral:7b` (default) | intelligence/reasoner.py (model-per-intent routing) |
| EXTRACTION_MODEL | `gemma2:9b` | fact_extraction_worker.py |
| SERVICE_PORT | `8309` | systemd service |
| FRONTEND_PORT | `8311` | Vite dev / nginx |

## Database Architecture

Echo Brain uses a **microservices database architecture** with dedicated databases:

| Database | Purpose |
|----------|---------|
| **echo_brain** | Main data: conversations, facts, learning, self-awareness, proposals |
| **tower_auth** | Authentication and service registry |
| **tower_kb** | Knowledge base (documents, articles) |
| **anime_production** | Animation production data |
| **Qdrant** | Vector embeddings: echo_memory (317,222 pts), story_bible |

Database separation from tower_consolidated is complete. See `MIGRATION_COMPLETE.md` for details.

## Contributing

Echo Brain is a personal project. Development happens through Claude Code sessions. See [ROADMAP.md](./docs/ROADMAP.md) for the current development plan.
