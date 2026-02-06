# Echo Brain

**Personal AI Assistant System — Self-Hosted Knowledge Layer & Agent Orchestrator**

Version: **0.3.0** (Phase 2c: Reasoning Pipeline & Knowledge Integration)
Last Updated: 2026-02-06

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
| Vector search over personal data | ✅ Working | 20,000+ vectors in Qdrant (1024D nomic-embed-text) |
| Fact extraction from vectors | ✅ Running | 40+ structured facts, worker runs every 30 min |
| Conversation ingestion | ✅ Running | Watches Claude conversation exports, 60 min cycles |
| Knowledge graph building | ✅ Running | Connections between facts, conflict detection |
| Domain knowledge ingestion | ✅ Phase 2c | Anime production, ComfyUI workflows, models |
| Reasoning pipeline | ✅ Phase 2c | CLASSIFY→EXTRACT→CONNECT→REASON→ACT |
| File system monitoring | ✅ Phase 2c | Watches models, workflows, outputs (10 min) |
| Self-awareness (own code) | ✅ Phase 2c | Codebase indexing, log monitoring, self-testing |
| Self-improvement proposals | ✅ Phase 2c | Detect issues → reason about cause → propose fix |
| Notification system | ✅ Phase 2c | Conflicts, new files, important changes |
| Intelligence layer (ask/query) | ⚠️ Partial | Works but context assembly has quality gaps |
| Voice interface | ❌ Planned | Wyoming/ESP32 satellite architecture designed |
| Home Assistant integration | ❌ Planned | Phase 3+ |
| Web dashboard (Vue3) | ❌ Planned | Frontend exists but limited |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TOWER SERVER                                 │
│  AMD Ryzen 9 24-core │ 96GB DDR6 │ RTX 3060 12GB │ RX 9070 XT  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ECHO BRAIN (port 8309)                      │    │
│  │                                                          │    │
│  │  FastAPI Application                                     │    │
│  │  ├── /api/echo/ask             (intelligence queries)    │    │
│  │  ├── /api/echo/memory          (memory management)       │    │
│  │  ├── /api/echo/search/domain   (domain knowledge)        │    │
│  │  ├── /api/echo/knowledge/facts (extracted facts)         │    │
│  │  ├── /api/echo/knowledge/stats (knowledge graph stats)   │    │
│  │  ├── /api/echo/notifications   (notification queue)      │    │
│  │  ├── /api/autonomous/*         (goal/task management)    │    │
│  │  ├── /api/workers/status       (worker monitoring)       │    │
│  │  └── /health                   (basic health check)      │    │
│  │                                                          │    │
│  │  Worker Scheduler (11 workers)                           │    │
│  │  ├── file_watcher           (10 min)    ← Phase 2c      │    │
│  │  ├── log_monitor            (15 min)                     │    │
│  │  ├── fact_extraction        (30 min)                     │    │
│  │  ├── reasoning_worker       (30 min)    ← Phase 2c      │    │
│  │  ├── conversation_watcher   (60 min)                     │    │
│  │  ├── domain_ingestor        (60 min)    ← Phase 2c      │    │
│  │  ├── codebase_indexer       (60 min)                     │    │
│  │  ├── schema_indexer         (60 min)                     │    │
│  │  ├── self_test_runner       (60 min)                     │    │
│  │  ├── improvement_engine     (2 hours)                    │    │
│  │  └── knowledge_graph        (daily)                      │    │
│  │                                                          │    │
│  │  Intelligence Layer                                      │    │
│  │  ├── Query Classification                                │    │
│  │  ├── Context Assembly (ParallelRetriever)                │    │
│  │  ├── LLM Reasoning (Ollama)                              │    │
│  │  └── Response Synthesis                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐     │
│  │  PostgreSQL   │  │   Qdrant     │  │     Ollama        │     │
│  │  (echo_brain) │  │  (6333)      │  │    (11434)        │     │
│  │  35+ tables   │  │  echo_memory │  │  mistral:7b       │     │
│  │  - knowledge  │  │  1024D vecs  │  │  nomic-embed-text │     │
│  │    facts      │  │  20,000+ pts │  │  gemma2:9b        │     │
│  └──────────────┘  └──────────────┘  └───────────────────┘     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ANIME PRODUCTION SYSTEM (secondary)                      │   │
│  │  ├── ComfyUI (port 8188)                                 │   │
│  │  ├── tower_anime database                                 │   │
│  │  └── LoRA training pipeline                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for detailed component documentation.
See [ROADMAP.md](./docs/ROADMAP.md) for the full development plan.
See [CHANGELOG.md](./CHANGELOG.md) for version history.

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

# Search domain knowledge (Phase 2c)
curl -s -X POST http://localhost:8309/api/echo/search/domain \
  -H "Content-Type: application/json" \
  -d '{"query": "LoRA training settings", "categories": ["anime:pipeline"]}'

# Check knowledge facts (Phase 2c)
curl -s http://localhost:8309/api/echo/knowledge/facts?limit=10 | python3 -m json.tool

# Get knowledge graph statistics (Phase 2c)
curl -s http://localhost:8309/api/echo/knowledge/stats | python3 -m json.tool

# Check notifications (Phase 2c)
curl -s http://localhost:8309/api/echo/notifications | python3 -m json.tool

# Check autonomous goals
sudo -u postgres psql echo_brain -c "SELECT name, status, priority FROM autonomous_goals ORDER BY priority DESC;"

# Check extracted facts (Phase 2c)
sudo -u postgres psql echo_brain -c "SELECT fact_text, fact_type, confidence FROM knowledge_facts LIMIT 5;"

# Check improvement proposals
sudo -u postgres psql echo_brain -c "SELECT title, status, risk_assessment FROM self_improvement_proposals WHERE status = 'pending';"
```

## Directory Structure

```
/opt/tower-echo-brain/
├── docs/
│   ├── ARCHITECTURE.md          # System architecture deep dive
│   ├── ROADMAP.md               # Development phases and milestones
│   └── prompts/
│       ├── PHASE1_AUTONOMOUS_CORE.md
│       ├── PHASE2_LEARNING_AUTONOMY.md
│       ├── PHASE2_FIX_AND_VERIFY.md
│       ├── PHASE2A_SELF_AWARENESS.md
│       └── PHASE2C_REASONING_PIPELINE.md   ← current
├── src/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   ├── endpoints/           # Route handlers
│   │   └── autonomous.py        # Autonomous system API
│   ├── intelligence/
│   │   └── reasoner.py          # Query processing pipeline
│   ├── context_assembly/
│   │   └── retriever.py         # Vector search + context building
│   ├── autonomous/
│   │   ├── core.py              # Autonomous orchestration
│   │   ├── goals.py             # Goal management
│   │   ├── safety.py            # Safety level enforcement
│   │   ├── audit.py             # Audit logging
│   │   ├── scheduler.py         # Task scheduling
│   │   ├── worker_scheduler.py  # Worker lifecycle management
│   │   └── workers/
│   │       ├── fact_extraction_worker.py
│   │       ├── conversation_watcher.py
│   │       ├── knowledge_graph_builder.py
│   │       ├── codebase_indexer.py      ← Phase 2a
│   │       ├── schema_indexer.py        ← Phase 2a
│   │       ├── log_monitor.py           ← Phase 2a
│   │       ├── self_test_runner.py      ← Phase 2a
│   │       └── improvement_engine.py    ← Phase 2a
│   ├── services/
│   │   └── embedding_service.py
│   └── pipeline/
│       └── context_layer.py
├── config/
│   └── self_tests.json          # Extensible self-test definitions
├── frontend/                    # Vue3 web dashboard
├── venv/                        # Python virtual environment
├── CHANGELOG.md
└── README.md                    # This file
```

## Configuration

| Variable | Value | Location |
|----------|-------|----------|
| DATABASE_URL | `postgresql://patrick:...@localhost/echo_brain` | systemd service env |
| QDRANT_URL | `http://localhost:6333` | Hardcoded in workers |
| OLLAMA_URL | `http://localhost:11434` | Hardcoded in workers |
| EMBEDDING_MODEL | `mxbai-embed-large` (1024D) | workers + retriever |
| REASONING_MODEL | `mistral:7b` | intelligence/reasoner.py |
| EXTRACTION_MODEL | `gemma2:9b` | fact_extraction_worker.py |
| SERVICE_USER | `echo` | systemd service |
| SERVICE_PORT | `8309` | systemd service |

## Contributing

Echo Brain is a personal project. Development happens through Claude Code prompts stored in `docs/prompts/`. Each prompt is versioned and gated — see [ROADMAP.md](./docs/ROADMAP.md) for the current development plan.
