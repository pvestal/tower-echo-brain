# Echo Brain Architecture

Version: 0.5.0
Last Updated: 2026-02-11

---

## Core Design Principle

Echo Brain operates on a three-stage loop:

```
INGEST → THINK → IMPROVE
  │         │         │
  │         │         └─→ Propose fixes, clean bad data, optimize queries
  │         └───────────→ Detect errors, validate output, reason about root causes
  └─────────────────────→ Index code, schemas, conversations, logs, external data
         ▲                                                          │
         └──────────────────────────────────────────────────────────┘
                              feedback loop
```

Without all three stages, the system is blind. Ingest without Think means data goes in but quality is unknown. Think without Improve means problems are detected but never fixed. The full loop makes Echo Brain self-correcting.

---

## Component Map

### 1. Data Layer

```
PostgreSQL (echo_brain)                    Qdrant (echo_memory)
┌────────────────────────────┐             ┌─────────────────────────┐
│ KNOWLEDGE                  │             │ 768D vectors            │
│  knowledge_facts (2,558)   │◄───────────►│ nomic-embed-text        │
│  conversations             │             │ 176,566 points          │
│  documents                 │             │                         │
│  graph_edges               │             │ Payload metadata:       │
│  extraction_coverage       │             │  source, type, text,    │
│                            │             │  file_path, functions   │
│ AUTONOMOUS                 │             └─────────────────────────┘
│  autonomous_goals          │
│  autonomous_tasks          │             Qdrant (story_bible)
│  autonomous_audit_log      │             ┌─────────────────────────┐
│  autonomous_notifications  │             │ Anime story/character   │
│                            │             │ reference collection    │
│ SELF-AWARENESS             │             └─────────────────────────┘
│  self_codebase_index       │
│  self_schema_index         │             Ollama (localhost:11434)
│  self_detected_issues      │             ┌─────────────────────────┐
│  self_test_results         │             │ mistral:7b   (reasoning)│
│  self_health_metrics       │             │ gemma2:9b    (extract.) │
│  self_improvement_proposals│             │ nomic-embed-text (emb.) │
│                            │             │ deepseek-r1:8b          │
│ DIAGNOSTICS                │             │ deepseek-coder-v2:16b   │
│  contract_monitor results  │             │ qwen2.5-coder:14b       │
│  diagnostic history        │             │ llava:7b / llava:13b    │
└────────────────────────────┘             │ llama3.2:3b             │
                                           └─────────────────────────┘
```

### 2. Worker System

Workers are the engine of the ingest→think→improve loop. Each worker implements `run_cycle()` and runs on a timer managed by `worker_scheduler.py`.

#### INGEST Workers (data in)

| Worker | Interval | What It Does | Safety |
|--------|----------|-------------|--------|
| `conversation_watcher` | 10 min | Scans `~/.claude/projects/` for new conversation exports, embeds and stores them | AUTO |
| `file_watcher` | 10 min | Watches models, workflows, outputs for changes | AUTO |
| `fact_extraction` | 30 min | Processes unprocessed vectors through Ollama to extract structured facts (subject/predicate/object) | AUTO |
| `domain_ingestor` | 60 min | Anime production, ComfyUI workflows, model metadata | AUTO |
| `codebase_indexer` | 6 hours | Parses Echo Brain's own Python source with `ast`, chunks, embeds into Qdrant. Detects file changes via SHA256. Also indexes anime pipeline code as secondary scope. | AUTO |
| `schema_indexer` | daily | Queries `information_schema` to map Echo Brain's own database structure, generates natural language descriptions, embeds them | AUTO |
| `knowledge_graph` | daily | Links facts by matching subjects/objects, creates `graph_edges` | AUTO |

#### THINK Workers (detect + validate)

| Worker | Interval | What It Does | Safety |
|--------|----------|-------------|--------|
| `contract_monitor` | 5 min | Tests critical API endpoints, tracks response times, detects breaking changes and API drift | AUTO |
| `log_monitor` | 15 min | Reads journalctl/log files for ERROR/WARNING/Traceback, classifies issues, stores in `self_detected_issues`, alerts on critical | AUTO |
| `reasoning_worker` | 30 min | Processes reasoning tasks, validates knowledge quality | AUTO |
| `self_test_runner` | 60 min | Fires known-good queries at own API, validates responses contain expected content, detects regressions, records pass rates | AUTO |

#### IMPROVE Workers (reason + propose)

| Worker | Interval | What It Does | Safety |
|--------|----------|-------------|--------|
| `improvement_engine` | 2 hours | Reads open issues from `self_detected_issues`, searches indexed codebase for relevant code, reasons about root cause via LLM, generates fix proposals stored in `self_improvement_proposals`. **NEVER auto-applies.** | AUTO (generate) / REVIEW (apply) |

### 3. Voice Service

The voice service provides speech-to-text, text-to-speech, and full voice chat with Echo Brain's reasoning pipeline.

```
Audio Input (mic/file)
    │
    ▼
Voice Activity Detection (Silero VAD)
    │  Segments speech from silence
    ▼
Speech-to-Text (Whisper large-v3, CUDA float16)
    │  Returns: text, language, confidence, segments
    ▼
Echo Brain Reasoning Pipeline
    │  Classify → Retrieve context → LLM reason → Respond
    ▼
Text-to-Speech (Piper en_US-lessac-medium, ONNX)
    │  Returns: WAV audio bytes (22050 Hz)
    ▼
Audio Output (browser playback)
```

**Endpoints:**
- `POST /api/echo/voice/transcribe` — STT only (file upload or base64)
- `POST /api/echo/voice/synthesize` — TTS only (returns WAV)
- `POST /api/echo/voice/synthesize/base64` — TTS (returns base64 JSON)
- `POST /api/echo/voice/chat` — Full pipeline: audio → text → reason → audio
- `GET /api/echo/voice/status` — Service health (STT loaded, TTS loaded, VAD available)
- `GET /api/echo/voice/voices` — Available Piper voice models
- `WS /api/echo/voice/ws` — Real-time streaming (bidirectional)

Voice conversations are persisted to Echo Brain memory via `voice_memory.py`.

### 4. Intelligence Layer

The intelligence layer handles user queries (via `POST /api/echo/ask`).

```
User Query
    │
    ▼
Query Classifier (src/context_assembly/classifier.py)
    │  Classifies: SYSTEM_QUERY, CODE_QUERY, GENERAL_KNOWLEDGE, etc.
    ▼
Context Assembly (src/context_assembly/retriever.py)
    │  ParallelRetriever searches Qdrant echo_memory
    │  limit=20, score_threshold=0.3
    │  Searches across ALL source types (facts, code, conversations, self_codebase)
    ▼
LLM Reasoning (Ollama mistral:7b)
    │  System prompt + assembled context + user query
    │  Generates response grounded in retrieved context
    ▼
Response (with confidence, sources, reasoning_time_ms)
```

### 5. Frontend Dashboard

Vue 3 + TypeScript single-page application served on port 8311.

| View | Route | Purpose |
|------|-------|---------|
| VoicePanel | `/echo-brain/voice/` | Full voice interface with recording, playback, mute/volume, collapsible debug panels, WS latency tracking |
| VoiceSimple | `/echo-brain/voice-simple/` | Minimal voice interface |
| VoiceTest | `/echo-brain/voice-test/` | Voice endpoint testing tool |
| SystemView | `/echo-brain/system/` | System diagnostics, workers, self-test triggers |
| AskView | `/echo-brain/ask/` | Intelligence query interface |
| MemoryView | `/echo-brain/memory/` | Vector search UI |
| EndpointsView | `/echo-brain/endpoints/` | API endpoint explorer |

API client: `frontend/src/api/echoApi.ts` — all endpoint bindings with proper base URL routing.

### 6. MCP Server

Model Context Protocol server at `/mcp` enables Claude Code integration.

**Available tools:**
- `search_memory` — Semantic vector search across echo_memory
- `get_facts` — Retrieve structured facts by topic

### 7. Safety System

Every autonomous action passes through the safety controller:

```
Task Created
    │
    ▼
Safety Level Check
    ├── AUTO      → Execute immediately, log to audit
    ├── NOTIFY    → Execute, create notification for Patrick
    ├── REVIEW    → Queue for approval, create notification, DO NOT execute
    └── FORBIDDEN → Reject, log attempt, alert
```

**Safety assignments:**

| Action Category | Level | Examples |
|----------------|-------|---------|
| Read filesystem | AUTO | Index code, read logs |
| Write to own DB/vectors | AUTO | Store facts, issues, metrics |
| Query own API | AUTO | Self-tests, contract monitoring |
| Generate text proposals | AUTO | LLM reasoning about fixes |
| Apply code changes | **REVIEW** | Improvement proposals — Patrick must approve |
| Delete vectors | **REVIEW** | Stale data cleanup |
| Modify systemd | **FORBIDDEN** | Never autonomous |

---

## Database Schema Map

### Core Knowledge Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `knowledge_facts` | Extracted knowledge triples (2,558 rows) | subject, predicate, object, confidence |
| `documents` | Source document metadata | title, source, content_hash |
| `conversations` | Chat history metadata | title, source, created_at |
| `graph_edges` | Fact-to-fact relationships | source_fact_id, target_fact_id, relationship_type |
| `extraction_coverage` | Tracks which vectors have been processed | source_collection, vector_id, facts_found |

### Autonomous System Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `autonomous_goals` | What Echo Brain is trying to accomplish | name, goal_type, status, priority |
| `autonomous_tasks` | Individual work items | goal_id, status, safety_level |
| `autonomous_audit_log` | Everything Echo Brain does autonomously | action, details, created_at |
| `autonomous_notifications` | Alerts for Patrick | title, message, read, task_id |

### Self-Awareness Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `self_codebase_index` | Map of own source code | file_path, file_hash, functions, classes, imports |
| `self_schema_index` | Map of own database | table_name, column_info, row_count, foreign_keys |
| `self_detected_issues` | Problems Echo Brain found in itself | issue_type, severity, title, stack_trace, suggested_fix |
| `self_test_results` | Query validation results | test_name, passed, expected_contains, actual_response |
| `self_health_metrics` | Performance tracking over time | metric_name, metric_value, recorded_at |
| `self_improvement_proposals` | Proposed code changes (REVIEW gated) | target_file, current_code, proposed_code, reasoning, status |

---

## Key Design Decisions

### Why Qdrant + PostgreSQL (not just one)?
PostgreSQL is the source of truth for structured data (facts, metadata, schemas). Qdrant enables semantic search — "find code related to model routing" doesn't work with SQL, but works great with vector similarity. They complement each other.

### Why nomic-embed-text (768D)?
Runs locally on Tower via Ollama, no API costs, no rate limits, no data leaving the network. 768D is a good balance between quality and storage. Originally used OpenAI 1536D, then mxbai-embed-large 1024D, then migrated to nomic-embed-text for better performance with Ollama.

### Why separate workers instead of one big loop?
Isolation. If the codebase indexer crashes, fact extraction keeps running. Each worker has its own error count, last run time, and can be independently restarted. The worker_scheduler handles lifecycle for all of them.

### Why REVIEW gate on improvements instead of auto-apply?
Trust is earned incrementally. Echo Brain can detect issues and reason about fixes, but applying code changes without human review is too risky at this stage. Once the improvement proposals prove reliable over time (tracked in the proposals table), the gate can be loosened.

### Why local Whisper + Piper for voice?
Same philosophy as embeddings — everything runs on Tower hardware (RX 9070 XT for Whisper CUDA, CPU for Piper). No cloud APIs, no latency, no data leaving the network. Whisper large-v3 provides excellent accuracy; Piper provides fast, natural-sounding TTS.
