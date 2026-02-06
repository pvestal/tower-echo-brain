# Echo Brain Architecture

Version: 0.3.0
Last Updated: 2026-02-06

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
│ KNOWLEDGE                  │             │ 1024D vectors           │
│  facts (6,389+)            │◄───────────►│ mxbai-embed-large       │
│  conversations             │             │ 2,473+ points           │
│  documents                 │             │                         │
│  graph_edges               │             │ Payload metadata:       │
│  extraction_coverage       │             │  source, type, text,    │
│                            │             │  file_path, functions   │
│ AUTONOMOUS                 │             └─────────────────────────┘
│  autonomous_goals          │
│  autonomous_tasks          │             Ollama (localhost:11434)
│  autonomous_audit_log      │             ┌─────────────────────────┐
│  autonomous_notifications  │             │ gemma2:9b  (extraction) │
│                            │             │ mistral:7b (reasoning)  │
│ SELF-AWARENESS             │             │ mxbai-embed-large       │
│  self_codebase_index       │             │   (embeddings)          │
│  self_schema_index         │             └─────────────────────────┘
│  self_detected_issues      │
│  self_test_results         │
│  self_health_metrics       │
│  self_improvement_proposals│
└────────────────────────────┘
```

### 2. Worker System

Workers are the engine of the ingest→think→improve loop. Each worker implements `run_cycle()` and runs on a timer managed by `worker_scheduler.py`.

#### INGEST Workers (data in)

| Worker | Interval | What It Does | Safety |
|--------|----------|-------------|--------|
| `conversation_watcher` | 10 min | Scans `~/.claude/` for new conversation exports, embeds and stores them | AUTO |
| `fact_extraction` | 30 min | Processes unprocessed vectors through Ollama to extract structured facts (subject/predicate/object) | AUTO |
| `codebase_indexer` | 6 hours | Parses Echo Brain's own Python source with `ast`, chunks, embeds into Qdrant. Detects file changes via SHA256. Also indexes anime pipeline code as secondary scope. | AUTO |
| `schema_indexer` | daily | Queries `information_schema` to map Echo Brain's own database structure, generates natural language descriptions, embeds them | AUTO |
| `knowledge_graph` | daily | Links facts by matching subjects/objects, creates `graph_edges` | AUTO |

#### THINK Workers (detect + validate)

| Worker | Interval | What It Does | Safety |
|--------|----------|-------------|--------|
| `log_monitor` | 15 min | Reads journalctl/log files for ERROR/WARNING/Traceback, classifies issues, stores in `self_detected_issues`, alerts on critical | AUTO |
| `self_test_runner` | 60 min | Fires known-good queries at own API, validates responses contain expected content, detects regressions, records pass rates | AUTO |

#### IMPROVE Workers (reason + propose)

| Worker | Interval | What It Does | Safety |
|--------|----------|-------------|--------|
| `improvement_engine` | 2 hours | Reads open issues from `self_detected_issues`, searches indexed codebase for relevant code, reasons about root cause via LLM, generates fix proposals stored in `self_improvement_proposals`. **NEVER auto-applies.** | AUTO (generate) / REVIEW (apply) |

#### Data Flow Through Workers

```
Raw Data Sources
├── Claude conversation exports (.jsonl)
├── Echo Brain source code (.py)
├── Database schemas (information_schema)
├── Service logs (journalctl)
└── Anime pipeline code (secondary)
        │
        ▼
   ┌─────────┐
   │ INGEST  │  conversation_watcher, codebase_indexer, schema_indexer
   └────┬────┘
        │
        ▼
   ┌──────────────────────┐
   │ STORE                │  PostgreSQL (metadata) + Qdrant (vectors)
   │  facts, code chunks, │
   │  schema descriptions │
   └────┬─────────────────┘
        │
        ▼
   ┌─────────┐
   │  THINK  │  self_test_runner, log_monitor
   └────┬────┘
        │ detected issues, test failures, regressions
        ▼
   ┌──────────┐
   │ IMPROVE  │  improvement_engine
   └────┬─────┘
        │ proposals (REVIEW gated)
        ▼
   ┌──────────────────────┐
   │ PATRICK REVIEWS      │  approve / reject via API or DB
   └────┬─────────────────┘
        │ if approved
        ▼
   ┌──────────┐
   │  APPLY   │  git branch → apply patch → run self-tests → commit if pass
   └──────────┘
```

### 3. Intelligence Layer

The intelligence layer handles user queries (via `/api/echo/ask`).

```
User Query
    │
    ▼
Query Classifier (src/intelligence/reasoner.py)
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
Response
```

**Known issue:** The intelligence layer occasionally returns stale information (e.g., DDR5 instead of DDR6) because context assembly may not surface the highest-quality vectors. The self_test_runner detects this; the improvement_engine proposes fixes.

### 4. Safety System

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

**Phase 2a safety assignments:**

| Action Category | Level | Examples |
|----------------|-------|---------|
| Read filesystem | AUTO | Index code, read logs |
| Write to own DB/vectors | AUTO | Store facts, issues, metrics |
| Query own API | AUTO | Self-tests |
| Generate text proposals | AUTO | LLM reasoning about fixes |
| Apply code changes | **REVIEW** | Improvement proposals — Patrick must approve |
| Delete vectors | **REVIEW** | Stale data cleanup |
| Modify systemd | **FORBIDDEN** | Never autonomous |

### 5. Anime Production System (Secondary Scope)

The anime pipeline is a separate system that Echo Brain is gaining awareness of:

```
/opt/tower-anime/ (or wherever anime code lives)
├── ComfyUI workflows
├── LoRA training pipeline
├── Character consistency system
└── tower_anime database
```

Echo Brain indexes this code into the same `echo_memory` vector store with `source: "anime_pipeline"` metadata so it can answer questions about the anime system and eventually orchestrate it (Phase 3).

---

## Database Schema Map

### Core Knowledge Tables

| Table | Purpose | Key Columns | Rows |
|-------|---------|-------------|------|
| `facts` | Extracted knowledge triples | subject, predicate, object, confidence | 6,389+ |
| `documents` | Source document metadata | title, source, content_hash | varies |
| `conversations` | Chat history metadata | title, source, created_at | varies |
| `graph_edges` | Fact-to-fact relationships | source_fact_id, target_fact_id, relationship_type | building |
| `extraction_coverage` | Tracks which vectors have been processed | source_collection, vector_id, facts_found | building |

### Autonomous System Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `autonomous_goals` | What Echo Brain is trying to accomplish | name, goal_type, status, priority |
| `autonomous_tasks` | Individual work items | goal_id, status, safety_level |
| `autonomous_audit_log` | Everything Echo Brain does autonomously | action, details, created_at |
| `autonomous_notifications` | Alerts for Patrick | title, message, read, task_id |

### Self-Awareness Tables (Phase 2a)

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

### Why mxbai-embed-large (1024D) instead of OpenAI?
Runs locally on Tower, no API costs, no rate limits, no data leaving the network. 1024D is a good balance between quality and storage. Originally used OpenAI 1536D but migrated to local for self-sufficiency.

### Why separate workers instead of one big loop?
Isolation. If the codebase indexer crashes, fact extraction keeps running. Each worker has its own error count, last run time, and can be independently restarted. The worker_scheduler handles lifecycle for all of them.

### Why REVIEW gate on improvements instead of auto-apply?
Trust is earned incrementally. Echo Brain can detect issues and reason about fixes, but applying code changes without human review is too risky at this stage. Once the improvement proposals prove reliable over time (tracked in the proposals table), the gate can be loosened.

### Why index anime code as secondary, not primary?
Echo Brain needs to understand itself before it can orchestrate other systems. Self-awareness is the foundation — once it can debug its own workers, it can debug anime pipeline issues too. The indexer handles both, but Echo Brain's own code gets priority in the chunking and embedding pipeline.
