# Echo Brain Architecture

Version: 0.6.1
Last Updated: 2026-02-16

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
│  facts (6,129 SPO triples) │             │ 317,222 points          │
│  conversations             │             │                         │
│  documents                 │             │ Payload indexes:        │
│  graph_edges               │             │  content (full-text)    │
│  extraction_coverage       │             │  type (keyword)         │
│                            │             │  category (keyword)     │
│ AUTONOMOUS                 │             │                         │
│  autonomous_goals          │             │ Payload metadata:       │
│  autonomous_tasks          │             │  source, type, content, │
│  autonomous_audit_log      │             │  file_path, category,   │
│  autonomous_notifications  │             │  chunk_index, role      │
│                            │             └─────────────────────────┘
│ SELF-AWARENESS             │
│  self_codebase_index       │             Ollama (localhost:11434)
│  self_schema_index         │             ┌─────────────────────────┐
│  self_detected_issues      │             │ mistral:7b   (reasoning)│
│  self_test_results         │             │ gemma2:9b    (extract.) │
│  self_health_metrics       │             │ nomic-embed-text (emb.) │
│  self_improvement_proposals│             │ deepseek-r1:8b          │
│                            │             │ deepseek-coder-v2:16b   │
│ DIAGNOSTICS                │             │ qwen2.5-coder:14b       │
│  contract_monitor results  │             │ llava:7b / llava:13b    │
│  diagnostic history        │             │ llama3.2:3b             │
│  domain_ingestion_log      │             └─────────────────────────┘
│  domain_category_stats     │
└────────────────────────────┘
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

#### QUALITY Workers (v0.6.0 — verify, decay, dedup, resolve)

| Worker | Interval | What It Does | Safety |
|--------|----------|-------------|--------|
| `decay` | daily | Applies logarithmic confidence decay to stale, low-access vectors. Vectors with access_count > 5 are exempt. | AUTO |
| `dedup` | 6 hours | Scans for near-duplicate vectors (cosine > 0.98), merges metadata into survivor, deletes duplicate. | AUTO |
| `fact_scrubber` | 2 hours | Verifies facts against source vectors via cosine similarity + LLM check. Demotes low-quality facts. | AUTO |
| `governor` | 12 hours | Resolves conflicting facts (same subject+predicate, different objects) by effective_score ranking. Logs to `governor_decisions`. | AUTO |

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
    │  Classifies into 7 domains: TECHNICAL, ANIME, PERSONAL, SYSTEM,
    │  GENERAL, CREATIVE, FINANCIAL
    │  Method: regex fast path + embedding cosine similarity fallback
    │  Score fusion: 0.4 × regex + 0.6 × embedding
    │  Expanded keyword filters (20+ per domain as of v0.6.1)
    ▼
Context Assembly (src/context_assembly/retriever.py)
    │  ParallelRetriever runs in parallel:
    │  ├─ Qdrant HYBRID SEARCH (vector + text, adaptive weighting)
    │  │    ├─ Vector: cosine similarity, garbage-filtered (readability check)
    │  │    └─ Text: OR-semantics `should` clauses, scored by term overlap
    │  ├─ PostgreSQL FTS (claude_conversations)
    │  └─ PostgreSQL facts (subject/predicate/object triples)
    │  Domain filters prevent cross-contamination
    ▼
Context Compiler (src/context_assembly/compiler.py)
    │  Token budget management per model (actual context windows)
    │  ALL facts included first (authoritative, compact)
    │  Hybrid/vector sources limited to 5 when facts exist (reduce noise)
    │  Emergency trim if over budget
    ▼
LLM Reasoning (Ollama — model selected by intent)
    │  CODING → qwen2.5-coder:7b
    │  REASONING → deepseek-r1:8b
    │  PERSONAL → llama3.1:8b
    │  FACTUAL → mistral:7b
    │  CREATIVE → gemma2:9b
    │  System prompt with CRITICAL INSTRUCTION to use KNOWN FACTS
    │  + compiled context + user query
    ▼
Response (with confidence, sources, reasoning_time_ms)
```

**Multi-hop reasoning** triggers when `best_score < 0.5` or domain confidence is low — performs a second retrieval pass with refined query.

#### Hybrid Search (v0.5.1, overhauled v0.6.1)

Inspired by OpenClaw's memory architecture. The Qdrant search combines two retrieval methods:

```
Query → [Parallel]
         ├─ Vector similarity (cosine, 50 fetched, garbage-filtered to 30)
         │    └─ Readability filter: space ratio, alphanumeric density, base64 detection
         └─ Full-text keyword match (OR semantics via `should` clauses)
         │    └─ Score by term overlap fraction (not AND — partial matches surface)
         ↓
Adaptive Weighted Score Fusion (weights vary by query type)
    KEYWORD:    score = 0.4 × vector + 0.6 × text
    CONCEPTUAL: score = 0.85 × vector + 0.15 × text
    MIXED:      score = 0.7 × vector + 0.3 × text
         ↓
Fallback: If 0 vector results survive garbage filter, text gets full weight (1.0)
         ↓
Dedup by point_id → Time decay (max 15% penalty) → Top K
```

**v0.6.1 fixes (2026-02-16):**
- Text search changed from AND semantics (`match.text`) to OR semantics (`should` clauses) — queries with 5+ terms no longer return 0 results
- Vector search garbage filter (`_is_readable_text()`) — rejects base64, binary, dense file paths before scoring
- Time decay formula fixed: removed `stored_confidence` multiplier that penalized all results by 30%+
- 43,523 garbage vectors (12.1%) purged from collection (360,745 → 317,222)
- Ingestion pipeline readability filter prevents re-ingestion of non-readable content

#### Adaptive Search Weighting (v0.6.0)

Queries are classified as KEYWORD, CONCEPTUAL, or MIXED, and weights are adjusted:

```
"retriever.py"              → KEYWORD    (vector=0.4, text=0.6)
"how does echo brain learn" → CONCEPTUAL (vector=0.85, text=0.15)
"echo brain architecture"   → MIXED      (vector=0.7, text=0.3)
```

#### Temporal Decay + Confidence (v0.6.0, fixed v0.6.1)

Every vector now carries `confidence` (0.2–1.0), `last_accessed`, and `access_count`. Retrieval scoring:

```
score = base_score × (0.85 + 0.15 × decay_factor)
```

- Logarithmic decay: `decay = 1 / (1 + ln(1 + age_days/halflife))`
- Maximum penalty: 15% (old formula penalized 20%+ and multiplied by stored_confidence)
- Vectors with `access_count > 5` are exempt (usage-validated)
- After retrieval, `last_accessed` and `access_count` are updated (fire-and-forget)
- **v0.6.1 fix:** Removed `stored_confidence` multiplier from decay formula — was penalizing all results by 30%+ (default confidence 0.7)

#### Graph Enrichment (v0.6.0)

After scoring, the top-5 results are used to query the knowledge graph:

```
Top-5 sources → extract entities → 1-hop graph traversal
    → graph_sources (type="graph", score = confidence × 0.5)
    → appended to results before final sort
```

### 4b. Knowledge Graph Engine (v0.6.0)

NetworkX-backed directed graph over facts and graph_edges. Lazy-loaded on first query.

```
PostgreSQL facts (subject/predicate/object)
    + graph_edges (from_entity/to_entity/relation_type)
    ↓
NetworkX DiGraph (nodes = entities, edges = predicates with confidence)
    ↓
API: /api/echo/graph/related/{entity}  — BFS traversal
     /api/echo/graph/path              — shortest path
     /api/echo/graph/neighborhood      — ego subgraph
     /api/echo/graph/stats             — node/edge/density
     /api/echo/graph/refresh           — incremental reload
    ↓
MCP: explore_graph tool for Claude Code
```

Incremental refresh adds only new facts; full rebuild every 24h.

### 4c. HMLR Pipeline (v0.6.0)

Inspired by the HMLR (Hierarchical Memory with Learned Retrieval) pattern. Two workers maintain fact quality:

```
FactScrubber (every 2h)                Governor (every 12h)
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│ For each unverified fact:   │        │ For each (subject, predicate)    │
│  1. Re-embed fact text      │        │ with multiple different objects:  │
│  2. Cosine sim vs source    │        │  1. effective = conf×0.6 +       │
│     vector                  │        │     recency×0.4                  │
│  3. sim < 0.5 → auto-demote│        │  2. If spread < 0.1 → flag      │
│  4. sim 0.5-0.7 → LLM check│        │     for human review             │
│  5. sim > 0.7 → verified   │        │  3. Else keep winner, demote     │
│  6. GPU check: skip LLM if │        │     losers (conf × 0.3)          │
│     busy > 80%              │        │  4. Log to governor_decisions    │
└─────────────────────────────┘        └──────────────────────────────────┘
```

### 4d. Semantic Deduplication (v0.6.0)

Prevents duplicate vectors at ingestion time and cleans existing duplicates:

```
Inline (all ingestion paths):           Background (DedupWorker, every 6h):
  Before upsert:                          Scroll batches of 50 (with vectors):
    search(embedding, threshold=0.97)       search(vector, threshold=0.98)
    If match → merge_metadata + bump        If match → keep higher access_count
    If no match → proceed with upsert       Merge metadata, delete loser
```

Merge strategy: earliest `ingested_at`, sum `access_counts`, highest `confidence`.

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
- `search_memory` — Semantic vector search across echo_memory (now with confidence, query_type, adaptive weights)
- `get_facts` — Retrieve structured facts by topic
- `store_fact` — Store a new fact with confidence score
- `explore_graph` — Knowledge graph traversal: related entities, paths, neighborhood stats
- `manage_ollama` — List, pull, delete, refresh Ollama models

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

---

## Embedding Model Migration History

Echo Brain has been through three embedding models:

1. **OpenAI text-embedding-ada-002** (1536D) — Original model, required API calls and cost money per request.
2. **mxbai-embed-large** (1024D) — First local model via Ollama. Only supported a **512-token context window**, meaning most code, conversations, and docs were truncated before embedding. Poor retrieval quality.
3. **nomic-embed-text** (768D) — Current model. **8,192-token context window** (16x larger). Content is embedded fully, not truncated. Slight reduction in vector precision (768 vs 1024) more than offset by context improvement.

The migration to nomic-embed-text required deleting the old `echo_memory` collection (1024D vectors incompatible), recreating at 768D, and complete re-ingestion. All references to mxbai-embed-large were cleaned from the facts database.

**Note:** CLIP is used for image/video embeddings, NOT nomic-embed-text. nomic is exclusively for text.

---

## Context Contamination Bug (Resolved)

Echo Brain and Tower Anime Production originally shared database tables and Qdrant collections. When queried about its own architecture, Echo Brain would retrieve anime content (LoRA configs, character descriptions) and mix it into technical answers.

**Resolution:** Complete separation — Echo Brain uses the `echo_brain` database and `echo_memory` collection, anime production uses `anime_production` database with separate collections. Memory middleware filters searches to Echo Brain's own collections only. The shared `tower_consolidated` database is being phased out.

---

## Known Architectural Weaknesses

### Retrieval & Memory
1. ~~**No retrieval confidence gate**~~ — **PARTIALLY RESOLVED v0.6.1**: Multi-hop trigger fires when `best_score < 0.5`. Compiler prioritizes authoritative facts over noisy hybrid results. System prompt instructs LLM to use KNOWN FACTS. Still no hard "I don't know" gate.
2. ~~**No temporal ordering in vectors**~~ — **RESOLVED v0.6.0**: Vectors now carry `confidence`, `last_accessed`, `access_count`. Logarithmic time decay in retriever.
3. ~~**No memory decay**~~ — **RESOLVED v0.6.0**: Logarithmic decay with usage-validated exemptions. Daily DecayWorker.
4. ~~**No memory consolidation**~~ — **RESOLVED v0.6.0**: Semantic dedup (0.97 inline, 0.98 background) with metadata merge.
5. ~~**No SHA-256 dedup on ingestion**~~ — **RESOLVED v0.6.0**: All ingestion paths check for near-duplicates before upsert.
6. ~~**Garbage vectors in collection**~~ — **RESOLVED v0.6.1**: 43,523 base64/binary garbage vectors purged (12.1% of collection). Ingestion pipeline now filters non-readable content before embedding. Runtime readability filter in vector search as safety net.
7. ~~**Text search AND semantics**~~ — **RESOLVED v0.6.1**: `match.text` required ALL tokens present, so multi-word queries returned 0 results. Changed to `should` clauses (OR semantics) with term overlap scoring.
8. ~~**Time decay destroying scores**~~ — **RESOLVED v0.6.1**: Formula multiplied by `stored_confidence` (default 0.7), penalizing ALL results by 30%+. Removed confidence multiplier; now max 15% penalty.

### Reasoning & Intelligence
9. **No true reasoning** — "Reasoning" is retrieval + single LLM call. No chain-of-thought extraction, no iterative refinement. Multi-hop retrieval now triggers on low-confidence results (v0.6.1).
10. ~~**No conflict detection**~~ — **RESOLVED v0.6.0**: Governor worker resolves conflicting facts; retriever detects contradictions in search results.
11. ~~**No fact verification**~~ — **RESOLVED v0.6.0**: FactScrubber verifies facts against source vectors (cosine + LLM).
12. **Orphaned reasoning implementations** — `ReasoningEngine`, `IntelligenceEngine`, `UnifiedKnowledgeLayer` exist in `src/core/` but are NOT connected to the main pipeline. Creates confusion about "where does reasoning happen."
13. **Chain-of-thought placeholder** — `thinking_steps=[]` in reasoning_layer.py is never populated, even when using deepseek-r1.
14. ~~**LLM ignoring retrieved context**~~ — **RESOLVED v0.6.1**: System prompt now has CRITICAL INSTRUCTION to use KNOWN FACTS. Compiler limits noisy hybrid sources to 5 when facts exist, preventing signal drowning.
15. **Two competing `/ask` endpoints** — `echo_main_router.py` (mounted first) shadows `reasoning_router.py` at the same path. The latter is dead code. Should be removed or consolidated.

### Routing & Classification
16. ~~**Regex-only domain classification**~~ — **PARTIALLY RESOLVED v0.6.1**: Now uses regex fast path + embedding cosine similarity fallback (0.4 regex + 0.6 embedding fusion). Not LLM-based but significantly more robust. Keyword filters expanded to 20+ per domain.
17. **No multi-intent support** — Complex queries like "search my anime conversations and compare to tower architecture" get classified as one domain only.

### Infrastructure
18. ~~**Embedding model mismatch**~~ — **RESOLVED**: All paths now use `nomic-embed-text` (768D). Qdrant collection is 768D. Old references to `mxbai-embed-large` (1024D) have been cleaned up.
19. **Model availability unverified** — Code assumes 5+ Ollama models loaded, but RTX 3060 12GB can only hold ~1-2 at a time. No runtime fallback chain.
20. **19 unmounted routers** — Dead code in `src/api/` — routers implemented but never mounted in main.py.
21. **Agent system is mock** — `src/services/agent_service.py` returns hardcoded responses. Not functional.
22. **No authentication** — Open API with no access control, rate limiting, or API keys.
23. **Credentials in source** — Database password embedded as env var default in 11+ files.

---

## Failure Modes

### Qdrant Crash (port 6333)
- **Impact:** All semantic search and memory retrieval fails
- **Behavior:** Queries still work but responses degrade to generic LLM answers with no personalization
- **Recovery:** `sudo systemctl restart qdrant` — vectors persist on disk

### Ollama Crash (port 11434)
- **Impact:** Total failure — no LLM inference, no embeddings
- **Behavior:** All query endpoints return errors or timeouts
- **Recovery:** `sudo systemctl restart ollama` — models cached on disk

### PostgreSQL Crash (port 5432)
- **Impact:** No routing config, no conversation history, no facts
- **Behavior:** Queries may work with cached routing but cannot log or access facts
- **Recovery:** `sudo systemctl restart postgresql`

### GPU Resource Contention
- **RTX 3060 (12GB):** Ollama LLM inference
- **RX 9070 XT (16GB):** Whisper STT (CUDA), ComfyUI image generation, FramePack video
- Running heavy workloads on both GPUs simultaneously risks VRAM exhaustion

---

## Ingestion Pipeline

```
Source Document (markdown, code, conversation, JSON)
    │
    ▼
Chunking (segments < 6,000 chars with overlap)
    │
    ▼
Readability Filter (v0.6.1)
    │  Rejects: base64 blobs, binary content, dense file paths
    │  Checks: space ratio (>2%), alphanumeric density (<85%), base64 regex
    │  Applied at message level AND chunk level
    ▼
Deduplication (semantic cosine similarity, threshold 0.97)
    │
    ▼
Embedding (768D via nomic-embed-text / Ollama)
    │
    ▼
Storage (Qdrant echo_memory + PostgreSQL metadata)
    │
    ▼
Logging (domain_ingestion_log table)
```

**Key files:**
- Orchestrator: `src/ingestion/orchestrator.py`
- Chunking: `src/ingestion/chunker.py`
- Embedding: `src/services/embedding_service.py`
- Conversation ingestion: `scripts/ingest_claude_conversations.py`
- Readability filter: `is_readable_text()` in `scripts/ingest_claude_conversations.py` and `_is_readable_text()` in `src/context_assembly/retriever.py`
