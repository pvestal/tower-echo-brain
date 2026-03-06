# Changelog

All notable changes to Echo Brain are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [0.6.2] - 2026-03-06 (Autonomous Worker Fixes + Google Re-auth)

### Fixed
- **Telegram bot**: Was polling wrong bot (@VeteranGuardianBot). Updated Vault to @PatricksEchobot token, bot now receives and responds to messages
- **Photo vision 500s**: gemma3:12b wasn't loaded before batch started. Added warmup call in `_prepare_gpu_for_vision()`. Also round-trip all images through PIL to catch corrupt/unsupported formats before sending to Ollama
- **Improvement engine 404s**: Was requesting deleted `gemma2:9b` model. Changed to `mistral:7b`
- **Google Calendar credentials**: Was using bare access token without refresh capability. Now loads full credentials (refresh_token, client_id, client_secret) from `config/google_credentials.json`
- **Duplicate daily briefings**: In-memory `_last_briefing_date` reset on service restart. Added file-based marker at `/tmp/echo-brain-briefing-last-date` that survives restarts
- **Short video keyframes**: Videos under 10s extracted 0 frames (fps=1/10). Now extracts 1 frame at midpoint for short clips
- **Drive ingestion scope**: Re-authed Google OAuth via tower-auth with `drive.readonly` scope (was `drive.file`). Now sees full Drive contents

### Changed
- Self-test assertions updated: `checkpoint_mario` expects 'illustrious' (was 'realcartoonPixar'), relaxed `patrick_info` and `echo_brain_purpose` string matching, increased `vector_search_functional` timeout to 60s
- Vision API error logs now include response body for faster debugging
- Telegram bot logs incoming message chat_id and text for troubleshooting
- Drive ingestion note no longer hardcodes misleading scope warning

### Stats
- Self-tests: 10/16 → 11/16 (68.8%)
- Improvement engine: 0/5 → 5/5 proposals per cycle
- Photo vision: 100% failure → 0 errors (with warmup)
- Qdrant echo_memory: 606,072 vectors
- Facts: 6,077 structured facts
- Photos/videos scanned: 73,081

---

## [0.6.1] - 2026-02-16 (Retrieval Quality Overhaul + Garbage Cleanup)

### Fixed
- Text search AND→OR semantics: Qdrant `match.text` required ALL tokens present, so multi-word queries returned 0 results. Changed to `should` clauses with term overlap fraction scoring.
- Time decay formula: removed `stored_confidence` multiplier that penalized all results by 30%+ (default confidence 0.7). New formula: `score × (0.85 + 0.15 × decay_factor)` — max 15% penalty.
- Context compiler model limits: mistral:7b was set to 4096 tokens (actual: 32,768), leaving only 1,496 tokens for context. Fixed all models to actual context windows.
- Multi-hop reasoning trigger: changed from quantity check (`total_returned < 3`, never fires) to quality check (`best_score < 0.5`).
- LLM ignoring retrieved context: system prompt now has CRITICAL INSTRUCTION to use KNOWN FACTS. Compiler limits noisy hybrid sources to 5 when authoritative facts exist.

### Added
- Garbage vector filtering: `_is_readable_text()` in retriever rejects base64, binary, dense file paths before scoring. Checks space ratio (<2% = garbage), alphanumeric density (>85% = garbage), base64 regex.
- Ingestion pipeline readability filter: `is_readable_text()` in `scripts/ingest_claude_conversations.py` prevents non-readable content from being embedded. Applied at both message and chunk level.
- Adaptive text weight fallback: when 0 vector results survive garbage filter, text search gets full weight (1.0) instead of 0.15.
- Context compiler fact prioritization: ALL facts included first (authoritative), then hybrid/vector limited to 5 when facts present.

### Changed
- Domain classifier keyword filters expanded from 2 to 20+ per domain (ANIME, TECHNICAL, PERSONAL, SYSTEM).
- Domain classifier min_scores lowered, max_sources increased for better recall.
- Qdrant echo_memory: 360,745 → 317,222 vectors (43,523 garbage purged, 12.1% of collection).

---

## [0.6.0] - 2026-02-15 (Semantic Memory Architecture)

### Added
- Temporal decay + confidence scoring: `confidence`, `last_accessed`, `access_count` on all Qdrant payloads
- Logarithmic decay in retriever with usage-validated exemptions (access_count > 5)
- Daily DecayWorker for background confidence decay
- Semantic deduplication: inline (0.97 threshold) + background DedupWorker (0.98, every 6h)
- `src/core/dedup.py` — shared module for check_duplicate, merge_metadata, bump_existing_point
- Adaptive search weighting: KEYWORD (0.4/0.6), CONCEPTUAL (0.85/0.15), MIXED (0.7/0.3)
- Knowledge graph engine: NetworkX DiGraph over facts + graph_edges
- Graph API: `/api/echo/graph/related/{entity}`, `/path`, `/neighborhood/{entity}`, `/stats`, `POST /refresh`
- `explore_graph` MCP tool for Claude Code integration
- FactScrubber worker: cosine similarity + LLM verification (every 2h)
- Governor worker: conflict resolution via effective_score ranking (every 12h)
- `governor_decisions` audit table

---

## [0.5.1] - 2026-02-15 (Ollama Model Management)

### Added
- Ollama model management REST endpoints under `/api/models/ollama`:
  - `GET /ollama` — List all models with size, family, quantization, parameters
  - `GET /ollama/running` — Show models currently loaded in GPU/CPU memory
  - `POST /ollama/pull` — Pull new models with background progress tracking
  - `GET /ollama/pull-status` — Check progress of active/recent pulls
  - `POST /ollama/{name}/refresh` — Re-pull existing model (delta download)
  - `DELETE /ollama/{name}` — Delete model from Ollama and tower_models registry
- MCP tool `manage_ollama` with actions: list, running, pull, delete, refresh, pull_status, show
- Auto-sync pulled models into tower_models DB registry after download completes
- All Ollama operations logged to tower_model_logs for audit trail

### Changed
- MCP tools/list now returns 4 tools (added manage_ollama alongside search_memory, get_facts, store_fact)
- models_manager.py uses httpx for Ollama API calls (async, streaming support) instead of subprocess

---

## [0.5.0] - 2026-02-11 (Voice Interface, Frontend Dashboard, Test Infrastructure)

### Added
- Voice interface: Whisper large-v3 STT (CUDA float16) + Piper TTS (en_US-lessac-medium)
- Voice REST endpoints: transcribe, synthesize, synthesize/base64, chat
- Voice WebSocket endpoint for real-time bidirectional streaming (`/api/echo/voice/ws`)
- Voice service health and voice model listing endpoints
- Voice conversation persistence to Echo Brain memory
- VoicePanel.vue: full voice UI with recording, playback, mute/volume, collapsible debug panels, WS latency/reconnect tracking
- VoiceSimple.vue: minimal voice interface with mute
- VoiceTest.vue: voice endpoint testing tool with dynamic URL support
- Mute button and volume slider (GainNode-based) in voice interfaces
- Per-message debug panels showing query type, confidence, sources, audio stats
- STT debug panels showing detected language and confidence
- WebSocket status bar with ping latency and reconnect count
- Pact V4 consumer contract tests for voice endpoints (status, voices)
- Pytest smoke tests for voice endpoints (status, voices, synthesize)
- TEST_INVENTORY.md documenting all test files

### Changed
- SystemView.vue: removed dead UI for non-existent endpoints (moltbook, thinking log, etc.)
- echoApi.ts: removed phantom endpoint bindings, fixed paths to match actual backend routes
- VoiceTest.vue: replaced hardcoded localhost:8309 with dynamic URL computation
- getAllEndpoints(): rewritten with verified backend routes including voice endpoints
- Worker count: 11 → 12 (added contract_monitor at 5 min interval)
- Qdrant vectors: 124,872 → 176,566
- Renamed test_actual_integration.py → test_auth_integration.py with cleaned function names

### Removed
- moltbookApi (no backend routes existed)
- intelligenceApi.compareKnowledge, testUnderstanding, thinkingLog (non-existent endpoints)
- conversationsApi.health, conversationsApi.test (non-existent endpoints)
- reasoningApi.health (non-existent endpoint)
- 5 stale/duplicate test files archived to tests/tmp_archive/

### Fixed
- echoApi.ts: modelsList path /models/list → /models
- echoApi.ts: knowledgeMap path /intelligence/knowledge-map → /intelligence/map
- echoApi.ts: operationsJobs using wrong baseURL (prepended /api/echo) → absolute path
- getAllEndpoints() contained ~15 phantom routes that don't exist on the backend

---

## [0.4.1] - 2026-02-11 (Enhanced Self-Diagnostic System)

### Added
- Comprehensive self-diagnostic system (`src/core/self_diagnostic.py`)
- System service monitoring (ports, systemd units)
- Knowledge source validation (facts, vectors, conversations)
- Ingestion pipeline health checks
- Resource usage tracking
- API health testing
- Data quality detection (duplicates, low-confidence, stale data)
- Codebase analysis (TODOs, issues)

---

## [0.4.0] - 2026-02-11 (Reasoning Pipeline & Domain Ingestion)

### Added
- Reasoning pipeline: CLASSIFY → RETRIEVE → REASON → RESPOND
- File watcher worker (10 min cycles)
- Domain ingestor worker (60 min cycles)
- Reasoning worker (30 min cycles)
- Contract monitor worker (5 min cycles, tests 15+ endpoints)
- Database separation complete (tower_consolidated migrated)

---

## [0.3.0] - 2026-02-06 (Phase 2a: Self-Awareness)

### Added
- Self-awareness database schema (6 tables: self_codebase_index, self_schema_index, self_detected_issues, self_test_results, self_health_metrics, self_improvement_proposals)
- Codebase indexer worker — indexes own Python source with AST parsing, anime pipeline (secondary), Tower architecture files
- Schema indexer worker — indexes own database structure with natural language descriptions
- Log monitor worker — detects errors from own service logs, classifies severity
- Self-test runner worker — validates own query responses, detects regressions
- Improvement engine worker — reasons about issues, proposes fixes (REVIEW gated, never auto-applies)
- Proposal review API (GET /api/echo/proposals, POST approve/reject)
- Health dashboard endpoint (`/api/echo/health/detailed`) with full system status
- Anime pipeline code indexing (secondary scope in codebase_indexer)
- Project documentation: README.md, docs/ARCHITECTURE.md, docs/ROADMAP.md, CHANGELOG.md

### Changed
- Worker count: 3 → 8 (+ codebase_indexer, schema_indexer, log_monitor, self_test_runner, improvement_engine)
- Safety gates updated: improvement proposals are AUTO to generate, REVIEW to apply

---

## [0.2.6] - 2026-02-05

### Added
- 13 autonomous goals seeded in database
- GPU-specific vectors (RTX 3060 12GB, RX 9070 XT 16GB)

### Fixed
- DDR5→DDR6 hallucination: deleted incorrect DDR5 vector (score 0.768), added correct DDR6 facts

---

## [0.2.5] - 2026-02-05

### Fixed
- Worker status endpoint (`/api/workers/status`) returning 404 — moved before frontend mount

---

## [0.2.4] - 2026-02-05

### Fixed
- Database permissions: granted SELECT/INSERT/UPDATE/DELETE on worker tables to both `patrick` and `echo` users
- Added DATABASE_URL environment variable to systemd service

---

## [0.2.3] - 2026-02-05

### Fixed
- File permissions on `src/autonomous/workers/` — changed from `drwx------` to `drwxr-xr-x` for echo user access

---

## [0.2.2] - 2026-02-05

### Fixed
- Autonomous API router loading: commented out missing `executor.py` and `events.py` imports
- Placeholder classes for Executor and TaskResult in `__init__.py`
- Celery tasks imports commented out (agents.coding_agent doesn't exist)

---

## [0.2.1] - 2026-02-05

### Added
- Worker scheduler (`src/autonomous/worker_scheduler.py`)
- Startup/shutdown events in main.py for worker lifecycle

---

## [0.2.0] - 2026-02-05 (Phase 2: Learning Autonomy)

### Added
- Fact extraction worker (30 min cycles)
- Conversation watcher worker (10 min cycles)
- Knowledge graph builder worker (daily cycles)
- Worker scheduler with independent failure handling

---

## [0.1.6] - 2026-01-20

### Verified
- Kill switch tested and confirmed working

---

## [0.1.5] - 2026-01-20

### Added
- Goal/task state persistence across service restarts

---

## [0.1.4] - 2026-01-20

### Added
- Notification system: `autonomous_notifications` table and API endpoints
- GET `/api/autonomous/notifications` — list unread
- POST `/api/autonomous/notifications/{id}/read` — mark as read
- GET `/api/autonomous/notifications/count` — unread count

---

## [0.1.3] - 2026-01-20

### Fixed
- Safety levels now enforced in executor before task execution
- REVIEW tasks properly queued for approval (not auto-executed)
- FORBIDDEN tasks rejected with audit log entry

---

## [0.1.0] - 2026-01-20 (Phase 1: Autonomous Core)

### Added
- Autonomous goal management (PostgreSQL-backed)
- Task queue with priorities
- Basic scheduler
- Audit logging for all autonomous actions
- Safety controller skeleton (AUTO/NOTIFY/REVIEW/FORBIDDEN)

---

## [0.0.5] - 2026-01-06

### Added
- 7,953 vectors indexed across 4 collections (documents, conversations, code, facts)

---

## [0.0.3] - 2026-01-06

### Changed
- Migrated from OpenAI 1536D embeddings to local mxbai-embed-large 1024D
- Recreated Qdrant collections with correct dimensions

---

## [0.0.1] - 2026-01-06 (Phase 0: Foundation)

### Added
- FastAPI application on port 8309
- PostgreSQL `echo_brain` database with core schema
- Qdrant vector database with initial collections
- Ollama integration for local LLM inference
- Unified embedding service
