# Changelog

All notable changes to Echo Brain are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

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
