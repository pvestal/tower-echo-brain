# Changelog

All notable changes to Echo Brain are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [0.3.0] - 2026-02-06 (Phase 2a: Self-Awareness — IN PROGRESS)

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
