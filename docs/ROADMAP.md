# Echo Brain Roadmap

Last Updated: 2026-02-11

---

## Version Scheme

Echo Brain uses semantic versioning: **MAJOR.MINOR.PATCH**

| Component | Meaning |
|-----------|---------|
| MAJOR | Fundamental capability shift (e.g., 0.x = pre-production, 1.0 = self-sustaining) |
| MINOR | Phase completion (new capability online) |
| PATCH | Bug fixes, worker fixes, data corrections within a phase |

---

## Version History & Phase Map

### v0.0.x — Phase 0: Foundation
**Goal:** Fix broken infrastructure, establish working data pipeline.
**Status:** COMPLETE (Jan 2026)

| Version | Milestone | Date | Details |
|---------|-----------|------|---------|
| v0.0.1 | Qdrant collections created | Jan 2026 | documents, conversations, facts, code |
| v0.0.3 | Migrated to local embeddings | Jan 2026 | OpenAI 1536D → mxbai-embed-large 1024D → nomic-embed-text 768D |
| v0.0.5 | Initial data load | Jan 2026 | 7,953 vectors indexed |

---

### v0.1.x — Phase 1: Autonomous Core
**Goal:** Build the safety-gated autonomous execution framework.
**Status:** COMPLETE (Jan 20, 2026)

- [x] Autonomous goals in PostgreSQL
- [x] Task queue with priorities
- [x] Safety level enforcement (AUTO/NOTIFY/REVIEW/FORBIDDEN)
- [x] Approval workflow for REVIEW tasks
- [x] Notification system
- [x] Persistence across restarts
- [x] Kill switch verified

---

### v0.2.x — Phase 2: Learning Autonomy
**Goal:** Make Echo Brain continuously learn from its data without manual intervention.
**Status:** COMPLETE (Feb 5, 2026)

| Version | Milestone | Date |
|---------|-----------|------|
| v0.2.0 | Worker scheduler operational | Feb 5 |
| v0.2.3 | `/api/workers/status` endpoint | Feb 5 |
| v0.2.4 | DDR5→DDR6 vector correction | Feb 5 |
| v0.2.6 | 13 autonomous goals seeded | Feb 5 |

- [x] Fact extraction worker (30 min cycles)
- [x] Conversation watcher (10 min cycles)
- [x] Knowledge graph builder (daily cycles)
- [x] Worker scheduler with independent failure handling

---

### v0.3.x — Phase 2a: Self-Awareness & Self-Improvement
**Goal:** Give Echo Brain the ability to know its own code, detect its own errors, validate its own output, and propose improvements.
**Status:** COMPLETE (Feb 6, 2026)

- [x] Self-awareness schema (6 tables)
- [x] Codebase indexer worker (6h cycles, AST parsing)
- [x] Schema indexer worker (daily)
- [x] Log monitor worker (15 min)
- [x] Self-test runner worker (60 min)
- [x] Improvement engine worker (2h, REVIEW gated)
- [x] Health dashboard endpoint (`/api/echo/health/detailed`)
- [x] Anime pipeline code indexed as secondary scope
- [x] Proposal review API (approve/reject)

---

### v0.4.x — Phase 2c/2d: Reasoning Pipeline & Self-Diagnostic
**Goal:** Enhanced reasoning, file watching, domain ingestion, comprehensive diagnostics.
**Status:** COMPLETE (Feb 2026)

| Version | Milestone | Date |
|---------|-----------|------|
| v0.4.0 | Reasoning pipeline | Feb 2026 |
| v0.4.1 | Enhanced self-diagnostic system | Feb 11 |

- [x] CLASSIFY → RETRIEVE → REASON → RESPOND pipeline
- [x] File watcher worker (10 min)
- [x] Domain ingestor worker (60 min)
- [x] Reasoning worker (30 min)
- [x] Contract monitor (5 min, tests 15+ endpoints)
- [x] Comprehensive system diagnostics
- [x] Database separation complete (tower_consolidated migrated)

---

### v0.5.0 — Voice Interface, Frontend Dashboard, Test Infrastructure ← CURRENT
**Goal:** Full voice pipeline, production-ready frontend, comprehensive test coverage.
**Status:** COMPLETE (Feb 11, 2026)

**Voice Interface:**
- [x] Whisper large-v3 STT (CUDA float16, AMD RX 9070 XT)
- [x] Piper TTS (en_US-lessac-medium, local ONNX)
- [x] Silero VAD (voice activity detection)
- [x] REST endpoints: transcribe, synthesize, chat
- [x] WebSocket real-time streaming (`/api/echo/voice/ws`)
- [x] Voice conversation persistence to memory
- [x] Voice service health/status endpoint

**Frontend Dashboard (Vue 3 + TypeScript):**
- [x] VoicePanel: full voice UI with recording, playback, mute/volume, debug panels
- [x] VoiceSimple: minimal voice interface with mute
- [x] VoiceTest: voice endpoint testing tool (dynamic URL, HTTPS-aware)
- [x] SystemView: diagnostics, workers, self-test triggers
- [x] AskView, MemoryView, EndpointsView
- [x] Frontend-to-backend endpoint audit — all API calls verified against live routes
- [x] Removed phantom endpoints and dead UI code

**Test Infrastructure:**
- [x] Pytest smoke suite: 31 tests (health, query, memory, embeddings, DB, voice)
- [x] Pact V4 consumer contracts: 11 tests (health, query, memory, ingestion, voice)
- [x] Test inventory documentation
- [x] Archived 5 stale/duplicate test files

**Data milestones:**
- 176,566 vectors in Qdrant echo_memory (768D nomic-embed-text)
- 2,558 structured facts in PostgreSQL
- 12 autonomous workers running (0 errors on 11 of 12)
- 176,566 embeddings created

---

### v0.6.x — Quality Feedback Loop (NEXT)
**Goal:** Close quality gaps in knowledge extraction and retrieval.

| Milestone | Details |
|-----------|---------|
| Fact validation | Second LLM pass to check extracted fact quality |
| Contradiction detector | Find conflicting facts automatically |
| Query regression suite | Expand self-tests to 50+ known-good Q&A pairs |
| Stale data cleanup | Automated detection and REVIEW-gated deletion |
| Extraction quality scoring | Rate each fact, discard low-quality |
| Embedding model evaluation | Compare nomic-embed-text vs alternatives |

---

### v0.7.x — Creation Autonomy (PLANNED)
**Goal:** Automate content creation pipelines.

| Milestone | Details |
|-----------|---------|
| Anime pipeline orchestration | Echo Brain triggers ComfyUI workflows from story context |
| LoRA training automation | Schedule and monitor training jobs |
| Content scheduling | Queue and prioritize generation tasks |
| Quality review | Auto-evaluate generated content |

---

### v0.8.x — External Integration (PLANNED)
**Goal:** Connect Echo Brain to external systems for proactive assistance.

| Milestone | Details |
|-----------|---------|
| Home Assistant bridge | Monitor sensors, control devices |
| Calendar intelligence | Scheduling awareness, proactive reminders |
| Frigate security | Camera event processing |
| Financial monitoring | Account alerts, spending patterns |

---

### v1.0.0 — Self-Sustaining Personal AI (VISION)
**Goal:** Echo Brain operates as a reliable personal assistant that learns, improves, and acts within safe boundaries.

**Criteria for v1.0.0:**
- [ ] Self-test pass rate > 95% sustained for 30 days
- [ ] Zero manual intervention needed for routine operations
- [ ] Improvement proposals accepted > 80% of the time
- [ ] Sub-5-second response time for knowledge queries
- [x] Voice interface operational
- [ ] 3+ external integrations active
- [ ] Full personal knowledge graph (10,000+ facts, 5,000+ edges)

---

## Development Workflow

Changes are made through Claude Code sessions on Tower. Each significant change is tested against the smoke suite and contract tests before committing.

```bash
# Run smoke tests
python3 -m pytest tests/echo_brain_smoke_test.py -v

# Run contract tests
cd contract-tests/consumer && npx vitest run --reporter=verbose

# Build frontend
cd frontend && npm run build
```
