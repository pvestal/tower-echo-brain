# Echo Brain Roadmap

Last Updated: 2026-02-12

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

### v0.5.0 — Voice Interface, Frontend Dashboard, Test Infrastructure
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

---

### v0.5.1 — Hybrid Search ← CURRENT
**Goal:** Improve retrieval quality with combined vector + keyword search.
**Status:** COMPLETE (Feb 12, 2026)

- [x] Qdrant full-text payload index on `content` field (word tokenizer, lowercase)
- [x] Qdrant keyword indexes on `type` and `category` fields
- [x] Hybrid search in `ParallelRetriever._search_qdrant()` — vector + text in parallel
- [x] Weighted score fusion: `0.7 × vector + 0.3 × text` (OpenClaw-inspired)
- [x] Text-only results surface hits that vector similarity missed
- [x] GENERAL domain now includes `echo_memory` (was empty → zero results)
- [x] ANIME domain expanded with checkpoint/safetensors/model name signals
- [x] OpenClaw analysis stored as KB fact

**Data milestones:**
- 194,921 vectors in Qdrant echo_memory (768D nomic-embed-text)
- 189,992 points with full-text index
- 6,129 structured facts in PostgreSQL
- 13 autonomous workers running

---

### v0.6.x — Close the Gaps (NEXT)
**Goal:** Address the critical gaps identified in the Feb 12 comprehensive audit.

#### Memory (Storing, Accessing, Forgetting)

| # | Milestone | Priority | Details |
|---|-----------|----------|---------|
| 1 | Memory decay / time-weighting | HIGH | All 194K vectors treated equally regardless of age. Implement time-decay scoring so recent context ranks higher. |
| 2 | Memory consolidation & dedup | HIGH | Similar vectors never merged. Some content appears 10+ times. Implement SHA-256 dedup on conversation ingestion + periodic consolidation worker. |
| 3 | Memory importance scoring | MEDIUM | No mechanism to weight critical facts higher than casual conversation snippets. Add importance field to payloads, score based on frequency of retrieval + explicit marking. |
| 4 | Pre-compaction session persistence | MEDIUM | When Claude's context window compresses, important session insights are lost. Add endpoint for "save session summary" before context loss. |
| 5 | Stale data cleanup worker | MEDIUM | Automated detection of outdated vectors (superseded facts, old conversation noise). REVIEW-gated deletion. |

#### Response & Dynamic Interactions

| # | Milestone | Priority | Details |
|---|-----------|----------|---------|
| 6 | Retrieval confidence gate | HIGH | LLM fabricates when retrieval is weak. Add threshold: if best score < 0.4, respond "I don't have specific information" instead of hallucinating. |
| 7 | Conflict detection in results | HIGH | Retrieval can return contradictory facts. Detect and flag before LLM synthesis. |
| 8 | Fact verification (post-generation) | MEDIUM | LLM response is never checked against sources. Add post-generation validation pass. |
| 9 | Multi-turn conversation context | MEDIUM | Each query is stateless. No conversation history carried between turns for follow-up questions. |
| 10 | Response streaming | LOW | Currently waits for full LLM response. Add SSE/streaming for better UX on voice and web. |

#### Thinking & Intelligence

| # | Milestone | Priority | Details |
|---|-----------|----------|---------|
| 11 | Chain-of-thought extraction | HIGH | `thinking_steps=[]` is never populated even with deepseek-r1. Extract `<think>` blocks from model output. |
| 12 | Multi-hop reasoning | HIGH | Currently single LLM call. For complex queries, implement: retrieve → reason → retrieve again → synthesize. |
| 13 | Clean up orphaned reasoning code | MEDIUM | `ReasoningEngine`, `IntelligenceEngine`, `UnifiedKnowledgeLayer` in `src/core/` are disconnected from the pipeline. Either integrate them or delete. |
| 14 | Agent system (real implementation) | LOW | `agent_service.py` returns mock responses. Either implement real agent collaboration or remove the pretense. |

#### Routing & Classification

| # | Milestone | Priority | Details |
|---|-----------|----------|---------|
| 15 | LLM-assisted classification | MEDIUM | Replace regex-only domain classifier with lightweight LLM classification for ambiguous queries. |
| 16 | Multi-intent query handling | MEDIUM | "Compare anime settings to tower architecture" → currently picks one domain. Support multi-domain queries. |
| 17 | Model availability verification | HIGH | Code assumes 5+ Ollama models loaded but RTX 3060 can hold ~1-2. Add runtime model check + graceful fallback chain. |

#### Infrastructure & Hygiene

| # | Milestone | Priority | Details |
|---|-----------|----------|---------|
| 18 | Fix embedding model mismatch | HIGH | Pipeline uses mxbai-embed-large (1024D), EmbeddingService defaults to nomic-embed-text (768D). Collection is 768D. Standardize everywhere. |
| 19 | Mount or delete 19 dead routers | MEDIUM | `src/api/` has 19 implemented but unmounted routers. Decision: mount needed ones, delete the rest. |
| 20 | Fix duplicate reasoning router mount | HIGH | `main.py` lines 177-198 mount two different reasoning routers under the same prefix. Second shadows first. |
| 21 | Conversation watcher embedding bug | HIGH | Worker uses `response.json()["embedding"]` but Ollama returns `["embeddings"]`. 0 chunks ingested. |
| 22 | Scrub credentials from source | MEDIUM | Database password in 11+ files as env var default. Move to Vault-only. |
| 23 | Complete requirements.txt | MEDIUM | Missing qdrant-client, torch, whisper, piper, and ~20 other dependencies. |
| 24 | Fix 9 syntax errors in video modules | LOW | `src/modules/generation/video/` — missing f-string prefixes. Dead code but should be fixed or deleted. |

#### Remaining from Original v0.6.x Plan

| # | Milestone | Priority | Details |
|---|-----------|----------|---------|
| 25 | Fact validation (second LLM pass) | MEDIUM | Verify extracted facts against source content. |
| 26 | Query regression suite expansion | MEDIUM | Currently 8 self-tests, target 50+ known-good Q&A pairs. |
| 27 | Extraction quality scoring | LOW | Rate each extracted fact, auto-discard low-quality. |

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
