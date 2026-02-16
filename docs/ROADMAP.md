# Echo Brain Roadmap

Last Updated: 2026-02-16

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

### v0.5.1 — Hybrid Search
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

### v0.6.0 — Close the Gaps
**Goal:** Address critical gaps: temporal decay, semantic dedup, adaptive search, knowledge graph traversal, HMLR fact verification pipeline.
**Status:** COMPLETE (Feb 15, 2026)
**Inspired by:** Competitive analysis of Mem0, Khoj, HMLR, Memento-MCP, and 16 other AI memory systems.

#### Feature 1: Temporal Decay + Confidence Scoring ✅
- [x] `confidence`, `last_accessed`, `access_count` fields on all Qdrant payloads
- [x] Logarithmic decay in retriever (gentler than exponential, floor at 0.2)
- [x] Usage-validated vectors (access_count > 5) exempt from decay
- [x] Fire-and-forget access tracking after every search
- [x] Daily `DecayWorker` for background confidence decay
- [x] One-time `scripts/backfill_confidence.py` for existing vectors

#### Feature 2: Semantic Deduplication ✅
- [x] `src/core/dedup.py` — shared module (check_duplicate, merge_metadata, bump_existing_point)
- [x] Inline dedup check (threshold=0.97) wired into all 5 ingestion paths
- [x] Background `DedupWorker` (threshold=0.98) scans for near-duplicates every 6h
- [x] Merge strategy: keep higher access_count, earliest ingested_at, highest confidence

#### Feature 3: Adaptive Search Weighting ✅
- [x] `QueryType` enum: KEYWORD / CONCEPTUAL / MIXED
- [x] `_classify_query_type()` — regex-based signal detection
- [x] Weight presets: keyword=0.4/0.6, conceptual=0.85/0.15, mixed=0.7/0.3
- [x] `query_type` and `search_weights` returned in retriever response

#### Feature 4: Knowledge Graph + HMLR Pipeline ✅
- [x] `src/core/graph_engine.py` — NetworkX DiGraph over facts + graph_edges
- [x] Lazy loading, incremental refresh, full rebuild every 24h
- [x] `GET /api/echo/graph/related/{entity}`, `/path`, `/neighborhood/{entity}`, `/stats`, `POST /refresh`
- [x] Graph enrichment in retriever: top-5 entities → 1-hop traversal → graph_sources
- [x] `explore_graph` MCP tool for Claude Code integration
- [x] `FactScrubber` worker: cosine similarity + LLM verification of extracted facts (every 2h)
- [x] `Governor` worker: conflict resolution via effective_score (confidence×0.6 + recency×0.4), every 12h
- [x] `governor_decisions` audit table for transparency

---

### v0.6.1 — Retrieval Quality Overhaul ← CURRENT
**Goal:** Fix the complete retrieval pipeline — context was not reaching the LLM despite 317K+ vectors of knowledge.
**Status:** COMPLETE (Feb 16, 2026)
**Root cause:** 8 compounding issues across retriever, classifier, compiler, reasoner, and ingestion pipeline.

#### Fix 1: Text Search OR Semantics ✅
- [x] Qdrant `match.text` uses AND semantics — multi-word queries returned 0 results
- [x] Changed to `should` clauses (OR semantics) with term overlap fraction scoring
- [x] Queries like "why did we switch from chilloutmix" now return relevant results

#### Fix 2: Garbage Vector Filtering ✅
- [x] ~12% of echo_memory was base64/binary from Claude JSONL tool results
- [x] Added `_is_readable_text()` runtime filter in vector search (space ratio, alphanumeric density, base64 regex)
- [x] Purged 43,523 garbage vectors (360,745 → 317,222) via cleanup script
- [x] Added readability filter to ingestion pipeline (`scripts/ingest_claude_conversations.py`) at message + chunk level

#### Fix 3: Time Decay Formula ✅
- [x] Removed `stored_confidence` multiplier (default 0.7 penalized all results by 30%+)
- [x] New formula: `score × (0.85 + 0.15 × decay_factor)` — max 15% penalty for old content

#### Fix 4: Domain Classifier Expansion ✅
- [x] ANIME facts_filter expanded from 2 to 20+ keywords
- [x] All domain filters (TECHNICAL, PERSONAL, SYSTEM) similarly expanded
- [x] Lowered min_scores, increased max_sources

#### Fix 5: Context Compiler Fact Prioritization ✅
- [x] ALL facts included first (authoritative, compact)
- [x] Hybrid/vector sources limited to 5 when facts exist (prevents noise drowning signal)
- [x] Model token limits fixed to actual context windows (mistral:7b was 4096, actual 32K)

#### Fix 6: System Prompt Strengthened ✅
- [x] Added CRITICAL INSTRUCTION to use KNOWN FACTS when present
- [x] LLM no longer ignores retrieved context in favor of general knowledge

#### Fix 7: Multi-hop Trigger Fixed ✅
- [x] Changed from quantity check (`total_returned < 3`) to quality check (`best_score < 0.5`)
- [x] Now triggers second retrieval pass when initial results are low confidence

#### Fix 8: Adaptive Text Weight Fallback ✅
- [x] When 0 vector results survive garbage filter, text search gets full weight (1.0)
- [x] Prevents total retrieval failure when vector search is degraded

#### Remaining v0.6.x items

| # | Milestone | Priority | Details |
|---|-----------|----------|---------|
| 4 | Pre-compaction session persistence | MEDIUM | Save session summary before context loss |
| 9 | Multi-turn conversation context | MEDIUM | Carry conversation history between turns |
| 10 | Response streaming | LOW | SSE/streaming for better UX |
| 11 | Chain-of-thought extraction | HIGH | Extract `<think>` blocks from deepseek-r1 |
| 15 | Dead code cleanup | MEDIUM | Remove `reasoning_router.py` dead `/ask` endpoint, consolidate reasoning paths |
| 17 | Model availability verification | HIGH | Runtime model check + fallback chain |

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
