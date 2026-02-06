# Echo Brain Roadmap

Last Updated: 2026-02-06

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
**Status:** ✅ COMPLETE (Jan 2026)

| Version | Milestone | Date | Details |
|---------|-----------|------|---------|
| v0.0.1 | Qdrant collections created | Jan 2026 | documents, conversations, facts, code |
| v0.0.2 | OpenAI 1536D embeddings working | Jan 2026 | Unified embedding service |
| v0.0.3 | Migrated to 1024D mxbai-embed-large | Jan 2026 | Local embeddings, no API dependency |
| v0.0.4 | PostgreSQL schema finalized | Jan 2026 | facts, documents, conversations tables |
| v0.0.5 | 7,953 vectors indexed | Jan 2026 | Initial data load complete |

**Deliverables:**
- [x] Qdrant running with correct dimensions
- [x] Embedding service (local mxbai-embed-large)
- [x] PostgreSQL schema for documents/chunks/facts
- [x] Async Ollama calls (non-blocking)
- [x] Real data indexed (not synthetic)

---

### v0.1.x — Phase 1: Autonomous Core
**Goal:** Build the safety-gated autonomous execution framework.
**Status:** ✅ COMPLETE (Jan 20, 2026)

| Version | Milestone | Date | Details |
|---------|-----------|------|---------|
| v0.1.0 | Goal storage + task queue | Jan 2026 | PostgreSQL-backed goal/task system |
| v0.1.1 | Safety controller | Jan 2026 | AUTO/NOTIFY/REVIEW/FORBIDDEN levels |
| v0.1.2 | Audit logging | Jan 2026 | All autonomous actions logged |
| v0.1.3 | Safety enforcement in executor | Jan 20 | REVIEW/FORBIDDEN properly gated |
| v0.1.4 | Notification system | Jan 20 | PostgreSQL notifications, API endpoints |
| v0.1.5 | State persistence | Jan 20 | Goals/tasks survive restarts |
| v0.1.6 | Kill switch verified | Jan 20 | Emergency stop tested |

**Deliverables:**
- [x] Autonomous goals in PostgreSQL
- [x] Task queue with priorities
- [x] Safety level enforcement before execution
- [x] Approval workflow for REVIEW tasks
- [x] Notification system (poll `/api/autonomous/notifications/count`)
- [x] Persistence across restarts
- [x] Kill switch

---

### v0.2.x — Phase 2: Learning Autonomy
**Goal:** Make Echo Brain continuously learn from its data without manual intervention.
**Status:** ⚠️ PARTIAL (workers running, quality unvalidated)

| Version | Milestone | Date | Details |
|---------|-----------|------|---------|
| v0.2.0 | Worker scheduler operational | Feb 5 | 3 workers registered and running |
| v0.2.1 | File permissions fixed | Feb 5 | echo user can access worker files |
| v0.2.2 | Database permissions fixed | Feb 5 | patrick user granted worker table access |
| v0.2.3 | Worker status endpoint | Feb 5 | `/api/workers/status` returns live data |
| v0.2.4 | DDR5→DDR6 vector correction | Feb 5 | Deleted bad vector, added correct facts |
| v0.2.5 | GPU facts added | Feb 5 | RTX 3060 12GB + RX 9070 XT 16GB indexed |
| v0.2.6 | Autonomous goals seeded | Feb 5 | 13 active goals in database |

**Deliverables:**
- [x] Fact extraction worker (30 min cycles)
- [x] Conversation watcher (10 min cycles)
- [x] Knowledge graph builder (daily cycles)
- [x] Worker scheduler with independent failure handling
- [x] `/api/workers/status` endpoint
- [ ] Fact extraction coverage > 50% (currently unknown — workers just started)
- [ ] Knowledge graph with meaningful edges
- [ ] Pattern detection across facts
- [ ] Insight generation from patterns

**Known Issues at v0.2.6:**
- Workers are running but output quality is unvalidated
- No feedback loop — errors go to logs that nothing reads
- Intelligence layer sometimes returns DDR5 despite correct vectors
- Knowledge graph builder has SQL syntax issues
- No way to measure if fact extraction produces useful facts or garbage

---

### v0.3.x — Phase 2a: Self-Awareness & Self-Improvement ← CURRENT
**Goal:** Give Echo Brain the ability to know its own code, detect its own errors, validate its own output, and propose improvements.
**Status:** 🔨 IN PROGRESS

| Version | Milestone | Target | Details |
|---------|-----------|--------|---------|
| v0.3.0 | Self-awareness schema | Feb 6 | 6 new tables: self_codebase_index, self_schema_index, self_detected_issues, self_test_results, self_health_metrics, self_improvement_proposals |
| v0.3.1 | Codebase indexer worker | Feb 6 | Indexes `/opt/tower-echo-brain/src/` with AST parsing |
| v0.3.2 | Schema indexer worker | Feb 6 | Indexes own database structure |
| v0.3.3 | Log monitor worker | Feb 6 | Reads own logs, classifies errors, creates issues |
| v0.3.4 | Self-test runner worker | Feb 6 | Validates own query responses, detects regressions |
| v0.3.5 | Improvement engine worker | Feb 6 | Reasons about issues, proposes fixes (REVIEW gated) |
| v0.3.6 | Health dashboard endpoint | Feb 6 | `/api/echo/health/detailed` with full system status |
| v0.3.7 | Anime pipeline indexed | Feb 6 | Secondary scope — anime code in vector store |
| v0.3.8 | Proposal review API | Feb 7+ | Approve/reject proposals via API |
| v0.3.9 | Proposal application | Feb 7+ | Apply approved proposals: branch → patch → test → commit |

**Deliverables:**
- [ ] Echo Brain can answer "What files handle model routing?" with actual file paths
- [ ] Echo Brain can answer "What's broken right now?" with detected issues
- [ ] Echo Brain can answer "How would you fix the DDR5 problem?" with a specific proposal
- [ ] Self-test pass rate tracked over time
- [ ] Health dashboard shows system status at a glance
- [ ] Anime pipeline code searchable via vector store
- [ ] Improvement proposals stored with reasoning and risk assessment
- [ ] Patrick can approve/reject proposals

**The Ingest→Think→Improve Loop:**
```
v0.3.1-v0.3.2: INGEST (codebase_indexer, schema_indexer)
    Echo Brain reads its own source code and database schemas.
    It can now answer questions about its own architecture.

v0.3.3-v0.3.4: THINK (log_monitor, self_test_runner)
    Echo Brain detects errors in its own logs and validates its own output.
    It knows when something is wrong and tracks quality over time.

v0.3.5: IMPROVE (improvement_engine)
    Echo Brain connects issues to code, reasons about root causes,
    and generates fix proposals. Patrick reviews before anything is applied.

v0.3.8-v0.3.9: APPLY (gated)
    Approved proposals are applied via git branches with automatic
    rollback if self-tests fail after the change.
```

---

### v0.4.x — Phase 2b: Quality Feedback Loop (PLANNED)
**Goal:** Close the quality gap in existing workers.
**Status:** ❌ NOT STARTED

| Milestone | Details |
|-----------|---------|
| Fact validation | After extraction, second LLM pass checks: "Is this fact actually useful and correct?" |
| Contradiction detector | Find facts that contradict each other (DDR5 vs DDR6 situation) |
| Query regression suite | Expand self-tests to 50+ known-good Q&A pairs |
| Stale data cleanup | Automated detection and REVIEW-gated deletion of bad vectors |
| Extraction quality scoring | Rate each extracted fact, discard low-quality |

---

### v0.5.x — Phase 3: Creation Autonomy (PLANNED)
**Goal:** Automate content creation pipelines.
**Status:** ❌ NOT STARTED

| Milestone | Details |
|-----------|---------|
| Anime pipeline orchestration | Echo Brain triggers ComfyUI workflows based on story context |
| LoRA training automation | Schedule and monitor training jobs |
| Content scheduling | Queue and prioritize generation tasks |
| Quality review | Auto-evaluate generated content |

**Depends on:** Phase 2a (Echo Brain must understand the anime codebase first)

---

### v0.6.x — Phase 4: External Integration (PLANNED)
**Goal:** Connect Echo Brain to external systems for proactive assistance.
**Status:** ❌ NOT STARTED

| Milestone | Details |
|-----------|---------|
| Home Assistant bridge | Monitor sensors, control devices |
| Calendar intelligence | Scheduling awareness, proactive reminders |
| Frigate security | Camera event processing |
| Financial monitoring | Account alerts, spending patterns |
| Victron RV monitoring | Battery/solar status |

---

### v1.0.0 — Self-Sustaining Personal AI (VISION)
**Goal:** Echo Brain operates as a reliable personal assistant that learns, improves, and acts within safe boundaries.

**Criteria for v1.0.0:**
- [ ] Self-test pass rate > 95% sustained for 30 days
- [ ] Zero manual intervention needed for routine operations
- [ ] Improvement proposals accepted > 80% of the time
- [ ] Sub-5-second response time for knowledge queries
- [ ] Voice interface operational
- [ ] 3+ external integrations active
- [ ] Full personal knowledge graph (10,000+ facts, 5,000+ edges)

---

## Development Workflow

### How Changes Are Made

```
1. Patrick defines the goal in a conversation with Claude (claude.ai)
2. Claude creates a versioned prompt document (docs/prompts/PHASE_X.md)
3. Prompt is sent to Claude Code on Tower via:
   scp prompt.md tower:/opt/tower-echo-brain/docs/prompts/
   cd /opt/tower-echo-brain && cat docs/prompts/PHASE_X.md | claude
4. Claude Code executes with gate verification at each step
5. Changes are committed to git with version tag
6. Results are reviewed in next claude.ai conversation
```

### Gate System

Every prompt includes numbered gates (GATE 1, GATE 2, etc.) that must pass before proceeding. Gates verify:
- Prerequisites are met (services running, permissions correct)
- Schema changes applied successfully
- New code imports without errors
- Workers register and run without crashing
- End-to-end functionality works as expected

If a gate fails, Claude Code stops and reports what went wrong. No "probably works" allowed.

### Branch Strategy

```
main                    ← stable, all gates passed
├── phase-2a/self-awareness  ← current development
├── phase-2a/improvement     ← improvement engine work
└── hotfix/*            ← urgent fixes to main
```

Tags follow version numbers: `v0.3.0`, `v0.3.1`, etc.
