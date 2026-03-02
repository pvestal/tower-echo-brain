# Echo Brain vs Perplexity AI — Feature Comparison

**Date:** 2026-02-27
**Echo Brain Version:** 0.6.1 (509K vectors, 213 endpoints, 5 agents)
**Perplexity Reference:** February 2026 feature set (post-Computer launch)

---

## Feature Matrix

| Capability | Perplexity AI | Echo Brain | Gap | Priority | Effort |
|---|---|---|---|---|---|
| **Web Search & Citations** | Real-time, 93.9% SimpleQA, Sonar models | `web_fetch` MCP tool (single URL, no search) | **CRITICAL** — no web search engine integration, no citation pipeline, no multi-source aggregation | P0 | 8-12h |
| **Deep Research (multi-step)** | Iterative search + reasoning + report, runs on Opus 4.6, HLE 21.1% | Single-turn `/ask` with vector context only | **CRITICAL** — no iterative research loop, no multi-step decomposition, no source aggregation | P0 | 20-30h |
| **Pro Search (enhanced)** | Multi-step search with thought streaming | Streaming SSE exists (`/ask/stream`) but no web search | **HIGH** — streaming infra exists, needs search backend | P1 | 12-16h |
| **Multi-Model Routing** | 19 models, dynamic per-task selection | 5 agents with fixed model assignment, fallback chains | **MEDIUM** — agent registry exists with hot-reload, needs dynamic routing based on task analysis rather than keyword intent | P1 | 10-15h |
| **Multi-Agent Orchestration** | Computer: decomposes goals into subtasks, assigns to specialized agents, runs hours/months | `collaborate_agents()` exists but agents are siloed, no task decomposition | **HIGH** — the `autonomous/` system has goals, tasks, executor, scheduler — this is the foundation for orchestration | P1 | 25-40h |
| **Memory & Personalization** | Cross-session, cross-model, structured preferences, transparency controls | 509K vectors, facts DB, knowledge graph, temporal decay, deduplication | **LOW** — Echo Brain's memory is more powerful than Perplexity's. Structured facts + graph > preference tags. Missing: memory transparency in responses (citing which memories were used) | P3 | 4-6h |
| **File Upload & Analysis** | PDF, Word, Excel, CSV, PowerPoint in Spaces | No file upload endpoint, no document parsing | **HIGH** — no document ingestion pipeline at all | P1 | 10-14h |
| **Finance Monitoring** | Real-time quotes, earnings hub, analyst ratings, FactSet/Morningstar data | `/api/finance/*` endpoints exist (accounts, balances, transactions via Plaid), daily briefing at 6:30 AM | **MEDIUM** — has Plaid integration for personal banking, missing: stock quotes, earnings, market analysis, portfolio tracking | P2 | 12-16h |
| **Voice Interface** | Mobile + "Hey Plex" + Samsung integration | `faster-whisper` STT + `piper-tts` TTS installed, `/api/echo/voice/*` endpoints exist | **MEDIUM** — backend exists, missing: always-listening wake word, mobile client, continuous conversation mode | P2 | 15-20h |
| **Image/Video Generation** | Nano Banana (images), Veo 3.1 (8s video) | Full anime production pipeline (ComfyUI, FramePack, Wan T2V), `trigger_generation` MCP tool | **NONE** — Echo Brain is far superior here. Multi-engine video, LoRA training, scene-based generation | - | - |
| **Autonomous Task Execution** | Computer: goal → subtasks → multi-model agents → hours of work | `autonomous/` system: goals, tasks, workers, safety gates, circuit breakers, scheduler | **MEDIUM** — architecture exists with 20 workers, needs: goal decomposition LLM, dynamic worker creation, longer-running task support | P1 | 20-30h |
| **Spaces / Workspaces** | Project workspaces with files, custom AI instructions, collaborators | Agent isolation by domain, no project workspace concept | **LOW** — single-user system, workspaces less valuable. Could map to "contexts" or "projects" | P3 | 8-10h |
| **Browser Integration** | Comet: AI-native Chromium, tab awareness, background assistant | None — API/CLI/Telegram only | **LOW for Patrick** — browser add-on could be useful but not core to self-hosted value prop | P3 | 20-30h |
| **Mobile Interface** | Native iOS/Android apps, camera queries | Telegram bot (polling), no native app | **MEDIUM** — Telegram covers basics. PWA would be the self-hosted approach | P2 | 15-20h |
| **Home/IoT Integration** | None | `home_assistant.py` integration exists (planned, code present), endpoints at `/api/home/*` | **Echo Brain advantage** — Perplexity cannot do this | P2 | 8-12h |
| **Security Cameras** | None | `tower-camera-service` running, Wyze Bridge active | **Echo Brain advantage** — Perplexity cannot do this | P2 | 10-15h |
| **Self-Hosted / Private** | Cloud-only, data retention policies | Full local control, all data on-premises | **Echo Brain advantage** — complete data sovereignty, zero subscription cost, no query limits | - | - |
| **Knowledge Graph** | None (flat memory + web search) | NetworkX graph, entity relationships, path finding, neighborhood stats | **Echo Brain advantage** — structured relationship understanding Perplexity lacks | - | - |
| **Code Intelligence** | None (uses external models) | Code indexing, dependency trees, codebase search, git operations | **Echo Brain advantage** — deep codebase awareness | - | - |
| **Anime/Creative Production** | Basic image/video gen | Full pipeline: 132+ routes, scene builder, LoRA training, multi-engine video, character management | **Echo Brain advantage** — specialized creative production system | - | - |
| **Custom Agent Isolation** | Model selection only | 5 specialized agents with domain-specific knowledge, hot-reload from markdown | **Echo Brain advantage** — true agent specialization with custom system prompts | - | - |
| **Self-Diagnostics** | None visible | 17 vector type contracts, circuit breakers, health monitoring, self-test suite | **Echo Brain advantage** — self-healing infrastructure | - | - |
| **Notification System** | Email + push (Tasks feature) | Telegram + ntfy + email, scheduled reminders | **PARITY** — both have multi-channel notifications | - | - |
| **API Access** | Sonar API, pay-per-token ($1-$15/1M tokens + request fees) | Full REST API (213 endpoints), MCP protocol (14 tools), free unlimited | **Echo Brain advantage** — unlimited free API access, MCP integration with Claude Code | - | - |
| **Enterprise / Multi-User** | SSO, audit logs, SCIM, shared Spaces, $40-325/user/mo | Single-user, no auth beyond Vault | **N/A** — single-user system, enterprise features not needed | P4 | - |

---

## Echo Brain's Unfair Advantages

Things Perplexity **fundamentally cannot do** due to its cloud architecture:

1. **Complete data sovereignty** — All 509K vectors, all conversations, all personal data stays on Patrick's hardware. No data retention policies, no third-party access, no terms of service changes.

2. **Home network integration** — Direct access to Home Assistant, Wyze cameras, local services. Perplexity can never control your lights or check your security cameras.

3. **Custom hardware acceleration** — Dual GPU (RTX 3060 12GB + RX 9070 XT 16GB), 96GB RAM, 24-core CPU. Dedicated resources with no throttling or rate limits.

4. **Direct database access** — PostgreSQL with personal financial data (Plaid), anime production data, structured facts. No API middleman.

5. **Specialized creative production** — Full anime pipeline with ComfyUI, FramePack, Wan T2V, LoRA training. 132+ production endpoints. Perplexity's 8-second Veo clips don't compare.

6. **Knowledge graph** — Structured entity relationships with path finding and neighborhood analysis. Perplexity has flat memory + web search, no relational understanding.

7. **Autonomous background workers** — 20 workers running continuously: fact extraction, conversation watching, codebase indexing, knowledge graph building, self-diagnostics, temporal decay. Perplexity's "Tasks" feature is scheduled searches, not autonomous intelligence.

8. **Zero marginal cost** — No per-query charges, no subscription tiers, no token metering. Every query is free after hardware investment.

9. **Code intelligence** — Deep codebase indexing, dependency analysis, git operations. Perplexity has no code-aware features.

10. **Full system observability** — Circuit breakers, vector health contracts, metrics tracking, self-repair. Perplexity is a black box.

---

## Perplexity Capabilities Worth Stealing

Ranked by **impact x feasibility** for self-hosted deployment:

### Tier 1 — High Impact, Feasible Now
1. **Web Search Integration** — SearXNG (self-hosted) or Brave Search API. The single biggest capability gap.
2. **Document Ingestion Pipeline** — Parse PDF/DOCX/CSV → chunk → embed → store. Libraries exist (unstructured, pypdf2, python-docx).
3. **Multi-Step Research Loop** — Decompose question → search → reason → search again → synthesize. Uses existing agent infrastructure.
4. **Dynamic Model Routing** — Route tasks to optimal models based on analysis, not just keyword matching. Agent registry already supports hot-reload.

### Tier 2 — High Impact, Moderate Effort
5. **Goal Decomposition Agent** — Take a complex goal, break into subtasks, assign to agents. The `autonomous/` system already has goals/tasks/executor.
6. **Citation Pipeline** — Track sources through the retrieval → reasoning → response chain. Show which memories/facts/web results informed each answer.
7. **Finance Enhancement** — Add stock quotes (Yahoo Finance API, free), earnings data, portfolio tracking on top of existing Plaid banking data.

### Tier 3 — Medium Impact
8. **File Upload Endpoint** — POST endpoint accepting multipart files, routing to appropriate parser.
9. **Streaming Research Progress** — Show intermediate search/reasoning steps via SSE (infrastructure already exists).
10. **Mobile PWA** — Progressive web app wrapping the Vue frontend for mobile access.

### Not Worth Stealing
- **Comet Browser** — Chromium fork is a massive undertaking with minimal payoff for single user.
- **Samsung/Hardware Integration** — Not relevant for self-hosted.
- **Enterprise Features** — Single-user system.
- **Shopping/Commerce** — Not a use case for Echo Brain.
