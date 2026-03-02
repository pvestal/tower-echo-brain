# Echo Brain Upgrade Roadmap

**Date:** 2026-02-27
**Baseline:** Echo Brain v0.6.1 (509K vectors, 213 endpoints, 5 agents, 20 autonomous workers)
**Goal:** Close critical gaps with Perplexity while leveraging self-hosted advantages

---

## Tier 1 — "Catch Perplexity" (Highest Impact, Enables Everything Else)

### 1.1 Web Search Integration

**What it does:** Gives Echo Brain the ability to search the live web, aggregate results from multiple sources, and cite them in responses.

**Why it matters:** Patrick currently has zero access to real-time information through Echo Brain. Every question about current events, documentation, prices, or news requires switching to a browser or Perplexity. This is the single largest capability gap — it blocks Deep Research, finance monitoring, and proactive alerting.

**Technical approach:**
- **Primary:** Self-hosted SearXNG instance (Docker, ~256MB RAM)
  - Aggregates results from Google, Bing, DuckDuckGo, Wikipedia, etc.
  - No API keys needed, no rate limits, no cost
  - JSON API at `http://localhost:8888/search?q=query&format=json`
- **Fallback:** Brave Search API ($0 for 2,000 queries/mo free tier, $3/1K after)
- **Integration point:** New `SearchService` in `src/services/search_service.py`
- **Pipeline:** Query → SearXNG → top-N results → fetch content (existing `web_fetch`) → chunk → embed → inject into context

**Dependencies:** Docker (already running for Wyze Bridge), nginx reverse proxy (already configured)

**Effort:** 8-12 hours

**Implementation sketch:**
```python
# src/services/search_service.py
class SearchService:
    def __init__(self, searxng_url="http://localhost:8888"):
        self.searxng_url = searxng_url
        self.brave_api_key = vault.get("brave_search_key")  # fallback

    async def search(self, query: str, num_results: int = 10,
                     categories: list[str] = ["general"]) -> list[SearchResult]:
        """Search web via SearXNG, fall back to Brave API."""
        params = {"q": query, "format": "json", "categories": ",".join(categories)}
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.searxng_url}/search", params=params)
            results = resp.json()["results"][:num_results]
        return [SearchResult(title=r["title"], url=r["url"],
                            snippet=r["content"], source=r.get("engine"))
                for r in results]

    async def search_and_fetch(self, query: str, max_pages: int = 5) -> list[WebDocument]:
        """Search, fetch top pages, chunk content."""
        results = await self.search(query)
        documents = []
        for result in results[:max_pages]:
            content = await self._fetch_and_clean(result.url)
            chunks = self._chunk_content(content, max_tokens=2000)
            documents.append(WebDocument(
                url=result.url, title=result.title,
                chunks=chunks, fetched_at=datetime.utcnow()
            ))
        return documents
```

**New endpoints:**
- `POST /api/echo/search/web` — Direct web search
- `GET /api/echo/search/web?q=...` — Quick web search

**MCP tool addition:**
- `web_search(query, num_results, categories)` — Search the web

**SearXNG deployment:**
```yaml
# docker-compose.searxng.yml
services:
  searxng:
    image: searxng/searxng:latest
    ports:
      - "8888:8080"
    volumes:
      - ./config/searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888
    restart: unless-stopped
```

---

### 1.2 Multi-Step Research Engine (Deep Research)

**What it does:** Takes a complex question, decomposes it into sub-questions, iteratively searches (web + memory), reasons through findings, and produces a cited research report.

**Why it matters:** This is Perplexity's killer feature (HLE 21.1%). Patrick needs this for: evaluating technical approaches, understanding new technologies, researching hardware/software decisions, investigating system issues with context from both local knowledge and the web.

**Technical approach:**
- **Orchestrator:** New `ResearchEngine` class using LangGraph (already in requirements)
- **Flow:** Question → Decompose (LLM) → [Search Web + Search Memory] per sub-question → Evaluate sufficiency → Iterate or Synthesize → Report with citations
- **Model:** Use deepseek-r1:8b for decomposition/reasoning, mistral:7b for synthesis. Optionally route to Claude API for complex research via existing Anthropic key.
- **Progress streaming:** SSE events at each stage (decomposing, searching, reasoning, synthesizing)

**Dependencies:** Web Search (1.1), existing LangGraph dependency, existing SSE infrastructure

**Effort:** 20-30 hours

**Implementation sketch:**
```python
# src/services/research_engine.py
class ResearchEngine:
    """Multi-step iterative research with web + memory search."""

    MAX_ITERATIONS = 5

    async def research(self, question: str, depth: str = "standard") -> ResearchReport:
        # Step 1: Decompose into sub-questions
        sub_questions = await self._decompose(question)

        # Step 2: Iterative search loop
        findings = []
        for iteration in range(self.MAX_ITERATIONS):
            for sq in sub_questions:
                # Parallel: web search + memory search
                web_results, memory_results = await asyncio.gather(
                    self.search_service.search_and_fetch(sq.query),
                    self.memory_service.search_memories(sq.query, limit=5)
                )
                findings.append(Finding(
                    question=sq, web_sources=web_results,
                    memory_sources=memory_results
                ))

            # Step 3: Evaluate — do we have enough to answer?
            evaluation = await self._evaluate_sufficiency(question, findings)
            if evaluation.sufficient:
                break
            # Generate follow-up questions for next iteration
            sub_questions = evaluation.follow_up_questions

        # Step 4: Synthesize report with citations
        report = await self._synthesize(question, findings)
        return report

    async def _decompose(self, question: str) -> list[SubQuestion]:
        """Use reasoning model to break question into searchable sub-questions."""
        prompt = f"""Break this question into 3-5 specific, searchable sub-questions:
        Question: {question}
        Return JSON: [{{"query": "...", "purpose": "..."}}]"""
        response = await self.llm.generate("reasoning", prompt)
        return [SubQuestion(**sq) for sq in json.loads(response)]

    async def _evaluate_sufficiency(self, question, findings) -> Evaluation:
        """Determine if gathered evidence is sufficient to answer."""
        prompt = f"""Given the original question and these findings,
        is there enough information to write a comprehensive answer?
        Question: {question}
        Findings: {self._format_findings(findings)}
        If not, what specific follow-up questions would fill the gaps?"""
        return await self.llm.generate("reasoning", prompt)

    async def _synthesize(self, question, findings) -> ResearchReport:
        """Produce final report with inline citations."""
        prompt = f"""Write a comprehensive answer with citations.
        Use [1], [2] etc. for web sources, [M1], [M2] for memory sources.
        Question: {question}
        Evidence: {self._format_findings(findings)}"""
        response = await self.llm.generate("general", prompt)
        return ResearchReport(
            question=question, answer=response,
            sources=[f.sources for f in findings],
            iterations=len(findings)
        )
```

**New endpoints:**
- `POST /api/echo/research` — Start deep research (returns job ID)
- `GET /api/echo/research/{job_id}` — Get research status/results
- `GET /api/echo/research/{job_id}/stream` — SSE stream of progress

**MCP tool addition:**
- `deep_research(question, depth)` — Run multi-step research

---

### 1.3 Memory Intelligence Extraction

**What it does:** Transforms raw vector data (509K points) into structured, actionable intelligence: patterns, preferences, recurring themes, temporal trends, and proactive insights.

**Why it matters:** Echo Brain has 509K vectors but most are raw conversation fragments. The fact extraction worker exists but runs infrequently. Patrick's preferences, decisions, and patterns are buried in noise. Perplexity's memory, while simpler (structured preferences), is more immediately useful because it's curated.

**Technical approach:**
- **Enhance existing workers:** `fact_extraction_worker.py` (30 min), `knowledge_graph_builder.py` (daily)
- **New: Pattern Extraction Worker** — Analyzes clusters of related memories to identify recurring patterns
- **New: Preference Learner** — Extracts and maintains structured preferences (tools, approaches, opinions)
- **New: Temporal Trend Analyzer** — Identifies what topics are trending in recent conversations vs historical
- **Consolidation pipeline:** Raw vectors → deduplicated → fact-extracted → graph-connected → pattern-analyzed → preference-updated

**Dependencies:** Existing autonomous worker infrastructure, existing fact extraction

**Effort:** 15-20 hours

**Implementation sketch:**
```python
# src/autonomous/workers/pattern_extraction_worker.py
class PatternExtractionWorker(BaseWorker):
    """Extract recurring patterns from memory clusters."""
    interval = timedelta(hours=2)

    async def execute(self):
        # 1. Get recent high-traffic memory clusters
        clusters = await self.memory.get_frequent_topics(
            window_days=30, min_access_count=3
        )

        for cluster in clusters:
            # 2. Retrieve all memories in cluster
            memories = await self.memory.search(cluster.topic, limit=50)

            # 3. Extract patterns via LLM
            patterns = await self.llm.generate("reasoning", f"""
                Analyze these {len(memories)} memories about "{cluster.topic}".
                Extract:
                1. Recurring preferences/decisions
                2. Evolution over time (did opinions change?)
                3. Unresolved questions or contradictions
                4. Actionable insights
                Return structured JSON.
            """)

            # 4. Store as high-confidence facts
            for pattern in patterns:
                await self.facts.upsert(
                    subject=pattern.entity,
                    predicate=pattern.relationship,
                    object=pattern.value,
                    confidence=pattern.confidence,
                    source="pattern_extraction"
                )

# src/autonomous/workers/preference_learner.py
class PreferenceLearner(BaseWorker):
    """Maintain structured preference profile from memory analysis."""
    interval = timedelta(hours=6)

    PREFERENCE_CATEGORIES = [
        "tools", "languages", "frameworks", "approaches",
        "communication_style", "workflow", "hardware", "food",
        "entertainment", "schedule"
    ]

    async def execute(self):
        for category in self.PREFERENCE_CATEGORIES:
            memories = await self.memory.search(
                f"Patrick prefers {category}", limit=20
            )
            if not memories:
                continue

            current_prefs = await self.facts.get(
                subject="patrick", predicate=f"prefers_{category}"
            )

            updated = await self.llm.generate("reasoning", f"""
                Current known preferences for {category}: {current_prefs}
                Recent relevant memories: {memories}
                Update the preference list. Only include preferences
                with clear evidence. Mark confidence level.
            """)

            await self.facts.upsert(
                subject="patrick",
                predicate=f"prefers_{category}",
                object=updated,
                confidence=0.85,
                source="preference_learner"
            )
```

---

### 1.4 Document Ingestion Pipeline

**What it does:** Accepts file uploads (PDF, DOCX, CSV, images, code files), parses them, chunks intelligently, embeds, and stores in vector memory with metadata.

**Why it matters:** Patrick can't feed documents to Echo Brain. Research papers, datasheets, RV manuals, equipment specs — all need to be queryable. Perplexity Spaces allow document upload; Echo Brain should too.

**Technical approach:**
- **Parser library:** `unstructured` (supports PDF, DOCX, XLSX, CSV, HTML, images with OCR, markdown)
- **Chunking:** Semantic chunking (by section/paragraph) rather than fixed-size. Use heading hierarchy for context.
- **OCR:** Tesseract (already available in most Linux distros) for scanned PDFs and images
- **Code files:** Language-aware chunking (by function/class) using tree-sitter
- **Metadata:** Preserve filename, page number, section heading, document type

**Dependencies:** `pip install unstructured[all-docs]` (~200MB), Tesseract for OCR

**Effort:** 10-14 hours

**Implementation sketch:**
```python
# src/services/document_service.py
from unstructured.partition.auto import partition

class DocumentService:
    SUPPORTED_TYPES = {
        ".pdf", ".docx", ".doc", ".xlsx", ".csv", ".txt",
        ".md", ".html", ".json", ".py", ".js", ".ts", ".yaml"
    }

    async def ingest_file(self, file: UploadFile,
                          collection: str = "echo_memory") -> IngestResult:
        """Parse, chunk, embed, and store a document."""
        # 1. Save to temp file
        tmp_path = await self._save_temp(file)

        # 2. Parse with unstructured
        elements = partition(filename=str(tmp_path))

        # 3. Chunk semantically
        chunks = self._semantic_chunk(elements, max_tokens=1500)

        # 4. Embed and store
        stored = 0
        for chunk in chunks:
            embedding = await self.embedding_service.embed(chunk.text)
            await self.qdrant.upsert(
                collection_name=collection,
                points=[PointStruct(
                    id=uuid4().hex,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "content": chunk.text,  # for full-text search
                        "type": "document",
                        "source": file.filename,
                        "page": chunk.page_number,
                        "section": chunk.section_heading,
                        "document_type": Path(file.filename).suffix,
                        "created_at": datetime.utcnow().isoformat(),
                        "confidence": 0.9
                    }
                )]
            )
            stored += 1

        return IngestResult(
            filename=file.filename, chunks=stored,
            pages=max(c.page_number for c in chunks if c.page_number)
        )

    def _semantic_chunk(self, elements, max_tokens=1500):
        """Group elements by section, respecting token limits."""
        chunks, current = [], []
        current_section = None
        current_tokens = 0

        for el in elements:
            if el.category == "Title":
                if current:
                    chunks.append(Chunk(
                        text="\n".join(current),
                        section_heading=current_section,
                        page_number=el.metadata.page_number
                    ))
                current = []
                current_section = el.text
                current_tokens = 0

            tokens = len(el.text.split()) * 1.3  # rough estimate
            if current_tokens + tokens > max_tokens:
                chunks.append(Chunk(text="\n".join(current),
                                   section_heading=current_section,
                                   page_number=el.metadata.page_number))
                current = [el.text]
                current_tokens = tokens
            else:
                current.append(el.text)
                current_tokens += tokens

        if current:
            chunks.append(Chunk(text="\n".join(current),
                               section_heading=current_section))
        return chunks
```

**New endpoints:**
- `POST /api/echo/documents/upload` — Upload and ingest a file
- `POST /api/echo/documents/upload-batch` — Upload multiple files
- `GET /api/echo/documents` — List ingested documents
- `GET /api/echo/documents/{doc_id}` — Get document metadata
- `DELETE /api/echo/documents/{doc_id}` — Remove document from memory

**MCP tool addition:**
- `ingest_document(file_path)` — Ingest a local file into memory

---

## Tier 2 — "Beat Perplexity" (Things Only Self-Hosted Can Do)

### 2.1 Home Assistant Bridge

**What it does:** Connects Echo Brain to Home Assistant for smart home awareness, control, and proactive monitoring.

**Why it matters:** Patrick has Home Assistant planned but not implemented. This gives Echo Brain environmental awareness — temperature, lighting, device states, energy usage, presence detection. "Is anyone home?" "Turn off the garage lights." "What's the power draw right now?" Perplexity can never do this.

**Technical approach:**
- `src/integrations/home_assistant.py` already exists with code structure
- Home Assistant REST API + WebSocket for real-time events
- **Event stream:** Subscribe to HA event bus → filter relevant events → store as temporal facts
- **Entity registry:** Cache HA entity list, expose as queryable knowledge
- **Action execution:** Map natural language to HA service calls (with safety gates from `autonomous/safety.py`)

**Dependencies:** Home Assistant instance (needs setup), existing `home_assistant.py` integration code

**Effort:** 8-12 hours (assuming HA is running)

**New endpoints (already stubbed):**
- `GET /api/home/entities` — List HA entities
- `POST /api/home/action` — Execute HA action
- `GET /api/home/state/{entity_id}` — Get entity state

---

### 2.2 Security Camera Intelligence (Frigate)

**What it does:** Integrates Frigate NVR for AI-powered security camera monitoring with event detection, person recognition, and proactive alerts.

**Why it matters:** Wyze Bridge is already running in Docker. Frigate adds AI object detection (persons, vehicles, animals). Echo Brain can answer "Who was at the front door at 3pm?" or proactively alert "Unknown person detected in driveway."

**Technical approach:**
- **Frigate:** Docker container with NVIDIA GPU access (RTX 3060 can handle detection alongside ComfyUI)
- **Event pipeline:** Frigate MQTT events → Echo Brain event listener → classify → store/alert
- **Snapshot analysis:** CLIP embeddings of camera snapshots for semantic search
- **Query interface:** "Show me the last person at the front door" → Frigate API + snapshot + CLIP match

**Dependencies:** Frigate setup, MQTT broker (Mosquitto), camera feeds from Wyze Bridge

**Effort:** 10-15 hours

---

### 2.3 Proactive Monitoring Loop

**What it does:** Continuously monitors financial accounts, security cameras, home state, and system health — proactively notifying Patrick of important changes rather than waiting to be asked.

**Why it matters:** Perplexity's "Tasks" feature does scheduled searches. Echo Brain can do true proactive monitoring with direct access to personal data. "Your electric bill spiked 40% this month." "Unusual login attempt on your server." "The garage door has been open for 2 hours."

**Technical approach:**
- **New autonomous worker:** `proactive_monitor.py` running every 5 minutes
- **Monitor types:**
  - Financial: Plaid balance changes > threshold, unusual transactions
  - Security: Frigate events outside normal patterns
  - Home: HA state anomalies (door open too long, temperature out of range)
  - System: Service failures, disk space, GPU temperature
- **Alert routing:** Severity → channel mapping (critical: Telegram + ntfy, info: daily briefing)
- **Uses existing:** notification_service.py, daily_briefing_worker.py

**Dependencies:** Frigate (2.2), Home Assistant (2.1), existing Plaid + notification infrastructure

**Effort:** 12-16 hours

---

### 2.4 Personal Data Indexing

**What it does:** Ingests and indexes personal data exports (Google Takeout, Claude conversations, browser bookmarks, email archives) into searchable vector memory.

**Why it matters:** Patrick has years of digital history across Google, Claude, and other services. Making it searchable means Echo Brain can answer "What did I say about X last year?" or "Find that email about the RV warranty."

**Technical approach:**
- **Google Takeout:** Parser for Gmail (mbox), Drive, Photos metadata, Chrome bookmarks, YouTube history
- **Claude conversations:** Already partially working via `ingest_claude_conversations.py`
- **Incremental:** Track ingestion watermarks to avoid re-processing
- **Privacy-first:** All processing local, no data leaves the Tower

**Dependencies:** Document ingestion pipeline (1.4), Google Takeout export

**Effort:** 15-20 hours

---

## Tier 3 — "Unique Value" (Patrick-Specific Power Features)

### 3.1 RV Systems Monitoring (Victron)

**What it does:** Monitors RV electrical system (batteries, solar, shore power, inverter) via Victron VE.Direct or MQTT and provides intelligent analysis.

**Why it matters:** Patrick has an RV. Victron systems expose data via VE.Direct serial protocol or MQTT (via Venus OS). "How's my battery bank?" "Am I generating enough solar for the AC?" "What's my power consumption trend this week?"

**Technical approach:**
- **VE.Direct:** Serial protocol parser (Python `vedirect` library)
- **Or MQTT:** If Venus GX/Cerbo GX is running, subscribe to MQTT topics
- **Metrics storage:** Time-series in PostgreSQL (or InfluxDB if preferred)
- **Intelligence:** Trend analysis, predictive alerts ("At current draw, batteries will hit 50% in 4 hours")

**Dependencies:** Physical connection to Victron system, either serial or network

**Effort:** 10-15 hours

---

### 3.2 Context-Aware Work Assistant

**What it does:** Provides work-context-aware assistance by understanding Patrick's current role, projects, and work patterns without storing classified information.

**Why it matters:** Echo Brain can understand "I have a meeting about X tomorrow" or "What was that procedure for Y?" based on unclassified work context, schedules, and general knowledge.

**Technical approach:**
- **Calendar integration:** Google Calendar already has code (`google_calendar.py`)
- **Context switching:** Detect work vs personal context based on time, calendar events, active projects
- **Knowledge boundary:** Explicit classification gate — never store or process anything marked sensitive

**Dependencies:** Google Calendar integration (already coded), preference learner (1.3)

**Effort:** 8-12 hours

---

### 3.3 Anime Production Pipeline Restoration

**What it does:** Fixes and stabilizes the anime production integration between Echo Brain and Anime Studio.

**Why it matters:** The anime production pipeline exists (132+ routes in Anime Studio) but the Echo Brain integration is broken. The anime_production agent exists but needs proper context from the production database.

**Technical approach:**
- **Fix:** `trigger_generation` MCP tool to work with current Anime Studio API
- **Enhance:** Give anime_production agent direct access to production DB queries
- **Dashboard:** Production status in Echo Brain frontend (episodes, scenes, shots, completion %)

**Dependencies:** Working Anime Studio service (already running on port 8401)

**Effort:** 8-12 hours

---

### 3.4 Voice Interface (Wake Word + Continuous Conversation)

**What it does:** Always-listening voice interface with wake word detection, streaming STT, and natural TTS responses.

**Why it matters:** Hands-free interaction while working, cooking, driving the RV. Perplexity has "Hey Plex" on Samsung; Echo Brain needs "Hey Echo" on the Tower or connected devices.

**Technical approach:**
- **Wake word:** OpenWakeWord (lightweight, runs on CPU) or Porcupine
- **STT:** faster-whisper (already installed, running on RX 9070 XT)
- **TTS:** piper-tts (already installed)
- **Client:** Python script on any device with a microphone (Raspberry Pi, Tower, laptop)
- **Protocol:** WebSocket for real-time audio streaming

**Dependencies:** Microphone hardware, faster-whisper + piper-tts (already installed)

**Effort:** 15-20 hours

---

## Tier 4 — "Polish" (Nice to Have)

### 4.1 Mobile Progressive Web App (PWA)

**What it does:** Wraps the Vue 3 frontend as a PWA installable on mobile devices.

**Why it matters:** Quick access to Echo Brain from phone without opening Telegram. PWA provides push notifications, offline caching, and app-like experience.

**Technical approach:**
- Add `manifest.json` and service worker to existing Vue 3 frontend
- Responsive design adjustments (if not already responsive)
- Push notification integration via Web Push API

**Dependencies:** Working Vue frontend (already has `dist/`)

**Effort:** 6-8 hours

---

### 4.2 Enhanced Notification Pipeline

**What it does:** Smart notification routing with priority, deduplication, quiet hours, and digest mode.

**Why it matters:** As proactive monitoring comes online, notification fatigue becomes a risk. Need intelligent batching and routing.

**Technical approach:**
- **Priority levels:** critical (immediate, all channels), high (immediate, primary channel), normal (batched hourly), low (daily digest)
- **Quiet hours:** Configurable, only critical breaks through
- **Dedup:** Suppress repeated alerts within configurable window
- **Digest:** Daily briefing already exists — extend it to include all accumulated notifications

**Dependencies:** Existing notification_service.py, proactive monitoring (2.3)

**Effort:** 6-8 hours

---

### 4.3 Usage Analytics Dashboard

**What it does:** Tracks and visualizes how Echo Brain is being used: query volumes, response times, agent utilization, memory growth, search effectiveness.

**Why it matters:** Understanding usage patterns helps prioritize future development. Also satisfies the engineer's need to see the system's metrics.

**Technical approach:**
- Extend existing `REQUEST_METRICS` tracking
- Store metrics in PostgreSQL time-series table
- Add dashboard page to Vue frontend with Chart.js or similar
- Expose Prometheus-compatible metrics (resilience router already has `/api/resilience/metrics`)

**Dependencies:** Vue frontend, existing metrics infrastructure

**Effort:** 8-12 hours

---

## Implementation Order (Recommended)

```
Week 1-2:  [1.1] Web Search Integration (SearXNG)
            └── Unblocks: Deep Research, Proactive Monitoring, Finance Enhancement

Week 3-4:  [1.4] Document Ingestion Pipeline
            └── Unblocks: Personal Data Indexing, Work Assistant

Week 5-7:  [1.2] Deep Research Engine
            └── Uses: Web Search + Memory + Document Knowledge

Week 8-9:  [1.3] Memory Intelligence Extraction
            └── Enhances: Everything (better memory = better responses)

Week 10-11: [2.1] Home Assistant Bridge
Week 12-13: [2.3] Proactive Monitoring Loop
Week 14-15: [2.2] Security Camera Intelligence
Week 16-17: [2.4] Personal Data Indexing

Week 18+:  Tier 3 & 4 based on priorities at that point
```

**Total estimated effort for Tiers 1-2:** 115-165 hours
**At 5-10 hours/week:** 12-33 weeks (3-8 months)
