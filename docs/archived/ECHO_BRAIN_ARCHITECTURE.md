# Echo Brain System Architecture

## Document Purpose
This is the authoritative architecture reference for Echo Brain, Patrick's personal AI assistant running on the Tower server. Every section is self-contained and covers one aspect of the system. This document is the single source of truth — if other sources conflict with this document, this document is correct.

---

## System Identity and Core Configuration

Echo Brain is a personal AI assistant built by Patrick. It runs as a FastAPI service on the Tower server at **port 8309**. The Tower server hardware includes an AMD Ryzen 9 24-core processor, 96GB DDR6 RAM, an NVIDIA RTX 3060 12GB GPU, and an AMD RX 9070 XT 16GB GPU.

Echo Brain's codebase consists of **108 modules across 29 directories**, organized in a modular architecture under `/opt/tower-echo-brain/`. The service is managed by systemd as `tower-echo-brain`.

Key configuration:
- API Port: **8309**
- Service name: `tower-echo-brain`
- Install path: `/opt/tower-echo-brain/`
- Python virtual environment: `/opt/tower-echo-brain/venv/`
- Log access: `journalctl -u tower-echo-brain`

---

## Databases and Storage

Echo Brain uses two primary databases:

1. **PostgreSQL** — Relational data storage
   - Database name: `echo_brain`
   - Host: localhost (default port 5432)
   - User: `patrick`
   - Stores: conversations, facts, agent history, domain ingestion logs, routing configuration
   - The `anime_production` database is separate and used exclusively by the Tower Anime Production system

2. **Qdrant** — Vector database for semantic search
   - Host: localhost, port **6333**
   - Collection: `echo_memory` configured at **768 dimensions** using Cosine distance
   - Stores: embedded document chunks, conversation vectors, codebase vectors
   - As of February 2026: approximately 61,932 vectors and 6,129 facts

Echo Brain does NOT use SQLite. It previously had vectors in a collection using 1024 dimensions (from mxbai-embed-large), but those were all invalidated and the collection was recreated at 768 dimensions for nomic-embed-text.

---

## Embedding Model: nomic-embed-text

Echo Brain uses **nomic-embed-text** as its embedding model, served locally through Ollama.

- Model: `nomic-embed-text`
- Dimensions: **768**
- Context window: **8,192 tokens**
- Qdrant collection: `echo_memory` at 768 dimensions
- API endpoint: Ollama at `http://localhost:11434/api/embed`

### Why We Switched from mxbai-embed-large to nomic-embed-text

The original embedding model was `mxbai-embed-large`, which produced 1024-dimensional vectors but only supported a **512-token context window**. This was critically insufficient for embedding code snippets, conversation chunks, and documentation — most content was being truncated before embedding, resulting in poor retrieval quality.

`nomic-embed-text` was chosen as the replacement because:
- **16x larger context window**: 8,192 tokens vs 512 tokens — content is embedded fully, not truncated
- **Acceptable dimension trade-off**: 768D vs 1024D — slight reduction in vector precision, more than offset by the context window improvement
- **Local inference**: Runs through Ollama with no external API dependency

The migration required:
1. Deleting the old `echo_memory` Qdrant collection (1024D vectors were incompatible)
2. Recreating the collection at 768 dimensions
3. Complete re-ingestion of all content with the new model
4. Removing all references to mxbai-embed-large from the facts database to eliminate confusion

**Important**: CLIP is used for image and video embeddings, NOT nomic-embed-text. nomic-embed-text is exclusively for text content.

---

## Agent System and Model Routing

Echo Brain uses three specialized agents, each backed by a different Ollama model selected for its strengths:

1. **CodingAgent** — Handles programming tasks, code generation, debugging
   - Model: `deepseek-coder-v2:16b`
   - Triggered by keywords: code, debug, function, script, error, syntax, compile, programming, refactor, algorithm

2. **ReasoningAgent** — Handles analytical thinking, comparisons, planning
   - Model: `deepseek-r1:8b`
   - Triggered by keywords: analyze, compare, reason, plan, evaluate, strategy, decision, trade-off, architecture

3. **NarrationAgent** — Handles creative writing, summaries, natural conversation
   - Model: `gemma2:9b`
   - Triggered by keywords: write, story, summarize, explain, describe, narrative, creative, email, blog

### Routing Logic

Agent routing is **database-driven** — the routing configuration lives in PostgreSQL, not hardcoded in Python. The `model_routing` table maps intent keywords to agent types and their models. This was a deliberate architectural decision to allow model swaps without code changes.

Previously, routing was fragmented across 46+ configuration sources including hardcoded dictionaries, environment variables, and YAML files. The consolidation into a single database table was completed to establish a single source of truth.

The banned model `tinyllama` was explicitly removed from all routing paths due to poor output quality.

---

## Frontend Stack

Echo Brain's web dashboard uses:
- **Vue 3** — Component framework (NOT React)
- **TypeScript** — Type-safe JavaScript
- **Tailwind CSS** — Utility-first CSS framework

The frontend provides status monitoring, conversation interface, and admin controls. It is a Single Page Application (SPA).

---

## API Endpoints

Echo Brain exposes the following API endpoints via FastAPI:

### Core Endpoints
- `GET /health` — Health check, returns service status and uptime
- `POST /api/echo/query` — Primary query endpoint, accepts `{"query": "..."}` and returns AI response
- `POST /api/echo/chat` — Alias for `/api/echo/query`, identical handler and response format

### Agent Endpoints
- `GET /api/agents` — List available agents and their configurations
- `POST /api/agents/{agent_type}/query` — Query a specific agent directly

### Memory and Knowledge
- `GET /api/memory/status` — Memory system health and vector counts
- `POST /api/memory/search` — Semantic search across stored knowledge
- `GET /api/facts` — Retrieve stored facts
- `POST /api/facts` — Add new facts

### MCP (Model Context Protocol)
- `POST /mcp` — MCP server endpoint for tool integration

### Coordination
- `GET /api/coordination/services` — List connected services and their status

### Environment Variables Required
- `ECHO_BRAIN_PORT` — API port (default: 8309)
- `DATABASE_URL` — PostgreSQL connection string (default: postgresql://patrick@localhost/echo_brain)
- `QDRANT_HOST` — Qdrant server host (default: localhost)
- `QDRANT_PORT` — Qdrant server port (default: 6333)
- `OLLAMA_BASE_URL` — Ollama API base URL (default: http://localhost:11434)

---

## Context Contamination Bug and Resolution

### What Happened
Echo Brain and the Tower Anime Production system originally shared database tables and vector collections. When a user asked Echo Brain a technical question (e.g., "What databases does Echo Brain use?"), the retrieval system would sometimes return anime-related content (LoRA training configurations, ComfyUI workflow settings, character descriptions) because those vectors existed in the same Qdrant collection.

This caused Echo Brain to give responses that mixed technical information with anime production details — for example, answering a question about system architecture by referencing anime character consistency settings.

### How It Was Fixed
The systems were **completely separated**:

1. **Database separation**: Echo Brain uses the `echo_brain` PostgreSQL database. The anime production system uses the `anime_production` database. They no longer share tables.

2. **Vector collection separation**: Echo Brain's vectors live in the `echo_memory` Qdrant collection. Anime production vectors were moved to a separate collection.

3. **Service isolation**: The anime production system runs as its own service, independent of Echo Brain. There is no code coupling between the two systems.

4. **Memory middleware filtering**: The context-building middleware was updated to only search Echo Brain's own collections, preventing cross-domain retrieval.

### Current State
As of February 2026, the systems are fully separated. Echo Brain cannot accidentally retrieve anime content, and the anime production system cannot access Echo Brain's personal knowledge base. The shared `tower_consolidated` database still exists but is being phased out.

---

## Known Architectural Weaknesses (February 2026)

1. **No retrieval confidence gate**: When vector search returns low-similarity results (or no results), the LLM falls back to its general knowledge and generates plausible-sounding but incorrect answers (e.g., claiming Echo Brain uses SQLite). There is no threshold that triggers "I don't have specific information about that."

2. **No temporal ordering in vectors**: Qdrant vectors have no timestamp metadata distinguishing current facts from historical ones. Echo Brain cannot tell if a retrieved fact is from last week or last year. This causes stale information to surface as if it were current.

3. **Code-heavy, context-light retrieval**: The vector store contains many code snippets but few architectural explanations. Echo Brain can find implementation details but struggles with "why was this decision made?" and "how do these components connect?" questions.

4. **Extreme duplication**: Some content (particularly the context contamination fix code) appears 10+ times identically in the vector store. This floods search results with redundant chunks and pushes relevant unique content out of the top results.

5. **Missing documentation in vector store**: API endpoint documentation, environment variable references, deployment runbooks, and Architecture Decision Records (ADRs) have not been ingested. This is the primary cause of Echo Brain answering "I don't know" to questions about its own configuration.

---

## Failure Modes and Dependencies

### If Qdrant Crashes (Port 6333)
- **Impact**: All semantic search and memory retrieval fails
- **Behavior**: Queries still work but Echo Brain loses all context — responses degrade to generic LLM answers with no personalization or system-specific knowledge
- **Recovery**: Restart Qdrant (`sudo systemctl restart qdrant`), vectors persist on disk
- **Detection**: `/api/memory/status` will report errors

### If Ollama Crashes (Port 11434)
- **Impact**: Total failure — no LLM inference, no embeddings
- **Behavior**: All query endpoints return errors or timeouts
- **Recovery**: Restart Ollama (`sudo systemctl restart ollama`), models are cached on disk
- **Detection**: `/health` endpoint will report unhealthy

### If PostgreSQL Crashes (Port 5432)
- **Impact**: No routing configuration, no conversation history, no facts database
- **Behavior**: Queries may still work using cached routing but cannot log responses or access facts
- **Recovery**: Restart PostgreSQL (`sudo systemctl restart postgresql`)
- **Detection**: `/health` endpoint will report database connection failure

### GPU Resource Contention
- The **RTX 3060 12GB** is primarily used by Ollama for LLM inference
- The **RX 9070 XT 16GB** is primarily used by ComfyUI for image generation
- When both systems run heavy workloads simultaneously, VRAM exhaustion causes OOM errors
- The anime production system should be stopped (`sudo systemctl stop tower-anime-production`) when running heavy Echo Brain workloads

---

## Tower Anime Production System

The Tower Anime Production system is a **separate system** from Echo Brain, running its own services and databases. It produces AI-generated anime content using Stable Diffusion and ComfyUI.

### Two Active Projects

1. **Tokyo Debt Desire**
   - Visual style: **Photorealistic**
   - Project ID: 24
   - Characters: 4 defined
   - Theme: Financial drama set in Tokyo

2. **Cyberpunk Goblin Slayer: Neon Shadows**
   - Visual style: **Arcane style** (stylized, painterly, inspired by Arcane animation)
   - Also described as "cyberpunk anime" in the database
   - Theme: Cyberpunk action/fantasy crossover

### Production Pipeline
The anime production pipeline flows: Script/Prompt → ComfyUI Generation → Frame Assembly → FramePack Video → Final Output

- **ComfyUI**: Runs on port **8188**, handles image generation using Stable Diffusion checkpoints
- **FramePack**: Video generation tool that creates up to **60-second** anime clips from generated frames using the FramePack I2V HY (FP8) model
- **LoRA Models**: Character-specific LoRA weights stored in the ComfyUI models directory, trained for character consistency across frames
- **Workflows**: Stored as JSON files in ComfyUI's workflow directory

### Production Bottlenecks
- **VRAM limitation**: Currently constrained to 768x768 resolution due to GPU memory limits
- **Video generation quality**: FramePack produces variable quality — temporal consistency between frames degrades in longer clips
- **Render time**: Each 60-second clip takes significant GPU time on the RTX 3060
- **Character consistency**: LoRA training helps but is not perfect — characters can drift across long sequences

### GPU Assignment for Production
- **NVIDIA RTX 3060 (12GB VRAM)**: Primary GPU for Ollama LLM inference and some ComfyUI generation
- **AMD RX 9070 XT (16GB VRAM)**: Primary GPU for ComfyUI image generation and FramePack video rendering
- The GPUs do NOT share VRAM — each has its own dedicated memory pool
- Running both Echo Brain (heavy LLM inference) and anime production (heavy image generation) simultaneously can cause VRAM exhaustion on the shared RTX 3060

---

## Ingestion Pipeline

Echo Brain's ingestion pipeline processes documents into searchable vectors:

1. **Source** → Read document (markdown, code, conversation export, JSON)
2. **Chunk** → Split into segments under 6,000 characters with overlap
3. **Hash** → SHA-256 hash of chunk content for deduplication
4. **Embed** → Generate 768D vector using nomic-embed-text via Ollama
5. **Store** → Insert vector into Qdrant `echo_memory` collection with metadata payload
6. **Log** → Record ingestion in PostgreSQL `domain_ingestion_log` table

### Key Files
- Ingestion orchestrator: `/opt/tower-echo-brain/src/ingestion/orchestrator.py`
- Chunking service: `/opt/tower-echo-brain/src/ingestion/chunker.py`
- Embedding service: `/opt/tower-echo-brain/src/services/embedding_service.py`
- Domain ingestor (conversations): `/opt/tower-echo-brain/scripts/ingest_conversations.py`

### Ingestion Best Practices
- Each document section should be **self-contained** — don't assume the reader has seen other chunks
- Include metadata: source file, section name, timestamp, content type
- Hash before embedding to avoid duplicate vectors
- Verify embedding dimensions match Qdrant collection (768D)