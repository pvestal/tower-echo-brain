# Echo Brain System Architecture & Context

## Overview

Echo Brain is a contextually-aware AI system that combines vector memory, fact extraction, and intelligent context assembly to provide domain-specific responses. The system operates as part of the Tower infrastructure at 192.168.50.135, serving both technical operations and creative anime production workflows.

## System Architecture

### Core Services

| Service | Port | Purpose | Status | GPU Usage |
|---------|------|---------|---------|-----------|
| **Echo Brain API** | 8309 | Main inference engine | ✅ Active | NVIDIA RTX 3060 |
| **Echo Brain MCP** | 8312 | Memory search API | ✅ Active | - |
| **Qdrant Vector DB** | 6333 | Vector storage | ✅ Active | - |
| **Semantic Memory** | 8310 | Additional memory API | ✅ Active | - |
| **Echo Frontend** | 8311 | Web interface | ✅ Active | - |
| **Ollama** | 11434 | LLM inference | ✅ Active | NVIDIA GPU (732MB) |
| **ComfyUI** | 8188 | Image generation | ✅ Active | RTX 3060 (12GB) |

### Database Architecture

#### PostgreSQL Databases

1. **echo_brain** - Primary Echo Brain data
   - Connection: `patrick@192.168.50.135:5432`
   - Password: `tower_echo_brain_secret_key_2025`

2. **anime_production** - Anime project data
   - Projects, characters, scenes, workflows

3. **tower_consolidated** - Main Tower system data
   - User accounts, system metrics, integration data

#### Qdrant Collections

| Collection | Dimensions | Purpose | Status |
|------------|------------|---------|---------|
| **echo_memory** | 1024 | Primary memory storage (MCP searches here) | 61,932 vectors |
| **claude_conversations** | 384 | Legacy collection (migrated to echo_memory) | Migrated |
| echo_memories | 768 | Additional memories | Active |
| scene_embeddings | - | Anime scene vectors | Active |
| anime_characters | - | Character embeddings | Active |

## Context Assembly Pipeline

### Current Problem
- 7,953 vectors indexed (documents: 435, conversations: 2,847, code: 4,150, facts: 521)
- **Only 6% fact extraction coverage**
- Context contamination (anime bleeding into technical queries)
- No systematic completeness verification

### Solution: Context Assembly Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      QUERY CLASSIFIER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Technical  │  │   Anime     │  │  Personal   │  │   General   │    │
│  │   Domain    │  │  Production │  │  /Context   │  │  Knowledge  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT ASSEMBLY ENGINE                               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     RETRIEVAL ORCHESTRATOR                        │  │
│  │   ┌────────────┐   ┌────────────┐   ┌────────────┐               │  │
│  │   │  Qdrant    │   │ PostgreSQL │   │   Fact     │               │  │
│  │   │  Vectors   │   │   Direct   │   │   Store    │               │  │
│  │   └────────────┘   └────────────┘   └────────────┘               │  │
│  │          │                │                │                      │  │
│  │          └────────────────┼────────────────┘                      │  │
│  │                           ▼                                        │  │
│  │                  ┌────────────────┐                               │  │
│  │                  │  RERANKER      │                               │  │
│  │                  │  (relevance +  │                               │  │
│  │                  │   recency)     │                               │  │
│  │                  └────────────────┘                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     CONTEXT COMPILER                              │  │
│  │   • Domain-specific system prompts                                │  │
│  │   • Fact injection (structured knowledge)                         │  │
│  │   • Conversation history (recent + relevant)                      │  │
│  │   • Code context (if technical query)                             │  │
│  │   • Token budget management                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OLLAMA / LLM INFERENCE                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Domain Isolation Strategy

| Domain | Triggers | Isolated From | Purpose |
|--------|----------|---------------|---------|
| `technical` | postgresql, fastapi, docker, kubernetes, debugging | anime content | Server/infrastructure queries |
| `anime` | lora, comfyui, tokyo debt desire, checkpoint, framepack | technical infrastructure | Creative production |
| `personal` | victron, tundra, rv, sundowner, schedule | both above | Personal preferences/context |
| `general` | everything else | none | Fallback domain |

### Token Budget Management

```
Token Budget: 8192 (configurable per model)
├── System Prompt:     ~500 tokens (reserved)
├── Facts:             ~1500 tokens (priority 1)
├── Conversation:      ~2000 tokens (priority 2)
├── Code Context:      ~2000 tokens (priority 3)
└── Query + Buffer:    ~2192 tokens (reserved)
```

## Database Schema

### Context Assembly Pipeline Tables

#### `ingestion_tracking`
```sql
CREATE TABLE ingestion_tracking (
    id UUID PRIMARY KEY,
    source_type VARCHAR(50),     -- 'document', 'conversation', 'code', 'external'
    source_path TEXT,
    source_hash VARCHAR(64),     -- SHA256 for deduplication

    -- Status tracking
    vector_id UUID,              -- Reference to Qdrant
    vectorized_at TIMESTAMP,
    fact_extracted BOOLEAN DEFAULT FALSE,
    fact_extracted_at TIMESTAMP,
    facts_count INTEGER DEFAULT 0,

    -- Classification
    domain VARCHAR(50),          -- 'technical', 'anime', 'personal', 'general'

    -- Metadata
    chunk_count INTEGER,
    token_count INTEGER,
    file_size_bytes BIGINT,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### `facts`
```sql
CREATE TABLE facts (
    id UUID PRIMARY KEY,
    source_id UUID REFERENCES ingestion_tracking(id),

    -- Fact content
    fact_text TEXT NOT NULL,
    fact_type VARCHAR(50),       -- 'entity', 'relationship', 'event', 'preference', 'technical'
    confidence FLOAT DEFAULT 1.0,

    -- Domain isolation
    domain VARCHAR(50) NOT NULL,

    -- Structured extraction (Subject-Predicate-Object)
    subject TEXT,                -- Who/what
    predicate TEXT,              -- Relationship/action
    object TEXT,                 -- Target/value

    -- Temporal bounds
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,       -- NULL = still valid

    -- Semantic search
    embedding VECTOR(384),       -- nomic-embed-text embeddings

    created_at TIMESTAMP DEFAULT NOW()
);
```

#### `context_assembly_log`
```sql
CREATE TABLE context_assembly_log (
    id UUID PRIMARY KEY,
    query_text TEXT NOT NULL,
    classified_domain VARCHAR(50),

    -- Retrieval metrics
    vectors_retrieved INTEGER,
    facts_retrieved INTEGER,
    conversation_turns INTEGER,

    -- Performance
    total_tokens INTEGER,
    assembly_time_ms INTEGER,

    -- Debug info
    retrieved_vector_ids UUID[],
    retrieved_fact_ids UUID[],

    created_at TIMESTAMP DEFAULT NOW()
);
```

## Tech Stack

### Core Infrastructure
- **Language**: Python 3.11+
- **Web Framework**: FastAPI 0.104+
- **Database**: PostgreSQL 15+ with pgvector extension
- **Vector Store**: Qdrant 1.8+
- **Message Queue**: (Future: Redis/RabbitMQ)
- **Caching**: In-memory + Redis (planned)

### AI/ML Components
- **LLM Inference**: Ollama (qwen2.5:14b, qwen2.5:7b)
- **Embeddings**: nomic-embed-text (384 dims)
- **Vector Search**: Qdrant with HNSW indexing
- **Image Generation**: ComfyUI + Custom LoRAs

### Monitoring & Observability
- **Health Checks**: `/health` endpoints on all services
- **Metrics**: PostgreSQL views + custom dashboards
- **Logging**: Structured JSON logs
- **Coverage Tracking**: Database views for extraction coverage
- **Real-time Operations**: Live dashboard at localhost:8311/interface
  - Background job monitoring via tmux session parsing
  - WebSocket streaming for real-time updates
  - System health checks (database, Qdrant, Ollama, ComfyUI)
  - Resource usage monitoring (CPU, RAM, GPU, disk)

## API Structure

### Core Endpoints

#### Echo Brain API (Port 8309)
- `GET /health` - Service health check
- `POST /api/echo/chat` - Main chat inference
- `GET /api/echo/status` - System status
- `POST /api/echo/search` - Memory search

#### Operations Monitoring (Port 8309)
- `GET /operations/status` - Real-time operations status
  - Returns: active jobs, system health, resource usage
- `GET /operations/jobs` - Background jobs list
- `WebSocket /operations/stream` - Live streaming updates (5s interval)

#### Echo Frontend (Port 8311)
- `GET /interface` - Live operations console
  - WebSocket-powered dashboard showing real-time job progress
  - System health monitoring for database, Qdrant, Ollama, ComfyUI
  - Resource usage visualization (CPU, RAM, GPU, disk)

#### MCP Server API (Port 8312)
- `POST /mcp` - Memory Component Protocol interface
  - `search_memory` - Semantic search across vectors
  - `get_facts` - Structured fact retrieval
  - `store_fact` - Fact storage
  - `get_anime_context` - Anime-specific context

### Agent Routing Logic

The system uses intelligent routing based on query analysis:

```python
# Pseudo-code for agent routing
async def route_query(query: str, user_context: dict):
    # 1. Classify domain
    domain = await classifier.classify(query)

    # 2. Retrieve relevant context
    context = await retriever.retrieve(
        query=query,
        domain=domain,
        user_context=user_context
    )

    # 3. Assemble prompt
    prompt = await compiler.compile(
        query=query,
        context=context,
        domain=domain
    )

    # 4. Route to appropriate model
    if domain == "anime":
        model = "qwen2.5:14b"  # Better creative writing
    elif domain == "technical":
        model = "qwen2.5:7b"   # Faster for code/docs
    else:
        model = "qwen2.5:7b"   # Default

    return await ollama.generate(model=model, prompt=prompt)
```

## Critical Configuration Patterns

### Environment Variables
```bash
# Database connections
POSTGRES_HOST=192.168.50.135
POSTGRES_USER=patrick
POSTGRES_PASSWORD=RP78eIrW7cI2jYvL5akt1yurE
ECHO_BRAIN_DB_PASSWORD=tower_echo_brain_secret_key_2025

# Service endpoints
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# Model configuration
DEFAULT_EMBEDDING_MODEL=nomic-embed-text
DEFAULT_LLM_MODEL=qwen2.5:7b
FACT_EXTRACTION_MODEL=qwen2.5:14b

# Token limits
MAX_CONTEXT_TOKENS=8192
MAX_FACT_TOKENS=1500
MAX_HISTORY_TOKENS=2000
```

### Model Recommendations by VRAM

| Task | Model | VRAM (Q4) | Purpose |
|------|-------|-----------|---------|
| Fact extraction | qwen2.5:14b | ~10GB | High quality extraction |
| Classification | qwen2.5:7b | ~5GB | Fast domain routing |
| Embeddings | nomic-embed-text | ~1GB | Vector generation |
| Chat inference | qwen2.5:7b | ~5GB | General responses |
| Creative writing | qwen2.5:14b | ~10GB | Anime production |

## Current State & Next Steps

### Current Status (Updated January 29, 2026)
✅ **Completed**
- Core Echo Brain service running
- Qdrant vector storage with 308k+ vectors (up from 61k)
- MCP interface for memory search
- Real-time operations monitoring with WebSocket streaming
- Database schema migration completed
- Fact extraction backfill completed (304,394 vectors tracked)
- Live operations dashboard at localhost:8311/interface
- System health monitoring (database, Qdrant, Ollama, ComfyUI)
- Background job monitoring via tmux session parsing

⚠️ **In Progress**
- Large-scale fact extraction (2,384 facts extracted, 304,394 remaining)
- Context Assembly Pipeline optimization
- Resource usage monitoring during intensive operations

🔄 **Planned**
- Domain isolation implementation
- Token budget optimization
- Coverage verification system
- CI/CD integration tests

### Implementation Priority

#### Phase 1: Fix Fact Extraction (Week 1)
1. Deploy new database schema (`001_create_tables.sql`)
2. Run backfill script for existing 7,953 vectors
3. Execute batch fact extraction to reach 100% coverage
4. Implement domain classification during extraction

#### Phase 2: Context Assembly (Week 2)
1. Deploy Query Classifier with domain routing
2. Build Retrieval Orchestrator with parallel retrieval
3. Implement Context Compiler with token budgeting
4. Add assembly performance logging

#### Phase 3: Testing & Verification (Week 3)
1. Build coverage monitoring dashboard
2. Implement completeness verification tests
3. Add fact extraction quality sampling
4. Create integration test suite

#### Phase 4: Optimization (Week 4)
1. Add intelligent caching for frequent query patterns
2. Tune reranking algorithms for relevance + recency
3. A/B test different context assembly strategies
4. Build feedback loop for continuous improvement

## File Structure

```
/opt/tower-echo-brain/
├── src/
│   ├── main_consolidated.py        # Main FastAPI application
│   ├── app_factory.py             # Application factory
│   ├── api/                       # REST API endpoints
│   │   ├── echo.py               # Core Echo Brain API
│   │   ├── routes.py             # Main routing
│   │   └── [25+ specialized APIs]
│   ├── context_assembly/          # Context Assembly Pipeline
│   │   ├── classifier.py         # Domain classification
│   │   ├── retriever.py          # Multi-source retrieval
│   │   ├── compiler.py           # Context assembly
│   │   └── models.py             # Pydantic models
│   ├── ingestion/                 # Data ingestion pipeline
│   ├── verification/              # Testing & verification
│   └── [50+ other modules]
├── echo-brain-context-pipeline/   # New Context Assembly implementation
│   ├── src/context_assembly/      # Core pipeline components
│   ├── scripts/backfill_tracking.py  # Migration scripts
│   └── migrations/001_create_tables.sql
├── requirements.txt               # Python dependencies
└── [100+ supporting files]
```

## Integration Points

### Tower System Integration
- **Auth Service** (8088): User authentication
- **Dashboard** (8307): Web interface
- **Anime Production** (8306): Creative workflows
- **Model Manager**: Centralized model path resolution

### External Integrations
- **GitHub**: Code ingestion and analysis
- **Google Calendar**: Scheduling and personal context
- **Home Assistant**: Personal environment data
- **Plaid**: Financial transaction analysis

## Monitoring & Health Checks

### Health Endpoints
```bash
# Check all services
curl http://localhost:8309/health  # Echo Brain
curl http://localhost:8312/health  # MCP Server
curl http://localhost:6333/        # Qdrant
curl http://localhost:11434/api/tags  # Ollama

# Database health
psql -h 192.168.50.135 -U patrick -d echo_brain -c "SELECT 1;"

# Coverage metrics
psql -h 192.168.50.135 -U patrick -d echo_brain -c "SELECT * FROM v_coverage_summary;"
```

### Key Metrics to Monitor
- **Vector Count**: Qdrant collection sizes
- **Fact Coverage**: Percentage of vectors with extracted facts
- **Response Time**: Context assembly + inference latency
- **Domain Classification**: Accuracy of query routing
- **VRAM Usage**: GPU utilization across models
- **Token Usage**: Context budget consumption patterns

---

*Last Updated: 2026-01-29*
*System Version: Echo Brain 4.0.0 with Context Assembly Pipeline*