# Echo Brain Context Assembly Pipeline

## Overview

The Context Assembly Pipeline (CAP) is the critical layer that transforms Echo Brain from a "dumb query-response" system into a contextually-aware, persistent intelligence. Think of it as the prefrontal cortex - the part that decides what memories and knowledge to pull before responding.

## The Problem

Currently Echo Brain has:
- 7,953 vectors indexed (documents: 435, conversations: 2,847, code: 4,150, facts: 521)
- Only 6% fact extraction coverage
- Context contamination issues (anime bleeding into technical queries)
- No systematic way to verify completeness

## Architecture

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
│  │                                                                    │  │
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
│  │                                                                    │  │
│  │   • Domain-specific system prompt selection                       │  │
│  │   • Fact injection (structured knowledge)                         │  │
│  │   • Conversation history (recent + relevant)                      │  │
│  │   • Code context (if technical query)                             │  │
│  │   • Token budget management                                        │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ASSEMBLED CONTEXT                                │
│                                                                          │
│   {                                                                      │
│     "system_prompt": "...",      // Domain-specific                     │
│     "facts": [...],              // Relevant extracted facts            │
│     "conversation_history": [...], // Recent + relevant turns           │
│     "code_context": [...],       // If applicable                       │
│     "metadata": {                                                        │
│       "domain": "technical",                                            │
│       "confidence": 0.87,                                               │
│       "token_count": 3200                                               │
│     }                                                                    │
│   }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OLLAMA / LLM INFERENCE                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Query Classifier

Prevents context contamination by routing queries to appropriate domains:

| Domain | Triggers | Isolated From |
|--------|----------|---------------|
| `technical` | code, server, database, API, debugging | anime content |
| `anime` | Tokyo Debt Desire, Cyberpunk Goblin Slayer, LoRA, ComfyUI | general tech |
| `personal` | schedule, preferences, history | all domains |
| `general` | everything else | none |

### 2. Retrieval Orchestrator

Parallel retrieval from multiple sources with domain filtering:

```python
async def retrieve(query: str, domain: str) -> RetrievalResult:
    tasks = [
        retrieve_vectors(query, domain_filter=domain),
        retrieve_facts(query, domain_filter=domain),
        retrieve_recent_conversation(limit=5),
        retrieve_code_context(query) if domain == "technical" else None
    ]
    results = await asyncio.gather(*tasks)
    return merge_and_dedupe(results)
```

### 3. Reranker

Two-stage ranking:
1. **Relevance score** - semantic similarity to query
2. **Recency boost** - newer information weighted higher for time-sensitive domains

### 4. Context Compiler

Assembles final context with token budget awareness:

```
Token Budget: 8192 (configurable per model)
├── System Prompt:     ~500 tokens (reserved)
├── Facts:             ~1500 tokens (priority 1)
├── Conversation:      ~2000 tokens (priority 2)
├── Code Context:      ~2000 tokens (priority 3)
└── Query + Buffer:    ~2192 tokens (reserved)
```

## Data Flow for Ingestion

### Current State
```
Documents (435) ──┐
Conversations (2847) ──┼──▶ Qdrant Vectors (7953)
Code (4150) ──────────┘           │
                                  ▼
                          Fact Extraction ──▶ Facts (521) ← ONLY 6%!
```

### Target State
```
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                                │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   SOURCE     │───▶│   CHUNKER    │───▶│  EMBEDDER    │          │
│  │  CRAWLERS    │    │              │    │  (Ollama)    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         │                   ▼                   ▼                   │
│         │            ┌──────────────┐    ┌──────────────┐          │
│         │            │    FACT      │    │   QDRANT     │          │
│         └───────────▶│  EXTRACTOR   │    │   STORE      │          │
│                      │   (LLM)      │    │              │          │
│                      └──────────────┘    └──────────────┘          │
│                             │                                       │
│                             ▼                                       │
│                      ┌──────────────┐                               │
│                      │  POSTGRESQL  │                               │
│                      │  FACT STORE  │                               │
│                      └──────────────┘                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  VERIFICATION LAYER                           │  │
│  │                                                                │  │
│  │  • Source tracking (what was ingested)                        │  │
│  │  • Coverage metrics (% of sources with facts extracted)       │  │
│  │  • Quality sampling (spot-check fact accuracy)                │  │
│  │  • Gap detection (identify missing sources)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Database Schema Additions

### ingestion_tracking
```sql
CREATE TABLE ingestion_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type VARCHAR(50) NOT NULL,  -- 'document', 'conversation', 'code', 'external'
    source_path TEXT NOT NULL,
    source_hash VARCHAR(64) NOT NULL,  -- SHA256 for deduplication
    
    -- Ingestion status
    vector_id UUID,                     -- Reference to Qdrant
    vectorized_at TIMESTAMP,
    fact_extracted BOOLEAN DEFAULT FALSE,
    fact_extracted_at TIMESTAMP,
    facts_count INTEGER DEFAULT 0,
    
    -- Metadata
    domain VARCHAR(50),                 -- 'technical', 'anime', 'personal', 'general'
    chunk_count INTEGER,
    token_count INTEGER,
    
    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(source_hash)
);

CREATE INDEX idx_ingestion_not_extracted ON ingestion_tracking(fact_extracted) 
    WHERE fact_extracted = FALSE;
CREATE INDEX idx_ingestion_domain ON ingestion_tracking(domain);
```

### facts
```sql
CREATE TABLE facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES ingestion_tracking(id),
    
    -- The fact itself
    fact_text TEXT NOT NULL,
    fact_type VARCHAR(50),              -- 'entity', 'relationship', 'event', 'preference', 'technical'
    confidence FLOAT DEFAULT 1.0,
    
    -- Domain isolation
    domain VARCHAR(50) NOT NULL,
    
    -- Structured extraction
    subject TEXT,                       -- Who/what the fact is about
    predicate TEXT,                     -- The relationship/action
    object TEXT,                        -- The target/value
    
    -- Temporal
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,              -- NULL = still valid
    
    -- Search
    embedding VECTOR(384),              -- For semantic search on facts
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_facts_domain ON facts(domain);
CREATE INDEX idx_facts_type ON facts(fact_type);
CREATE INDEX idx_facts_subject ON facts(subject);
```

### context_assembly_log
```sql
CREATE TABLE context_assembly_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Query info
    query_text TEXT NOT NULL,
    query_embedding VECTOR(384),
    classified_domain VARCHAR(50),
    
    -- What was retrieved
    vectors_retrieved INTEGER,
    facts_retrieved INTEGER,
    conversation_turns INTEGER,
    
    -- Assembly metrics
    total_tokens INTEGER,
    assembly_time_ms INTEGER,
    
    -- For debugging/improvement
    retrieved_vector_ids UUID[],
    retrieved_fact_ids UUID[],
    
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Verification & Testing Strategy

### 1. Coverage Metrics

```python
class CoverageMetrics:
    """Track what percentage of data has been fully processed."""
    
    async def get_coverage_report(self) -> dict:
        return {
            "vectors": {
                "total": await self.count_vectors(),
                "with_facts": await self.count_vectors_with_facts(),
                "coverage_pct": ...
            },
            "by_domain": {
                "technical": {...},
                "anime": {...},
                "personal": {...}
            },
            "by_source_type": {
                "documents": {...},
                "conversations": {...},
                "code": {...}
            }
        }
```

### 2. Ingestion Completeness Test

```python
async def test_ingestion_completeness():
    """Verify all known sources are tracked."""
    
    # Get all files in known directories
    known_sources = await scan_all_source_directories()
    
    # Get all tracked sources
    tracked = await db.fetch("SELECT source_path FROM ingestion_tracking")
    tracked_paths = {r['source_path'] for r in tracked}
    
    # Find gaps
    missing = known_sources - tracked_paths
    
    assert len(missing) == 0, f"Missing sources: {missing}"
```

### 3. Fact Extraction Quality Test

```python
async def test_fact_extraction_quality():
    """Spot-check fact extraction accuracy."""
    
    # Sample 100 random vectors
    samples = await get_random_vectors(100)
    
    for sample in samples:
        # Re-extract facts
        new_facts = await extract_facts(sample.content)
        
        # Compare with stored facts
        stored_facts = await get_facts_for_vector(sample.id)
        
        # Check for major discrepancies
        assert similarity(new_facts, stored_facts) > 0.8
```

### 4. Context Assembly Integration Test

```python
async def test_context_assembly():
    """Verify context assembly produces expected output."""
    
    test_cases = [
        {
            "query": "How do I fix the PostgreSQL connection pooling issue?",
            "expected_domain": "technical",
            "should_contain": ["database", "connection"],
            "should_not_contain": ["anime", "LoRA", "Tokyo Debt"]
        },
        {
            "query": "What LoRA settings work best for character consistency?",
            "expected_domain": "anime",
            "should_contain": ["LoRA", "checkpoint"],
            "should_not_contain": ["PostgreSQL", "FastAPI"]
        }
    ]
    
    for case in test_cases:
        context = await assemble_context(case["query"])
        
        assert context.domain == case["expected_domain"]
        
        context_text = str(context)
        for term in case["should_contain"]:
            assert term.lower() in context_text.lower()
        for term in case["should_not_contain"]:
            assert term.lower() not in context_text.lower()
```

## Implementation Priority

### Phase 1: Fix Fact Extraction (Week 1)
1. Create `ingestion_tracking` table
2. Backfill tracking for all existing vectors
3. Build background job to extract facts from remaining 94%
4. Add domain classification during extraction

### Phase 2: Build Context Assembly (Week 2)
1. Implement Query Classifier
2. Build Retrieval Orchestrator with domain filtering
3. Implement Context Compiler with token budgeting
4. Add assembly logging

### Phase 3: Testing & Verification (Week 3)
1. Build coverage dashboard
2. Implement completeness tests
3. Add quality sampling
4. Create CI/CD integration tests

### Phase 4: Optimization (Week 4)
1. Add caching for frequent patterns
2. Tune reranking weights
3. A/B test context assembly strategies
4. Build feedback loop for continuous improvement

## File Structure

```
tower-echo-brain/
├── src/
│   ├── context_assembly/
│   │   ├── __init__.py
│   │   ├── classifier.py        # Query domain classification
│   │   ├── retriever.py         # Multi-source retrieval
│   │   ├── reranker.py          # Relevance + recency scoring
│   │   ├── compiler.py          # Final context assembly
│   │   └── models.py            # Pydantic models
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── crawler.py           # Source discovery
│   │   ├── chunker.py           # Text chunking
│   │   ├── embedder.py          # Vector generation
│   │   ├── fact_extractor.py    # LLM-based fact extraction
│   │   └── tracker.py           # Ingestion tracking
│   │
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── coverage.py          # Coverage metrics
│   │   ├── completeness.py      # Gap detection
│   │   └── quality.py           # Fact quality checks
│   │
│   └── api/
│       └── routes/
│           └── context.py       # Context assembly endpoints
│
├── tests/
│   ├── test_classifier.py
│   ├── test_retriever.py
│   ├── test_compiler.py
│   ├── test_ingestion.py
│   └── test_integration.py
│
├── scripts/
│   ├── backfill_tracking.py     # One-time backfill
│   ├── extract_all_facts.py     # Batch fact extraction
│   └── verify_coverage.py       # Coverage report
│
└── migrations/
    ├── 001_ingestion_tracking.sql
    ├── 002_facts_table.sql
    └── 003_context_assembly_log.sql
```
