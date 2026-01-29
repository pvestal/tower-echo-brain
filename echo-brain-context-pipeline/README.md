# Echo Brain Context Assembly Pipeline

Transform Echo Brain from a "dumb query-response" system into a contextually-aware, persistent intelligence.

## Quick Start

### 1. Install Dependencies

```bash
cd echo-brain-context-pipeline
pip install -r requirements.txt --break-system-packages
```

### 2. Run Database Migration

```bash
psql -d echo_brain -f migrations/001_create_tables.sql
```

### 3. Backfill Tracking from Existing Qdrant Vectors

```bash
python scripts/backfill_tracking.py \
    --postgres-dsn "postgresql://localhost/echo_brain" \
    --qdrant-host localhost \
    --qdrant-port 6333
```

### 4. Run Fact Extraction (This Gets You from 6% to 100%)

```bash
python -m src.ingestion.fact_extractor "postgresql://localhost/echo_brain"
```

**Note:** This will take time depending on how many vectors you have. With 7,953 vectors and a 14B model, expect several hours. You can:
- Run it in a tmux session
- Use a smaller model (qwen2.5:7b) for faster but lower quality extraction
- Run in batches by modifying the script

### 5. Verify Coverage

```bash
python -m src.verification.coverage "postgresql://localhost/echo_brain"
```

## Architecture Overview

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────┐
│        QUERY CLASSIFIER             │
│  (technical/anime/personal/general) │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     RETRIEVAL ORCHESTRATOR          │
│  ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │ Qdrant  │ │  Facts  │ │ Conv  │ │
│  │ Vectors │ │   DB    │ │History│ │
│  └─────────┘ └─────────┘ └───────┘ │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│       CONTEXT COMPILER              │
│  (token budgeting, prioritization)  │
└─────────────────────────────────────┘
    │
    ▼
ASSEMBLED CONTEXT → LLM INFERENCE
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Query Classifier | `src/context_assembly/classifier.py` | Domain routing to prevent context contamination |
| Retriever | `src/context_assembly/retriever.py` | Parallel retrieval from multiple sources |
| Compiler | `src/context_assembly/compiler.py` | Token budgeting and context assembly |
| Fact Extractor | `src/ingestion/fact_extractor.py` | LLM-based fact extraction |
| Coverage Verifier | `src/verification/coverage.py` | Testing and completeness verification |

## Integration with Echo Brain

### Option A: Direct Integration

```python
from src.context_assembly.classifier import QueryClassifier
from src.context_assembly.retriever import create_retriever
from src.context_assembly.compiler import ContextAssembler, create_assembler

# Initialize components
classifier = QueryClassifier()
retriever = await create_retriever(
    qdrant_host="localhost",
    postgres_dsn="postgresql://localhost/echo_brain",
    ollama_client=your_ollama_client
)
assembler = create_assembler(classifier, retriever)

# Use in your inference loop
async def handle_query(user_query: str):
    # Assemble context
    context = await assembler.assemble(user_query)
    
    # Convert to Ollama format
    messages = assembler.to_ollama_messages(context, user_query)
    
    # Run inference
    response = await ollama.chat(model="your-model", messages=messages)
    
    return response
```

### Option B: FastAPI Endpoint

Add this route to your existing FastAPI app:

```python
from fastapi import APIRouter
from src.context_assembly.compiler import ContextAssembler

router = APIRouter()

@router.post("/v1/context")
async def assemble_context(query: str):
    context = await assembler.assemble(query)
    return context.to_prompt_components()
```

## Monitoring Coverage

Check the PostgreSQL views:

```sql
-- Overall coverage summary
SELECT * FROM v_coverage_summary;

-- Items pending extraction
SELECT * FROM v_extraction_queue LIMIT 10;

-- Fact distribution
SELECT * FROM v_fact_distribution;
```

## Domain Isolation

The classifier prevents context contamination:

| Domain | Keywords | Isolated From |
|--------|----------|---------------|
| `technical` | postgresql, fastapi, echo brain, docker... | anime content |
| `anime` | lora, comfyui, tokyo debt desire, checkpoint... | technical infra |
| `personal` | victron, tundra, rv, sundowner... | both above |
| `general` | (fallback) | none |

## File Structure

```
echo-brain-context-pipeline/
├── ARCHITECTURE.md          # Detailed design doc
├── README.md               # This file
├── requirements.txt
├── migrations/
│   └── 001_create_tables.sql
├── scripts/
│   ├── backfill_tracking.py
│   └── verify_coverage.py
└── src/
    ├── context_assembly/
    │   ├── models.py       # Pydantic models
    │   ├── classifier.py   # Query classification
    │   ├── retriever.py    # Multi-source retrieval
    │   └── compiler.py     # Context assembly
    ├── ingestion/
    │   └── fact_extractor.py
    └── verification/
        └── coverage.py
```

## VRAM Considerations

Your RTX 3060 has 12GB VRAM. Recommended models:

| Task | Model | VRAM (Q4) |
|------|-------|-----------|
| Fact extraction | qwen2.5:14b | ~10GB |
| Classification | qwen2.5:7b | ~5GB |
| Embeddings | nomic-embed-text | ~1GB |

For faster extraction, use `qwen2.5:7b` but expect lower quality facts.

## Next Steps After Setup

1. ✅ Run migration
2. ✅ Backfill tracking  
3. ✅ Extract all facts (get to 100% coverage)
4. ⬜ Verify coverage
5. ⬜ Run integration tests
6. ⬜ Integrate with Echo Brain's main inference loop
7. ⬜ Set up cron job for continuous ingestion of new content
