# Echo Brain Fact Extraction: Practical Execution Plan

## The Reality Check

| Metric | Value |
|--------|-------|
| Total vectors | 393,578 (299k + 94k) |
| Available GPU | RTX 3060 12GB (single) |
| Extraction rate | ~3 vectors/minute (qwen2.5:7b) |
| Full extraction time | **2,186 hours (91 days)** |

This is not a "run overnight" task. It's a background operation that runs for weeks.

---

## Recommended Strategy: Phased Priority Extraction

Instead of extracting everything, we prioritize what matters most:

### Phase 1: Technical Core (Week 1)
**Goal**: Get Echo Brain's infrastructure knowledge complete

```bash
# Extract only technical/code content first
python scripts/scalable_extractor.py \
    --collection echo_memory \
    --priority-only \
    --postgres "postgresql://patrick@localhost:5432/echo_brain"
```

Expected: ~30k vectors (technical + code domain)
Time: ~170 hours (7 days running 24/7)
Result: Echo Brain understands its own infrastructure

### Phase 2: Recent Conversations (Week 2-3)
**Goal**: Context from recent interactions

```sql
-- Check recent conversation volume
SELECT COUNT(*) FROM ingestion_tracking 
WHERE domain = 'conversation' 
AND created_at > NOW() - INTERVAL '90 days';
```

### Phase 3: Background Long-tail (Ongoing)
**Goal**: Extract remaining content opportunistically

```bash
# Run in tmux, let it grind
tmux new -s extraction
python scripts/scalable_extractor.py \
    --collection echo_memory \
    --resume \
    --postgres "postgresql://patrick@localhost:5432/echo_brain"
# Ctrl+B, D to detach
```

---

## Quick Start Commands

### Step 1: Run Migration
```bash
sudo -u postgres psql echo_brain -f migrations/002_context_pipeline_v2.sql
```

### Step 2: Test Extraction (100 vectors)
```bash
# Quick validation that everything works
python scripts/scalable_extractor.py \
    --collection echo_memory \
    --limit 100 \
    --postgres "postgresql://patrick@localhost:5432/echo_brain"
```

### Step 3: Start Priority Extraction
```bash
# In tmux for persistence
tmux new -s extraction
python scripts/scalable_extractor.py \
    --collection echo_memory \
    --priority-only \
    --postgres "postgresql://patrick@localhost:5432/echo_brain"
```

### Step 4: Monitor Progress
```bash
# In another terminal
watch -n 60 'sudo -u postgres psql echo_brain -c "SELECT * FROM v_extraction_progress;"'
```

---

## Checkpoint & Resume

The extractor automatically saves checkpoints every 100 vectors. If it dies or you need to restart:

```bash
# Resume from last checkpoint
python scripts/scalable_extractor.py \
    --collection echo_memory \
    --resume
```

Graceful shutdown: Press Ctrl+C once. It will save checkpoint and exit cleanly.

---

## Monitoring Queries

### Overall Progress
```sql
SELECT * FROM v_extraction_progress;
```

### Extraction Velocity (recent)
```sql
SELECT * FROM v_extraction_velocity;
```

### Domain Coverage
```sql
SELECT * FROM v_domain_stats;
```

### Estimated Completion
```sql
SELECT 
    collection_name,
    vectors_processed,
    last_checkpoint_at,
    CASE 
        WHEN vectors_processed > 0 THEN
            NOW() + (
                (SELECT COUNT(*) FROM ingestion_tracking WHERE fact_extracted = FALSE) 
                / (vectors_processed::float / EXTRACT(EPOCH FROM NOW() - started_at))
            ) * INTERVAL '1 second'
    END as estimated_completion
FROM extraction_checkpoints
WHERE checkpoint_type = 'batch';
```

---

## Alternative: Aggressive Optimization

If 91 days is unacceptable, here are acceleration options:

### Option 1: Smaller Model (qwen2.5:3b)
- 2-3x faster
- Lower quality facts
- ~30-40 days total

```bash
# If you have qwen2.5:3b installed
python scripts/scalable_extractor.py --model qwen2.5:3b
```

### Option 2: Skip Low-Value Content
```sql
-- Mark low-value content as "extracted" without processing
UPDATE ingestion_tracking SET
    fact_extracted = TRUE,
    fact_extracted_at = NOW(),
    facts_count = 0
WHERE content_length < 200  -- Very short content
OR domain = 'general';      -- Non-specific content
```

### Option 3: Sampling Strategy
Extract facts from 10% random sample, extrapolate for queries:

```sql
-- Mark 90% as low priority
UPDATE ingestion_tracking SET
    priority_score = 0.1
WHERE random() > 0.1
AND domain NOT IN ('technical', 'code');
```

### Option 4: Cloud Burst (if budget allows)
Rent GPU time for a week:
- RunPod: RTX 4090 @ $0.74/hr = ~$50 for priority extraction
- Lambda Labs: Similar pricing

---

## File Inventory

| File | Purpose |
|------|---------|
| `migrations/002_context_pipeline_v2.sql` | Database schema with correct dimensions |
| `scripts/scalable_extractor.py` | Main extraction engine with checkpointing |
| `scripts/backfill_tracking.py` | Populate tracking from Qdrant (called automatically) |

---

## Integration Point

Once you have facts extracted, update your Echo Brain inference to use them:

```python
# In your chat endpoint
async def get_context(query: str):
    # 1. Classify query domain
    domain = classify_query(query)  # From classifier.py
    
    # 2. Search facts
    facts = await search_facts(query, domain)
    
    # 3. Get relevant vectors
    vectors = await search_vectors(query, domain)
    
    # 4. Assemble context
    return compile_context(facts, vectors, domain)
```

The `src/context_assembly/` modules are ready for this integration once you have facts populated.

---

## Success Metrics

After Phase 1 (technical extraction), you should have:

| Metric | Target |
|--------|--------|
| Technical domain coverage | 100% |
| Code domain coverage | 100% |
| Facts in technical domain | 5,000+ |
| Context contamination | Eliminated |
| Query latency impact | <100ms added |

---

## Questions?

If you hit issues:

1. **"No vectors being processed"**: Check tracking table was populated
   ```sql
   SELECT COUNT(*) FROM ingestion_tracking WHERE qdrant_collection = 'echo_memory';
   ```

2. **"Ollama timeout"**: Reduce batch size or check GPU memory
   ```bash
   nvidia-smi
   ```

3. **"Database connection failed"**: Verify PostgreSQL is on localhost
   ```bash
   ss -tlnp | grep 5432
   ```

4. **"Embedding dimension mismatch"**: Ensure using `mxbai-embed-large` for 1024-dim facts
