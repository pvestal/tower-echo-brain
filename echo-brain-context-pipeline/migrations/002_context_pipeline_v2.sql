-- Echo Brain Context Assembly Pipeline
-- Migration 002: Updated for actual system configuration
-- 
-- Key findings:
--   - echo_memory: 299,432 vectors @ 1024 dimensions (mxbai-embed-large)
--   - claude_conversations: 94,146 vectors @ 384 dimensions (nomic-embed-text)
--   - PostgreSQL on localhost:5432 (not remote)
--
-- Run: sudo -u postgres psql echo_brain -f 002_context_pipeline_v2.sql

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ============================================================================
-- INGESTION TRACKING (updated for multi-collection)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ingestion_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Source identification
    source_type VARCHAR(50) NOT NULL,
    source_path TEXT NOT NULL,
    source_hash VARCHAR(64) NOT NULL,
    
    -- Qdrant reference (updated for multi-collection)
    qdrant_collection VARCHAR(100),       -- 'echo_memory', 'claude_conversations', etc.
    vector_id TEXT,                        -- Qdrant point ID (can be UUID or int)
    vector_dimensions INTEGER,             -- 1024, 768, or 384
    vectorized_at TIMESTAMP WITH TIME ZONE,
    
    -- Fact extraction status
    fact_extracted BOOLEAN DEFAULT FALSE,
    fact_extracted_at TIMESTAMP WITH TIME ZONE,
    facts_count INTEGER DEFAULT 0,
    extraction_error TEXT,
    extraction_attempts INTEGER DEFAULT 0,
    
    -- Domain classification
    domain VARCHAR(50),
    
    -- Content metadata
    chunk_count INTEGER,
    token_count INTEGER,
    content_length INTEGER,                -- Characters (for filtering short content)
    
    -- Priority scoring
    priority_score FLOAT DEFAULT 0.5,      -- 0-1, higher = extract first
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(source_hash)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tracking_pending ON ingestion_tracking(fact_extracted, priority_score DESC) 
    WHERE fact_extracted = FALSE;
CREATE INDEX IF NOT EXISTS idx_tracking_collection ON ingestion_tracking(qdrant_collection);
CREATE INDEX IF NOT EXISTS idx_tracking_domain ON ingestion_tracking(domain);


-- ============================================================================
-- VECTOR CONTENT CACHE
-- ============================================================================

CREATE TABLE IF NOT EXISTS vector_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tracking_id UUID NOT NULL REFERENCES ingestion_tracking(id) ON DELETE CASCADE,
    
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tracking_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_content_tracking ON vector_content(tracking_id);


-- ============================================================================
-- FACTS (with correct dimension for mxbai-embed-large)
-- ============================================================================

CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES ingestion_tracking(id) ON DELETE CASCADE,
    
    fact_text TEXT NOT NULL,
    fact_type VARCHAR(50) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    
    domain VARCHAR(50) NOT NULL,
    
    -- SPO triple
    subject TEXT,
    predicate TEXT,
    object TEXT,
    
    -- Temporal
    valid_from TIMESTAMP WITH TIME ZONE,
    valid_until TIMESTAMP WITH TIME ZONE,
    
    -- Embedding: 1024 dims for mxbai-embed-large
    -- If you need 384-dim compatibility, create a separate column
    embedding_1024 VECTOR(1024),
    embedding_384 VECTOR(384),             -- For nomic-embed-text compatibility
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts(domain);
CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source_id);

-- Vector indexes (IVFFlat for large collections)
CREATE INDEX IF NOT EXISTS idx_facts_emb_1024 ON facts 
    USING ivfflat (embedding_1024 vector_cosine_ops) WITH (lists = 500)
    WHERE embedding_1024 IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_facts_emb_384 ON facts 
    USING ivfflat (embedding_384 vector_cosine_ops) WITH (lists = 100)
    WHERE embedding_384 IS NOT NULL;


-- ============================================================================
-- EXTRACTION CHECKPOINTS (for resume capability)
-- ============================================================================

CREATE TABLE IF NOT EXISTS extraction_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Checkpoint identification
    collection_name VARCHAR(100) NOT NULL,
    checkpoint_type VARCHAR(50) NOT NULL,  -- 'offset', 'last_id', 'batch'
    
    -- State
    last_processed_id TEXT,
    last_offset INTEGER,
    vectors_processed INTEGER DEFAULT 0,
    facts_extracted INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    
    -- Timing
    started_at TIMESTAMP WITH TIME ZONE,
    last_checkpoint_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    estimated_completion TIMESTAMP WITH TIME ZONE,
    
    -- Config snapshot
    model_used VARCHAR(100),
    batch_size INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_checkpoint_collection 
    ON extraction_checkpoints(collection_name, checkpoint_type);


-- ============================================================================
-- EXTRACTION QUEUE (prioritized work items)
-- ============================================================================

CREATE TABLE IF NOT EXISTS extraction_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tracking_id UUID NOT NULL REFERENCES ingestion_tracking(id) ON DELETE CASCADE,
    
    -- Priority factors
    priority_score FLOAT NOT NULL DEFAULT 0.5,
    domain VARCHAR(50),
    content_length INTEGER,
    
    -- Queue state
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
    locked_by TEXT,                         -- Worker ID if being processed
    locked_at TIMESTAMP WITH TIME ZONE,
    
    -- Retry handling
    attempts INTEGER DEFAULT 0,
    last_error TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tracking_id)
);

CREATE INDEX IF NOT EXISTS idx_queue_pending 
    ON extraction_queue(priority_score DESC, created_at)
    WHERE status = 'pending';


-- ============================================================================
-- MONITORING VIEWS
-- ============================================================================

-- Overall progress view
CREATE OR REPLACE VIEW v_extraction_progress AS
SELECT 
    qdrant_collection,
    COUNT(*) as total_vectors,
    COUNT(*) FILTER (WHERE fact_extracted = TRUE) as extracted,
    COUNT(*) FILTER (WHERE fact_extracted = FALSE) as pending,
    COUNT(*) FILTER (WHERE extraction_error IS NOT NULL) as errors,
    ROUND(
        COUNT(*) FILTER (WHERE fact_extracted = TRUE)::numeric / 
        NULLIF(COUNT(*), 0) * 100, 2
    ) as coverage_pct,
    SUM(facts_count) as total_facts,
    AVG(facts_count) FILTER (WHERE fact_extracted = TRUE) as avg_facts_per_vector
FROM ingestion_tracking
GROUP BY qdrant_collection
ORDER BY total_vectors DESC;

-- Priority queue view
CREATE OR REPLACE VIEW v_priority_queue AS
SELECT 
    it.id,
    it.qdrant_collection,
    it.domain,
    it.content_length,
    it.priority_score,
    LEFT(vc.content, 100) as content_preview
FROM ingestion_tracking it
JOIN vector_content vc ON vc.tracking_id = it.id
WHERE it.fact_extracted = FALSE
ORDER BY it.priority_score DESC, it.created_at
LIMIT 1000;

-- Domain distribution
CREATE OR REPLACE VIEW v_domain_stats AS
SELECT 
    domain,
    qdrant_collection,
    COUNT(*) as vectors,
    COUNT(*) FILTER (WHERE fact_extracted = TRUE) as with_facts,
    SUM(facts_count) as total_facts
FROM ingestion_tracking
GROUP BY domain, qdrant_collection
ORDER BY vectors DESC;

-- Extraction velocity (for time estimates)
CREATE OR REPLACE VIEW v_extraction_velocity AS
SELECT 
    date_trunc('hour', fact_extracted_at) as hour,
    COUNT(*) as vectors_extracted,
    SUM(facts_count) as facts_extracted,
    AVG(facts_count) as avg_facts
FROM ingestion_tracking
WHERE fact_extracted_at IS NOT NULL
AND fact_extracted_at > NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', fact_extracted_at)
ORDER BY hour DESC;


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Auto-update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_tracking_timestamp ON ingestion_tracking;
CREATE TRIGGER update_tracking_timestamp
    BEFORE UPDATE ON ingestion_tracking
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Calculate priority score based on content characteristics
CREATE OR REPLACE FUNCTION calculate_priority(
    p_domain VARCHAR,
    p_content_length INTEGER,
    p_created_at TIMESTAMP WITH TIME ZONE
) RETURNS FLOAT AS $$
DECLARE
    domain_weight FLOAT;
    length_weight FLOAT;
    recency_weight FLOAT;
BEGIN
    -- Domain priority
    domain_weight := CASE p_domain
        WHEN 'technical' THEN 1.0
        WHEN 'code' THEN 0.9
        WHEN 'conversation' THEN 0.7
        WHEN 'anime' THEN 0.6
        WHEN 'personal' THEN 0.8
        ELSE 0.5
    END;
    
    -- Content length (prefer substantial content, max at 2000 chars)
    length_weight := LEAST(p_content_length::float / 2000.0, 1.0);
    
    -- Recency (prefer recent, decay over 90 days)
    recency_weight := GREATEST(
        1.0 - (EXTRACT(EPOCH FROM NOW() - p_created_at) / (90 * 86400)),
        0.1
    );
    
    RETURN (domain_weight * 0.4 + length_weight * 0.3 + recency_weight * 0.3);
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- INITIAL DATA SETUP
-- ============================================================================

-- Create initial checkpoint records
INSERT INTO extraction_checkpoints (collection_name, checkpoint_type, model_used, batch_size)
VALUES 
    ('echo_memory', 'batch', 'qwen2.5:7b', 100),
    ('claude_conversations', 'batch', 'qwen2.5:7b', 100)
ON CONFLICT (collection_name, checkpoint_type) DO NOTHING;


-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Migration complete!';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - ingestion_tracking (with multi-collection support)';
    RAISE NOTICE '  - vector_content';
    RAISE NOTICE '  - facts (with 1024 + 384 dim embeddings)';
    RAISE NOTICE '  - extraction_checkpoints';
    RAISE NOTICE '  - extraction_queue';
    RAISE NOTICE '';
    RAISE NOTICE 'Views created:';
    RAISE NOTICE '  - v_extraction_progress';
    RAISE NOTICE '  - v_priority_queue';
    RAISE NOTICE '  - v_domain_stats';
    RAISE NOTICE '  - v_extraction_velocity';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '  1. Run backfill script to populate tracking from Qdrant';
    RAISE NOTICE '  2. Run prioritized fact extraction';
    RAISE NOTICE '  3. Monitor progress with: SELECT * FROM v_extraction_progress;';
END $$;
