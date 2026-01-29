-- Echo Brain Context Assembly Pipeline
-- Migration 001: Create ingestion tracking and facts tables
-- Run this in PostgreSQL to set up the required schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for embeddings

-- ============================================================================
-- INGESTION TRACKING
-- Tracks every source that has been or should be ingested
-- ============================================================================

CREATE TABLE IF NOT EXISTS ingestion_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Source identification
    source_type VARCHAR(50) NOT NULL,  -- 'document', 'conversation', 'code', 'external'
    source_path TEXT NOT NULL,
    source_hash VARCHAR(64) NOT NULL,  -- SHA256 for deduplication
    
    -- Ingestion status
    vector_id UUID,                     -- Reference to Qdrant vector ID
    vectorized_at TIMESTAMP WITH TIME ZONE,
    
    -- Fact extraction status
    fact_extracted BOOLEAN DEFAULT FALSE,
    fact_extracted_at TIMESTAMP WITH TIME ZONE,
    facts_count INTEGER DEFAULT 0,
    extraction_error TEXT,              -- Store error message if extraction failed
    
    -- Domain classification
    domain VARCHAR(50),                 -- 'technical', 'anime', 'personal', 'general'
    
    -- Metadata
    chunk_count INTEGER,
    token_count INTEGER,
    file_size_bytes BIGINT,
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(source_hash)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ingestion_not_extracted 
    ON ingestion_tracking(fact_extracted) 
    WHERE fact_extracted = FALSE;

CREATE INDEX IF NOT EXISTS idx_ingestion_domain 
    ON ingestion_tracking(domain);

CREATE INDEX IF NOT EXISTS idx_ingestion_source_type 
    ON ingestion_tracking(source_type);

CREATE INDEX IF NOT EXISTS idx_ingestion_source_path 
    ON ingestion_tracking(source_path);

CREATE INDEX IF NOT EXISTS idx_ingestion_created 
    ON ingestion_tracking(created_at);


-- ============================================================================
-- VECTOR CONTENT CACHE
-- Stores the actual content for vectors (for fact extraction)
-- ============================================================================

CREATE TABLE IF NOT EXISTS vector_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tracking_id UUID NOT NULL REFERENCES ingestion_tracking(id) ON DELETE CASCADE,
    
    -- The content
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    
    -- Chunking info
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tracking_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_vector_content_tracking 
    ON vector_content(tracking_id);


-- ============================================================================
-- FACTS
-- Extracted facts from content
-- ============================================================================

CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES ingestion_tracking(id) ON DELETE CASCADE,
    
    -- The fact itself
    fact_text TEXT NOT NULL,
    fact_type VARCHAR(50) NOT NULL,     -- 'entity', 'relationship', 'event', 'preference', 'technical', 'temporal'
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Domain isolation (denormalized for query performance)
    domain VARCHAR(50) NOT NULL,
    
    -- Structured extraction (Subject-Predicate-Object triple)
    subject TEXT,                       -- Who/what the fact is about
    predicate TEXT,                     -- The relationship/action  
    object TEXT,                        -- The target/value
    
    -- Temporal bounds
    valid_from TIMESTAMP WITH TIME ZONE,
    valid_until TIMESTAMP WITH TIME ZONE,  -- NULL = still valid
    
    -- Embedding for semantic search on facts (384 dims for nomic-embed-text)
    embedding VECTOR(384),
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for fact retrieval
CREATE INDEX IF NOT EXISTS idx_facts_domain 
    ON facts(domain);

CREATE INDEX IF NOT EXISTS idx_facts_type 
    ON facts(fact_type);

CREATE INDEX IF NOT EXISTS idx_facts_subject 
    ON facts(subject) 
    WHERE subject IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_facts_source 
    ON facts(source_id);

CREATE INDEX IF NOT EXISTS idx_facts_valid 
    ON facts(valid_until) 
    WHERE valid_until IS NULL OR valid_until > NOW();

-- Vector similarity index for semantic search on facts
CREATE INDEX IF NOT EXISTS idx_facts_embedding 
    ON facts 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);


-- ============================================================================
-- CONVERSATION TURNS
-- Stores conversation history for context retrieval
-- ============================================================================

CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL,      -- Groups turns into conversations
    
    -- Turn data
    role VARCHAR(20) NOT NULL,          -- 'user' or 'assistant'
    content TEXT NOT NULL,
    
    -- Metadata
    domain VARCHAR(50),                 -- Classified domain of this turn
    token_count INTEGER,
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_conv 
    ON conversation_turns(conversation_id);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_created 
    ON conversation_turns(created_at DESC);


-- ============================================================================
-- CONTEXT ASSEMBLY LOG
-- Logs context assembly operations for debugging and optimization
-- ============================================================================

CREATE TABLE IF NOT EXISTS context_assembly_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Query info
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64),             -- For finding duplicate queries
    classified_domain VARCHAR(50),
    classification_confidence FLOAT,
    
    -- What was retrieved
    vectors_retrieved INTEGER,
    facts_retrieved INTEGER,
    conversation_turns INTEGER,
    code_files INTEGER,
    
    -- Assembly metrics
    total_tokens INTEGER,
    assembly_time_ms INTEGER,
    
    -- For debugging/analysis
    retrieved_vector_ids UUID[],
    retrieved_fact_ids UUID[],
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_context_log_created 
    ON context_assembly_log(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_context_log_domain 
    ON context_assembly_log(classified_domain);


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for ingestion_tracking
DROP TRIGGER IF EXISTS update_ingestion_tracking_updated_at ON ingestion_tracking;
CREATE TRIGGER update_ingestion_tracking_updated_at
    BEFORE UPDATE ON ingestion_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- VIEWS FOR MONITORING
-- ============================================================================

-- Coverage summary view
CREATE OR REPLACE VIEW v_coverage_summary AS
SELECT 
    source_type,
    domain,
    COUNT(*) as total_sources,
    COUNT(*) FILTER (WHERE vector_id IS NOT NULL) as vectorized,
    COUNT(*) FILTER (WHERE fact_extracted = TRUE) as facts_extracted,
    ROUND(
        COUNT(*) FILTER (WHERE fact_extracted = TRUE)::numeric / 
        NULLIF(COUNT(*), 0) * 100, 
        2
    ) as coverage_pct,
    SUM(facts_count) as total_facts
FROM ingestion_tracking
GROUP BY source_type, domain
ORDER BY source_type, domain;

-- Extraction queue view (pending items)
CREATE OR REPLACE VIEW v_extraction_queue AS
SELECT 
    it.id,
    it.source_type,
    it.source_path,
    it.domain,
    it.created_at,
    LENGTH(vc.content) as content_length
FROM ingestion_tracking it
JOIN vector_content vc ON vc.tracking_id = it.id
WHERE it.fact_extracted = FALSE
AND it.vector_id IS NOT NULL
ORDER BY it.created_at;

-- Fact distribution view
CREATE OR REPLACE VIEW v_fact_distribution AS
SELECT 
    domain,
    fact_type,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence
FROM facts
WHERE valid_until IS NULL OR valid_until > NOW()
GROUP BY domain, fact_type
ORDER BY domain, count DESC;


-- ============================================================================
-- GRANT PERMISSIONS (adjust as needed for your setup)
-- ============================================================================

-- If using a specific application user, grant permissions:
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO echo_brain_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO echo_brain_app;


-- ============================================================================
-- VERIFICATION QUERY
-- Run this to verify the migration worked
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Migration complete. Tables created:';
    RAISE NOTICE '  - ingestion_tracking';
    RAISE NOTICE '  - vector_content';
    RAISE NOTICE '  - facts';
    RAISE NOTICE '  - conversation_turns';
    RAISE NOTICE '  - context_assembly_log';
    RAISE NOTICE 'Views created:';
    RAISE NOTICE '  - v_coverage_summary';
    RAISE NOTICE '  - v_extraction_queue';
    RAISE NOTICE '  - v_fact_distribution';
END $$;
