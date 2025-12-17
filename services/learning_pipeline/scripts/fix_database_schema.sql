-- Echo Brain Learning Pipeline Database Schema
-- This script fixes the conflicting table issues and creates the unified schema

-- First, backup existing data (if any exists)
CREATE TABLE IF NOT EXISTS conversations_backup AS
SELECT * FROM conversations WHERE 1=0;

CREATE TABLE IF NOT EXISTS echo_unified_interactions_backup AS
SELECT * FROM echo_unified_interactions WHERE 1=0;

-- Insert existing data into backup tables (if tables exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'conversations') THEN
        INSERT INTO conversations_backup SELECT * FROM conversations;
        RAISE NOTICE 'Backed up % rows from conversations table', (SELECT COUNT(*) FROM conversations_backup);
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'echo_unified_interactions') THEN
        INSERT INTO echo_unified_interactions_backup SELECT * FROM echo_unified_interactions;
        RAISE NOTICE 'Backed up % rows from echo_unified_interactions table', (SELECT COUNT(*) FROM echo_unified_interactions_backup);
    END IF;
END $$;

-- Drop conflicting tables
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS echo_unified_interactions CASCADE;

-- Create unified learning schema
CREATE TABLE learning_conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL DEFAULT 'claude', -- 'claude', 'kb_article', 'user_input'
    file_path TEXT,
    processed_at TIMESTAMP DEFAULT NOW(),
    last_modified TIMESTAMP NOT NULL DEFAULT NOW(),
    content_hash VARCHAR(64) UNIQUE, -- SHA-256 for duplicate detection
    metadata JSONB DEFAULT '{}',
    processing_status VARCHAR(20) DEFAULT 'pending', -- pending, processed, failed
    error_message TEXT,
    vector_id VARCHAR(255), -- Reference to Qdrant vector
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE learning_items (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) REFERENCES learning_conversations(conversation_id),
    item_type VARCHAR(50) NOT NULL, -- 'insight', 'code_example', 'solution', 'error_fix'
    title VARCHAR(500),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    importance_score FLOAT DEFAULT 0.0, -- ML-generated importance (0-1)
    confidence_score FLOAT DEFAULT 0.0, -- Extraction confidence (0-1)
    categories TEXT[] DEFAULT ARRAY[]::TEXT[], -- Categorization tags
    tags TEXT[] DEFAULT ARRAY[]::TEXT[], -- Additional tags
    content_hash VARCHAR(64) NOT NULL, -- SHA-256 for duplicate detection
    extracted_at TIMESTAMP DEFAULT NOW(),
    vector_embedding_id VARCHAR(255), -- Qdrant vector ID
    embedding_model VARCHAR(100), -- Model used for embedding
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running', -- running, completed, failed
    conversations_processed INTEGER DEFAULT 0,
    articles_processed INTEGER DEFAULT 0,
    learning_items_extracted INTEGER DEFAULT 0,
    vectors_updated INTEGER DEFAULT 0,
    errors_encountered INTEGER DEFAULT 0,
    error_message TEXT,
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_learning_conversations_processed_at ON learning_conversations(processed_at);
CREATE INDEX idx_learning_conversations_status ON learning_conversations(processing_status);
CREATE INDEX idx_learning_conversations_source_type ON learning_conversations(source_type);
CREATE INDEX idx_learning_conversations_hash ON learning_conversations(content_hash);
CREATE INDEX idx_learning_conversations_modified ON learning_conversations(last_modified);

CREATE INDEX idx_learning_items_conversation ON learning_items(conversation_id);
CREATE INDEX idx_learning_items_type ON learning_items(item_type);
CREATE INDEX idx_learning_items_importance ON learning_items(importance_score);
CREATE INDEX idx_learning_items_extracted ON learning_items(extracted_at);
CREATE INDEX idx_learning_items_hash ON learning_items(content_hash);
CREATE INDEX idx_learning_items_categories ON learning_items USING GIN(categories);
CREATE INDEX idx_learning_items_tags ON learning_items USING GIN(tags);

CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX idx_pipeline_runs_started ON pipeline_runs(started_at);
CREATE INDEX idx_pipeline_runs_run_id ON pipeline_runs(run_id);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_learning_conversations_updated_at
    BEFORE UPDATE ON learning_conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_learning_items_updated_at
    BEFORE UPDATE ON learning_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pipeline_runs_updated_at
    BEFORE UPDATE ON pipeline_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for easy querying
CREATE VIEW recent_learning_items AS
SELECT
    li.*,
    lc.source_type,
    lc.file_path,
    lc.processed_at
FROM learning_items li
JOIN learning_conversations lc ON li.conversation_id = lc.conversation_id
WHERE li.extracted_at >= NOW() - INTERVAL '30 days'
ORDER BY li.extracted_at DESC;

CREATE VIEW pipeline_performance AS
SELECT
    DATE_TRUNC('day', started_at) as day,
    COUNT(*) as runs,
    AVG(conversations_processed) as avg_conversations,
    AVG(learning_items_extracted) as avg_items_extracted,
    AVG(vectors_updated) as avg_vectors_updated,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds
FROM pipeline_runs
WHERE status = 'completed'
GROUP BY DATE_TRUNC('day', started_at)
ORDER BY day DESC;

-- Insert configuration data
INSERT INTO learning_conversations (conversation_id, source_type, content_hash, processing_status)
VALUES ('SYSTEM_INIT', 'system', 'system_init_hash', 'processed')
ON CONFLICT (conversation_id) DO NOTHING;

-- Grant permissions (adjust user as needed)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO patrick;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO patrick;

-- Create function to get pipeline stats
CREATE OR REPLACE FUNCTION get_pipeline_stats()
RETURNS TABLE (
    total_conversations INT,
    total_learning_items INT,
    pending_conversations INT,
    failed_conversations INT,
    recent_runs INT,
    avg_processing_time FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*)::INT FROM learning_conversations),
        (SELECT COUNT(*)::INT FROM learning_items),
        (SELECT COUNT(*)::INT FROM learning_conversations WHERE processing_status = 'pending'),
        (SELECT COUNT(*)::INT FROM learning_conversations WHERE processing_status = 'failed'),
        (SELECT COUNT(*)::INT FROM pipeline_runs WHERE started_at >= NOW() - INTERVAL '7 days'),
        (SELECT AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))::FLOAT
         FROM pipeline_runs
         WHERE status = 'completed' AND started_at >= NOW() - INTERVAL '30 days');
END;
$$ LANGUAGE plpgsql;

-- Create function to cleanup old data
CREATE OR REPLACE FUNCTION cleanup_old_pipeline_data(days_to_keep INT DEFAULT 90)
RETURNS TABLE (
    deleted_conversations INT,
    deleted_items INT,
    deleted_runs INT
) AS $$
DECLARE
    del_convs INT;
    del_items INT;
    del_runs INT;
BEGIN
    -- Delete old learning items
    DELETE FROM learning_items
    WHERE extracted_at < NOW() - (days_to_keep || ' days')::INTERVAL;
    GET DIAGNOSTICS del_items = ROW_COUNT;

    -- Delete old conversations with no associated items
    DELETE FROM learning_conversations
    WHERE processed_at < NOW() - (days_to_keep || ' days')::INTERVAL
    AND id NOT IN (SELECT DISTINCT conversation_id FROM learning_items WHERE conversation_id IS NOT NULL);
    GET DIAGNOSTICS del_convs = ROW_COUNT;

    -- Delete old pipeline runs
    DELETE FROM pipeline_runs
    WHERE started_at < NOW() - (days_to_keep || ' days')::INTERVAL;
    GET DIAGNOSTICS del_runs = ROW_COUNT;

    RETURN QUERY SELECT del_convs, del_items, del_runs;
END;
$$ LANGUAGE plpgsql;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Echo Brain Learning Pipeline database schema created successfully';
    RAISE NOTICE 'Tables created: learning_conversations, learning_items, pipeline_runs';
    RAISE NOTICE 'Views created: recent_learning_items, pipeline_performance';
    RAISE NOTICE 'Functions created: get_pipeline_stats(), cleanup_old_pipeline_data()';
END $$;