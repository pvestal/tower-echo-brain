-- Track fact extraction progress
CREATE TABLE IF NOT EXISTS fact_extraction_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_collection VARCHAR(50) NOT NULL,
    source_point_id VARCHAR(100) NOT NULL,
    content_hash VARCHAR(64),
    extraction_model VARCHAR(100),
    facts_extracted INTEGER DEFAULT 0,
    processed_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'completed',
    error_message TEXT,
    UNIQUE(source_collection, source_point_id)
);

CREATE INDEX IF NOT EXISTS idx_extraction_log_collection ON fact_extraction_log(source_collection);
CREATE INDEX IF NOT EXISTS idx_extraction_log_status ON fact_extraction_log(status);

-- Coverage view
CREATE OR REPLACE VIEW extraction_coverage AS
SELECT
    source_collection,
    COUNT(*) as processed,
    SUM(facts_extracted) as facts_found,
    COUNT(*) FILTER (WHERE status = 'failed') as failures,
    MAX(processed_at) as last_run
FROM fact_extraction_log
GROUP BY source_collection;