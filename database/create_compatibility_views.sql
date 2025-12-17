-- Compatibility Views for Legacy Echo Brain Code
-- Creates views that map old table/column names to new schema

-- Drop existing views if they exist
DROP VIEW IF EXISTS conversations CASCADE;

-- Create compatibility view for 'conversations' table
-- Maps to echo_unified_interactions
CREATE VIEW conversations AS
SELECT
    id,
    conversation_id,
    user_id,
    query AS query_text,
    response AS response_text,
    model_used,
    processing_time,
    escalation_path,
    intent,
    confidence,
    requires_clarification,
    clarifying_questions,
    metadata,
    timestamp
FROM echo_unified_interactions;

-- Grant permissions on view
GRANT ALL ON conversations TO patrick;

-- Add comment explaining this is a compatibility view
COMMENT ON VIEW conversations IS 'Compatibility view for legacy code - maps to echo_unified_interactions';

-- Test the view works
SELECT COUNT(*) as test_count FROM conversations;