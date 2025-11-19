-- ==========================================
-- DATABASE PERFORMANCE OPTIMIZATION
-- ==========================================
-- Database: echo_brain
-- Analysis Date: November 19, 2025
-- Focus: Query performance and index optimization

-- ==========================================
-- QUERY PERFORMANCE ANALYSIS RESULTS
-- ==========================================
-- Slow Query Patterns Identified:
-- 1. Character lookup queries (missing indexes)
-- 2. Conversation history retrieval (sequential scans)
-- 3. Context registry searches (bloated indexes)
-- 4. Photo index queries (inefficient file path searches)

-- ==========================================
-- CHARACTER PROFILE OPTIMIZATIONS
-- ==========================================

-- Current character_profiles table is well-indexed but could be optimized
-- for common query patterns

-- Add composite index for character searches
CREATE INDEX IF NOT EXISTS idx_character_search_composite
ON character_profiles (creator, source_franchise, is_active)
WHERE is_active = true;

-- Add index for generation statistics
CREATE INDEX IF NOT EXISTS idx_character_generation_stats
ON character_profiles (generation_count DESC, consistency_score DESC, last_generated DESC)
WHERE is_active = true;

-- Add text search index for character descriptions
CREATE INDEX IF NOT EXISTS idx_character_text_search
ON character_profiles USING gin(
    to_tsvector('english',
        coalesce(character_name, '') || ' ' ||
        coalesce(personality_traits, '') || ' ' ||
        coalesce(background_story, '')
    )
);

-- ==========================================
-- CONVERSATION OPTIMIZATION
-- ==========================================

-- Current conversation queries are inefficient for recent message retrieval
-- Add optimized index for conversation browsing
CREATE INDEX IF NOT EXISTS idx_conversations_recent_active
ON echo_conversations (last_interaction DESC, created_at DESC)
WHERE created_at > CURRENT_DATE - INTERVAL '7 days';

-- Index for conversation search by content
CREATE INDEX IF NOT EXISTS idx_conversations_content_search
ON echo_conversations USING gin(
    to_tsvector('english', coalesce(context, '') || ' ' || coalesce(intent_history, ''))
);

-- ==========================================
-- ECHO TASKS OPTIMIZATION
-- ==========================================

-- Current echo_tasks table (74MB) needs better indexing
-- Add index for task queue processing
CREATE INDEX IF NOT EXISTS idx_tasks_queue_processing
ON echo_tasks (status, priority, created_at)
WHERE status IN ('pending', 'running');

-- Index for task completion tracking
CREATE INDEX IF NOT EXISTS idx_tasks_completion_stats
ON echo_tasks (task_type, completed_at DESC, processing_time)
WHERE completed_at IS NOT NULL;

-- ==========================================
-- PHOTO INDEX OPTIMIZATION
-- ==========================================

-- Photo index (33MB, 49,842 photos, 348GB total) needs better search
-- Current text search is inefficient

-- Replace existing text search with optimized version
DROP INDEX IF EXISTS idx_photo_path;
CREATE INDEX idx_photo_path_optimized
ON photo_index USING gin(
    to_tsvector('english',
        regexp_replace(file_path, '[/\\]', ' ', 'g') -- Convert paths to searchable text
    )
);

-- Add index for file management queries
CREATE INDEX IF NOT EXISTS idx_photo_size_date
ON photo_index (file_size DESC, indexed_date DESC, modified_date DESC);

-- ==========================================
-- MODEL DECISIONS OPTIMIZATION
-- ==========================================

-- model_decisions table (7.6MB) could be optimized
-- Add index for decision analysis
CREATE INDEX IF NOT EXISTS idx_model_decisions_analysis
ON model_decisions (created_at DESC, model_used, processing_time);

-- ==========================================
-- VETERAN RESOURCES OPTIMIZATION
-- ==========================================

-- veteran_support_resources is the largest table (123MB)
-- Needs optimization for resource searches
CREATE INDEX IF NOT EXISTS idx_veteran_resources_search
ON veteran_support_resources USING gin(
    to_tsvector('english',
        coalesce(resource_title, '') || ' ' ||
        coalesce(resource_description, '') || ' ' ||
        coalesce(tags::text, '')
    )
) WHERE resource_status = 'active';

-- ==========================================
-- CONTEXT REGISTRY SEARCH OPTIMIZATION
-- ==========================================

-- After cleanup, optimize remaining context searches
-- Remove bloated search vector and rebuild efficiently
DROP INDEX IF EXISTS idx_context_search_vector;
CREATE INDEX idx_context_search_optimized
ON echo_context_registry USING gin(search_vector)
WHERE source_type IN ('conversation', 'database_schema')
  AND indexed_at > CURRENT_DATE - INTERVAL '30 days';

-- ==========================================
-- AUTONOMOUS TASKS OPTIMIZATION
-- ==========================================

-- Optimize autonomous task queue processing
CREATE INDEX IF NOT EXISTS idx_autonomous_tasks_queue
ON autonomous_tasks (status, priority_level, created_at)
WHERE status IN ('pending', 'running');

-- ==========================================
-- MAINTENANCE CONFIGURATION
-- ==========================================

-- Configure autovacuum for optimal performance
-- High-churn tables need aggressive autovacuum

ALTER TABLE echo_tasks SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05,
    autovacuum_vacuum_cost_delay = 10
);

ALTER TABLE echo_conversations SET (
    autovacuum_vacuum_scale_factor = 0.15,
    autovacuum_analyze_scale_factor = 0.1
);

ALTER TABLE model_decisions SET (
    autovacuum_vacuum_scale_factor = 0.2,
    autovacuum_analyze_scale_factor = 0.1
);

-- ==========================================
-- QUERY PERFORMANCE TESTING
-- ==========================================

-- Test character lookup performance
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT * FROM character_profiles
-- WHERE creator = 'Patrick Vestal'
--   AND is_active = true
--   AND generation_count > 0
-- ORDER BY consistency_score DESC;

-- Test conversation retrieval performance
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT * FROM echo_conversations
-- WHERE last_interaction > CURRENT_DATE - INTERVAL '7 days'
-- ORDER BY last_interaction DESC
-- LIMIT 20;

-- Test context search performance
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT title, content_preview, importance_score
-- FROM echo_context_registry
-- WHERE search_vector @@ plainto_tsquery('anime character')
--   AND source_type = 'conversation'
-- ORDER BY importance_score DESC
-- LIMIT 10;