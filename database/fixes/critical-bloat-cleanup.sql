-- ==========================================
-- CRITICAL DATABASE BLOAT CLEANUP FIXES
-- ==========================================
-- Database: echo_brain
-- Analysis Date: November 19, 2025
-- Critical Issue: 5.1GB echo_context_registry table bloat

-- ==========================================
-- ISSUE 1: MASSIVE FILE CONTEXT BLOAT
-- ==========================================
-- Problem: 895,274 file records in echo_context_registry (5.1GB table)
-- Impact: Database performance severely degraded
-- Root Cause: Aggressive file indexing without retention limits

-- Step 1: Identify duplicate content by hash
-- 78,474 records have the same hash (5f5388d2c7aa0affe33fedf30989dded)
-- 35,107 records have empty content hash (d41d8cd98f00b204e9800998ecf8427e)

-- WARNING: These operations will permanently delete data
-- Create backup before running:
-- pg_dump -h 192.168.50.135 -U patrick echo_brain > /tmp/echo_brain_backup_$(date +%Y%m%d_%H%M%S).sql

-- Remove duplicate content (keep newest record for each hash)
BEGIN;

-- Remove empty content files (35,107 records)
DELETE FROM echo_context_registry
WHERE source_type = 'file'
  AND (content_full IS NULL OR content_full = '' OR TRIM(content_full) = '');

-- Remove duplicate content (keep most recently indexed)
DELETE FROM echo_context_registry
WHERE id IN (
    SELECT id
    FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY content_hash
                   ORDER BY indexed_at DESC
               ) as rn
        FROM echo_context_registry
        WHERE content_hash IS NOT NULL
          AND source_type = 'file'
    ) ranked
    WHERE rn > 1
);

-- Archive old file references (older than 90 days with low importance)
DELETE FROM echo_context_registry
WHERE source_type = 'file'
  AND indexed_at < CURRENT_DATE - INTERVAL '90 days'
  AND importance_score < 0.3
  AND access_frequency = 0;

COMMIT;

-- ==========================================
-- ISSUE 2: MISSING PRIMARY KEYS
-- ==========================================

-- Fix missing primary keys on tables
ALTER TABLE veteran_resources_archive ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY;
ALTER TABLE anime_prompt_templates ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY;
ALTER TABLE anime_generation_history ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY;
ALTER TABLE anime_style_feedback ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY;

-- ==========================================
-- ISSUE 3: INDEX OPTIMIZATION
-- ==========================================

-- Remove redundant indexes on echo_context_registry
-- Current indexes total 417MB - many are redundant
DROP INDEX IF EXISTS idx_context_hash; -- 26MB - content_hash rarely queried alone
DROP INDEX IF EXISTS idx_context_modified; -- 14MB - source_modified_at rarely used
DROP INDEX IF EXISTS idx_context_category; -- 9.6MB - covered by composite index

-- Add missing performance indexes
CREATE INDEX IF NOT EXISTS idx_context_registry_cleanup
ON echo_context_registry (source_type, indexed_at, importance_score)
WHERE source_type = 'file';

-- Optimize conversation retrieval
CREATE INDEX IF NOT EXISTS idx_conversations_user_recent
ON echo_conversations (created_at DESC, last_interaction DESC)
WHERE created_at > CURRENT_DATE - INTERVAL '30 days';

-- ==========================================
-- ISSUE 4: TABLE MAINTENANCE
-- ==========================================

-- Full vacuum on bloated table (will take significant time)
-- Run during low usage period
VACUUM FULL echo_context_registry;

-- Update table statistics
ANALYZE echo_context_registry;
ANALYZE echo_conversations;
ANALYZE echo_tasks;
ANALYZE photo_index;

-- ==========================================
-- ISSUE 5: AUTOVACUUM TUNING
-- ==========================================

-- Increase autovacuum frequency for high-churn tables
ALTER TABLE echo_context_registry SET (
    autovacuum_vacuum_scale_factor = 0.05,  -- Vacuum when 5% dead tuples
    autovacuum_analyze_scale_factor = 0.05  -- Analyze when 5% changes
);

ALTER TABLE echo_conversations SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.1
);

-- ==========================================
-- POST-CLEANUP VERIFICATION QUERIES
-- ==========================================

-- Check reduction in table size
-- SELECT
--     pg_size_pretty(pg_total_relation_size('echo_context_registry')) as table_size,
--     COUNT(*) as record_count
-- FROM echo_context_registry;

-- Verify no orphaned records
-- SELECT COUNT(*) as orphaned_usage
-- FROM echo_context_usage ecu
-- LEFT JOIN echo_context_registry ecr ON ecu.registry_id = ecr.id
-- WHERE ecr.id IS NULL;

-- Check dead tuple reduction
-- SELECT n_dead_tup FROM pg_stat_user_tables WHERE relname = 'echo_context_registry';