-- ==========================================
-- DATA RETENTION POLICY IMPLEMENTATION
-- ==========================================
-- Database: echo_brain
-- Analysis Date: November 19, 2025
-- Purpose: Implement intelligent data retention to prevent future bloat

-- ==========================================
-- RETENTION POLICY CONFIGURATION
-- ==========================================

-- Create table to manage retention policies
CREATE TABLE IF NOT EXISTS data_retention_policies (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL UNIQUE,
    retention_days INTEGER NOT NULL,
    cleanup_field VARCHAR(100) NOT NULL,
    conditions JSONB DEFAULT '{}',
    last_cleanup TIMESTAMP DEFAULT NULL,
    next_cleanup TIMESTAMP DEFAULT CURRENT_DATE + INTERVAL '1 day',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert retention policies for all major tables
INSERT INTO data_retention_policies (table_name, retention_days, cleanup_field, conditions) VALUES
('echo_context_registry', 90, 'indexed_at', '{"source_type": "file", "min_importance_score": 0.3}'),
('echo_conversations', 180, 'created_at', '{"exclude_important": true}'),
('echo_tasks', 30, 'completed_at', '{"status": "completed"}'),
('echo_unified_interactions', 60, 'timestamp', '{}'),
('model_decisions', 90, 'created_at', '{}'),
('autonomous_tasks', 14, 'completed_at', '{"status": "completed"}'),
('echo_context_usage', 45, 'used_at', '{}'),
('restart_cooldowns', 7, 'updated_at', '{}'),
('photo_index', 365, 'indexed_date', '{"exclude_recent_access": true}'),
('personal_data_insights', 180, 'learned_at', '{}'),
('anime_echo_sessions', 30, 'expires_at', '{}')
ON CONFLICT (table_name) DO UPDATE SET
    retention_days = EXCLUDED.retention_days,
    cleanup_field = EXCLUDED.cleanup_field,
    conditions = EXCLUDED.conditions;

-- ==========================================
-- AUTOMATED CLEANUP FUNCTIONS
-- ==========================================

-- Function to clean up echo_context_registry based on retention policy
CREATE OR REPLACE FUNCTION cleanup_context_registry()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    policy_record RECORD;
BEGIN
    SELECT * INTO policy_record
    FROM data_retention_policies
    WHERE table_name = 'echo_context_registry' AND is_active = true;

    IF NOT FOUND THEN
        RETURN 0;
    END IF;

    -- Delete old file records with low importance
    DELETE FROM echo_context_registry
    WHERE source_type = 'file'
      AND indexed_at < CURRENT_DATE - (policy_record.retention_days || ' days')::INTERVAL
      AND importance_score < (policy_record.conditions->>'min_importance_score')::FLOAT
      AND access_frequency <= 1;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Update cleanup timestamp
    UPDATE data_retention_policies
    SET last_cleanup = CURRENT_TIMESTAMP,
        next_cleanup = CURRENT_DATE + INTERVAL '1 day'
    WHERE table_name = 'echo_context_registry';

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up completed tasks
CREATE OR REPLACE FUNCTION cleanup_completed_tasks()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    policy_record RECORD;
BEGIN
    SELECT * INTO policy_record
    FROM data_retention_policies
    WHERE table_name = 'echo_tasks' AND is_active = true;

    IF NOT FOUND THEN
        RETURN 0;
    END IF;

    -- Archive successful tasks older than retention period
    DELETE FROM echo_tasks
    WHERE status = 'completed'
      AND completed_at < CURRENT_DATE - (policy_record.retention_days || ' days')::INTERVAL
      AND task_type NOT IN ('CRITICAL_SYSTEM_REPAIR', 'SECURITY_EVENT');

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    UPDATE data_retention_policies
    SET last_cleanup = CURRENT_TIMESTAMP,
        next_cleanup = CURRENT_DATE + INTERVAL '1 day'
    WHERE table_name = 'echo_tasks';

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old interactions
CREATE OR REPLACE FUNCTION cleanup_old_interactions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    policy_record RECORD;
BEGIN
    SELECT * INTO policy_record
    FROM data_retention_policies
    WHERE table_name = 'echo_unified_interactions' AND is_active = true;

    IF NOT FOUND THEN
        RETURN 0;
    END IF;

    -- Keep important conversations, remove routine interactions
    DELETE FROM echo_unified_interactions
    WHERE timestamp < CURRENT_DATE - (policy_record.retention_days || ' days')::INTERVAL
      AND interaction_type NOT IN ('important_decision', 'learning_milestone', 'error_resolution')
      AND processing_time < 5000; -- Keep complex processing events

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    UPDATE data_retention_policies
    SET last_cleanup = CURRENT_TIMESTAMP,
        next_cleanup = CURRENT_DATE + INTERVAL '1 day'
    WHERE table_name = 'echo_unified_interactions';

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Master cleanup function
CREATE OR REPLACE FUNCTION run_retention_cleanup()
RETURNS TABLE(table_name TEXT, deleted_records INTEGER, execution_time INTERVAL) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    result_record RECORD;
BEGIN
    FOR result_record IN
        SELECT p.table_name
        FROM data_retention_policies p
        WHERE p.is_active = true
          AND p.next_cleanup <= CURRENT_TIMESTAMP
        ORDER BY p.table_name
    LOOP
        start_time := CURRENT_TIMESTAMP;

        -- Call appropriate cleanup function based on table
        CASE result_record.table_name
            WHEN 'echo_context_registry' THEN
                SELECT cleanup_context_registry() INTO deleted_records;
            WHEN 'echo_tasks' THEN
                SELECT cleanup_completed_tasks() INTO deleted_records;
            WHEN 'echo_unified_interactions' THEN
                SELECT cleanup_old_interactions() INTO deleted_records;
            ELSE
                deleted_records := 0;
        END CASE;

        end_time := CURRENT_TIMESTAMP;
        table_name := result_record.table_name;
        execution_time := end_time - start_time;

        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- BLOAT MONITORING
-- ==========================================

-- Create table to track table bloat over time
CREATE TABLE IF NOT EXISTS table_bloat_monitoring (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    table_size_bytes BIGINT NOT NULL,
    dead_tuples BIGINT NOT NULL,
    live_tuples BIGINT NOT NULL,
    bloat_percentage NUMERIC(5,2) DEFAULT 0,
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to record current bloat statistics
CREATE OR REPLACE FUNCTION record_bloat_stats()
RETURNS INTEGER AS $$
DECLARE
    table_record RECORD;
    inserted_count INTEGER := 0;
BEGIN
    FOR table_record IN
        SELECT
            schemaname,
            tablename,
            n_tup_ins + n_tup_upd + n_tup_del as total_operations,
            n_dead_tup,
            n_live_tup,
            CASE
                WHEN n_live_tup > 0 THEN
                    ROUND((n_dead_tup::NUMERIC / (n_live_tup + n_dead_tup)) * 100, 2)
                ELSE 0
            END as bloat_pct
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
          AND (n_dead_tup > 1000 OR n_live_tup > 10000)
    LOOP
        INSERT INTO table_bloat_monitoring (
            table_name,
            table_size_bytes,
            dead_tuples,
            live_tuples,
            bloat_percentage
        )
        SELECT
            table_record.tablename,
            pg_total_relation_size(table_record.tablename::regclass),
            table_record.n_dead_tup,
            table_record.n_live_tup,
            table_record.bloat_pct;

        inserted_count := inserted_count + 1;
    END LOOP;

    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- SCHEDULED MAINTENANCE
-- ==========================================

-- Create function to run daily maintenance
CREATE OR REPLACE FUNCTION daily_maintenance()
RETURNS TEXT AS $$
DECLARE
    cleanup_results TEXT;
    bloat_stats INTEGER;
    maintenance_log TEXT;
BEGIN
    maintenance_log := 'Daily Maintenance Report - ' || CURRENT_TIMESTAMP || E'\n';
    maintenance_log := maintenance_log || '=' || repeat('=', 50) || E'\n\n';

    -- Run retention cleanup
    SELECT string_agg(
        'Table: ' || table_name || ', Deleted: ' || deleted_records || ', Time: ' || execution_time,
        E'\n'
    ) INTO cleanup_results
    FROM run_retention_cleanup();

    maintenance_log := maintenance_log || 'Retention Cleanup Results:' || E'\n';
    maintenance_log := maintenance_log || COALESCE(cleanup_results, 'No cleanup required') || E'\n\n';

    -- Record bloat statistics
    SELECT record_bloat_stats() INTO bloat_stats;
    maintenance_log := maintenance_log || 'Bloat monitoring: ' || bloat_stats || ' tables recorded' || E'\n\n';

    -- Update table statistics for query planner
    ANALYZE echo_context_registry;
    ANALYZE echo_conversations;
    ANALYZE echo_tasks;
    maintenance_log := maintenance_log || 'Table statistics updated for query planner' || E'\n';

    RETURN maintenance_log;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- USAGE INSTRUCTIONS
-- ==========================================

-- Run immediate cleanup (use with caution in production):
-- SELECT * FROM run_retention_cleanup();

-- Check current bloat status:
-- SELECT record_bloat_stats();
-- SELECT * FROM table_bloat_monitoring ORDER BY measured_at DESC LIMIT 10;

-- Run daily maintenance:
-- SELECT daily_maintenance();

-- View retention policies:
-- SELECT * FROM data_retention_policies ORDER BY table_name;

-- Disable retention for specific table:
-- UPDATE data_retention_policies SET is_active = false WHERE table_name = 'table_name';