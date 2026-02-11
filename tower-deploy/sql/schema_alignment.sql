-- ============================================================
-- Tower Anime Production — Schema Alignment
-- ============================================================
-- Run on: anime_production database
-- Purpose: Create validation tracking tables and link to existing schema
--
-- Usage:
--   PGPASSWORD=$DB_PASSWORD \
--   psql -h localhost -U patrick -d anime_production -f schema_alignment.sql
-- ============================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. generation_validation — tracks output validation results
--    Every generation that completes gets a validation record here.
--    This is what the output validator writes to.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS generation_validation (
    id                  SERIAL PRIMARY KEY,
    comfyui_prompt_id   VARCHAR(255) UNIQUE NOT NULL,
    prompt_text         TEXT,
    model_name          VARCHAR(255),
    loras               JSONB DEFAULT '[]',
    validation_status   VARCHAR(50) NOT NULL,        -- passed, failed, partial, no_output
    quality_score       FLOAT DEFAULT 0.0,           -- 0.0 to 1.0
    output_paths        JSONB DEFAULT '[]',          -- list of file paths
    total_images        INTEGER DEFAULT 0,
    passed_images       INTEGER DEFAULT 0,
    failed_images       INTEGER DEFAULT 0,
    issues              JSONB DEFAULT '[]',          -- list of issue strings
    generation_time_s   FLOAT,
    workflow_used       TEXT,
    image_details       JSONB DEFAULT '[]',          -- per-file validation detail
    used_vhs_fallback   BOOLEAN DEFAULT FALSE,       -- true if found via disk scan
    validated_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE generation_validation IS
    'Output validation results for every ComfyUI generation. '
    'Written by generation_output_validator.py.';

-- ---------------------------------------------------------------------------
-- 2. generation_metrics — performance and resource tracking
--    For monitoring dashboard and optimization.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS generation_metrics (
    id                      SERIAL PRIMARY KEY,
    comfyui_prompt_id       VARCHAR(255) REFERENCES generation_validation(comfyui_prompt_id),
    gpu_time_seconds        FLOAT,
    model_load_time_seconds FLOAT,
    generation_time_seconds FLOAT,
    vram_peak_mb            INTEGER,
    queue_wait_seconds      FLOAT,
    success                 BOOLEAN,
    error_message           TEXT,
    workflow_name           TEXT,
    checkpoint_name         VARCHAR(255),
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE generation_metrics IS
    'Performance metrics per generation for monitoring and optimization.';

-- ---------------------------------------------------------------------------
-- 3. Link generation_history → generation_validation
--    The existing generation_history table uses generation_id (UUID).
--    We add a column to cross-reference validation results.
-- ---------------------------------------------------------------------------

-- Check if generation_history exists before altering
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'generation_history'
    ) THEN
        -- Add validation_id FK if column doesn't exist
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'generation_history' AND column_name = 'validation_id'
        ) THEN
            ALTER TABLE generation_history
            ADD COLUMN validation_id INTEGER REFERENCES generation_validation(id);
        END IF;

        -- Add comfyui_prompt_id for direct join if missing
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'generation_history' AND column_name = 'comfyui_prompt_id'
        ) THEN
            ALTER TABLE generation_history
            ADD COLUMN comfyui_prompt_id VARCHAR(255);
        END IF;
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 4. Indexes for monitoring queries
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_gv_status
    ON generation_validation(validation_status);

CREATE INDEX IF NOT EXISTS idx_gv_validated_at
    ON generation_validation(validated_at);

CREATE INDEX IF NOT EXISTS idx_gv_quality
    ON generation_validation(quality_score);

CREATE INDEX IF NOT EXISTS idx_gv_model
    ON generation_validation(model_name);

CREATE INDEX IF NOT EXISTS idx_gm_created
    ON generation_metrics(created_at);

CREATE INDEX IF NOT EXISTS idx_gm_success
    ON generation_metrics(success);

-- ---------------------------------------------------------------------------
-- 5. Monitoring views
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW v_generation_stats_24h AS
SELECT
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN validation_status = 'passed' THEN 1 ELSE 0 END)  AS passed,
    SUM(CASE WHEN validation_status = 'failed' THEN 1 ELSE 0 END)  AS failed,
    SUM(CASE WHEN validation_status = 'partial' THEN 1 ELSE 0 END) AS partial,
    SUM(CASE WHEN validation_status = 'no_output' THEN 1 ELSE 0 END) AS no_output,
    ROUND(AVG(quality_score)::numeric, 3)                       AS avg_quality,
    ROUND(AVG(generation_time_s)::numeric, 1)                   AS avg_gen_time_s,
    SUM(CASE WHEN used_vhs_fallback THEN 1 ELSE 0 END)         AS vhs_fallback_count,
    COUNT(DISTINCT model_name)                                  AS models_used
FROM generation_validation
WHERE validated_at > NOW() - INTERVAL '24 hours';

COMMENT ON VIEW v_generation_stats_24h IS
    'Quick dashboard: generation stats for the last 24 hours.';

CREATE OR REPLACE VIEW v_recent_failures AS
SELECT
    comfyui_prompt_id,
    validation_status,
    model_name,
    workflow_used,
    total_images,
    passed_images,
    issues,
    used_vhs_fallback,
    validated_at
FROM generation_validation
WHERE validation_status IN ('failed', 'no_output')
ORDER BY validated_at DESC
LIMIT 20;

COMMENT ON VIEW v_recent_failures IS
    'Most recent generation failures for debugging.';

COMMIT;

-- ---------------------------------------------------------------------------
-- Verification
-- ---------------------------------------------------------------------------
\echo '--- Schema alignment complete ---'
\echo ''

SELECT 'generation_validation' AS table_name, COUNT(*) AS rows FROM generation_validation
UNION ALL
SELECT 'generation_metrics', COUNT(*) FROM generation_metrics;

\echo ''
\echo 'Views created: v_generation_stats_24h, v_recent_failures'
\echo 'Run: SELECT * FROM v_generation_stats_24h;'
