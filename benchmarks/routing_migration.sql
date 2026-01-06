-- Echo Brain Model Routing Consolidation
-- Generated from benchmark results: 2026-01-06
-- Replaces hardcoded routing with data-driven decisions

BEGIN;

-- Clean up existing routing chaos
TRUNCATE TABLE intent_model_mapping CASCADE;
TRUNCATE TABLE model_routing CASCADE;

-- Insert benchmark-optimized routing
INSERT INTO intent_model_mapping (intent, recommended_model, confidence, reasoning)
VALUES ('classification', 'qwen2.5:3b', 0.864, 
        'Benchmark winner: 94ms TTFT, 9.7 TPS');

-- Update model capabilities table
INSERT INTO model_capabilities (model_name, category, ttft_ms, tps, vram_mb, last_benchmarked)
VALUES
  ('qwen2.5:3b', 'classification', 94, 9.7, 10523.0, NOW()),
  ('qwen2.5-coder:7b', 'classification', 94, 8.7, 10525.0, NOW());

COMMIT;

-- Verification queries:
SELECT intent, recommended_model, confidence FROM intent_model_mapping;
SELECT model_name, category, ttft_ms, tps FROM model_capabilities ORDER BY category, ttft_ms;