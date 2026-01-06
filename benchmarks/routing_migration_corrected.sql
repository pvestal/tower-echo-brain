-- Echo Brain Model Routing Consolidation (Corrected)
-- Generated from benchmark results: 2026-01-06
-- Replaces hardcoded routing with data-driven decisions
-- Uses actual database schema

BEGIN;

-- First, update model capabilities based on benchmark data
-- Add or update qwen2.5:3b
INSERT INTO model_capabilities (
    model_name,
    display_name,
    code_generation,
    code_debugging,
    reasoning,
    creative_writing,
    structured_output,
    conversation,
    technical_depth,
    max_context_tokens,
    avg_tokens_per_second,
    vram_required_gb,
    is_installed,
    is_active,
    updated_at
) VALUES (
    'qwen2.5:3b',
    'Qwen 2.5 3B (Classification Optimized)',
    60,  -- Decent for simple coding
    55,  -- Decent for debugging
    65,  -- Good reasoning for 3B model
    70,  -- Good creative writing
    90,  -- Excellent structured output (classification winner)
    85,  -- Good conversation
    60,  -- Moderate technical depth
    32768, -- 32k context
    9.7,   -- From benchmark: 9.7 TPS
    2.5,   -- ~2.5GB VRAM (from VRAM estimates)
    true,  -- Currently installed
    true,  -- Active
    NOW()
) ON CONFLICT (model_name) DO UPDATE SET
    avg_tokens_per_second = 9.7,
    vram_required_gb = 2.5,
    structured_output = 90,  -- Benchmark winner for classification
    is_installed = true,
    updated_at = NOW();

-- Add or update qwen2.5-coder:7b
INSERT INTO model_capabilities (
    model_name,
    display_name,
    code_generation,
    code_debugging,
    reasoning,
    creative_writing,
    structured_output,
    conversation,
    technical_depth,
    max_context_tokens,
    avg_tokens_per_second,
    vram_required_gb,
    is_installed,
    is_active,
    updated_at
) VALUES (
    'qwen2.5-coder:7b',
    'Qwen 2.5 Coder 7B (Coding Optimized)',
    95,  -- Excellent coding generation
    90,  -- Excellent debugging
    80,  -- Good reasoning
    60,  -- Moderate creative writing (coding focus)
    85,  -- Good structured output
    70,  -- Moderate conversation
    90,  -- High technical depth
    32768, -- 32k context
    8.7,   -- From benchmark: 8.7 TPS
    5.0,   -- ~5GB VRAM
    true,  -- Currently installed
    true,  -- Active
    NOW()
) ON CONFLICT (model_name) DO UPDATE SET
    avg_tokens_per_second = 8.7,
    vram_required_gb = 5.0,
    code_generation = 95,
    code_debugging = 90,
    is_installed = true,
    updated_at = NOW();

-- Clear existing routing mappings to avoid conflicts
DELETE FROM intent_model_mapping WHERE intent IN ('classification', 'coding', 'greeting', 'question');

-- Insert benchmark-optimized routing rules
-- Classification: qwen2.5:3b won with 94ms TTFT and 80% accuracy
INSERT INTO intent_model_mapping (
    intent,
    domain,
    recommended_model,
    reason,
    min_complexity,
    max_complexity,
    priority
) VALUES
-- Classification tasks
('classification', 'general', 'qwen2.5:3b',
 'Benchmark winner: 94ms TTFT, 9.7 TPS, 80% accuracy on classification tests',
 0, 100, 10),

-- Greeting and simple interactions (use fast classification model)
('greeting', 'general', 'qwen2.5:3b',
 'Fast response for greetings: 94ms TTFT',
 0, 30, 20),

-- Question answering (use classification model for speed)
('question', 'general', 'qwen2.5:3b',
 'Fast question classification and simple answers',
 0, 50, 30),

-- Coding tasks (use specialized coding model)
('coding', 'general', 'qwen2.5-coder:7b',
 'Specialized coding model with high technical capabilities',
 0, 100, 40),

-- Code debugging (use specialized coding model)
('debugging', 'general', 'qwen2.5-coder:7b',
 'Optimized for code debugging and error analysis',
 0, 100, 45);

COMMIT;

-- Verification queries
SELECT
    intent,
    domain,
    recommended_model,
    reason,
    priority
FROM intent_model_mapping
ORDER BY priority;

SELECT
    model_name,
    avg_tokens_per_second,
    vram_required_gb,
    code_generation,
    structured_output,
    is_installed
FROM model_capabilities
WHERE model_name IN ('qwen2.5:3b', 'qwen2.5-coder:7b');