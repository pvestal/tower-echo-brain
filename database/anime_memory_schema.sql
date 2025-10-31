-- Echo Brain Anime Memory Enhancement Schema
-- Adds anime production context and creative preference learning to Echo's database

-- Table for storing user's creative preferences learned over time
CREATE TABLE IF NOT EXISTS anime_creative_preferences (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'patrick',
    preference_type VARCHAR(50) NOT NULL, -- style, character_type, scene_type, quality
    preference_value TEXT NOT NULL,
    confidence_score FLOAT DEFAULT 0.5, -- How confident we are in this preference
    evidence_count INTEGER DEFAULT 1, -- How many examples support this preference
    last_reinforced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, preference_type, preference_value)
);

-- Table for anime generation learning records
CREATE TABLE IF NOT EXISTS anime_learning_records (
    id SERIAL PRIMARY KEY,
    generation_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'patrick',
    request_data JSONB NOT NULL, -- Original request parameters
    result_data JSONB, -- Generation results
    learning_metadata JSONB, -- Extracted style elements, quality metrics
    user_rating INTEGER, -- User feedback (1-5 stars)
    user_feedback TEXT, -- Text feedback from user
    preference_updates JSONB, -- What preferences were updated based on this
    rated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for character consistency tracking across projects
CREATE TABLE IF NOT EXISTS anime_character_memory (
    id SERIAL PRIMARY KEY,
    character_name TEXT NOT NULL,
    project_id INTEGER, -- Can link to anime_production database
    canonical_description TEXT NOT NULL,
    visual_consistency_score FLOAT DEFAULT 0.8,
    generation_count INTEGER DEFAULT 0,
    successful_generations INTEGER DEFAULT 0,
    reference_images JSONB, -- Array of reference image paths
    comfyui_workflow_template JSONB, -- Stored ComfyUI workflow for this character
    style_elements JSONB, -- Common style elements that work for this character
    last_generated TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for project context and storyline memory
CREATE TABLE IF NOT EXISTS anime_project_memory (
    id SERIAL PRIMARY KEY,
    project_id INTEGER, -- Links to anime_production database
    project_name TEXT NOT NULL,
    storyline_context JSONB, -- Current story state, character relationships
    style_guide JSONB, -- Project-specific style preferences
    generation_history JSONB, -- Timeline of generations
    character_list JSONB, -- Characters involved in this project
    consistency_rules JSONB, -- Rules for maintaining consistency
    creative_direction TEXT, -- Overall creative vision
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id)
);

-- Table for conversation context linking anime discussions to projects
CREATE TABLE IF NOT EXISTS anime_conversation_context (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL, -- Links to Echo's conversation table
    project_id INTEGER, -- Links to anime project
    character_mentioned TEXT[], -- Characters discussed
    generation_requested BOOLEAN DEFAULT FALSE,
    context_type VARCHAR(50), -- planning, feedback, generation, review
    extracted_preferences JSONB, -- Preferences mentioned in conversation
    follow_up_actions JSONB, -- Actions to take based on conversation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_anime_preferences_user_type ON anime_creative_preferences(user_id, preference_type);
CREATE INDEX IF NOT EXISTS idx_anime_learning_user_rated ON anime_learning_records(user_id, rated_at) WHERE user_rating IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_anime_character_name_project ON anime_character_memory(character_name, project_id);
CREATE INDEX IF NOT EXISTS idx_anime_project_memory_name ON anime_project_memory(project_name);
CREATE INDEX IF NOT EXISTS idx_anime_conversation_project ON anime_conversation_context(project_id);

-- Insert initial creative preferences based on Patrick's known preferences
INSERT INTO anime_creative_preferences (preference_type, preference_value, confidence_score, evidence_count) VALUES
('model', 'counterfeit_v3.safetensors', 0.9, 10),
('vae', 'vae-ft-mse-840000-ema-pruned.safetensors', 0.85, 8),
('style', 'professional anime style', 0.8, 15),
('style', 'detailed character design', 0.8, 12),
('style', 'cinematic lighting', 0.7, 8),
('quality', 'high resolution', 0.9, 20),
('scene_type', 'character portrait', 0.7, 10),
('scene_type', 'action scene', 0.6, 5)
ON CONFLICT (user_id, preference_type, preference_value) DO UPDATE SET
    confidence_score = EXCLUDED.confidence_score,
    evidence_count = EXCLUDED.evidence_count,
    last_reinforced = CURRENT_TIMESTAMP;

-- Create view for easy preference analysis
CREATE OR REPLACE VIEW anime_preference_analysis AS
SELECT
    preference_type,
    preference_value,
    confidence_score,
    evidence_count,
    CASE
        WHEN confidence_score >= 0.8 AND evidence_count >= 5 THEN 'strong'
        WHEN confidence_score >= 0.6 AND evidence_count >= 3 THEN 'moderate'
        ELSE 'weak'
    END as preference_strength,
    last_reinforced
FROM anime_creative_preferences
WHERE user_id = 'patrick'
ORDER BY preference_type, confidence_score DESC;

-- Function to update preferences based on user feedback
CREATE OR REPLACE FUNCTION update_anime_preferences(
    p_generation_id TEXT,
    p_user_rating INTEGER,
    p_feedback TEXT DEFAULT NULL
) RETURNS void AS $$
DECLARE
    learning_record RECORD;
    style_element TEXT;
    pref_exists BOOLEAN;
BEGIN
    -- Get the learning record
    SELECT * INTO learning_record
    FROM anime_learning_records
    WHERE generation_id = p_generation_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Generation record not found: %', p_generation_id;
    END IF;

    -- Update the learning record with feedback
    UPDATE anime_learning_records
    SET user_rating = p_user_rating,
        user_feedback = p_feedback,
        rated_at = CURRENT_TIMESTAMP
    WHERE generation_id = p_generation_id;

    -- If positive feedback (4-5 stars), reinforce preferences
    IF p_user_rating >= 4 THEN
        -- Process style elements from the generation
        FOR style_element IN
            SELECT jsonb_array_elements_text(learning_record.learning_metadata->'style_elements')
        LOOP
            -- Check if preference exists
            SELECT EXISTS(
                SELECT 1 FROM anime_creative_preferences
                WHERE user_id = 'patrick'
                AND preference_type = 'style'
                AND preference_value = style_element
            ) INTO pref_exists;

            IF pref_exists THEN
                -- Reinforce existing preference
                UPDATE anime_creative_preferences
                SET confidence_score = LEAST(confidence_score + 0.1, 1.0),
                    evidence_count = evidence_count + 1,
                    last_reinforced = CURRENT_TIMESTAMP
                WHERE user_id = 'patrick'
                AND preference_type = 'style'
                AND preference_value = style_element;
            ELSE
                -- Create new preference
                INSERT INTO anime_creative_preferences
                (preference_type, preference_value, confidence_score, evidence_count)
                VALUES ('style', style_element, 0.6, 1);
            END IF;
        END LOOP;
    END IF;

    -- If negative feedback (1-2 stars), reduce preference confidence
    IF p_user_rating <= 2 THEN
        FOR style_element IN
            SELECT jsonb_array_elements_text(learning_record.learning_metadata->'style_elements')
        LOOP
            UPDATE anime_creative_preferences
            SET confidence_score = GREATEST(confidence_score - 0.15, 0.1)
            WHERE user_id = 'patrick'
            AND preference_type = 'style'
            AND preference_value = style_element;
        END LOOP;
    END IF;

END;
$$ LANGUAGE plpgsql;