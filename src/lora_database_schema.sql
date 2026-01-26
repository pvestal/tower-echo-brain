-- LoRA Training Database Schema
-- Comprehensive tracking for all LoRA types and training status

-- Categories of LoRAs
CREATE TABLE IF NOT EXISTS lora_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    priority INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default categories
INSERT INTO lora_categories (name, description, priority) VALUES
('pose', 'Body positions and poses', 10),
('action', 'Actions and movements', 9),
('style', 'Visual styles and aesthetics', 8),
('scene', 'Environments and settings', 7),
('character', 'Specific character appearances', 6),
('clothing', 'Clothing and outfits', 5),
('emotion', 'Facial expressions and emotions', 5),
('object', 'Objects and props', 4)
ON CONFLICT (name) DO NOTHING;

-- Specific LoRA definitions
CREATE TABLE IF NOT EXISTS lora_definitions (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES lora_categories(id),
    name VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    trigger_word VARCHAR(255),
    description TEXT,
    base_prompt TEXT,
    negative_prompt TEXT,
    training_priority INTEGER DEFAULT 5,
    is_nsfw BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training status tracking
CREATE TABLE IF NOT EXISTS lora_training_status (
    id SERIAL PRIMARY KEY,
    definition_id INTEGER REFERENCES lora_definitions(id),
    status VARCHAR(50) DEFAULT 'not_started', -- not_started, queued, training, completed, failed
    model_path TEXT,
    training_started_at TIMESTAMP,
    training_completed_at TIMESTAMP,
    training_steps INTEGER,
    final_loss FLOAT,
    error_message TEXT,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training datasets
CREATE TABLE IF NOT EXISTS lora_datasets (
    id SERIAL PRIMARY KEY,
    definition_id INTEGER REFERENCES lora_definitions(id),
    dataset_path TEXT,
    image_count INTEGER,
    video_sources TEXT[], -- Array of video paths used
    caption_template TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training queue
CREATE TABLE IF NOT EXISTS training_queue (
    id SERIAL PRIMARY KEY,
    definition_id INTEGER REFERENCES lora_definitions(id),
    priority INTEGER DEFAULT 5,
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    scheduled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    worker_id VARCHAR(100),
    error_message TEXT
);

-- Insert comprehensive LoRA definitions for poses
INSERT INTO lora_definitions (category_id, name, display_name, trigger_word, description, base_prompt, is_nsfw)
VALUES
-- Poses (NSFW)
((SELECT id FROM lora_categories WHERE name='pose'), 'cowgirl_position', 'Cowgirl Position', 'cowgirl_position', 'Woman on top position', 'a woman in cowgirl position, straddling', true),
((SELECT id FROM lora_categories WHERE name='pose'), 'missionary_position', 'Missionary Position', 'missionary_position', 'Traditional intimate position', 'missionary position, lying on back', true),
((SELECT id FROM lora_categories WHERE name='pose'), 'doggy_style', 'Doggy Style', 'doggy_style', 'From behind position', 'doggy style position, from behind', true),
((SELECT id FROM lora_categories WHERE name='pose'), 'standing_sex', 'Standing Position', 'standing_sex', 'Standing intimate position', 'standing intimate position', true),
((SELECT id FROM lora_categories WHERE name='pose'), 'spooning', 'Spooning Position', 'spooning', 'Side by side position', 'spooning position, lying on side', true),

-- Actions (NSFW)
((SELECT id FROM lora_categories WHERE name='action'), 'kissing', 'Kissing', 'kissing', 'Romantic kissing', 'two people kissing passionately', true),
((SELECT id FROM lora_categories WHERE name='action'), 'oral_sex', 'Oral Sex', 'oral_sex', 'Oral intimate act', 'oral sex act', true),
((SELECT id FROM lora_categories WHERE name='action'), 'masturbation', 'Masturbation', 'masturbation', 'Self pleasure', 'masturbation, self pleasure', true),
((SELECT id FROM lora_categories WHERE name='action'), 'orgasm', 'Orgasm', 'orgasm', 'Climax expression', 'orgasm, climax expression', true),
((SELECT id FROM lora_categories WHERE name='action'), 'undressing', 'Undressing', 'undressing', 'Removing clothes', 'undressing, removing clothes', true),

-- Poses (SFW)
((SELECT id FROM lora_categories WHERE name='pose'), 'sitting_pose', 'Sitting Pose', 'sitting_pose', 'Person sitting', 'person sitting in chair', false),
((SELECT id FROM lora_categories WHERE name='pose'), 'standing_pose', 'Standing Pose', 'standing_pose', 'Person standing', 'person standing upright', false),
((SELECT id FROM lora_categories WHERE name='pose'), 'walking', 'Walking', 'walking', 'Person walking', 'person walking', false),
((SELECT id FROM lora_categories WHERE name='pose'), 'running', 'Running', 'running', 'Person running', 'person running', false),

-- Styles
((SELECT id FROM lora_categories WHERE name='style'), 'anime_style', 'Anime Style', 'anime_style', 'Japanese anime aesthetic', 'anime style artwork', false),
((SELECT id FROM lora_categories WHERE name='style'), 'realistic', 'Realistic', 'realistic', 'Photorealistic style', 'photorealistic, highly detailed', false),
((SELECT id FROM lora_categories WHERE name='style'), 'cyberpunk', 'Cyberpunk', 'cyberpunk', 'Cyberpunk aesthetic', 'cyberpunk style, neon lights', false),
((SELECT id FROM lora_categories WHERE name='style'), 'fantasy', 'Fantasy', 'fantasy', 'Fantasy aesthetic', 'fantasy style, magical', false),

-- Scenes
((SELECT id FROM lora_categories WHERE name='scene'), 'bedroom', 'Bedroom', 'bedroom', 'Bedroom setting', 'in a bedroom, bed visible', false),
((SELECT id FROM lora_categories WHERE name='scene'), 'bathroom', 'Bathroom', 'bathroom', 'Bathroom setting', 'in a bathroom', false),
((SELECT id FROM lora_categories WHERE name='scene'), 'outdoors', 'Outdoors', 'outdoors', 'Outdoor setting', 'outdoors, natural lighting', false),
((SELECT id FROM lora_categories WHERE name='scene'), 'office', 'Office', 'office', 'Office setting', 'in an office', false),

-- Emotions
((SELECT id FROM lora_categories WHERE name='emotion'), 'happy', 'Happy', 'happy', 'Happy expression', 'happy, smiling', false),
((SELECT id FROM lora_categories WHERE name='emotion'), 'sad', 'Sad', 'sad', 'Sad expression', 'sad, crying', false),
((SELECT id FROM lora_categories WHERE name='emotion'), 'angry', 'Angry', 'angry', 'Angry expression', 'angry, frowning', false),
((SELECT id FROM lora_categories WHERE name='emotion'), 'pleasure', 'Pleasure', 'pleasure', 'Expression of pleasure', 'expression of pleasure', true)
ON CONFLICT (name) DO NOTHING;

-- Create view for training status overview
CREATE OR REPLACE VIEW lora_training_overview AS
SELECT
    c.name as category,
    d.name as lora_name,
    d.display_name,
    d.trigger_word,
    d.is_nsfw,
    COALESCE(s.status, 'not_started') as training_status,
    s.model_path,
    s.training_completed_at,
    s.version,
    CASE
        WHEN s.model_path IS NOT NULL THEN 'Ready'
        WHEN s.status = 'training' THEN 'In Progress'
        WHEN s.status = 'queued' THEN 'Queued'
        WHEN s.status = 'failed' THEN 'Failed'
        ELSE 'Not Started'
    END as status_display
FROM lora_definitions d
JOIN lora_categories c ON d.category_id = c.id
LEFT JOIN lora_training_status s ON d.id = s.definition_id
ORDER BY c.priority DESC, d.training_priority DESC, d.name;

-- Function to get training statistics
CREATE OR REPLACE FUNCTION get_training_stats()
RETURNS TABLE(
    total_loras INTEGER,
    trained INTEGER,
    in_progress INTEGER,
    queued INTEGER,
    not_started INTEGER,
    failed INTEGER,
    nsfw_trained INTEGER,
    nsfw_total INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_loras,
        COUNT(CASE WHEN s.status = 'completed' THEN 1 END)::INTEGER as trained,
        COUNT(CASE WHEN s.status = 'training' THEN 1 END)::INTEGER as in_progress,
        COUNT(CASE WHEN s.status = 'queued' THEN 1 END)::INTEGER as queued,
        COUNT(CASE WHEN s.status IS NULL OR s.status = 'not_started' THEN 1 END)::INTEGER as not_started,
        COUNT(CASE WHEN s.status = 'failed' THEN 1 END)::INTEGER as failed,
        COUNT(CASE WHEN d.is_nsfw AND s.status = 'completed' THEN 1 END)::INTEGER as nsfw_trained,
        COUNT(CASE WHEN d.is_nsfw THEN 1 END)::INTEGER as nsfw_total
    FROM lora_definitions d
    LEFT JOIN lora_training_status s ON d.id = s.definition_id;
END;
$$ LANGUAGE plpgsql;