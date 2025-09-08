-- Character Evolution System Migration
-- Extends existing anime_characters and character_relationships tables
-- with timeline tracking and emotional impact monitoring

-- Character Evolution Timeline - tracks major character development moments
CREATE TABLE character_evolution_timeline (
    id SERIAL PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES anime_characters(id) ON DELETE CASCADE,
    scene_id INTEGER REFERENCES anime_scenes(id) ON DELETE CASCADE,
    evolution_type VARCHAR(50) NOT NULL, -- 'personality_shift', 'skill_gain', 'relationship_change', 'trauma', 'growth'
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    impact_level INTEGER CHECK (impact_level BETWEEN 1 AND 10), -- 1=minor, 10=life changing
    previous_state JSONB, -- Character state before this evolution
    new_state JSONB, -- Character state after this evolution
    triggers TEXT[], -- What caused this evolution (events, other characters, etc.)
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Character State History - snapshots of character state at different points
CREATE TABLE character_state_history (
    id SERIAL PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES anime_characters(id) ON DELETE CASCADE,
    evolution_timeline_id INTEGER REFERENCES character_evolution_timeline(id) ON DELETE CASCADE,
    state_snapshot JSONB NOT NULL, -- Complete character state at this point
    personality_traits JSONB, -- Key-value pairs of traits and their strength
    skills JSONB, -- Abilities, powers, talents with levels
    emotional_state JSONB, -- Current emotional baseline and temperament
    relationships_snapshot JSONB, -- Summary of all relationships at this time
    story_arc_position VARCHAR(100), -- Where in their story arc: 'introduction', 'rising_action', 'climax', 'resolution'
    character_level INTEGER DEFAULT 1, -- Overall character development level
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Character Relationship Dynamics - extends character_relationships with evolution tracking
CREATE TABLE character_relationship_dynamics (
    id SERIAL PRIMARY KEY,
    relationship_id INTEGER NOT NULL REFERENCES character_relationships(id) ON DELETE CASCADE,
    evolution_timeline_id INTEGER REFERENCES character_evolution_timeline(id) ON DELETE CASCADE,
    relationship_strength INTEGER CHECK (relationship_strength BETWEEN -10 AND 10), -- -10=hatred, 0=neutral, 10=deep bond
    relationship_status VARCHAR(50), -- 'developing', 'stable', 'deteriorating', 'broken', 'restored'
    interaction_frequency VARCHAR(20), -- 'constant', 'frequent', 'occasional', 'rare', 'none'
    emotional_intensity INTEGER CHECK (emotional_intensity BETWEEN 1 AND 10),
    conflict_level INTEGER CHECK (conflict_level BETWEEN 0 AND 10),
    trust_level INTEGER CHECK (trust_level BETWEEN 0 AND 10),
    dependency_level INTEGER CHECK (dependency_level BETWEEN 0 AND 10),
    recent_interactions JSONB, -- Last few significant interactions
    relationship_milestones TEXT[], -- Key moments in this relationship
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Emotional Impact Tracking - tracks how events affect character emotions
CREATE TABLE emotional_impact_tracking (
    id SERIAL PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES anime_characters(id) ON DELETE CASCADE,
    scene_id INTEGER REFERENCES anime_scenes(id) ON DELETE CASCADE,
    evolution_timeline_id INTEGER REFERENCES character_evolution_timeline(id) ON DELETE CASCADE,
    trigger_event TEXT NOT NULL, -- What happened
    trigger_character_id INTEGER REFERENCES anime_characters(id), -- Who caused it (if applicable)
    emotional_response JSONB NOT NULL, -- Immediate emotional reaction
    intensity_level INTEGER CHECK (intensity_level BETWEEN 1 AND 10),
    duration_category VARCHAR(20), -- 'momentary', 'short_term', 'lasting', 'permanent'
    baseline_impact JSONB, -- How this affects their baseline emotional state
    coping_mechanism TEXT, -- How the character deals with this emotion
    long_term_effects JSONB, -- Lasting changes to personality/behavior
    recovery_timeline INTEGER, -- Days/scenes to recover (null if permanent)
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_char_evolution_timeline_char_id ON character_evolution_timeline(character_id);
CREATE INDEX idx_char_evolution_timeline_timestamp ON character_evolution_timeline(timestamp);
CREATE INDEX idx_char_state_history_char_id ON character_state_history(character_id);
CREATE INDEX idx_char_state_history_timeline_id ON character_state_history(evolution_timeline_id);
CREATE INDEX idx_char_relationship_dynamics_rel_id ON character_relationship_dynamics(relationship_id);
CREATE INDEX idx_emotional_impact_char_id ON emotional_impact_tracking(character_id);
CREATE INDEX idx_emotional_impact_scene_id ON emotional_impact_tracking(scene_id);
CREATE INDEX idx_emotional_impact_timestamp ON emotional_impact_tracking(timestamp);

-- Add helpful triggers for automatic timestamping
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.timestamp = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_char_evolution_timeline_timestamp BEFORE UPDATE ON character_evolution_timeline FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_char_state_history_timestamp BEFORE UPDATE ON character_state_history FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_char_relationship_dynamics_timestamp BEFORE UPDATE ON character_relationship_dynamics FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_emotional_impact_tracking_timestamp BEFORE UPDATE ON emotional_impact_tracking FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
