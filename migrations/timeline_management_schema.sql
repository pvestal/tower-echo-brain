-- Timeline Management System with Decision Branching
-- Database Schema for Tower Anime Production Suite
-- Integrates with existing character evolution system

-- Main Timelines Table
CREATE TABLE IF NOT EXISTS timelines (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    root_branch_id INTEGER, -- Will be set after root branch is created
    created_by VARCHAR(100) DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active', -- active, paused, completed, archived
    timeline_type VARCHAR(50) DEFAULT 'narrative', -- narrative, character_arc, production_flow
    metadata JSONB DEFAULT '{}',
    CONSTRAINT unique_timeline_name UNIQUE(name)
);

-- Timeline Branches (Story Paths)
CREATE TABLE IF NOT EXISTS timeline_branches (
    id SERIAL PRIMARY KEY,
    timeline_id INTEGER NOT NULL REFERENCES timelines(id) ON DELETE CASCADE,
    parent_branch_id INTEGER REFERENCES timeline_branches(id), -- For tracking divergence
    branch_name VARCHAR(255) NOT NULL,
    description TEXT,
    branch_type VARCHAR(50) DEFAULT 'linear', -- linear, parallel, divergent, convergent
    start_decision_id INTEGER, -- Decision that created this branch
    merge_target_branch_id INTEGER REFERENCES timeline_branches(id), -- For convergence
    character_states JSONB DEFAULT '{}', -- Current character states in this branch
    branch_priority INTEGER DEFAULT 1, -- For main/side story ranking
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    narrative_weight FLOAT DEFAULT 1.0, -- Impact on overall story
    completion_status VARCHAR(50) DEFAULT 'in_progress'
);

-- Decision Points (Critical Story Moments)
CREATE TABLE IF NOT EXISTS timeline_decisions (
    id SERIAL PRIMARY KEY,
    timeline_id INTEGER NOT NULL REFERENCES timelines(id) ON DELETE CASCADE,
    branch_id INTEGER NOT NULL REFERENCES timeline_branches(id) ON DELETE CASCADE,
    decision_text TEXT NOT NULL,
    decision_type VARCHAR(50) DEFAULT 'character_action', -- character_action, plot_device, external_event
    position_in_branch INTEGER DEFAULT 0, -- Sequential position
    required_conditions JSONB DEFAULT '{}', -- Prerequisites for this decision
    character_impact JSONB DEFAULT '{}', -- Which characters are affected
    scene_context JSONB DEFAULT '{}', -- Scene information for production
    urgency_level INTEGER DEFAULT 1 CHECK (urgency_level >= 1 AND urgency_level <= 10),
    reversible BOOLEAN DEFAULT true, -- Can this decision be undone?
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    narrative_importance INTEGER DEFAULT 5 CHECK (narrative_importance >= 1 AND narrative_importance <= 10),
    production_complexity INTEGER DEFAULT 1 CHECK (production_complexity >= 1 AND production_complexity <= 10)
);

-- Decision Options (Possible Choices)
CREATE TABLE IF NOT EXISTS decision_options (
    id SERIAL PRIMARY KEY,
    decision_id INTEGER NOT NULL REFERENCES timeline_decisions(id) ON DELETE CASCADE,
    option_text TEXT NOT NULL,
    option_type VARCHAR(50) DEFAULT 'direct_action', -- direct_action, dialogue, passive, skip
    target_branch_id INTEGER REFERENCES timeline_branches(id), -- Branch this option leads to
    probability_weight FLOAT DEFAULT 1.0, -- For weighted random selection
    character_requirements JSONB DEFAULT '{}', -- Character state requirements
    immediate_consequences JSONB DEFAULT '{}', -- Instant effects
    delayed_consequences JSONB DEFAULT '{}', -- Long-term effects
    production_notes TEXT, -- Notes for anime production
    emotional_tone VARCHAR(50), -- happy, sad, dramatic, action, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Decision Consequences (Impact Tracking)
CREATE TABLE IF NOT EXISTS decision_consequences (
    id SERIAL PRIMARY KEY,
    decision_id INTEGER NOT NULL REFERENCES timeline_decisions(id) ON DELETE CASCADE,
    option_id INTEGER NOT NULL REFERENCES decision_options(id) ON DELETE CASCADE,
    consequence_type VARCHAR(50) NOT NULL, -- character_change, story_shift, relationship_impact
    target_character_id INTEGER, -- Links to character system
    impact_description TEXT NOT NULL,
    impact_magnitude INTEGER DEFAULT 1 CHECK (impact_magnitude >= 1 AND impact_magnitude <= 10),
    duration_category VARCHAR(50) DEFAULT 'permanent', -- temporary, short_term, lasting, permanent
    state_changes JSONB DEFAULT '{}', -- Specific state modifications
    triggers_events JSONB DEFAULT '[]', -- Future events this consequence triggers
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP, -- When this consequence was actually applied
    applied_to_branch_id INTEGER REFERENCES timeline_branches(id)
);

-- Timeline State Snapshots (Version Control for Branching)
CREATE TABLE IF NOT EXISTS timeline_states (
    id SERIAL PRIMARY KEY,
    timeline_id INTEGER NOT NULL REFERENCES timelines(id) ON DELETE CASCADE,
    branch_id INTEGER NOT NULL REFERENCES timeline_branches(id) ON DELETE CASCADE,
    snapshot_name VARCHAR(255),
    state_data JSONB NOT NULL, -- Complete state snapshot
    character_states JSONB DEFAULT '{}', -- Character-specific states
    relationship_matrix JSONB DEFAULT '{}', -- Character relationships
    world_state JSONB DEFAULT '{}', -- Environmental/world conditions
    narrative_position JSONB DEFAULT '{}', -- Story arc progress
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    snapshot_type VARCHAR(50) DEFAULT 'auto', -- auto, manual, checkpoint, rollback
    parent_state_id INTEGER REFERENCES timeline_states(id), -- For state history
    is_checkpoint BOOLEAN DEFAULT false -- Major story milestones
);

-- Branch Relationships (Tracking Convergence/Divergence)
CREATE TABLE IF NOT EXISTS branch_relationships (
    id SERIAL PRIMARY KEY,
    source_branch_id INTEGER NOT NULL REFERENCES timeline_branches(id) ON DELETE CASCADE,
    target_branch_id INTEGER NOT NULL REFERENCES timeline_branches(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- diverged_from, converges_to, parallel_to, replaces
    decision_point_id INTEGER REFERENCES timeline_decisions(id), -- Decision that created relationship
    strength FLOAT DEFAULT 1.0, -- Relationship strength (0.0 to 1.0)
    narrative_coherence JSONB DEFAULT '{}', -- How branches maintain story consistency
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT true,
    CONSTRAINT unique_branch_relationship UNIQUE(source_branch_id, target_branch_id, relationship_type)
);

-- Narrative Coherence Tracking
CREATE TABLE IF NOT EXISTS narrative_coherence_checks (
    id SERIAL PRIMARY KEY,
    timeline_id INTEGER NOT NULL REFERENCES timelines(id) ON DELETE CASCADE,
    check_type VARCHAR(50) NOT NULL, -- character_consistency, plot_logic, timeline_validity
    check_description TEXT NOT NULL,
    affected_branches JSONB DEFAULT '[]', -- List of branch IDs affected
    severity VARCHAR(50) DEFAULT 'minor', -- minor, major, critical
    auto_fixable BOOLEAN DEFAULT false,
    fix_suggestions JSONB DEFAULT '[]',
    resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100)
);

-- Integration with existing character evolution system
-- This table links timeline events to character evolution records
CREATE TABLE IF NOT EXISTS timeline_character_evolution_links (
    id SERIAL PRIMARY KEY,
    timeline_id INTEGER NOT NULL REFERENCES timelines(id) ON DELETE CASCADE,
    branch_id INTEGER NOT NULL REFERENCES timeline_branches(id) ON DELETE CASCADE,
    decision_id INTEGER REFERENCES timeline_decisions(id),
    character_evolution_id INTEGER, -- Links to existing character_evolution table
    evolution_trigger VARCHAR(100) NOT NULL, -- What timeline event caused the evolution
    timeline_position INTEGER DEFAULT 0, -- When in timeline this evolution occurs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_timelines_status ON timelines(status);
CREATE INDEX IF NOT EXISTS idx_timeline_branches_timeline_id ON timeline_branches(timeline_id);
CREATE INDEX IF NOT EXISTS idx_timeline_branches_parent_branch ON timeline_branches(parent_branch_id);
CREATE INDEX IF NOT EXISTS idx_timeline_decisions_branch_id ON timeline_decisions(branch_id);
CREATE INDEX IF NOT EXISTS idx_timeline_decisions_position ON timeline_decisions(branch_id, position_in_branch);
CREATE INDEX IF NOT EXISTS idx_decision_options_decision_id ON decision_options(decision_id);
CREATE INDEX IF NOT EXISTS idx_decision_consequences_decision_id ON decision_consequences(decision_id);
CREATE INDEX IF NOT EXISTS idx_timeline_states_branch_id ON timeline_states(branch_id);
CREATE INDEX IF NOT EXISTS idx_timeline_states_created_at ON timeline_states(created_at);
CREATE INDEX IF NOT EXISTS idx_branch_relationships_source ON branch_relationships(source_branch_id);
CREATE INDEX IF NOT EXISTS idx_branch_relationships_target ON branch_relationships(target_branch_id);

-- Sample Data for Testing
INSERT INTO timelines (name, description, timeline_type) VALUES
('Tokyo Debt Desire - Main Story', 'Primary narrative timeline for Tokyo Debt Desire anime', 'narrative'),
('Character Development Arc - Sakura', 'Personal growth timeline for protagonist Sakura', 'character_arc')
ON CONFLICT (name) DO NOTHING;

COMMENT ON TABLE timelines IS 'Main timelines for anime production stories';
COMMENT ON TABLE timeline_branches IS 'Story branches representing different narrative paths';
COMMENT ON TABLE timeline_decisions IS 'Critical decision points that drive story branching';
COMMENT ON TABLE decision_options IS 'Available choices at each decision point';
COMMENT ON TABLE decision_consequences IS 'Impact tracking for each decision option';
COMMENT ON TABLE timeline_states IS 'State snapshots for branch version control';
COMMENT ON TABLE branch_relationships IS 'Tracking how branches relate to each other';
COMMENT ON TABLE narrative_coherence_checks IS 'Automated story consistency validation';
COMMENT ON TABLE timeline_character_evolution_links IS 'Integration with character evolution system';
