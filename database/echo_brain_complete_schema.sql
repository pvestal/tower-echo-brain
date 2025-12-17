-- Echo Brain Complete Database Schema
-- PostgreSQL schema for tower_consolidated database
-- Consolidates all Echo Brain memory and conversation requirements

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
-- Note: vector extension requires superuser, skip if not available
-- CREATE EXTENSION IF NOT EXISTS "vector" SCHEMA public;

-- ================================
-- CORE CONVERSATION TABLES
-- ================================

-- Main conversations table
CREATE TABLE IF NOT EXISTS echo_conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(100) UNIQUE NOT NULL,
    user_id VARCHAR(100) DEFAULT 'default',
    username VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    intent_history JSONB,
    context JSONB,
    message_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

-- Conversation messages
CREATE TABLE IF NOT EXISTS echo_messages (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(100) REFERENCES echo_conversations(conversation_id) ON DELETE CASCADE,
    message_id UUID DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user', -- user, assistant, system
    content TEXT NOT NULL,
    model_used VARCHAR(100),
    response_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Unified interactions log
CREATE TABLE IF NOT EXISTS echo_unified_interactions (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(100),
    user_id VARCHAR(100) DEFAULT 'default',
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    processing_time FLOAT NOT NULL,
    escalation_path JSONB,
    intent VARCHAR(50),
    confidence FLOAT,
    requires_clarification BOOLEAN DEFAULT FALSE,
    clarifying_questions JSONB,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- MEMORY SYSTEMS
-- ================================

-- Episodic memory for detailed conversation memories
CREATE TABLE IF NOT EXISTS echo_episodic_memory (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(100),
    memory_type VARCHAR(50), -- episodic, semantic, procedural
    content TEXT NOT NULL,
    emotional_valence FLOAT DEFAULT 0.0, -- -1 to 1
    importance FLOAT DEFAULT 0.5, -- 0 to 1
    query_text TEXT,
    echo_response TEXT,
    model_used VARCHAR(100),
    learned_fact TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

-- User context and preferences
CREATE TABLE IF NOT EXISTS echo_user_contexts (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    user_id UUID DEFAULT uuid_generate_v4(),
    display_name VARCHAR(255),
    permissions JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    interaction_count INTEGER DEFAULT 0,
    first_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context_metadata JSONB DEFAULT '{}',
    learned_patterns JSONB DEFAULT '{}',
    creative_preferences JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

-- Learning records for pattern recognition
CREATE TABLE IF NOT EXISTS echo_learnings (
    id SERIAL PRIMARY KEY,
    learning_id UUID DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100),
    learning_type VARCHAR(50), -- pattern, preference, fact, correction
    content JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 1,
    source_conversation_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_reinforced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Vector embeddings for semantic search (requires pgvector extension)
-- Uncomment if pgvector is installed with superuser privileges
-- CREATE TABLE IF NOT EXISTS echo_embeddings (
--     id SERIAL PRIMARY KEY,
--     content_id UUID DEFAULT uuid_generate_v4(),
--     content_type VARCHAR(50), -- message, memory, learning
--     reference_id VARCHAR(100), -- ID in source table
--     embedding vector(1536), -- Adjust dimension based on model
--     content_text TEXT,
--     metadata JSONB DEFAULT '{}',
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- ================================
-- AGENT & MODEL MANAGEMENT
-- ================================

-- Agent interactions and delegations
CREATE TABLE IF NOT EXISTS echo_agent_interactions (
    id SERIAL PRIMARY KEY,
    interaction_id UUID DEFAULT uuid_generate_v4(),
    conversation_id VARCHAR(100),
    agent_type VARCHAR(100),
    request_data JSONB,
    response_data JSONB,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model usage tracking
CREATE TABLE IF NOT EXISTS echo_model_usage (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    conversation_id VARCHAR(100),
    user_id VARCHAR(100),
    tokens_input INTEGER,
    tokens_output INTEGER,
    processing_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- TELEGRAM INTEGRATION
-- ================================

-- Telegram conversation mapping
CREATE TABLE IF NOT EXISTS echo_telegram_conversations (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT UNIQUE NOT NULL,
    echo_conversation_id VARCHAR(100) REFERENCES echo_conversations(conversation_id),
    telegram_username VARCHAR(100),
    first_message_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

-- ================================
-- TRAINING & FEEDBACK
-- ================================

-- User feedback for continuous improvement
CREATE TABLE IF NOT EXISTS echo_feedback (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(100),
    message_id VARCHAR(100),
    user_id VARCHAR(100),
    feedback_type VARCHAR(20), -- thumbs_up, thumbs_down, correction
    feedback_text TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training data queue
CREATE TABLE IF NOT EXISTS echo_training_queue (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(100),
    query TEXT NOT NULL,
    expected_response TEXT,
    actual_response TEXT,
    needs_review BOOLEAN DEFAULT TRUE,
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP,
    training_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- INDEXES FOR PERFORMANCE
-- ================================

-- Conversation indexes
CREATE INDEX IF NOT EXISTS idx_echo_conversations_user ON echo_conversations(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_echo_conversations_active ON echo_conversations(is_active, last_interaction DESC);

-- Message indexes
CREATE INDEX IF NOT EXISTS idx_echo_messages_conversation ON echo_messages(conversation_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_echo_messages_user ON echo_messages(user_id, timestamp DESC);

-- Memory indexes
CREATE INDEX IF NOT EXISTS idx_echo_episodic_conversation ON echo_episodic_memory(conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_echo_episodic_importance ON echo_episodic_memory(importance DESC) WHERE importance > 0.5;
CREATE INDEX IF NOT EXISTS idx_echo_episodic_type ON echo_episodic_memory(memory_type);

-- User context indexes
CREATE INDEX IF NOT EXISTS idx_echo_user_contexts_username ON echo_user_contexts(username);
CREATE INDEX IF NOT EXISTS idx_echo_user_contexts_active ON echo_user_contexts(is_active, last_interaction DESC);

-- Learning indexes
CREATE INDEX IF NOT EXISTS idx_echo_learnings_user ON echo_learnings(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_echo_learnings_type ON echo_learnings(learning_type, confidence DESC);

-- Model usage indexes
CREATE INDEX IF NOT EXISTS idx_echo_model_usage_time ON echo_model_usage(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_echo_model_usage_model ON echo_model_usage(model_name, timestamp DESC);

-- Telegram indexes
CREATE INDEX IF NOT EXISTS idx_echo_telegram_chat ON echo_telegram_conversations(chat_id);

-- Full text search indexes
CREATE INDEX IF NOT EXISTS idx_echo_messages_content_fts ON echo_messages USING GIN(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_echo_episodic_content_fts ON echo_episodic_memory USING GIN(to_tsvector('english', content));

-- ================================
-- INITIAL DATA & PERMISSIONS
-- ================================

-- Create default user context for patrick
INSERT INTO echo_user_contexts (username, display_name, permissions, preferences)
VALUES ('patrick', 'Patrick',
    '{"creator": true, "system_commands": true, "file_access": true, "database_access": true}'::jsonb,
    '{"model_preference": "auto", "response_style": "technical", "verbosity": "normal"}'::jsonb)
ON CONFLICT (username) DO UPDATE SET
    last_interaction = CURRENT_TIMESTAMP,
    interaction_count = echo_user_contexts.interaction_count + 1;

-- Grant appropriate permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO patrick;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO patrick;

-- ================================
-- HELPER FUNCTIONS
-- ================================

-- Function to clean old conversations
CREATE OR REPLACE FUNCTION clean_old_conversations(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM echo_conversations
    WHERE last_interaction < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep
    AND is_active = FALSE;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get user conversation summary
CREATE OR REPLACE FUNCTION get_user_conversation_summary(p_username VARCHAR)
RETURNS TABLE(
    total_conversations BIGINT,
    total_messages BIGINT,
    avg_messages_per_conversation NUMERIC,
    first_interaction TIMESTAMP,
    last_interaction TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(DISTINCT c.conversation_id) as total_conversations,
        COUNT(m.id) as total_messages,
        AVG(c.message_count) as avg_messages_per_conversation,
        MIN(c.created_at) as first_interaction,
        MAX(c.last_interaction) as last_interaction
    FROM echo_conversations c
    LEFT JOIN echo_messages m ON c.conversation_id = m.conversation_id
    WHERE c.username = p_username;
END;
$$ LANGUAGE plpgsql;

-- Comment on tables
COMMENT ON TABLE echo_conversations IS 'Main conversation tracking table for Echo Brain';
COMMENT ON TABLE echo_messages IS 'Individual messages within conversations';
COMMENT ON TABLE echo_episodic_memory IS 'Detailed episodic memories with emotional and importance weighting';
COMMENT ON TABLE echo_user_contexts IS 'User preferences, permissions, and learned patterns';
COMMENT ON TABLE echo_learnings IS 'Machine learning insights and patterns discovered from conversations';

-- ================================
-- MIGRATION FROM EXISTING TABLES
-- ================================

-- If anime tables exist, link them
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'anime_conversation_context') THEN
        -- Link anime conversations to echo conversations
        INSERT INTO echo_conversations (conversation_id, username, created_at, context)
        SELECT
            conversation_id,
            'patrick',
            created_at,
            jsonb_build_object('source', 'anime_system', 'project_id', project_id)
        FROM anime_conversation_context
        ON CONFLICT (conversation_id) DO NOTHING;
    END IF;
END $$;