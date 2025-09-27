-- Echo Brain Family Multi-User Database Schema
-- PostgreSQL with Row Level Security (RLS) for proper user isolation

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create family database
CREATE DATABASE echo_family;
\c echo_family;

-- ================================
-- USER MANAGEMENT
-- ================================

-- Family members table
CREATE TABLE family_members (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'adult', 'teen', 'child', 'guest')),
    parent_id UUID REFERENCES family_members(user_id),
    birth_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    preferences JSONB DEFAULT '{}',
    restrictions JSONB DEFAULT '[]'
);

-- Create Patrick as admin
INSERT INTO family_members (username, display_name, role, email)
VALUES ('patrick', 'Patrick', 'admin', 'patrick.vestal.digital@gmail.com');

-- Session management
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES family_members(user_id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    device_id VARCHAR(100),
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create index for session lookups
CREATE INDEX idx_sessions_token ON user_sessions(token_hash);
CREATE INDEX idx_sessions_user ON user_sessions(user_id, is_active);

-- ================================
-- CONVERSATIONS WITH USER ISOLATION
-- ================================

CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES family_members(user_id) ON DELETE CASCADE,
    thread_id UUID,  -- For conversation threading
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Messages table with proper foreign keys
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES family_members(user_id),
    message_text TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'user',  -- user, assistant, system
    model_used VARCHAR(100),
    response_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_sensitive BOOLEAN DEFAULT FALSE,  -- Flag for sensitive content
    metadata JSONB DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX idx_messages_conversation ON messages(conversation_id, timestamp);
CREATE INDEX idx_messages_user ON messages(user_id, timestamp);

-- ================================
-- ROW LEVEL SECURITY (RLS)
-- ================================

-- Enable RLS on sensitive tables
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own conversations
CREATE POLICY user_conversations ON conversations
    FOR ALL
    TO PUBLIC
    USING (
        user_id = current_setting('app.current_user_id')::UUID
        OR
        EXISTS (
            SELECT 1 FROM family_members
            WHERE user_id = current_setting('app.current_user_id')::UUID
            AND role = 'admin'
        )
    );

-- Policy: Users can only see their own messages (with admin override)
CREATE POLICY user_messages ON messages
    FOR ALL
    TO PUBLIC
    USING (
        user_id = current_setting('app.current_user_id')::UUID
        OR
        -- Admin can see all
        EXISTS (
            SELECT 1 FROM family_members
            WHERE user_id = current_setting('app.current_user_id')::UUID
            AND role = 'admin'
        )
        OR
        -- Parents can see children's messages
        EXISTS (
            SELECT 1 FROM family_members child
            JOIN family_members parent ON child.parent_id = parent.user_id
            WHERE child.user_id = messages.user_id
            AND parent.user_id = current_setting('app.current_user_id')::UUID
        )
    );

-- ================================
-- SHARED FAMILY RESOURCES
-- ================================

CREATE TABLE family_calendar (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_by UUID NOT NULL REFERENCES family_members(user_id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    location VARCHAR(255),
    attendees UUID[] DEFAULT '{}',  -- Array of user_ids
    is_recurring BOOLEAN DEFAULT FALSE,
    recurrence_rule JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shopping_list (
    item_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    added_by UUID NOT NULL REFERENCES family_members(user_id),
    item_name VARCHAR(255) NOT NULL,
    quantity VARCHAR(50),
    category VARCHAR(50),
    is_purchased BOOLEAN DEFAULT FALSE,
    purchased_by UUID REFERENCES family_members(user_id),
    purchased_at TIMESTAMP,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE family_notes (
    note_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_by UUID NOT NULL REFERENCES family_members(user_id),
    title VARCHAR(255),
    content TEXT NOT NULL,
    is_shared BOOLEAN DEFAULT TRUE,
    shared_with UUID[] DEFAULT '{}',  -- Specific users or empty for all
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- KNOWLEDGE BASE WITH PERMISSIONS
-- ================================

CREATE TABLE knowledge_articles (
    article_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_by UUID NOT NULL REFERENCES family_members(user_id),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    tags TEXT[] DEFAULT '{}',
    access_level VARCHAR(20) DEFAULT 'family',  -- public, family, private, admin
    min_age INTEGER,  -- Minimum age to view (for content filtering)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    view_count INTEGER DEFAULT 0
);

-- ================================
-- USER PREFERENCES & SETTINGS
-- ================================

CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY REFERENCES family_members(user_id) ON DELETE CASCADE,
    theme VARCHAR(20) DEFAULT 'auto',
    language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    notification_settings JSONB DEFAULT '{}',
    privacy_settings JSONB DEFAULT '{}',
    model_preferences JSONB DEFAULT '{}',  -- Preferred AI models
    content_filters JSONB DEFAULT '{}',    -- For children
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- AUDIT LOGGING
-- ================================

CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES family_members(user_id),
    action VARCHAR(100) NOT NULL,
    target_user_id UUID REFERENCES family_members(user_id),  -- If accessing another user's data
    target_resource VARCHAR(100),
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for audit queries
CREATE INDEX idx_audit_user ON audit_log(user_id, timestamp);
CREATE INDEX idx_audit_target ON audit_log(target_user_id, timestamp);
CREATE INDEX idx_audit_action ON audit_log(action, timestamp);

-- ================================
-- SENSITIVE DATA ENCRYPTION
-- ================================

-- Table for encrypted sensitive data (API keys, passwords, etc.)
CREATE TABLE secure_vault (
    vault_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES family_members(user_id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    encrypted_value TEXT NOT NULL,  -- Encrypted with user's key
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(user_id, key_name)
);

-- Enable RLS on secure vault
ALTER TABLE secure_vault ENABLE ROW LEVEL SECURITY;

-- Only owner can access their vault (even admin needs special function)
CREATE POLICY vault_owner_only ON secure_vault
    FOR ALL
    TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- ================================
-- FUNCTIONS FOR DATA ACCESS
-- ================================

-- Function to set current user context (called at session start)
CREATE OR REPLACE FUNCTION set_user_context(p_user_id UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_user_id', p_user_id::TEXT, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function for admin to access user data (with audit logging)
CREATE OR REPLACE FUNCTION admin_access_user_data(
    p_admin_id UUID,
    p_target_user_id UUID,
    p_reason TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    v_is_admin BOOLEAN;
BEGIN
    -- Verify admin status
    SELECT role = 'admin' INTO v_is_admin
    FROM family_members
    WHERE user_id = p_admin_id;

    IF NOT v_is_admin THEN
        RETURN FALSE;
    END IF;

    -- Log the admin access
    INSERT INTO audit_log (user_id, action, target_user_id, details)
    VALUES (
        p_admin_id,
        'admin_data_access',
        p_target_user_id,
        jsonb_build_object('reason', p_reason, 'timestamp', CURRENT_TIMESTAMP)
    );

    -- Set context to allow access
    PERFORM set_config('app.admin_override', 'true', true);
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get user's conversation history (with privacy)
CREATE OR REPLACE FUNCTION get_user_conversations(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    conversation_id UUID,
    message_count BIGINT,
    last_message TIMESTAMP,
    metadata JSONB
) AS $$
BEGIN
    -- Set user context
    PERFORM set_user_context(p_user_id);

    RETURN QUERY
    SELECT
        c.conversation_id,
        COUNT(m.message_id) as message_count,
        MAX(m.timestamp) as last_message,
        c.metadata
    FROM conversations c
    LEFT JOIN messages m ON c.conversation_id = m.conversation_id
    WHERE c.user_id = p_user_id
    GROUP BY c.conversation_id, c.metadata
    ORDER BY last_message DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ================================
-- VIEWS FOR COMMON QUERIES
-- ================================

-- Family dashboard view (shows appropriate data per role)
CREATE OR REPLACE VIEW family_dashboard AS
SELECT
    fm.user_id,
    fm.display_name,
    fm.role,
    (
        SELECT COUNT(*) FROM shopping_list
        WHERE NOT is_purchased
    ) as shopping_items,
    (
        SELECT COUNT(*) FROM family_calendar
        WHERE start_time > CURRENT_TIMESTAMP
        AND start_time < CURRENT_TIMESTAMP + INTERVAL '7 days'
    ) as upcoming_events,
    (
        SELECT COUNT(*) FROM messages
        WHERE user_id = fm.user_id
        AND timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
    ) as recent_messages
FROM family_members fm
WHERE fm.is_active = TRUE;

-- ================================
-- TRIGGERS FOR DATA INTEGRITY
-- ================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_conversations_timestamp BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_knowledge_timestamp BEFORE UPDATE ON knowledge_articles
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- ================================
-- INDEXES FOR PERFORMANCE
-- ================================

CREATE INDEX idx_calendar_upcoming ON family_calendar(start_time)
    WHERE start_time > CURRENT_TIMESTAMP;

CREATE INDEX idx_shopping_pending ON shopping_list(is_purchased)
    WHERE NOT is_purchased;

CREATE INDEX idx_notes_shared ON family_notes(is_shared, created_by);

-- ================================
-- SAMPLE DATA FOR TESTING
-- ================================

-- Add family members (customize as needed)
INSERT INTO family_members (username, display_name, role, parent_id)
VALUES
    ('partner', 'Partner', 'adult', NULL),
    ('teen1', 'Teen', 'teen', (SELECT user_id FROM family_members WHERE username = 'patrick')),
    ('child1', 'Child', 'child', (SELECT user_id FROM family_members WHERE username = 'patrick'));

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO echo_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO echo_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO echo_user;