-- User preferences (flexible key-value)
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category VARCHAR(100) NOT NULL,
    key VARCHAR(200) NOT NULL,
    value JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(category, key)
);

-- Integration connections
CREATE TABLE IF NOT EXISTS integrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(200),
    status VARCHAR(50) DEFAULT 'disconnected',
    config JSONB DEFAULT '{}',
    scopes TEXT[],
    last_sync_at TIMESTAMP,
    connected_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vault registry (tracks keys, not values)
CREATE TABLE IF NOT EXISTS vault_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_name VARCHAR(200) NOT NULL UNIQUE,
    key_type VARCHAR(50),
    service VARCHAR(100),
    description TEXT,
    is_set BOOLEAN DEFAULT FALSE,
    last_updated TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_prefs_category ON user_preferences(category);
CREATE INDEX IF NOT EXISTS idx_integrations_status ON integrations(status);

-- Insert default integrations
INSERT INTO integrations (provider, display_name, status) VALUES
('google', 'Google (Drive, Gmail, Calendar)', 'disconnected'),
('apple_music', 'Apple Music', 'disconnected'),
('spotify', 'Spotify', 'disconnected'),
('plaid', 'Plaid (Financial)', 'disconnected'),
('home_assistant', 'Home Assistant', 'disconnected'),
('frigate', 'Frigate (Security)', 'disconnected')
ON CONFLICT (provider) DO NOTHING;