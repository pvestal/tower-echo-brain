-- Documents and chunks schema for Echo Brain knowledge store
-- PostgreSQL is SSOT, Qdrant stores vectors with document_id references

-- Main documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source information
    source_type VARCHAR(50) NOT NULL,  -- 'google_drive', 'gmail', 'local_file', 'conversation', 'knowledge_base', etc.
    source_id VARCHAR(500),            -- Original ID from source (Google doc ID, file path, etc.)
    source_url TEXT,                   -- URL or path to original

    -- Content metadata
    title TEXT,
    content_type VARCHAR(100),         -- MIME type or content category
    content_hash VARCHAR(64),          -- SHA-256 for deduplication

    -- Timestamps
    source_created_at TIMESTAMP,       -- When created in source
    source_modified_at TIMESTAMP,      -- When last modified in source
    indexed_at TIMESTAMP DEFAULT NOW(),
    last_sync_at TIMESTAMP DEFAULT NOW(),

    -- Status
    status VARCHAR(20) DEFAULT 'active',  -- 'active', 'deleted', 'archived'

    -- Flexible metadata as JSONB
    metadata JSONB DEFAULT '{}',

    -- Constraints
    UNIQUE(source_type, source_id)
);

-- Document chunks (for RAG)
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Chunk content
    chunk_index INTEGER NOT NULL,      -- Order within document
    content TEXT NOT NULL,             -- Actual text content
    content_hash VARCHAR(64),          -- For dedup

    -- Chunk metadata
    token_count INTEGER,
    char_count INTEGER,

    -- Vector reference (stored in Qdrant)
    qdrant_collection VARCHAR(50) DEFAULT 'documents',
    qdrant_point_id VARCHAR(100),      -- UUID in Qdrant

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(document_id, chunk_index)
);

-- Extracted facts (structured knowledge)
CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Fact content
    subject VARCHAR(200) NOT NULL,     -- "Patrick", "Tower server", etc.
    predicate VARCHAR(200) NOT NULL,   -- "drives", "runs on", etc.
    object TEXT NOT NULL,              -- "2022 Toyota Tundra", "AMD Ryzen 9", etc.

    -- Source tracking
    source_document_id UUID REFERENCES documents(id),
    source_conversation_id UUID,       -- If extracted from conversation
    confidence FLOAT DEFAULT 1.0,      -- Extraction confidence

    -- Temporal
    valid_from TIMESTAMP,              -- When fact became true
    valid_until TIMESTAMP,             -- When fact stopped being true (NULL = still true)

    -- Vector reference
    qdrant_point_id VARCHAR(100),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Prevent duplicate facts
    UNIQUE(subject, predicate, object)
);

-- Conversations (chat history)
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Conversation metadata
    title TEXT,
    source VARCHAR(50) DEFAULT 'echo_brain',  -- 'echo_brain', 'telegram', etc.

    -- Timestamps
    started_at TIMESTAMP DEFAULT NOW(),
    last_message_at TIMESTAMP DEFAULT NOW(),

    -- Status
    status VARCHAR(20) DEFAULT 'active',

    -- Flexible metadata
    metadata JSONB DEFAULT '{}'
);

-- Conversation messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

    -- Message content
    role VARCHAR(20) NOT NULL,         -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,

    -- Vector reference
    qdrant_point_id VARCHAR(100),

    -- Agent tracking
    agent_used VARCHAR(50),            -- Which agent responded
    model_used VARCHAR(100),           -- Which model was used

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),

    -- Ordering
    message_index INTEGER NOT NULL
);

-- Calendar events (synced from Google)
CREATE TABLE IF NOT EXISTS calendar_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source
    google_event_id VARCHAR(200) UNIQUE,
    calendar_id VARCHAR(200),

    -- Event details
    title TEXT NOT NULL,
    description TEXT,
    location TEXT,

    -- Timing
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    all_day BOOLEAN DEFAULT FALSE,
    timezone VARCHAR(50),

    -- Recurrence
    is_recurring BOOLEAN DEFAULT FALSE,
    recurrence_rule TEXT,

    -- Status
    status VARCHAR(20) DEFAULT 'confirmed',

    -- Sync tracking
    last_sync_at TIMESTAMP DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant ON document_chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, message_index);
CREATE INDEX IF NOT EXISTS idx_calendar_time ON calendar_events(start_time);

-- Full text search indexes
CREATE INDEX IF NOT EXISTS idx_documents_title_fts ON documents USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_facts_fts ON facts USING gin(to_tsvector('english', subject || ' ' || predicate || ' ' || object));