-- Create missing Echo Brain tables

-- 1. echo_context_registry (for context persistence)
CREATE TABLE IF NOT EXISTS echo_context_registry (
    id SERIAL PRIMARY KEY,
    context_type VARCHAR(100) NOT NULL,
    context_value TEXT,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE
);

-- 2. task_queue (for background task management)
CREATE TABLE IF NOT EXISTS task_queue (
    id SERIAL PRIMARY KEY,
    task_type VARCHAR(100) NOT NULL,
    task_data JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    result JSONB
);

-- 3. vector_memories (for vector memory storage)
CREATE TABLE IF NOT EXISTS vector_memories (
    id SERIAL PRIMARY KEY,
    memory_type VARCHAR(100),
    content TEXT NOT NULL,
    embedding VECTOR(384),
    metadata JSONB,
    importance_score FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1
);

-- 4. agent_state (for autonomous agent state)
CREATE TABLE IF NOT EXISTS agent_state (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) UNIQUE NOT NULL,
    state JSONB,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enabled BOOLEAN DEFAULT TRUE,
    configuration JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_context_registry_type ON echo_context_registry(context_type);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);
CREATE INDEX IF NOT EXISTS idx_vector_memories_type ON vector_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_agent_state_name ON agent_state(agent_name);