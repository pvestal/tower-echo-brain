-- ============================================================================
-- Echo Brain Board of Directors - Complete Database Schema
-- Comprehensive PostgreSQL schema for transparent AI decision tracking
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ============================================================================
-- CORE BOARD SYSTEM TABLES
-- ============================================================================

-- Board tasks: Main tasks submitted for board evaluation
CREATE TABLE IF NOT EXISTS board_tasks (
    task_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(255) NOT NULL,
    original_request TEXT NOT NULL,
    task_context JSONB DEFAULT '{}',
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'critical')),
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'in_progress', 'completed', 'rejected', 'overridden', 'appealed', 'cancelled')
    ),
    final_recommendation TEXT,
    consensus_score FLOAT CHECK (consensus_score >= 0.0 AND consensus_score <= 1.0),
    confidence_score FLOAT CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    total_processing_time FLOAT DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    director_participation TEXT[],
    completion_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Director evaluations: Individual director assessments
CREATE TABLE IF NOT EXISTS director_evaluations (
    evaluation_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    task_id VARCHAR(255) REFERENCES board_tasks(task_id) ON DELETE CASCADE,
    decision_point_id VARCHAR(255),
    director_id VARCHAR(255) NOT NULL,
    director_name VARCHAR(255) NOT NULL,
    recommendation VARCHAR(50) NOT NULL CHECK (
        recommendation IN ('approve', 'reject', 'modify', 'escalate', 'defer')
    ),
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    reasoning TEXT,
    processing_time FLOAT DEFAULT 0.0,
    risk_score FLOAT DEFAULT 0.0 CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Decision evidence: Supporting evidence for evaluations
CREATE TABLE IF NOT EXISTS decision_evidence (
    evidence_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    evaluation_id VARCHAR(255) REFERENCES director_evaluations(evaluation_id) ON DELETE CASCADE,
    evidence_type VARCHAR(100) NOT NULL CHECK (
        evidence_type IN ('performance_data', 'security_analysis', 'quality_metrics',
                         'user_feedback', 'historical_pattern', 'resource_impact', 'risk_assessment')
    ),
    source VARCHAR(255) NOT NULL,
    weight FLOAT NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    reasoning TEXT,
    data JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Decision timeline: Complete audit trail of decision process
CREATE TABLE IF NOT EXISTS decision_timeline (
    point_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    task_id VARCHAR(255) REFERENCES board_tasks(task_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) NOT NULL CHECK (
        status IN ('pending', 'in_progress', 'completed', 'rejected', 'overridden', 'appealed', 'error')
    ),
    description TEXT,
    consensus_score FLOAT CHECK (consensus_score >= 0.0 AND consensus_score <= 1.0),
    total_confidence FLOAT CHECK (total_confidence >= 0.0 AND total_confidence <= 1.0),
    evidence_summary JSONB DEFAULT '{}',
    user_feedback TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- USER FEEDBACK AND OVERRIDE SYSTEM
-- ============================================================================

-- User feedback: All user feedback on board decisions
CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    task_id VARCHAR(255) REFERENCES board_tasks(task_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    feedback_type VARCHAR(50) NOT NULL CHECK (
        feedback_type IN ('approval', 'rejection', 'modification', 'escalation', 'comment', 'rating')
    ),
    content TEXT,
    rating FLOAT CHECK (rating >= 0.0 AND rating <= 1.0),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    original_recommendation TEXT,
    user_recommendation TEXT,
    reasoning TEXT,
    context JSONB DEFAULT '{}',
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User overrides: Specific user overrides of board decisions
CREATE TABLE IF NOT EXISTS user_overrides (
    override_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    task_id VARCHAR(255) REFERENCES board_tasks(task_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    override_type VARCHAR(50) NOT NULL CHECK (
        override_type IN ('approve', 'reject', 'modify')
    ),
    original_recommendation TEXT,
    new_recommendation TEXT,
    reasoning TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- USER PREFERENCES AND CUSTOMIZATION
-- ============================================================================

-- User profiles: Basic user profile information
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(255) PRIMARY KEY,
    display_name VARCHAR(255),
    risk_tolerance FLOAT DEFAULT 0.5 CHECK (risk_tolerance >= 0.0 AND risk_tolerance <= 1.0),
    automation_level FLOAT DEFAULT 0.7 CHECK (automation_level >= 0.0 AND automation_level <= 1.0),
    profile_version VARCHAR(20) DEFAULT '1.0',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User preferences: Individual preference settings
CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(255) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
    preference_type VARCHAR(50) NOT NULL CHECK (
        preference_type IN ('director_weight', 'risk_tolerance', 'automation_level',
                           'notification_setting', 'approval_threshold', 'escalation_rule',
                           'context_filter', 'priority_rule')
    ),
    key VARCHAR(100) NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    UNIQUE(user_id, preference_type, key)
);

-- User constraints: User-defined constraints for board decisions
CREATE TABLE IF NOT EXISTS user_constraints (
    constraint_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(255) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
    constraint_type VARCHAR(50) NOT NULL CHECK (
        constraint_type IN ('time_limit', 'resource_limit', 'approval_required',
                           'blacklist_pattern', 'whitelist_pattern', 'director_exclusion',
                           'minimum_consensus')
    ),
    rule_expression TEXT NOT NULL,
    description TEXT,
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- ============================================================================
-- MACHINE LEARNING AND FEEDBACK SYSTEM
-- ============================================================================

-- Learning insights: ML-derived insights from user feedback
CREATE TABLE IF NOT EXISTS learning_insights (
    insight_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    learning_type VARCHAR(50) NOT NULL CHECK (
        learning_type IN ('director_weight_adjustment', 'pattern_recognition',
                         'preference_extraction', 'confidence_calibration')
    ),
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    description TEXT,
    affected_directors TEXT[],
    weight_adjustments JSONB DEFAULT '{}',
    pattern_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validation_score FLOAT CHECK (validation_score >= 0.0 AND validation_score <= 1.0),
    applied BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User behavior patterns: Detected patterns in user behavior
CREATE TABLE IF NOT EXISTS user_behavior_patterns (
    pattern_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(255) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_strength FLOAT NOT NULL CHECK (pattern_strength >= 0.0 AND pattern_strength <= 1.0),
    context_conditions JSONB DEFAULT '{}',
    examples TEXT[],
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Director weight adjustments: Personalized director weights per user
CREATE TABLE IF NOT EXISTS director_weight_adjustments (
    adjustment_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(255) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
    director_id VARCHAR(255) NOT NULL,
    original_weight FLOAT DEFAULT 1.0,
    adjusted_weight FLOAT NOT NULL CHECK (adjusted_weight >= 0.0),
    adjustment_reason TEXT,
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feedback processing queue: Queue for batch processing of feedback
CREATE TABLE IF NOT EXISTS feedback_processing_queue (
    queue_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    feedback_id VARCHAR(255) REFERENCES user_feedback(feedback_id) ON DELETE CASCADE,
    processing_type VARCHAR(50) NOT NULL,
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_attempts INTEGER DEFAULT 0,
    last_error TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- KNOWLEDGE BASE AND VERSIONING
-- ============================================================================

-- Knowledge items: Core knowledge base entries
CREATE TABLE IF NOT EXISTS knowledge_items (
    item_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    knowledge_type VARCHAR(50) NOT NULL CHECK (
        knowledge_type IN ('decision_pattern', 'director_behavior', 'user_preference',
                          'system_metric', 'error_pattern', 'performance_data',
                          'security_event', 'learning_insight')
    ),
    title TEXT NOT NULL,
    content JSONB NOT NULL,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    created_by VARCHAR(255),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    relevance_score FLOAT DEFAULT 1.0 CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
    metadata JSONB DEFAULT '{}'
);

-- Knowledge versions: Version control for knowledge base
CREATE TABLE IF NOT EXISTS knowledge_versions (
    version_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    version_number INTEGER NOT NULL,
    action VARCHAR(50) NOT NULL CHECK (
        action IN ('create', 'update', 'delete', 'merge', 'rollback')
    ),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    affected_items TEXT[],
    snapshot_hash VARCHAR(255),
    rollback_data BYTEA
);

-- Knowledge analytics: Periodic analytics snapshots
CREATE TABLE IF NOT EXISTS knowledge_analytics (
    analytics_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    date DATE DEFAULT CURRENT_DATE,
    total_items INTEGER,
    items_by_type JSONB DEFAULT '{}',
    items_by_tag JSONB DEFAULT '{}',
    avg_confidence FLOAT,
    query_count INTEGER DEFAULT 0,
    cache_hit_rate FLOAT CHECK (cache_hit_rate >= 0.0 AND cache_hit_rate <= 1.0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Knowledge relationships: Relationships between knowledge items
CREATE TABLE IF NOT EXISTS knowledge_relationships (
    relationship_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    source_item_id VARCHAR(255) REFERENCES knowledge_items(item_id) ON DELETE CASCADE,
    target_item_id VARCHAR(255) REFERENCES knowledge_items(item_id) ON DELETE CASCADE,
    relationship_type VARCHAR(50),
    strength FLOAT DEFAULT 1.0 CHECK (strength >= 0.0 AND strength <= 1.0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- ============================================================================
-- DIRECTOR PERFORMANCE AND METRICS
-- ============================================================================

-- Director performance: Historical performance metrics for directors
CREATE TABLE IF NOT EXISTS director_performance (
    performance_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    director_id VARCHAR(255) NOT NULL,
    director_name VARCHAR(255) NOT NULL,
    date DATE DEFAULT CURRENT_DATE,
    evaluations_count INTEGER DEFAULT 0,
    avg_confidence FLOAT CHECK (avg_confidence >= 0.0 AND avg_confidence <= 1.0),
    avg_processing_time FLOAT DEFAULT 0.0,
    avg_risk_score FLOAT CHECK (avg_risk_score >= 0.0 AND avg_risk_score <= 1.0),
    approval_rate FLOAT CHECK (approval_rate >= 0.0 AND approval_rate <= 1.0),
    user_satisfaction_score FLOAT CHECK (user_satisfaction_score >= 0.0 AND user_satisfaction_score <= 1.0),
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Director specializations: Director areas of expertise
CREATE TABLE IF NOT EXISTS director_specializations (
    specialization_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    director_id VARCHAR(255) NOT NULL,
    specialization_area VARCHAR(100) NOT NULL,
    proficiency_level FLOAT DEFAULT 1.0 CHECK (proficiency_level >= 0.0 AND proficiency_level <= 1.0),
    context_patterns TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- SYSTEM MONITORING AND AUDITING
-- ============================================================================

-- System events: General system events and monitoring
CREATE TABLE IF NOT EXISTS system_events (
    event_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    event_type VARCHAR(50) NOT NULL,
    event_category VARCHAR(50) NOT NULL CHECK (
        event_category IN ('info', 'warning', 'error', 'security', 'performance')
    ),
    description TEXT,
    details JSONB DEFAULT '{}',
    user_id VARCHAR(255),
    task_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT
);

-- Audit log: Comprehensive audit trail
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    table_name VARCHAR(100) NOT NULL,
    record_id VARCHAR(255) NOT NULL,
    operation VARCHAR(20) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    user_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Performance metrics: System performance tracking
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    context JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- SANDBOX AND SECURITY
-- ============================================================================

-- Sandbox executions: Record of code execution in sandbox
CREATE TABLE IF NOT EXISTS sandbox_executions (
    execution_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    task_id VARCHAR(255) REFERENCES board_tasks(task_id) ON DELETE CASCADE,
    director_id VARCHAR(255),
    code_hash VARCHAR(255),
    execution_result VARCHAR(20) CHECK (
        execution_result IN ('success', 'error', 'timeout', 'resource_limit', 'security_violation', 'blocked')
    ),
    execution_time FLOAT DEFAULT 0.0,
    memory_used INTEGER DEFAULT 0,
    cpu_time FLOAT DEFAULT 0.0,
    security_violations TEXT[],
    stdout_excerpt TEXT,
    stderr_excerpt TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Security events: Security-related events and violations
CREATE TABLE IF NOT EXISTS security_events (
    event_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    source VARCHAR(100),
    description TEXT,
    details JSONB DEFAULT '{}',
    user_id VARCHAR(255),
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    investigated BOOLEAN DEFAULT FALSE,
    investigation_notes TEXT
);

-- ============================================================================
-- PREFERENCE TEMPLATES AND SHARING
-- ============================================================================

-- Preference templates: Reusable preference configurations
CREATE TABLE IF NOT EXISTS preference_templates (
    template_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    template_name VARCHAR(255) NOT NULL,
    description TEXT,
    template_data JSONB NOT NULL,
    created_by VARCHAR(255),
    is_public BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Board tasks indexes
CREATE INDEX IF NOT EXISTS idx_board_tasks_user_id ON board_tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_board_tasks_status ON board_tasks(status);
CREATE INDEX IF NOT EXISTS idx_board_tasks_submitted_at ON board_tasks(submitted_at);
CREATE INDEX IF NOT EXISTS idx_board_tasks_priority ON board_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_board_tasks_completion_timestamp ON board_tasks(completion_timestamp);

-- Director evaluations indexes
CREATE INDEX IF NOT EXISTS idx_director_evaluations_task_id ON director_evaluations(task_id);
CREATE INDEX IF NOT EXISTS idx_director_evaluations_director_id ON director_evaluations(director_id);
CREATE INDEX IF NOT EXISTS idx_director_evaluations_recommendation ON director_evaluations(recommendation);
CREATE INDEX IF NOT EXISTS idx_director_evaluations_timestamp ON director_evaluations(timestamp);

-- Decision timeline indexes
CREATE INDEX IF NOT EXISTS idx_decision_timeline_task_id ON decision_timeline(task_id);
CREATE INDEX IF NOT EXISTS idx_decision_timeline_timestamp ON decision_timeline(timestamp);
CREATE INDEX IF NOT EXISTS idx_decision_timeline_status ON decision_timeline(status);

-- User feedback indexes
CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_task_id ON user_feedback(task_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_feedback_type ON user_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp ON user_feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_feedback_processed ON user_feedback(processed);

-- User preferences indexes
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_type ON user_preferences(preference_type);
CREATE INDEX IF NOT EXISTS idx_user_preferences_active ON user_preferences(active);

-- User constraints indexes
CREATE INDEX IF NOT EXISTS idx_user_constraints_user_id ON user_constraints(user_id);
CREATE INDEX IF NOT EXISTS idx_user_constraints_type ON user_constraints(constraint_type);
CREATE INDEX IF NOT EXISTS idx_user_constraints_active ON user_constraints(active);
CREATE INDEX IF NOT EXISTS idx_user_constraints_priority ON user_constraints(priority);

-- User behavior patterns indexes
CREATE INDEX IF NOT EXISTS idx_behavior_patterns_user_id ON user_behavior_patterns(user_id);
CREATE INDEX IF NOT EXISTS idx_behavior_patterns_type ON user_behavior_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_behavior_patterns_active ON user_behavior_patterns(active);

-- Director weight adjustments indexes
CREATE INDEX IF NOT EXISTS idx_weight_adjustments_user_id ON director_weight_adjustments(user_id);
CREATE INDEX IF NOT EXISTS idx_weight_adjustments_director_id ON director_weight_adjustments(director_id);
CREATE INDEX IF NOT EXISTS idx_weight_adjustments_active ON director_weight_adjustments(active);

-- Knowledge items indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_items_type ON knowledge_items(knowledge_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_tags ON knowledge_items USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_created_at ON knowledge_items(created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_updated_at ON knowledge_items(updated_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_confidence ON knowledge_items(confidence);

-- Knowledge versions indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_versions_number ON knowledge_versions(version_number);
CREATE INDEX IF NOT EXISTS idx_knowledge_versions_action ON knowledge_versions(action);
CREATE INDEX IF NOT EXISTS idx_knowledge_versions_created_at ON knowledge_versions(created_at);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_items_content_search
ON knowledge_items USING GIN(to_tsvector('english', title || ' ' || content::text));

CREATE INDEX IF NOT EXISTS idx_board_tasks_request_search
ON board_tasks USING GIN(to_tsvector('english', original_request));

-- Performance monitoring indexes
CREATE INDEX IF NOT EXISTS idx_director_performance_director_id ON director_performance(director_id);
CREATE INDEX IF NOT EXISTS idx_director_performance_date ON director_performance(date);
CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_events_category ON system_events(event_category);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_board_tasks_user_status ON board_tasks(user_id, status);
CREATE INDEX IF NOT EXISTS idx_director_eval_task_director ON director_evaluations(task_id, director_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_user_type ON user_feedback(user_id, feedback_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_type_confidence ON knowledge_items(knowledge_type, confidence);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updating timestamps
CREATE TRIGGER update_board_tasks_updated_at
    BEFORE UPDATE ON board_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_user_constraints_updated_at
    BEFORE UPDATE ON user_constraints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_knowledge_items_updated_at
    BEFORE UPDATE ON knowledge_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Function to calculate consensus score
CREATE OR REPLACE FUNCTION calculate_consensus_score(task_uuid VARCHAR(255))
RETURNS FLOAT AS $$
DECLARE
    total_evaluations INTEGER;
    recommendation_counts JSONB;
    max_count INTEGER;
    consensus_score FLOAT;
BEGIN
    -- Count total evaluations for the task
    SELECT COUNT(*) INTO total_evaluations
    FROM director_evaluations
    WHERE task_id = task_uuid;

    IF total_evaluations = 0 THEN
        RETURN 0.0;
    END IF;

    -- Count recommendations by type
    SELECT jsonb_object_agg(recommendation, cnt) INTO recommendation_counts
    FROM (
        SELECT recommendation, COUNT(*) as cnt
        FROM director_evaluations
        WHERE task_id = task_uuid
        GROUP BY recommendation
    ) t;

    -- Find the maximum count
    SELECT MAX((value::text)::integer) INTO max_count
    FROM jsonb_each(recommendation_counts);

    -- Calculate consensus score
    consensus_score := max_count::FLOAT / total_evaluations::FLOAT;

    RETURN consensus_score;
END;
$$ LANGUAGE plpgsql;

-- Function to update task consensus when evaluations change
CREATE OR REPLACE FUNCTION update_task_consensus()
RETURNS TRIGGER AS $$
DECLARE
    task_uuid VARCHAR(255);
    new_consensus_score FLOAT;
    avg_confidence FLOAT;
BEGIN
    -- Get task ID from the evaluation
    IF TG_OP = 'DELETE' THEN
        task_uuid := OLD.task_id;
    ELSE
        task_uuid := NEW.task_id;
    END IF;

    -- Calculate new consensus score
    SELECT calculate_consensus_score(task_uuid) INTO new_consensus_score;

    -- Calculate average confidence
    SELECT AVG(confidence) INTO avg_confidence
    FROM director_evaluations
    WHERE task_id = task_uuid;

    -- Update the board task
    UPDATE board_tasks
    SET
        consensus_score = new_consensus_score,
        confidence_score = COALESCE(avg_confidence, 0.0),
        updated_at = NOW()
    WHERE board_tasks.task_id = task_uuid;

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Trigger to update consensus when evaluations change
CREATE TRIGGER update_consensus_on_evaluation_change
    AFTER INSERT OR UPDATE OR DELETE ON director_evaluations
    FOR EACH ROW EXECUTE FUNCTION update_task_consensus();

-- Function for audit logging
CREATE OR REPLACE FUNCTION audit_log_changes()
RETURNS TRIGGER AS $$
DECLARE
    audit_user_id VARCHAR(255) := current_setting('application.user_id', true);
    audit_ip_address INET := current_setting('application.ip_address', true)::INET;
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, operation, old_values, user_id, ip_address)
        VALUES (TG_TABLE_NAME, OLD.user_id, TG_OP, row_to_json(OLD), audit_user_id, audit_ip_address);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, operation, old_values, new_values, user_id, ip_address)
        VALUES (TG_TABLE_NAME, NEW.user_id, TG_OP, row_to_json(OLD), row_to_json(NEW), audit_user_id, audit_ip_address);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, operation, new_values, user_id, ip_address)
        VALUES (TG_TABLE_NAME, NEW.user_id, TG_OP, row_to_json(NEW), audit_user_id, audit_ip_address);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply audit logging to sensitive tables
CREATE TRIGGER audit_user_preferences
    AFTER INSERT OR UPDATE OR DELETE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION audit_log_changes();

CREATE TRIGGER audit_user_overrides
    AFTER INSERT OR UPDATE OR DELETE ON user_overrides
    FOR EACH ROW EXECUTE FUNCTION audit_log_changes();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for task summary with director participation
CREATE OR REPLACE VIEW task_summary AS
SELECT
    bt.task_id,
    bt.user_id,
    bt.original_request,
    bt.status,
    bt.consensus_score,
    bt.confidence_score,
    bt.submitted_at,
    bt.completion_timestamp,
    COUNT(de.evaluation_id) as director_count,
    AVG(de.confidence) as avg_director_confidence,
    STRING_AGG(DISTINCT de.director_name, ', ') as participating_directors
FROM board_tasks bt
LEFT JOIN director_evaluations de ON bt.task_id = de.task_id
GROUP BY bt.task_id, bt.user_id, bt.original_request, bt.status,
         bt.consensus_score, bt.confidence_score, bt.submitted_at, bt.completion_timestamp;

-- View for director performance summary
CREATE OR REPLACE VIEW director_performance_summary AS
SELECT
    de.director_id,
    de.director_name,
    COUNT(*) as total_evaluations,
    AVG(de.confidence) as avg_confidence,
    AVG(de.processing_time) as avg_processing_time,
    AVG(de.risk_score) as avg_risk_score,
    COUNT(CASE WHEN de.recommendation = 'approve' THEN 1 END)::FLOAT / COUNT(*)::FLOAT as approval_rate,
    COUNT(CASE WHEN bt.status = 'completed' THEN 1 END) as completed_tasks,
    COUNT(CASE WHEN uo.override_id IS NOT NULL THEN 1 END) as override_count
FROM director_evaluations de
LEFT JOIN board_tasks bt ON de.task_id = bt.task_id
LEFT JOIN user_overrides uo ON de.task_id = uo.task_id
GROUP BY de.director_id, de.director_name;

-- View for user feedback analytics
CREATE OR REPLACE VIEW user_feedback_analytics AS
SELECT
    uf.user_id,
    COUNT(*) as total_feedback,
    AVG(uf.rating) as avg_rating,
    COUNT(CASE WHEN uf.feedback_type = 'approval' THEN 1 END)::FLOAT / COUNT(*)::FLOAT as approval_rate,
    COUNT(CASE WHEN uf.feedback_type = 'rejection' THEN 1 END)::FLOAT / COUNT(*)::FLOAT as rejection_rate,
    COUNT(CASE WHEN uf.feedback_type = 'modification' THEN 1 END)::FLOAT / COUNT(*)::FLOAT as modification_rate,
    MAX(uf.timestamp) as last_feedback_date
FROM user_feedback uf
GROUP BY uf.user_id;

-- ============================================================================
-- INITIAL DATA AND SETUP
-- ============================================================================

-- Insert default system events
INSERT INTO system_events (event_id, event_type, event_category, description, details)
VALUES
    (uuid_generate_v4()::text, 'system_start', 'info', 'Board system initialized', '{"version": "1.0", "schema_version": "1.0"}'),
    (uuid_generate_v4()::text, 'schema_created', 'info', 'Database schema created successfully', '{"tables_created": 25, "indexes_created": 35}')
ON CONFLICT (event_id) DO NOTHING;

-- Insert default preference templates
INSERT INTO preference_templates (template_id, template_name, description, template_data, is_public)
VALUES
    (uuid_generate_v4()::text, 'Conservative User', 'Conservative settings for risk-averse users',
     '{"risk_tolerance": 0.2, "automation_level": 0.5, "approval_threshold": 0.9, "director_weights": {"security_director": 1.5, "ethics_director": 1.3}}', true),
    (uuid_generate_v4()::text, 'Balanced User', 'Balanced settings for average users',
     '{"risk_tolerance": 0.5, "automation_level": 0.7, "approval_threshold": 0.8, "director_weights": {"security_director": 1.0, "performance_director": 1.0, "quality_director": 1.0, "ethics_director": 1.0, "ux_director": 1.0}}', true),
    (uuid_generate_v4()::text, 'Progressive User', 'Progressive settings for risk-tolerant users',
     '{"risk_tolerance": 0.8, "automation_level": 0.9, "approval_threshold": 0.6, "director_weights": {"performance_director": 1.3, "ux_director": 1.2}}', true)
ON CONFLICT (template_id) DO NOTHING;

-- ============================================================================
-- CLEANUP AND MAINTENANCE FUNCTIONS
-- ============================================================================

-- Function to cleanup old audit logs
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM audit_log
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    INSERT INTO system_events (event_type, event_category, description, details)
    VALUES ('audit_cleanup', 'info', 'Cleaned up old audit logs',
            jsonb_build_object('deleted_count', deleted_count, 'days_kept', days_to_keep));

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to generate daily analytics
CREATE OR REPLACE FUNCTION generate_daily_analytics()
RETURNS VOID AS $$
BEGIN
    INSERT INTO knowledge_analytics (date, total_items, items_by_type, items_by_tag, avg_confidence)
    SELECT
        CURRENT_DATE,
        COUNT(*),
        jsonb_object_agg(knowledge_type, type_count),
        jsonb_object_agg(tag, tag_count),
        AVG(confidence)
    FROM (
        SELECT
            knowledge_type,
            COUNT(*) OVER (PARTITION BY knowledge_type) as type_count,
            unnest(tags) as tag,
            COUNT(*) OVER (PARTITION BY unnest(tags)) as tag_count,
            confidence
        FROM knowledge_items
    ) t
    ON CONFLICT (date) DO UPDATE SET
        total_items = EXCLUDED.total_items,
        items_by_type = EXCLUDED.items_by_type,
        items_by_tag = EXCLUDED.items_by_tag,
        avg_confidence = EXCLUDED.avg_confidence;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant appropriate permissions (adjust as needed for your security model)
-- These are basic grants - you should customize based on your application needs

-- Application role for the Echo Brain service
-- CREATE ROLE echo_brain_app;
-- GRANT CONNECT ON DATABASE your_database TO echo_brain_app;
-- GRANT USAGE ON SCHEMA public TO echo_brain_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO echo_brain_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO echo_brain_app;

-- Read-only role for reporting
-- CREATE ROLE echo_brain_readonly;
-- GRANT CONNECT ON DATABASE your_database TO echo_brain_readonly;
-- GRANT USAGE ON SCHEMA public TO echo_brain_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO echo_brain_readonly;

-- ============================================================================
-- SCHEMA VALIDATION
-- ============================================================================

-- Verify schema was created successfully
DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
    trigger_count INTEGER;
    function_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name LIKE '%board%' OR table_name LIKE '%knowledge%' OR table_name LIKE '%user_%';

    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'public' AND indexname LIKE 'idx_%';

    SELECT COUNT(*) INTO trigger_count
    FROM information_schema.triggers
    WHERE trigger_schema = 'public';

    SELECT COUNT(*) INTO function_count
    FROM information_schema.routines
    WHERE routine_schema = 'public' AND routine_type = 'FUNCTION';

    INSERT INTO system_events (event_type, event_category, description, details)
    VALUES ('schema_validation', 'info', 'Schema validation completed',
            jsonb_build_object(
                'tables_created', table_count,
                'indexes_created', index_count,
                'triggers_created', trigger_count,
                'functions_created', function_count,
                'validation_timestamp', NOW()
            ));

    RAISE NOTICE 'Schema validation completed: % tables, % indexes, % triggers, % functions',
                 table_count, index_count, trigger_count, function_count;
END;
$$;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

COMMIT;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE 'Echo Brain Board of Directors database schema created successfully!';
    RAISE NOTICE 'Schema includes:';
    RAISE NOTICE '  - Complete board decision tracking system';
    RAISE NOTICE '  - User feedback and learning system';
    RAISE NOTICE '  - Knowledge base with versioning';
    RAISE NOTICE '  - Performance monitoring and analytics';
    RAISE NOTICE '  - Security and audit logging';
    RAISE NOTICE '  - Comprehensive indexing for performance';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '  1. Configure application connection settings';
    RAISE NOTICE '  2. Set up user roles and permissions';
    RAISE NOTICE '  3. Initialize default directors and preferences';
    RAISE NOTICE '  4. Start the Echo Brain Board API service';
END;
$$;