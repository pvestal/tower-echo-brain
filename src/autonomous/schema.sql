-- Autonomous Core Foundation Database Schema
-- Echo Brain Phase 1 - Autonomous Operations

-- Goals table: High-level autonomous objectives
CREATE TABLE IF NOT EXISTS autonomous_goals (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    goal_type VARCHAR(100) NOT NULL, -- e.g., 'research', 'coding', 'analysis', 'maintenance'
    status VARCHAR(50) NOT NULL DEFAULT 'active', -- 'active', 'paused', 'completed', 'failed'
    priority INTEGER NOT NULL DEFAULT 5, -- 1 (highest) to 10 (lowest)
    progress_percent DECIMAL(5,2) NOT NULL DEFAULT 0.00, -- 0.00 to 100.00
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}' -- Flexible storage for goal-specific data
);

-- Tasks table: Specific executable actions under goals
CREATE TABLE IF NOT EXISTS autonomous_tasks (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES autonomous_goals(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL, -- e.g., 'api_call', 'file_operation', 'analysis', 'model_inference'
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed', 'needs_approval', 'interrupted', 'rejected'
    safety_level VARCHAR(50) NOT NULL DEFAULT 'auto', -- 'auto', 'notify', 'review', 'forbidden'
    priority INTEGER NOT NULL DEFAULT 5, -- 1 (highest) to 10 (lowest)
    scheduled_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    result TEXT, -- Task execution result/output
    error TEXT, -- Error message if task failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}' -- Task metadata including safety_level and other details
);

-- Approvals table: Human approval workflow for sensitive tasks
CREATE TABLE IF NOT EXISTS autonomous_approvals (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES autonomous_tasks(id) ON DELETE CASCADE,
    action_description TEXT NOT NULL, -- Human-readable description of what will happen
    risk_assessment TEXT NOT NULL, -- Risk analysis and safety considerations
    proposed_action JSONB NOT NULL, -- Detailed action parameters
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by VARCHAR(255), -- Username or identifier of reviewer
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Notifications table: User notifications for tasks requiring attention
CREATE TABLE IF NOT EXISTS autonomous_notifications (
    id SERIAL PRIMARY KEY,
    notification_type VARCHAR(100) NOT NULL, -- 'approval_required', 'task_executed', 'forbidden_attempt', etc.
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    task_id INTEGER REFERENCES autonomous_tasks(id) ON DELETE CASCADE,
    read BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit log table: Comprehensive logging of all autonomous operations
CREATE TABLE IF NOT EXISTS autonomous_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL, -- e.g., 'goal_created', 'task_executed', 'approval_requested'
    goal_id INTEGER REFERENCES autonomous_goals(id) ON DELETE SET NULL,
    task_id INTEGER REFERENCES autonomous_tasks(id) ON DELETE SET NULL,
    action VARCHAR(255) NOT NULL, -- Brief description of action taken
    details JSONB DEFAULT '{}', -- Detailed information about the event
    safety_level VARCHAR(50) NOT NULL DEFAULT 'auto',
    outcome VARCHAR(100) -- 'success', 'failure', 'pending', 'blocked'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_autonomous_goals_status ON autonomous_goals(status);
CREATE INDEX IF NOT EXISTS idx_autonomous_goals_priority ON autonomous_goals(priority);
CREATE INDEX IF NOT EXISTS idx_autonomous_goals_created_at ON autonomous_goals(created_at);

CREATE INDEX IF NOT EXISTS idx_autonomous_tasks_goal_id ON autonomous_tasks(goal_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_tasks_status ON autonomous_tasks(status);
CREATE INDEX IF NOT EXISTS idx_autonomous_tasks_safety_level ON autonomous_tasks(safety_level);
CREATE INDEX IF NOT EXISTS idx_autonomous_tasks_scheduled_at ON autonomous_tasks(scheduled_at);

CREATE INDEX IF NOT EXISTS idx_autonomous_approvals_task_id ON autonomous_approvals(task_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_approvals_status ON autonomous_approvals(status);
CREATE INDEX IF NOT EXISTS idx_autonomous_approvals_created_at ON autonomous_approvals(created_at);

CREATE INDEX IF NOT EXISTS idx_autonomous_notifications_task_id ON autonomous_notifications(task_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_notifications_read ON autonomous_notifications(read);
CREATE INDEX IF NOT EXISTS idx_autonomous_notifications_type ON autonomous_notifications(notification_type);
CREATE INDEX IF NOT EXISTS idx_autonomous_notifications_created_at ON autonomous_notifications(created_at);

CREATE INDEX IF NOT EXISTS idx_autonomous_audit_log_timestamp ON autonomous_audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_autonomous_audit_log_event_type ON autonomous_audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_autonomous_audit_log_goal_id ON autonomous_audit_log(goal_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_audit_log_task_id ON autonomous_audit_log(task_id);

-- Triggers to automatically update timestamps
CREATE OR REPLACE FUNCTION update_autonomous_goals_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_autonomous_goals_updated_at
    BEFORE UPDATE ON autonomous_goals
    FOR EACH ROW
    EXECUTE FUNCTION update_autonomous_goals_updated_at();

-- Comments for documentation
COMMENT ON TABLE autonomous_goals IS 'High-level autonomous objectives and their tracking';
COMMENT ON TABLE autonomous_tasks IS 'Specific executable actions under autonomous goals';
COMMENT ON TABLE autonomous_approvals IS 'Human approval workflow for sensitive autonomous tasks';
COMMENT ON TABLE autonomous_audit_log IS 'Comprehensive audit trail of all autonomous operations';

COMMENT ON COLUMN autonomous_goals.goal_type IS 'Type of goal: research, coding, analysis, maintenance, etc.';
COMMENT ON COLUMN autonomous_goals.status IS 'Current status: active, paused, completed, failed';
COMMENT ON COLUMN autonomous_goals.priority IS 'Priority level from 1 (highest) to 10 (lowest)';
COMMENT ON COLUMN autonomous_goals.progress_percent IS 'Completion percentage from 0.00 to 100.00';
COMMENT ON COLUMN autonomous_goals.metadata IS 'Flexible JSONB storage for goal-specific configuration';

COMMENT ON COLUMN autonomous_tasks.task_type IS 'Type of task: api_call, file_operation, analysis, model_inference, etc.';
COMMENT ON COLUMN autonomous_tasks.status IS 'Current status: pending, in_progress, completed, failed, needs_approval, interrupted, rejected';
COMMENT ON COLUMN autonomous_tasks.safety_level IS 'Safety classification: auto, notify, review, forbidden';
COMMENT ON COLUMN autonomous_tasks.priority IS 'Priority level from 1 (highest) to 10 (lowest)';

COMMENT ON COLUMN autonomous_approvals.action_description IS 'Human-readable description of the proposed action';
COMMENT ON COLUMN autonomous_approvals.risk_assessment IS 'Risk analysis and safety considerations';
COMMENT ON COLUMN autonomous_approvals.proposed_action IS 'JSONB containing detailed action parameters';
COMMENT ON COLUMN autonomous_approvals.status IS 'Approval status: pending, approved, rejected';

COMMENT ON COLUMN autonomous_audit_log.event_type IS 'Type of event: goal_created, task_executed, approval_requested, etc.';
COMMENT ON COLUMN autonomous_audit_log.action IS 'Brief description of the action taken';
COMMENT ON COLUMN autonomous_audit_log.details IS 'JSONB containing detailed event information';
COMMENT ON COLUMN autonomous_audit_log.outcome IS 'Result of the action: success, failure, pending, blocked';