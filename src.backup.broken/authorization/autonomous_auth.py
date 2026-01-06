#!/usr/bin/env python3
"""
Autonomous Action Authorization System for Echo Brain
Multi-tier approval workflow for safe autonomous operations
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncpg

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for autonomous actions"""
    SAFE = 1        # Read-only, monitoring, logging
    LOW = 2         # Service restarts, log rotation
    MEDIUM = 3      # Code formatting, minor config changes
    HIGH = 4        # Service config changes, code refactoring
    CRITICAL = 5    # Production deployments, database changes


class ApprovalMode(Enum):
    """Approval requirements by risk level"""
    AUTO_APPROVE = "auto"           # No approval needed
    NOTIFY_ONLY = "notify"          # Execute but notify Patrick
    APPROVAL_REQUIRED = "approval"  # Must get approval first
    TWO_FACTOR = "two_factor"       # Requires 2FA confirmation


class AutonomousAuthorization:
    """Authorization system for Echo's autonomous actions"""

    def __init__(self, db_url: str = "postgresql://patrick:***REMOVED***@localhost/echo_brain"):
        self.db_url = db_url
        self.pool = None

        # Risk-based approval mapping
        self.approval_rules = {
            RiskLevel.SAFE: ApprovalMode.AUTO_APPROVE,
            RiskLevel.LOW: ApprovalMode.NOTIFY_ONLY,
            RiskLevel.MEDIUM: ApprovalMode.APPROVAL_REQUIRED,
            RiskLevel.HIGH: ApprovalMode.APPROVAL_REQUIRED,
            RiskLevel.CRITICAL: ApprovalMode.TWO_FACTOR
        }

        # Action type → Risk level mapping
        self.action_risks = {
            # Safe operations
            "monitor_service": RiskLevel.SAFE,
            "analyze_code": RiskLevel.SAFE,
            "scan_logs": RiskLevel.SAFE,
            "check_health": RiskLevel.SAFE,

            # Low risk operations
            "restart_service": RiskLevel.LOW,
            "rotate_logs": RiskLevel.LOW,
            "cleanup_temp": RiskLevel.LOW,

            # Medium risk operations
            "format_code": RiskLevel.MEDIUM,
            "update_comments": RiskLevel.MEDIUM,
            "optimize_imports": RiskLevel.MEDIUM,

            # High risk operations
            "refactor_code": RiskLevel.HIGH,
            "modify_config": RiskLevel.HIGH,
            "update_dependencies": RiskLevel.HIGH,

            # Critical operations
            "deploy_service": RiskLevel.CRITICAL,
            "modify_database": RiskLevel.CRITICAL,
            "change_production": RiskLevel.CRITICAL
        }

    async def initialize(self):
        """Initialize database connection and tables"""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        await self.ensure_tables()

    async def ensure_tables(self):
        """Create authorization tables"""
        async with self.pool.acquire() as conn:
            # Approval requests table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS autonomous_approval_requests (
                    id SERIAL PRIMARY KEY,
                    action_type VARCHAR(100) NOT NULL,
                    risk_level VARCHAR(20) NOT NULL,
                    description TEXT,
                    details JSONB,
                    status VARCHAR(20) DEFAULT 'pending',
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    approved_at TIMESTAMP,
                    approved_by VARCHAR(100),
                    expires_at TIMESTAMP,
                    execution_result JSONB,
                    rollback_plan JSONB
                )
            """)

            # Action history for learning
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS autonomous_action_history (
                    id SERIAL PRIMARY KEY,
                    action_type VARCHAR(100),
                    risk_level VARCHAR(20),
                    approval_mode VARCHAR(20),
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN,
                    duration_seconds FLOAT,
                    error_message TEXT,
                    metrics JSONB,
                    rollback_performed BOOLEAN DEFAULT FALSE
                )
            """)

            # Patrick's authorization preferences
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_preferences (
                    id SERIAL PRIMARY KEY,
                    action_pattern VARCHAR(200),
                    approval_mode VARCHAR(20),
                    auto_approve_conditions JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def assess_risk(self, action_type: str, context: Dict[str, Any]) -> RiskLevel:
        """
        Assess risk level for an autonomous action

        Args:
            action_type: Type of action (e.g., 'restart_service', 'refactor_code')
            context: Context including target, scope, impact

        Returns:
            RiskLevel enum
        """
        # Base risk from action type
        base_risk = self.action_risks.get(action_type, RiskLevel.HIGH)

        # Adjust based on context
        target = context.get('target', '')
        scope = context.get('scope', 'unknown')

        # Elevate risk for production services
        if 'production' in target.lower() or context.get('env') == 'production':
            if base_risk.value < RiskLevel.HIGH.value:
                base_risk = RiskLevel.HIGH

        # Reduce risk for test/dev environments
        if 'test' in target.lower() or context.get('env') == 'test':
            if base_risk.value > RiskLevel.LOW.value:
                base_risk = RiskLevel(base_risk.value - 1)

        # Elevate if affecting multiple services
        if scope == 'multiple' or context.get('service_count', 1) > 1:
            base_risk = RiskLevel(min(base_risk.value + 1, 5))

        logger.info(f"🎯 Risk assessment: {action_type} → {base_risk.name} (context: {scope})")
        return base_risk

    async def request_approval(
        self,
        action_type: str,
        description: str,
        details: Dict[str, Any],
        risk_level: Optional[RiskLevel] = None,
        expires_in_hours: int = 24
    ) -> int:
        """
        Request approval for an autonomous action

        Returns:
            Approval request ID
        """
        if risk_level is None:
            risk_level = await self.assess_risk(action_type, details)

        approval_mode = self.approval_rules[risk_level]

        # Auto-approve safe actions
        if approval_mode == ApprovalMode.AUTO_APPROVE:
            logger.info(f"✅ Auto-approved: {action_type} (SAFE)")
            return -1  # No approval needed

        async with self.pool.acquire() as conn:
            request_id = await conn.fetchval("""
                INSERT INTO autonomous_approval_requests
                (action_type, risk_level, description, json.dumps(details), expires_at, rollback_plan)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                RETURNING id
            """, action_type, risk_level.name, description, json.dumps(details),
                datetime.now() + timedelta(hours=expires_in_hours),
                self._generate_rollback_plan(action_type, details))

        # Send notification based on mode
        if approval_mode == ApprovalMode.NOTIFY_ONLY:
            await self._notify_patrick(request_id, action_type, description, risk_level, notify_only=True)
            # Auto-approve but log
            await self.approve_request(request_id, "auto-notify")
        else:
            await self._notify_patrick(request_id, action_type, description, risk_level, notify_only=False)

        logger.info(f"📋 Approval requested: {action_type} (ID: {request_id}, Risk: {risk_level.name})")
        return request_id

    async def approve_request(self, request_id: int, approved_by: str = "patrick") -> bool:
        """Approve an autonomous action request"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE autonomous_approval_requests
                SET status = 'approved', approved_at = CURRENT_TIMESTAMP, approved_by = $1
                WHERE id = $2 AND status = 'pending'
            """, approved_by, request_id)

            if result == "UPDATE 1":
                logger.info(f"✅ Approved: Request #{request_id} by {approved_by}")
                return True
            else:
                logger.warning(f"⚠️ Failed to approve: Request #{request_id} not found or already processed")
                return False

    async def check_approval_status(self, request_id: int) -> Optional[str]:
        """Check if an approval request has been approved"""
        if request_id == -1:
            return "approved"  # Auto-approved

        async with self.pool.acquire() as conn:
            status = await conn.fetchval("""
                SELECT status FROM autonomous_approval_requests
                WHERE id = $1
            """, request_id)

            return status

    async def wait_for_approval(self, request_id: int, timeout_seconds: int = 300) -> bool:
        """
        Wait for approval with timeout

        Args:
            request_id: Approval request ID
            timeout_seconds: Maximum wait time (default 5 minutes)

        Returns:
            True if approved, False if rejected/timeout
        """
        if request_id == -1:
            return True  # Auto-approved

        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout_seconds:
            status = await self.check_approval_status(request_id)

            if status == "approved":
                return True
            elif status in ["rejected", "expired"]:
                logger.warning(f"❌ Request #{request_id} was {status}")
                return False

            await asyncio.sleep(5)  # Check every 5 seconds

        logger.warning(f"⏱️ Timeout waiting for approval: Request #{request_id}")
        return False

    async def log_action_result(
        self,
        action_type: str,
        risk_level: RiskLevel,
        success: bool,
        duration: float,
        error: Optional[str] = None,
        metrics: Optional[Dict] = None
    ):
        """Log action result for learning and analytics"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO autonomous_action_history
                (action_type, risk_level, approval_mode, success, duration_seconds, error_message, metrics)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
            """, action_type, risk_level.name, self.approval_rules[risk_level].value,
                success, duration, error, metrics)

    def _generate_rollback_plan(self, action_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rollback plan for an action"""
        rollback = {
            "type": action_type,
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }

        if action_type == "restart_service":
            rollback["steps"] = [
                f"Check service status: systemctl status {details.get('service_name')}",
                f"View logs: journalctl -u {details.get('service_name')} -n 50",
                "Manual restart if needed"
            ]
            rollback["details"] = details
        elif action_type == "modify_config":
            rollback["steps"] = [
                f"Restore backup: cp {details.get('backup_path')} {details.get('config_path')}",
                f"Restart affected service: systemctl restart {details.get('service_name')}"
            ]
            rollback["details"] = details
        elif action_type == "refactor_code":
            rollback["steps"] = [
                f"Git revert: cd {details.get('repo_path')} && git revert HEAD",
                "Run tests to verify rollback",
                "Restart services if needed"
            ]
            rollback["details"] = details

        return rollback

    async def _notify_patrick(
        self,
        request_id: int,
        action_type: str,
        description: str,
        risk_level: RiskLevel,
        notify_only: bool = False
    ):
        """Send notification to Patrick (email/Telegram/dashboard)"""
        message_type = "Notification" if notify_only else "Approval Required"

        message = f"""
🤖 Echo Brain - Autonomous Action {message_type}

Action: {action_type}
Risk Level: {risk_level.name}
Description: {description}
Request ID: #{request_id}

{'✅ This action will be executed automatically.' if notify_only else '⚠️ This action requires your approval.'}

{'Review at: https://vestal-garcia.duckdns.org/echo/approvals/' + str(request_id) if not notify_only else ''}
"""

        # TODO: Integrate with notification service, Telegram, email
        logger.info(f"📧 Notification: {message_type} for {action_type}")

    async def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, action_type, risk_level, description, requested_at, expires_at
                FROM autonomous_approval_requests
                WHERE status = 'pending' AND expires_at > CURRENT_TIMESTAMP
                ORDER BY requested_at ASC
            """)

            return [dict(row) for row in rows]


# Global instance
_auth_instance = None

async def get_autonomous_auth() -> AutonomousAuthorization:
    """Get or create autonomous authorization instance"""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = AutonomousAuthorization()
        await _auth_instance.initialize()
    return _auth_instance
