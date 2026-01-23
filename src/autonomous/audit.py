"""
Audit Logger for Echo Brain Autonomous Operations

The AuditLogger provides comprehensive logging and audit trail capabilities
for all autonomous operations, ensuring full traceability and compliance
with security and operational requirements.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncpg
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Comprehensive audit logging system for autonomous operations.

    Provides detailed logging to database, retrieval capabilities,
    and audit trail management for all autonomous activities.
    """

    def __init__(self):
        """Initialize the AuditLogger with database configuration."""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }
        self._pool = None

        # Event type definitions for consistency
        self.event_types = {
            'goal_created': 'New autonomous goal created',
            'goal_updated': 'Autonomous goal updated',
            'goal_completed': 'Autonomous goal completed',
            'goal_failed': 'Autonomous goal failed',
            'task_created': 'New task created under goal',
            'task_started': 'Task execution started',
            'task_completed': 'Task execution completed',
            'task_failed': 'Task execution failed',
            'approval_requested': 'Human approval requested for task',
            'approval_granted': 'Human approval granted for task',
            'approval_rejected': 'Human approval rejected for task',
            'safety_check': 'Safety constraint check performed',
            'rate_limit_hit': 'Rate limit threshold reached',
            'kill_switch_activated': 'Emergency kill switch activated',
            'kill_switch_deactivated': 'Emergency kill switch deactivated',
            'emergency_halt': 'Emergency halt of all operations',
            'system_error': 'System error occurred during operation'
        }

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)

        async with self._pool.acquire() as connection:
            yield connection

    async def log(self, event_type: str, action: str, details: Optional[Dict[str, Any]] = None,
                 goal_id: Optional[int] = None, task_id: Optional[int] = None,
                 safety_level: str = 'auto', outcome: str = 'success') -> int:
        """
        Log an autonomous operation event to the audit trail.

        Args:
            event_type: Type of event (should be from self.event_types keys)
            action: Brief description of the action taken
            details: Optional detailed information about the event
            goal_id: Optional ID of associated goal
            task_id: Optional ID of associated task
            safety_level: Safety level of the operation ('auto', 'notify', 'review', 'forbidden')
            outcome: Outcome of the action ('success', 'failure', 'pending', 'blocked')

        Returns:
            int: ID of the created audit log entry

        Raises:
            ValueError: If required parameters are invalid
        """
        if not event_type or not action:
            raise ValueError("Event type and action are required")

        if safety_level not in ['auto', 'notify', 'review', 'forbidden']:
            raise ValueError("Safety level must be one of: auto, notify, review, forbidden")

        if outcome not in ['success', 'failure', 'pending', 'blocked']:
            raise ValueError("Outcome must be one of: success, failure, pending, blocked")

        details = details or {}

        # Convert details to JSON string if it's a dict
        if isinstance(details, dict):
            details = json.dumps(details)

        try:
            async with self.get_connection() as conn:
                audit_id = await conn.fetchval(
                    """
                    INSERT INTO autonomous_audit_log
                    (event_type, goal_id, task_id, action, details, safety_level, outcome)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """,
                    event_type, goal_id, task_id, action, details, safety_level, outcome
                )

                logger.debug(f"Logged audit event {event_type}: {action} (ID: {audit_id})")
                return audit_id

        except Exception as e:
            logger.error(f"Failed to log audit event {event_type}: {e}")
            # Don't raise - we don't want audit logging failures to break operations
            return -1

    async def log_goal_created(self, goal_id: int, goal_name: str, goal_type: str,
                              priority: int, metadata: Dict[str, Any] = None) -> int:
        """
        Log goal creation event.

        Args:
            goal_id: ID of the created goal
            goal_name: Name of the goal
            goal_type: Type of goal
            priority: Priority level
            metadata: Optional goal metadata

        Returns:
            int: Audit log entry ID
        """
        details = {
            'goal_name': goal_name,
            'goal_type': goal_type,
            'priority': priority,
            'metadata': metadata or {}
        }

        return await self.log(
            event_type='goal_created',
            action=f"Created goal '{goal_name}' (type: {goal_type})",
            details=details,
            goal_id=goal_id,
            outcome='success'
        )

    async def log_goal_completed(self, goal_id: int, goal_name: str,
                                final_progress: float, completion_time: datetime) -> int:
        """
        Log goal completion event.

        Args:
            goal_id: ID of the completed goal
            goal_name: Name of the goal
            final_progress: Final progress percentage
            completion_time: When the goal was completed

        Returns:
            int: Audit log entry ID
        """
        details = {
            'goal_name': goal_name,
            'final_progress': final_progress,
            'completion_time': completion_time.isoformat()
        }

        return await self.log(
            event_type='goal_completed',
            action=f"Completed goal '{goal_name}' at {final_progress}% progress",
            details=details,
            goal_id=goal_id,
            outcome='success'
        )

    async def log_task_execution(self, task_id: int, goal_id: int, task_name: str,
                               task_type: str, safety_level: str, outcome: str,
                               result: Optional[str] = None, error: Optional[str] = None,
                               execution_time_seconds: Optional[float] = None) -> int:
        """
        Log task execution event.

        Args:
            task_id: ID of the task
            goal_id: ID of the parent goal
            task_name: Name of the task
            task_type: Type of task
            safety_level: Safety level of the task
            outcome: Execution outcome ('success', 'failure')
            result: Optional execution result
            error: Optional error message
            execution_time_seconds: Optional execution duration

        Returns:
            int: Audit log entry ID
        """
        details = {
            'task_name': task_name,
            'task_type': task_type,
            'execution_time_seconds': execution_time_seconds,
            'result_length': len(result) if result else 0,
            'error': error
        }

        if result:
            # Truncate very long results for storage
            details['result_preview'] = result[:500] + '...' if len(result) > 500 else result

        event_type = 'task_completed' if outcome == 'success' else 'task_failed'
        action = f"Executed task '{task_name}' (type: {task_type})"

        if execution_time_seconds:
            action += f" in {execution_time_seconds:.2f}s"

        return await self.log(
            event_type=event_type,
            action=action,
            details=details,
            goal_id=goal_id,
            task_id=task_id,
            safety_level=safety_level,
            outcome=outcome
        )

    async def log_approval_request(self, task_id: int, goal_id: int, approval_id: int,
                                  action_description: str, risk_assessment: str,
                                  safety_level: str = 'review') -> int:
        """
        Log approval request event.

        Args:
            task_id: ID of the task requiring approval
            goal_id: ID of the parent goal
            approval_id: ID of the approval request
            action_description: Description of the action
            risk_assessment: Risk assessment details
            safety_level: Safety level (typically 'review')

        Returns:
            int: Audit log entry ID
        """
        details = {
            'approval_id': approval_id,
            'action_description': action_description,
            'risk_assessment': risk_assessment
        }

        return await self.log(
            event_type='approval_requested',
            action=f"Requested approval for task (approval ID: {approval_id})",
            details=details,
            goal_id=goal_id,
            task_id=task_id,
            safety_level=safety_level,
            outcome='pending'
        )

    async def log_approval_decision(self, approval_id: int, task_id: int, goal_id: int,
                                   decision: str, reviewed_by: str, reason: Optional[str] = None) -> int:
        """
        Log approval decision event.

        Args:
            approval_id: ID of the approval request
            task_id: ID of the associated task
            goal_id: ID of the parent goal
            decision: 'approved' or 'rejected'
            reviewed_by: Who made the decision
            reason: Optional reason for the decision

        Returns:
            int: Audit log entry ID
        """
        details = {
            'approval_id': approval_id,
            'decision': decision,
            'reviewed_by': reviewed_by,
            'reason': reason
        }

        event_type = 'approval_granted' if decision == 'approved' else 'approval_rejected'
        outcome = 'success' if decision == 'approved' else 'blocked'
        action = f"Approval {decision} by {reviewed_by} (approval ID: {approval_id})"

        return await self.log(
            event_type=event_type,
            action=action,
            details=details,
            goal_id=goal_id,
            task_id=task_id,
            safety_level='review',
            outcome=outcome
        )

    async def log_safety_event(self, event_type: str, action: str, details: Dict[str, Any],
                              safety_level: str = 'review', outcome: str = 'blocked') -> int:
        """
        Log safety-related events like rate limiting, kill switch activation, etc.

        Args:
            event_type: Type of safety event
            action: Description of the safety action
            details: Event details
            safety_level: Safety level (typically 'review' for safety events)
            outcome: Event outcome (typically 'blocked' for safety interventions)

        Returns:
            int: Audit log entry ID
        """
        return await self.log(
            event_type=event_type,
            action=action,
            details=details,
            safety_level=safety_level,
            outcome=outcome
        )

    async def get_recent_logs(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve recent audit log entries.

        Args:
            hours: Number of hours back to retrieve logs (default: 24)
            limit: Maximum number of entries to return (default: 100)

        Returns:
            List[Dict]: List of audit log entries
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT l.id, l.timestamp, l.event_type, l.goal_id, l.task_id,
                           l.action, l.details, l.safety_level, l.outcome,
                           g.name as goal_name, t.name as task_name
                    FROM autonomous_audit_log l
                    LEFT JOIN autonomous_goals g ON l.goal_id = g.id
                    LEFT JOIN autonomous_tasks t ON l.task_id = t.id
                    WHERE l.timestamp >= $1
                    ORDER BY l.timestamp DESC
                    LIMIT $2
                    """,
                    cutoff_time, limit
                )

                logs = [dict(row) for row in rows]
                logger.debug(f"Retrieved {len(logs)} recent log entries from last {hours} hours")
                return logs

        except Exception as e:
            logger.error(f"Failed to retrieve recent logs: {e}")
            raise

    async def get_logs_for_goal(self, goal_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all audit logs for a specific goal.

        Args:
            goal_id: ID of the goal
            limit: Maximum number of entries to return (default: 100)

        Returns:
            List[Dict]: List of audit log entries for the goal
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT l.id, l.timestamp, l.event_type, l.goal_id, l.task_id,
                           l.action, l.details, l.safety_level, l.outcome,
                           t.name as task_name
                    FROM autonomous_audit_log l
                    LEFT JOIN autonomous_tasks t ON l.task_id = t.id
                    WHERE l.goal_id = $1
                    ORDER BY l.timestamp DESC
                    LIMIT $2
                    """,
                    goal_id, limit
                )

                logs = [dict(row) for row in rows]
                logger.debug(f"Retrieved {len(logs)} log entries for goal {goal_id}")
                return logs

        except Exception as e:
            logger.error(f"Failed to retrieve logs for goal {goal_id}: {e}")
            raise

    async def get_logs_by_event_type(self, event_type: str, hours: int = 24,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs by event type.

        Args:
            event_type: Type of event to filter by
            hours: Number of hours back to search (default: 24)
            limit: Maximum number of entries to return (default: 100)

        Returns:
            List[Dict]: List of audit log entries matching the event type
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT l.id, l.timestamp, l.event_type, l.goal_id, l.task_id,
                           l.action, l.details, l.safety_level, l.outcome,
                           g.name as goal_name, t.name as task_name
                    FROM autonomous_audit_log l
                    LEFT JOIN autonomous_goals g ON l.goal_id = g.id
                    LEFT JOIN autonomous_tasks t ON l.task_id = t.id
                    WHERE l.event_type = $1 AND l.timestamp >= $2
                    ORDER BY l.timestamp DESC
                    LIMIT $3
                    """,
                    event_type, cutoff_time, limit
                )

                logs = [dict(row) for row in rows]
                logger.debug(f"Retrieved {len(logs)} log entries for event type '{event_type}'")
                return logs

        except Exception as e:
            logger.error(f"Failed to retrieve logs by event type '{event_type}': {e}")
            raise

    async def get_audit_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get audit statistics for the specified time period.

        Args:
            hours: Number of hours back to analyze (default: 24)

        Returns:
            Dict: Statistics about audit log activity
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            async with self.get_connection() as conn:
                # Get total event counts
                total_events = await conn.fetchval(
                    "SELECT COUNT(*) FROM autonomous_audit_log WHERE timestamp >= $1",
                    cutoff_time
                )

                # Get event type breakdown
                event_type_stats = await conn.fetch(
                    """
                    SELECT event_type, COUNT(*) as count
                    FROM autonomous_audit_log
                    WHERE timestamp >= $1
                    GROUP BY event_type
                    ORDER BY count DESC
                    """,
                    cutoff_time
                )

                # Get outcome breakdown
                outcome_stats = await conn.fetch(
                    """
                    SELECT outcome, COUNT(*) as count
                    FROM autonomous_audit_log
                    WHERE timestamp >= $1
                    GROUP BY outcome
                    ORDER BY count DESC
                    """,
                    cutoff_time
                )

                # Get safety level breakdown
                safety_level_stats = await conn.fetch(
                    """
                    SELECT safety_level, COUNT(*) as count
                    FROM autonomous_audit_log
                    WHERE timestamp >= $1
                    GROUP BY safety_level
                    ORDER BY count DESC
                    """,
                    cutoff_time
                )

                statistics = {
                    'time_period_hours': hours,
                    'total_events': total_events,
                    'event_types': [dict(row) for row in event_type_stats],
                    'outcomes': [dict(row) for row in outcome_stats],
                    'safety_levels': [dict(row) for row in safety_level_stats],
                    'generated_at': datetime.now().isoformat()
                }

                logger.debug(f"Generated audit statistics for {hours} hours: {total_events} total events")
                return statistics

        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            raise

    async def cleanup_old_logs(self, days_to_keep: int = 90):
        """
        Clean up old audit log entries.

        Args:
            days_to_keep: Number of days of logs to keep (default: 90)

        Returns:
            int: Number of records deleted
        """
        try:
            async with self.get_connection() as conn:
                result = await conn.execute(
                    f"""
                    DELETE FROM autonomous_audit_log
                    WHERE timestamp < NOW() - INTERVAL '{days_to_keep} days'
                    """
                )

                # Extract count from result
                deleted_count = int(result.split()[-1]) if result and result.split()[-1].isdigit() else 0

                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old audit log entries")

                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old audit logs: {e}")
            return 0

    async def close(self):
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            logger.debug("Closed AuditLogger database pool")