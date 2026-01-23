"""
Safety Controller for Echo Brain Autonomous Operations

The SafetyController enforces safety limits, manages approval workflows,
and provides rate limiting and kill switch mechanisms for autonomous operations.
This is a critical safety component that prevents runaway autonomous behavior.
"""

import logging
import asyncio
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncpg
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)


class SafetyController:
    """
    Controls and monitors safety aspects of autonomous operations.

    Provides rate limiting, approval workflows, safety level checking,
    and emergency controls to ensure safe autonomous operation.
    """

    def __init__(self, max_ollama_calls_per_minute: int = 10):
        """
        Initialize the SafetyController with rate limiting configuration.

        Args:
            max_ollama_calls_per_minute: Maximum allowed Ollama API calls per minute (default: 10)
        """
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }
        self._pool = None

        # Rate limiting configuration
        self.max_ollama_calls_per_minute = max_ollama_calls_per_minute
        self.ollama_call_timestamps = []

        # Kill switch state
        self._kill_switch_active = False
        self._kill_switch_reason = None
        self._kill_switch_activated_at = None

        # Safety level definitions
        self.safety_levels = {
            'auto': 'Automatically execute without human intervention',
            'notify': 'Execute but notify humans of the action',
            'review': 'Require human approval before execution',
            'forbidden': 'Never allow execution under any circumstances'
        }

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)

        async with self._pool.acquire() as connection:
            yield connection

    def activate_kill_switch(self, reason: str) -> None:
        """
        Activate the emergency kill switch to halt all autonomous operations.

        Args:
            reason: Reason for activating the kill switch
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._kill_switch_activated_at = datetime.now()
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self) -> None:
        """Deactivate the kill switch to resume autonomous operations."""
        if self._kill_switch_active:
            logger.warning(f"Kill switch deactivated after {datetime.now() - self._kill_switch_activated_at}")
            self._kill_switch_active = False
            self._kill_switch_reason = None
            self._kill_switch_activated_at = None

    def is_kill_switch_active(self) -> Tuple[bool, Optional[str]]:
        """
        Check if the kill switch is currently active.

        Returns:
            Tuple[bool, Optional[str]]: (is_active, reason)
        """
        return self._kill_switch_active, self._kill_switch_reason

    def check_rate_limit(self) -> Tuple[bool, str]:
        """
        Check if the current Ollama API call rate is within limits.

        Returns:
            Tuple[bool, str]: (is_within_limit, message)
        """
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Remove timestamps older than 1 minute
        self.ollama_call_timestamps = [
            ts for ts in self.ollama_call_timestamps if ts > cutoff
        ]

        current_calls = len(self.ollama_call_timestamps)

        if current_calls >= self.max_ollama_calls_per_minute:
            return False, f"Rate limit exceeded: {current_calls}/{self.max_ollama_calls_per_minute} calls per minute"

        return True, f"Rate limit OK: {current_calls}/{self.max_ollama_calls_per_minute} calls per minute"

    def record_ollama_call(self) -> None:
        """Record an Ollama API call for rate limiting purposes."""
        self.ollama_call_timestamps.append(datetime.now())

    async def can_execute(self, task_id: int, safety_level: str, task_details: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a task can be executed based on safety constraints.

        Args:
            task_id: ID of the task to check
            safety_level: Safety level of the task ('auto', 'notify', 'review', 'forbidden')
            task_details: Details about the task for safety assessment

        Returns:
            Tuple[bool, str]: (can_execute, reason)
        """
        # Check kill switch first
        if self._kill_switch_active:
            return False, f"Kill switch active: {self._kill_switch_reason}"

        # Check safety level
        if safety_level == 'forbidden':
            return False, "Task is marked as forbidden and cannot be executed"

        if safety_level == 'review':
            # Check if approval exists
            approval_exists = await self._check_task_approval(task_id)
            if not approval_exists:
                return False, "Task requires human approval but none exists"

        # Check rate limits for tasks involving model calls
        if task_details.get('involves_model_call', False):
            within_limit, rate_message = self.check_rate_limit()
            if not within_limit:
                return False, f"Rate limit check failed: {rate_message}"

        # Additional safety checks can be added here
        # For example: resource usage, system load, time windows, etc.

        return True, f"Safety checks passed for {safety_level} level task"

    async def _check_task_approval(self, task_id: int) -> bool:
        """Check if a task has been approved for execution."""
        try:
            async with self.get_connection() as conn:
                approval = await conn.fetchrow(
                    """
                    SELECT status FROM autonomous_approvals
                    WHERE task_id = $1 AND status = 'approved'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    task_id
                )
                return approval is not None

        except Exception as e:
            logger.error(f"Failed to check task approval for {task_id}: {e}")
            return False

    async def request_approval(self, task_id: int, action_description: str,
                             risk_assessment: str, proposed_action: Dict[str, Any]) -> int:
        """
        Request human approval for a sensitive task.

        Args:
            task_id: ID of the task requiring approval
            action_description: Human-readable description of the action
            risk_assessment: Assessment of risks and safety considerations
            proposed_action: Detailed action parameters as JSON

        Returns:
            int: ID of the approval request

        Raises:
            ValueError: If required parameters are missing
        """
        if not action_description or not risk_assessment:
            raise ValueError("Action description and risk assessment are required")

        try:
            async with self.get_connection() as conn:
                # Update task status to require approval
                await conn.execute(
                    """
                    UPDATE autonomous_tasks
                    SET status = 'requires_approval'
                    WHERE id = $1
                    """,
                    task_id
                )

                # Create approval request
                approval_id = await conn.fetchval(
                    """
                    INSERT INTO autonomous_approvals (task_id, action_description, risk_assessment, proposed_action)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    task_id, action_description, risk_assessment, proposed_action
                )

                logger.info(f"Created approval request {approval_id} for task {task_id}")
                return approval_id

        except Exception as e:
            logger.error(f"Failed to request approval for task {task_id}: {e}")
            raise

    async def approve_task(self, approval_id: int, reviewed_by: str) -> bool:
        """
        Approve a task for execution.

        Args:
            approval_id: ID of the approval request
            reviewed_by: Identifier of the person approving the task

        Returns:
            bool: True if approval was successful, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Get the task ID
                task_id = await conn.fetchval(
                    """
                    SELECT task_id FROM autonomous_approvals
                    WHERE id = $1 AND status = 'pending'
                    """,
                    approval_id
                )

                if not task_id:
                    logger.warning(f"Approval {approval_id} not found or already processed")
                    return False

                # Update approval status
                await conn.execute(
                    """
                    UPDATE autonomous_approvals
                    SET status = 'approved', reviewed_at = NOW(), reviewed_by = $1
                    WHERE id = $2
                    """,
                    reviewed_by, approval_id
                )

                # Update task status back to pending
                await conn.execute(
                    """
                    UPDATE autonomous_tasks
                    SET status = 'pending'
                    WHERE id = $1
                    """,
                    task_id
                )

                logger.info(f"Approved task {task_id} via approval {approval_id} by {reviewed_by}")
                return True

        except Exception as e:
            logger.error(f"Failed to approve task via approval {approval_id}: {e}")
            raise

    async def reject_task(self, approval_id: int, reviewed_by: str, rejection_reason: str = None) -> bool:
        """
        Reject a task approval request.

        Args:
            approval_id: ID of the approval request
            reviewed_by: Identifier of the person rejecting the task
            rejection_reason: Optional reason for rejection

        Returns:
            bool: True if rejection was successful, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Get the task ID
                task_id = await conn.fetchval(
                    """
                    SELECT task_id FROM autonomous_approvals
                    WHERE id = $1 AND status = 'pending'
                    """,
                    approval_id
                )

                if not task_id:
                    logger.warning(f"Approval {approval_id} not found or already processed")
                    return False

                # Update approval status
                await conn.execute(
                    """
                    UPDATE autonomous_approvals
                    SET status = 'rejected', reviewed_at = NOW(), reviewed_by = $1
                    WHERE id = $2
                    """,
                    reviewed_by, approval_id
                )

                # Update task status to failed with rejection reason
                error_message = f"Rejected by {reviewed_by}"
                if rejection_reason:
                    error_message += f": {rejection_reason}"

                await conn.execute(
                    """
                    UPDATE autonomous_tasks
                    SET status = 'failed', error = $1
                    WHERE id = $2
                    """,
                    error_message, task_id
                )

                logger.info(f"Rejected task {task_id} via approval {approval_id} by {reviewed_by}")
                return True

        except Exception as e:
            logger.error(f"Failed to reject task via approval {approval_id}: {e}")
            raise

    async def get_pending_approvals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all pending approval requests.

        Args:
            limit: Maximum number of approvals to return, default 50

        Returns:
            List[Dict]: List of pending approval dictionaries
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT a.id, a.task_id, a.action_description, a.risk_assessment,
                           a.proposed_action, a.status, a.created_at,
                           t.name as task_name, t.task_type, t.priority,
                           g.name as goal_name
                    FROM autonomous_approvals a
                    JOIN autonomous_tasks t ON a.task_id = t.id
                    JOIN autonomous_goals g ON t.goal_id = g.id
                    WHERE a.status = 'pending'
                    ORDER BY t.priority ASC, a.created_at ASC
                    LIMIT $1
                    """,
                    limit
                )

                approvals = [dict(row) for row in rows]
                logger.debug(f"Retrieved {len(approvals)} pending approvals")
                return approvals

        except Exception as e:
            logger.error(f"Failed to retrieve pending approvals: {e}")
            raise

    async def get_safety_status(self) -> Dict[str, Any]:
        """
        Get comprehensive safety status information.

        Returns:
            Dict: Safety status including kill switch, rate limits, and pending approvals
        """
        try:
            kill_switch_active, kill_switch_reason = self.is_kill_switch_active()
            within_rate_limit, rate_message = self.check_rate_limit()

            async with self.get_connection() as conn:
                pending_approvals_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM autonomous_approvals WHERE status = 'pending'"
                )

                forbidden_tasks_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM autonomous_tasks WHERE safety_level = 'forbidden'"
                )

            return {
                'kill_switch': {
                    'active': kill_switch_active,
                    'reason': kill_switch_reason,
                    'activated_at': self._kill_switch_activated_at.isoformat() if self._kill_switch_activated_at else None
                },
                'rate_limiting': {
                    'within_limit': within_rate_limit,
                    'message': rate_message,
                    'calls_per_minute': len(self.ollama_call_timestamps),
                    'max_calls_per_minute': self.max_ollama_calls_per_minute
                },
                'approvals': {
                    'pending_count': pending_approvals_count
                },
                'safety_levels': {
                    'forbidden_tasks_count': forbidden_tasks_count
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get safety status: {e}")
            raise

    async def emergency_halt_all_tasks(self, reason: str, halted_by: str) -> int:
        """
        Emergency function to halt all pending and in-progress tasks.

        Args:
            reason: Reason for the emergency halt
            halted_by: Identifier of who initiated the halt

        Returns:
            int: Number of tasks halted
        """
        logger.critical(f"EMERGENCY HALT initiated by {halted_by}: {reason}")

        try:
            # Activate kill switch
            self.activate_kill_switch(f"Emergency halt by {halted_by}: {reason}")

            async with self.get_connection() as conn:
                # Halt all pending and in-progress tasks
                result = await conn.execute(
                    """
                    UPDATE autonomous_tasks
                    SET status = 'failed', error = $1
                    WHERE status IN ('pending', 'in_progress', 'requires_approval')
                    """,
                    f"Emergency halt by {halted_by}: {reason}"
                )

                # Extract count from result string (e.g., "UPDATE 5")
                halted_count = int(result.split()[-1]) if result and result.split()[-1].isdigit() else 0

                logger.critical(f"Emergency halt complete: {halted_count} tasks halted")
                return halted_count

        except Exception as e:
            logger.error(f"Failed to execute emergency halt: {e}")
            raise

    async def evaluate_task_safety(self, task: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict]]:
        """
        Evaluate if a task is safe to execute.

        Args:
            task: Task dictionary with details

        Returns:
            Tuple of (is_safe, safety_level, approval_data)
        """
        # Check kill switch first
        kill_active, kill_reason = self.is_kill_switch_active()
        if kill_active:
            return False, "forbidden", {"reason": f"Kill switch active: {kill_reason}"}

        # Check rate limits
        within_limit, rate_msg = self.check_rate_limit()
        if not within_limit:
            return False, "review", {"reason": rate_msg}

        # Determine safety level based on task type
        task_type = task.get('task_type', 'unknown')
        safety_level = task.get('safety_level', 'auto')

        # Override safety levels for certain task types
        dangerous_types = ['system_modification', 'data_deletion', 'security_change']
        if task_type in dangerous_types:
            safety_level = 'review'

        # Forbidden task types
        forbidden_types = ['credential_access', 'private_data_exposure']
        if task_type in forbidden_types:
            safety_level = 'forbidden'

        # System review tasks are auto-approved
        if task_type in ['system_review', 'monitoring', 'analysis', 'testing']:
            safety_level = 'auto'

        # Return evaluation
        if safety_level == 'forbidden':
            return False, safety_level, {"reason": f"Task type {task_type} is forbidden"}
        elif safety_level == 'review':
            approval_data = {
                "task_id": task.get('id'),
                "action_description": task.get('name', 'Unknown task'),
                "risk_assessment": f"Task type {task_type} requires human approval",
                "proposed_action": task
            }
            return False, safety_level, approval_data
        elif safety_level == 'notify':
            return True, safety_level, {"notification": f"Executing {task_type} task"}
        else:  # auto
            return True, safety_level, None

    async def cleanup(self):
        """Clean up old approval records."""
        try:
            async with self.get_connection() as conn:
                # Clean up old approved/rejected approvals older than 30 days
                result = await conn.execute("""
                    DELETE FROM autonomous_approvals
                    WHERE status IN ('approved', 'rejected')
                    AND created_at < NOW() - INTERVAL '30 days'
                """)
                logger.info(f"Cleaned up old approval records")
        except Exception as e:
            logger.error(f"Failed to cleanup approvals: {e}")

    async def close(self):
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            logger.debug("Closed SafetyController database pool")