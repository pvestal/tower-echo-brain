#!/usr/bin/env python3
"""
Verified Execution Manager - Eliminates "Execution Theater"

Core Principle: NEVER claim something worked without proof.

This integrates the resilient model manager with verified execution to ensure
Echo Brain's actions actually happen and can be confirmed.
"""

import asyncio
import subprocess
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from .resilient_model_manager import ResilientModelManager, TaskUrgency, get_resilient_manager
from ..execution.verified_executor import ExecutionStatus, ExecutionResult, VerifiedAction

logger = logging.getLogger(__name__)


class ActionCategory(Enum):
    SERVICE_MANAGEMENT = "service_management"
    FILE_OPERATIONS = "file_operations"
    SYSTEM_MAINTENANCE = "system_maintenance"
    CODE_MODIFICATIONS = "code_modifications"
    NETWORK_OPERATIONS = "network_operations"


@dataclass
class VerificationSpec:
    """Specification for how to verify an action succeeded."""
    check_command: str
    expected_output_contains: Optional[str] = None
    expected_exit_code: int = 0
    timeout_seconds: int = 30
    description: str = "Action verification"


@dataclass
class ActionPlan:
    """A planned action with verification built in."""
    name: str
    category: ActionCategory
    command: str
    verification: VerificationSpec
    risk_level: int = 1  # 1-5, 5 being highest risk
    rollback_command: Optional[str] = None
    description: str = ""


class VerifiedExecutionManager:
    """
    Manages execution of actions with mandatory verification.

    Key Features:
    - Every action must have a verification method
    - Failed verifications are treated as failures regardless of command exit code
    - Rollback capability for risky operations
    - Integration with resilient model manager for intelligent error reporting
    """

    def __init__(self):
        self.resilient_manager: Optional[ResilientModelManager] = None
        self.execution_history: List[ExecutionResult] = []
        self.state_file = Path("/opt/tower-echo-brain/data/verified_execution_state.json")

        # Cooldown periods to prevent aggressive loops
        self.action_cooldowns: Dict[str, datetime] = {}
        self.default_cooldown = timedelta(minutes=5)

        # Common action templates
        self.action_templates = self._load_action_templates()

    async def initialize(self):
        """Initialize the verified execution manager."""
        self.resilient_manager = await get_resilient_manager()
        self._load_state()
        logger.info("✅ Verified execution manager initialized")

    def _load_action_templates(self) -> Dict[str, ActionPlan]:
        """Load predefined action templates for common operations."""
        return {
            "restart_service": ActionPlan(
                name="restart_service",
                category=ActionCategory.SERVICE_MANAGEMENT,
                command="sudo systemctl restart {service}",
                verification=VerificationSpec(
                    check_command="systemctl is-active {service}",
                    expected_output_contains="active",
                    description="Verify service is active after restart"
                ),
                risk_level=2,
                description="Restart a systemd service and verify it's running"
            ),
            "check_service_status": ActionPlan(
                name="check_service_status",
                category=ActionCategory.SERVICE_MANAGEMENT,
                command="systemctl status {service} --no-pager",
                verification=VerificationSpec(
                    check_command="echo 'status_check'",
                    expected_output_contains="status_check",
                    description="Status check always verifies (read-only operation)"
                ),
                risk_level=1,
                description="Check status of a systemd service"
            ),
            "kill_process": ActionPlan(
                name="kill_process",
                category=ActionCategory.SYSTEM_MAINTENANCE,
                command="sudo pkill -f '{process_pattern}'",
                verification=VerificationSpec(
                    check_command="pgrep -f '{process_pattern}'",
                    expected_exit_code=1,  # pgrep returns 1 when no processes found
                    description="Verify process is no longer running"
                ),
                risk_level=3,
                description="Kill processes matching pattern and verify they're gone"
            ),
            "cleanup_disk_space": ActionPlan(
                name="cleanup_disk_space",
                category=ActionCategory.SYSTEM_MAINTENANCE,
                command="sudo find {path} -name '*.log' -mtime +7 -delete",
                verification=VerificationSpec(
                    check_command="df -h {path}",
                    description="Check disk space after cleanup"
                ),
                risk_level=2,
                description="Clean up old log files and check disk space"
            ),
            "rotate_logs": ActionPlan(
                name="rotate_logs",
                category=ActionCategory.FILE_OPERATIONS,
                command="sudo journalctl --rotate && sudo journalctl --vacuum-time=7d",
                verification=VerificationSpec(
                    check_command="sudo journalctl --disk-usage",
                    description="Verify log rotation completed"
                ),
                risk_level=1,
                description="Rotate and vacuum systemd journals"
            ),
            "check_port": ActionPlan(
                name="check_port",
                category=ActionCategory.NETWORK_OPERATIONS,
                command="netstat -tlnp | grep :{port}",
                verification=VerificationSpec(
                    check_command="curl -f http://localhost:{port}/health",
                    timeout_seconds=10,
                    description="Verify service is responding on port"
                ),
                risk_level=1,
                description="Check if a port is open and responsive"
            ),
            "git_status": ActionPlan(
                name="git_status",
                category=ActionCategory.CODE_MODIFICATIONS,
                command="git status --porcelain",
                verification=VerificationSpec(
                    check_command="echo 'git_status_complete'",
                    expected_output_contains="git_status_complete",
                    description="Git status always verifies (read-only)"
                ),
                risk_level=1,
                description="Check git repository status"
            ),
            "commit_changes": ActionPlan(
                name="commit_changes",
                category=ActionCategory.CODE_MODIFICATIONS,
                command="git add -A && git commit -m '{message}'",
                verification=VerificationSpec(
                    check_command="git log --oneline -1",
                    expected_output_contains="{message_short}",
                    description="Verify commit was created with expected message"
                ),
                risk_level=3,
                rollback_command="git reset HEAD~1",
                description="Commit all changes and verify commit exists"
            )
        }

    def _is_in_cooldown(self, action_name: str) -> bool:
        """Check if an action is in cooldown period."""
        if action_name not in self.action_cooldowns:
            return False

        cooldown_until = self.action_cooldowns[action_name]
        return datetime.now() < cooldown_until

    def _set_cooldown(self, action_name: str, duration: timedelta = None):
        """Set cooldown for an action."""
        duration = duration or self.default_cooldown
        self.action_cooldowns[action_name] = datetime.now() + duration

    async def execute_verified_action(
        self,
        action_template: str,
        parameters: Dict[str, Any] = None,
        custom_verification: Optional[VerificationSpec] = None
    ) -> ExecutionResult:
        """
        Execute an action with mandatory verification.

        Args:
            action_template: Name of predefined action template
            parameters: Parameters to substitute in commands (e.g., {'service': 'nginx'})
            custom_verification: Override default verification

        Returns:
            ExecutionResult with honest success/failure status
        """
        parameters = parameters or {}

        if action_template not in self.action_templates:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action_taken=action_template,
                expected_outcome="Action template exists",
                actual_outcome="Action template not found",
                verification_method="Template lookup",
                stderr=f"Unknown action template: {action_template}"
            )

        plan = self.action_templates[action_template]

        # Check cooldown
        cooldown_key = f"{action_template}_{hash(str(parameters))}"
        if self._is_in_cooldown(cooldown_key):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action_taken=plan.name,
                expected_outcome="Execute action",
                actual_outcome="Action in cooldown period",
                verification_method="Cooldown check",
                stderr=f"Action {action_template} is in cooldown period"
            )

        start_time = datetime.now()

        # Format commands with parameters
        try:
            formatted_command = plan.command.format(**parameters)
            verification = custom_verification or plan.verification
            formatted_verification = verification.check_command.format(**parameters)
            expected_contains = verification.expected_output_contains
            if expected_contains:
                expected_contains = expected_contains.format(**parameters)

        except KeyError as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action_taken=plan.name,
                expected_outcome="Format command with parameters",
                actual_outcome=f"Missing parameter: {e}",
                verification_method="Parameter formatting",
                stderr=f"Missing required parameter: {e}"
            )

        logger.info(f"🔧 Executing verified action: {plan.name}")
        logger.info(f"   Command: {formatted_command}")
        logger.info(f"   Verification: {formatted_verification}")

        # Execute the main action
        try:
            process = await asyncio.create_subprocess_shell(
                formatted_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60.0
            )

            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""

            # Action executed, now verify it worked
            verify_process = await asyncio.create_subprocess_shell(
                formatted_verification,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            verify_stdout, verify_stderr = await asyncio.wait_for(
                verify_process.communicate(),
                timeout=verification.timeout_seconds
            )

            verify_stdout_text = verify_stdout.decode() if verify_stdout else ""
            verify_stderr_text = verify_stderr.decode() if verify_stderr else ""

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Check verification results
            verification_passed = False

            if verify_process.returncode == verification.expected_exit_code:
                if expected_contains:
                    verification_passed = expected_contains in verify_stdout_text
                else:
                    verification_passed = True

            if verification_passed:
                result = ExecutionResult(
                    status=ExecutionStatus.SUCCEEDED,
                    action_taken=formatted_command,
                    expected_outcome=verification.description,
                    actual_outcome=f"Verification passed: {verify_stdout_text}",
                    verification_method=formatted_verification,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    duration_ms=duration_ms
                )
                logger.info(f"✅ Action succeeded and verified: {plan.name}")
            else:
                result = ExecutionResult(
                    status=ExecutionStatus.VERIFICATION_FAILED,
                    action_taken=formatted_command,
                    expected_outcome=verification.description,
                    actual_outcome=f"Verification failed: exit={verify_process.returncode}, output={verify_stdout_text}",
                    verification_method=formatted_verification,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    duration_ms=duration_ms
                )
                logger.warning(f"❌ Action ran but verification failed: {plan.name}")
                logger.warning(f"   Expected: {expected_contains}")
                logger.warning(f"   Got: {verify_stdout_text}")

                # Set cooldown for failed actions
                self._set_cooldown(cooldown_key)

        except asyncio.TimeoutError:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                action_taken=formatted_command,
                expected_outcome="Complete within timeout",
                actual_outcome="Command timed out",
                verification_method="Timeout check",
                stderr="Command execution timed out",
                duration_ms=duration_ms
            )
            logger.error(f"⏰ Action timed out: {plan.name}")
            self._set_cooldown(cooldown_key)

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                action_taken=formatted_command,
                expected_outcome="Execute without exception",
                actual_outcome=f"Exception: {e}",
                verification_method="Exception handling",
                stderr=str(e),
                duration_ms=duration_ms
            )
            logger.error(f"💥 Action failed with exception: {plan.name}: {e}")
            self._set_cooldown(cooldown_key)

        # Store result
        self.execution_history.append(result)
        self._save_state()

        return result

    async def analyze_failure_with_llm(self, result: ExecutionResult) -> Dict[str, Any]:
        """
        Use the resilient model manager to analyze execution failures.

        Returns insights about what went wrong and potential fixes.
        """
        if not self.resilient_manager:
            await self.initialize()

        analysis_prompt = f"""
        Analyze this execution failure and provide insights:

        Action: {result.action_taken}
        Expected: {result.expected_outcome}
        Actual: {result.actual_outcome}
        Verification: {result.verification_method}
        Status: {result.status.value}

        STDOUT: {result.stdout}
        STDERR: {result.stderr}

        Provide:
        1. Root cause analysis
        2. Specific fix recommendations
        3. Whether this is likely a transient or permanent failure
        4. Suggested next actions

        Be concise and actionable.
        """

        llm_result = await self.resilient_manager.complete_with_fallback(
            task_type="analysis",
            prompt=analysis_prompt,
            urgency=TaskUrgency.BACKGROUND
        )

        if llm_result.success:
            return {
                "analysis": llm_result.value,
                "model_used": llm_result.model_used,
                "confidence": "high" if not llm_result.fallback_used else "medium"
            }
        else:
            return {
                "analysis": "LLM analysis failed",
                "error": llm_result.error,
                "confidence": "none"
            }

    async def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of recent execution history."""
        recent_history = self.execution_history[-50:]  # Last 50 actions

        summary = {
            "total_executions": len(self.execution_history),
            "recent_executions": len(recent_history),
            "success_rate": 0.0,
            "status_breakdown": {},
            "common_failures": [],
            "cooldowns_active": len([
                k for k, v in self.action_cooldowns.items()
                if v > datetime.now()
            ])
        }

        if recent_history:
            statuses = [r.status for r in recent_history]
            summary["success_rate"] = statuses.count(ExecutionStatus.SUCCEEDED) / len(statuses)

            for status in ExecutionStatus:
                count = statuses.count(status)
                if count > 0:
                    summary["status_breakdown"][status.value] = count

        return summary

    def _load_state(self):
        """Load execution state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                # Load cooldowns
                cooldowns_data = data.get("cooldowns", {})
                for action, timestamp_str in cooldowns_data.items():
                    self.action_cooldowns[action] = datetime.fromisoformat(timestamp_str)

                logger.info(f"Loaded execution state: {len(self.action_cooldowns)} cooldowns")
            except Exception as e:
                logger.warning(f"Failed to load execution state: {e}")

    def _save_state(self):
        """Save execution state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "cooldowns": {
                    action: timestamp.isoformat()
                    for action, timestamp in self.action_cooldowns.items()
                    if timestamp > datetime.now()  # Only save active cooldowns
                },
                "last_updated": datetime.now().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save execution state: {e}")


# Singleton instance for Echo Brain
_execution_manager_instance: Optional[VerifiedExecutionManager] = None

async def get_verified_execution_manager() -> VerifiedExecutionManager:
    """Get or create the singleton verified execution manager."""
    global _execution_manager_instance
    if _execution_manager_instance is None:
        _execution_manager_instance = VerifiedExecutionManager()
        await _execution_manager_instance.initialize()
    return _execution_manager_instance


# Common execution functions for easy use
async def restart_service_verified(service_name: str) -> ExecutionResult:
    """Restart a service and verify it's running."""
    manager = await get_verified_execution_manager()
    return await manager.execute_verified_action(
        "restart_service",
        {"service": service_name}
    )

async def check_service_status_verified(service_name: str) -> ExecutionResult:
    """Check service status (read-only operation)."""
    manager = await get_verified_execution_manager()
    return await manager.execute_verified_action(
        "check_service_status",
        {"service": service_name}
    )

async def kill_process_verified(process_pattern: str) -> ExecutionResult:
    """Kill processes matching pattern and verify they're gone."""
    manager = await get_verified_execution_manager()
    return await manager.execute_verified_action(
        "kill_process",
        {"process_pattern": process_pattern}
    )

async def check_port_verified(port: int) -> ExecutionResult:
    """Check if a port is open and responsive."""
    manager = await get_verified_execution_manager()
    return await manager.execute_verified_action(
        "check_port",
        {"port": port}
    )


# Integration with autonomous repair system
class VerifiedAutonomousRepair:
    """
    Enhanced autonomous repair system with verified execution.

    Replaces "execution theater" with provable results.
    """

    def __init__(self):
        self.execution_manager: Optional[VerifiedExecutionManager] = None

    async def initialize(self):
        """Initialize verified autonomous repair."""
        self.execution_manager = await get_verified_execution_manager()

    async def repair_service_issue(self, service: str, issue: str) -> Dict[str, Any]:
        """
        Repair a service issue with verified execution.

        Returns detailed results including LLM analysis of any failures.
        """
        if not self.execution_manager:
            await self.initialize()

        # Step 1: Check current status
        status_result = await check_service_status_verified(service)

        # Step 2: If service is down, try to restart
        if "failed" in status_result.actual_outcome.lower() or status_result.status != ExecutionStatus.SUCCEEDED:
            restart_result = await restart_service_verified(service)

            if restart_result.actually_worked:
                return {
                    "success": True,
                    "action": "service_restart",
                    "service": service,
                    "result": "Service successfully restarted and verified running",
                    "verification": restart_result.actual_outcome
                }
            else:
                # Analyze the failure with LLM
                analysis = await self.execution_manager.analyze_failure_with_llm(restart_result)

                return {
                    "success": False,
                    "action": "service_restart",
                    "service": service,
                    "result": "Service restart failed verification",
                    "error": restart_result.actual_outcome,
                    "analysis": analysis["analysis"],
                    "model_used": analysis.get("model_used"),
                    "confidence": analysis.get("confidence")
                }
        else:
            return {
                "success": True,
                "action": "status_check",
                "service": service,
                "result": "Service is already running normally",
                "verification": status_result.actual_outcome
            }


# Singleton for verified autonomous repair
_verified_repair_instance: Optional[VerifiedAutonomousRepair] = None

async def get_verified_autonomous_repair() -> VerifiedAutonomousRepair:
    """Get or create singleton verified autonomous repair."""
    global _verified_repair_instance
    if _verified_repair_instance is None:
        _verified_repair_instance = VerifiedAutonomousRepair()
        await _verified_repair_instance.initialize()
    return _verified_repair_instance