#!/usr/bin/env python3
"""
Execution framework that verifies actions actually occurred.
Core principle: NEVER report success without confirmation.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from enum import Enum
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path

class ExecutionStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    VERIFICATION_FAILED = "verification_failed"  # Action ran but didn't achieve goal

@dataclass
class ExecutionResult:
    """Honest result of an execution attempt."""
    status: ExecutionStatus
    action_taken: str
    expected_outcome: str
    actual_outcome: str
    verification_method: str
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def actually_worked(self) -> bool:
        """Did the action achieve its intended effect?"""
        return self.status == ExecutionStatus.SUCCEEDED

@dataclass
class VerifiedAction:
    """
    An action paired with its verification.

    Both execute() and verify() must be defined.
    """
    name: str
    execute: Callable[[], subprocess.CompletedProcess]
    verify: Callable[[], bool]
    description: str

class VerifiedExecutor:
    """
    Executor that confirms actions achieved their intended effects.

    Usage:
        executor = VerifiedExecutor()
        result = await executor.run(
            action=VerifiedAction(
                name="restart_service",
                execute=lambda: subprocess.run(["docker", "compose", "restart", "myservice"]),
                verify=lambda: check_service_running("myservice"),
                description="Restart myservice container"
            )
        )

        if result.actually_worked:
            print("Service restarted successfully")
        else:
            print(f"Failed: {result.actual_outcome}")
    """

    def __init__(self, max_retries: int = 2, verify_delay_seconds: float = 2.0):
        self.max_retries = max_retries
        self.verify_delay = verify_delay_seconds
        self.execution_log: list[ExecutionResult] = []

    async def run(self, action: VerifiedAction) -> ExecutionResult:
        """
        Execute an action and verify it worked.

        Returns honest result including whether verification passed.
        """
        start_time = datetime.now()

        for attempt in range(self.max_retries + 1):
            try:
                # Execute the action
                proc_result = action.execute()

                # Wait for effects to propagate
                await asyncio.sleep(self.verify_delay)

                # Verify the outcome
                verification_passed = action.verify()

                duration = int((datetime.now() - start_time).total_seconds() * 1000)

                if verification_passed:
                    result = ExecutionResult(
                        status=ExecutionStatus.SUCCEEDED,
                        action_taken=action.name,
                        expected_outcome=action.description,
                        actual_outcome="Verified successful",
                        verification_method=f"{action.verify.__name__}() returned True",
                        stdout=proc_result.stdout.decode() if proc_result.stdout else "",
                        stderr=proc_result.stderr.decode() if proc_result.stderr else "",
                        duration_ms=duration
                    )
                else:
                    if attempt < self.max_retries:
                        continue  # Retry

                    result = ExecutionResult(
                        status=ExecutionStatus.VERIFICATION_FAILED,
                        action_taken=action.name,
                        expected_outcome=action.description,
                        actual_outcome="Action completed but verification failed",
                        verification_method=f"{action.verify.__name__}() returned False",
                        stdout=proc_result.stdout.decode() if proc_result.stdout else "",
                        stderr=proc_result.stderr.decode() if proc_result.stderr else "",
                        duration_ms=duration
                    )

                self.execution_log.append(result)
                return result

            except Exception as e:
                if attempt < self.max_retries:
                    continue

                result = ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    action_taken=action.name,
                    expected_outcome=action.description,
                    actual_outcome=f"Exception: {type(e).__name__}: {str(e)}",
                    verification_method="N/A - execution failed",
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                )
                self.execution_log.append(result)
                return result

        # Should not reach here, but safety fallback
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            action_taken=action.name,
            expected_outcome=action.description,
            actual_outcome="Exhausted retries without success",
            verification_method="N/A"
        )


async def test_verified_executor():
    """Verify executor catches fake successes."""

    executor = VerifiedExecutor(max_retries=0, verify_delay_seconds=0.1)

    # Test 1: Action that claims success but verification fails
    fake_action = VerifiedAction(
        name="fake_restart",
        execute=lambda: subprocess.CompletedProcess(args=[], returncode=0),
        verify=lambda: False,  # Verification fails
        description="Fake restart that doesn't work"
    )

    result = await executor.run(fake_action)
    assert result.status == ExecutionStatus.VERIFICATION_FAILED, \
        f"Should catch fake success, got {result.status}"
    assert not result.actually_worked, "Should report failure"

    # Test 2: Action that actually works
    real_action = VerifiedAction(
        name="create_file",
        execute=lambda: subprocess.run(["touch", "/tmp/test_verified_exec"], capture_output=True),
        verify=lambda: Path("/tmp/test_verified_exec").exists(),
        description="Create test file"
    )

    result = await executor.run(real_action)
    assert result.status == ExecutionStatus.SUCCEEDED, \
        f"Should succeed, got {result.status}"
    assert result.actually_worked, "Should report success"

    # Cleanup
    Path("/tmp/test_verified_exec").unlink(missing_ok=True)

    print("✅ Verified executor correctly distinguishes real from fake success")


if __name__ == "__main__":
    asyncio.run(test_verified_executor())