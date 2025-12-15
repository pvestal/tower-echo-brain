#!/usr/bin/env python3
"""
API routes for verified execution management.

Provides endpoints to execute actions with mandatory verification,
eliminating "execution theater" where Echo claims success without proof.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.managers.verified_execution_manager import (
    get_verified_execution_manager,
    get_verified_autonomous_repair,
    restart_service_verified,
    check_service_status_verified,
    kill_process_verified,
    check_port_verified,
    ExecutionStatus
)

logger = logging.getLogger(__name__)

# API Router
router = APIRouter(prefix="/api/echo/verified", tags=["verified-execution"])


# Request/Response Models
class ExecuteActionRequest(BaseModel):
    """Request to execute a verified action."""
    action_template: str
    parameters: Dict[str, Any] = {}
    description: Optional[str] = None


class ServiceActionRequest(BaseModel):
    """Request for service-specific actions."""
    service_name: str
    description: Optional[str] = None


class ProcessActionRequest(BaseModel):
    """Request for process management actions."""
    process_pattern: str
    description: Optional[str] = None


class PortCheckRequest(BaseModel):
    """Request to check port status."""
    port: int
    description: Optional[str] = None


class FailureAnalysisRequest(BaseModel):
    """Request to analyze an execution failure."""
    execution_id: Optional[str] = None
    action_taken: str
    expected_outcome: str
    actual_outcome: str
    stderr: Optional[str] = None


class RepairServiceRequest(BaseModel):
    """Request to repair a service issue."""
    service: str
    issue: str
    description: Optional[str] = None


# =============================================================================
# EXECUTION ENDPOINTS
# =============================================================================

@router.post("/execute")
async def execute_verified_action(request: ExecuteActionRequest):
    """
    Execute an action with mandatory verification.

    Returns success only if both execution AND verification succeed.
    """
    try:
        manager = await get_verified_execution_manager()

        result = await manager.execute_verified_action(
            action_template=request.action_template,
            parameters=request.parameters
        )

        return {
            "success": result.actually_worked,
            "status": result.status.value,
            "action_taken": result.action_taken,
            "expected_outcome": result.expected_outcome,
            "actual_outcome": result.actual_outcome,
            "verification_method": result.verification_method,
            "duration_ms": result.duration_ms,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": result.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to execute verified action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SERVICE MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/service/restart")
async def restart_service(request: ServiceActionRequest):
    """
    Restart a service and verify it's running.

    Returns success only if service actually restarts and passes health check.
    """
    try:
        result = await restart_service_verified(request.service_name)

        if result.actually_worked:
            return {
                "success": True,
                "message": f"Service {request.service_name} successfully restarted and verified",
                "verification": result.actual_outcome,
                "duration_ms": result.duration_ms
            }
        else:
            # Analyze failure with LLM
            manager = await get_verified_execution_manager()
            analysis = await manager.analyze_failure_with_llm(result)

            return {
                "success": False,
                "message": f"Service {request.service_name} restart failed verification",
                "error": result.actual_outcome,
                "analysis": analysis["analysis"],
                "model_used": analysis.get("model_used"),
                "stderr": result.stderr
            }

    except Exception as e:
        logger.error(f"Failed to restart service {request.service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/service/{service_name}/status")
async def check_service_status(service_name: str):
    """
    Check service status with verification.

    Returns actual systemctl output, not guesses.
    """
    try:
        result = await check_service_status_verified(service_name)

        return {
            "success": result.actually_worked,
            "service": service_name,
            "status_output": result.stdout,
            "verification": result.actual_outcome,
            "duration_ms": result.duration_ms,
            "timestamp": result.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to check service status for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PROCESS MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/process/kill")
async def kill_process(request: ProcessActionRequest):
    """
    Kill processes matching pattern and verify they're gone.

    Returns success only if processes are actually terminated.
    """
    try:
        result = await kill_process_verified(request.process_pattern)

        if result.actually_worked:
            return {
                "success": True,
                "message": f"Processes matching '{request.process_pattern}' successfully terminated",
                "verification": result.actual_outcome,
                "duration_ms": result.duration_ms
            }
        else:
            return {
                "success": False,
                "message": f"Failed to kill processes matching '{request.process_pattern}'",
                "error": result.actual_outcome,
                "stderr": result.stderr
            }

    except Exception as e:
        logger.error(f"Failed to kill process {request.process_pattern}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NETWORK VERIFICATION ENDPOINTS
# =============================================================================

@router.post("/port/check")
async def check_port(request: PortCheckRequest):
    """
    Check if a port is open and responsive.

    Verifies both that port is listening AND service responds to requests.
    """
    try:
        result = await check_port_verified(request.port)

        return {
            "success": result.actually_worked,
            "port": request.port,
            "listening": result.status == ExecutionStatus.SUCCEEDED,
            "verification": result.actual_outcome,
            "response_time_ms": result.duration_ms,
            "details": result.stdout
        }

    except Exception as e:
        logger.error(f"Failed to check port {request.port}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ANALYSIS & MONITORING ENDPOINTS
# =============================================================================

@router.post("/analyze/failure")
async def analyze_execution_failure(request: FailureAnalysisRequest):
    """
    Use LLM to analyze why an execution failed and suggest fixes.

    Provides intelligent insights into execution problems.
    """
    try:
        manager = await get_verified_execution_manager()

        # Create a mock execution result for analysis
        from src.managers.verified_execution_manager import ExecutionResult, ExecutionStatus
        from datetime import datetime

        mock_result = ExecutionResult(
            status=ExecutionStatus.FAILED,
            action_taken=request.action_taken,
            expected_outcome=request.expected_outcome,
            actual_outcome=request.actual_outcome,
            verification_method="External analysis request",
            stderr=request.stderr or "",
            timestamp=datetime.now()
        )

        analysis = await manager.analyze_failure_with_llm(mock_result)

        return {
            "success": True,
            "analysis": analysis["analysis"],
            "model_used": analysis.get("model_used"),
            "confidence": analysis.get("confidence"),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to analyze execution failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_execution_summary():
    """
    Get summary of recent execution history and performance.

    Shows success rates, common failures, and system health.
    """
    try:
        manager = await get_verified_execution_manager()
        summary = await manager.get_execution_summary()

        return summary

    except Exception as e:
        logger.error(f"Failed to get execution summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions")
async def list_available_actions():
    """
    List all available action templates.

    Shows what actions can be executed with their risk levels and descriptions.
    """
    try:
        manager = await get_verified_execution_manager()

        actions = []
        for name, template in manager.action_templates.items():
            actions.append({
                "name": name,
                "category": template.category.value,
                "risk_level": template.risk_level,
                "description": template.description,
                "command_template": template.command,
                "verification_description": template.verification.description,
                "has_rollback": template.rollback_command is not None
            })

        return {
            "success": True,
            "actions": actions,
            "total_actions": len(actions)
        }

    except Exception as e:
        logger.error(f"Failed to list available actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AUTONOMOUS REPAIR ENDPOINTS
# =============================================================================

@router.post("/repair/service")
async def repair_service_issue(request: RepairServiceRequest):
    """
    Autonomously repair a service issue with verified execution.

    Combines intelligent analysis with verified actions to fix problems.
    """
    try:
        repair_system = await get_verified_autonomous_repair()
        result = await repair_system.repair_service_issue(request.service, request.issue)

        return result

    except Exception as e:
        logger.error(f"Failed to repair service {request.service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def verified_execution_health():
    """Health check for verified execution system."""
    try:
        manager = await get_verified_execution_manager()
        summary = await manager.get_execution_summary()

        return {
            "status": "healthy",
            "service": "verified-execution",
            "recent_success_rate": summary.get("success_rate", 0.0),
            "total_executions": summary.get("total_executions", 0),
            "active_cooldowns": summary.get("cooldowns_active", 0),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Verified execution health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "verified-execution",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }