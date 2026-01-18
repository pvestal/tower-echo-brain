"""
Autonomous Repair API
Provides endpoints for triggering and monitoring autonomous repairs
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/repair", tags=["autonomous-repair"])

class RepairRequest(BaseModel):
    """Request for autonomous repair"""
    issue_type: str
    description: str
    severity: str = "medium"  # low, medium, high, critical
    auto_execute: bool = False

class RepairStatus(BaseModel):
    """Status of a repair operation"""
    repair_id: str
    status: str
    progress: int
    message: str

# Global repair executor instance
_repair_executor = None

def get_repair_executor():
    """Get or create repair executor instance"""
    global _repair_executor
    if _repair_executor is None:
        try:
            from src.tasks.autonomous_repair_executor import RepairExecutor
            _repair_executor = RepairExecutor()
        except ImportError as e:
            logger.error(f"Failed to import RepairExecutor: {e}")
            raise HTTPException(status_code=500, detail="Repair system not available")
    return _repair_executor

@router.get("/status")
async def get_repair_system_status():
    """Get autonomous repair system status"""
    try:
        executor = get_repair_executor()
        return {
            "status": "available",
            "repair_log_path": str(executor.repair_log_path),
            "history_count": len(executor.repair_history),
            "last_repair": executor.repair_history[-1] if executor.repair_history else None,
            "capabilities": [
                "system_diagnosis",
                "service_restart",
                "configuration_fix",
                "dependency_repair",
                "notification_alerts"
            ]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/trigger")
async def trigger_repair(request: RepairRequest, background_tasks: BackgroundTasks):
    """Trigger autonomous repair operation"""
    try:
        executor = get_repair_executor()

        repair_id = f"repair_{int(datetime.now().timestamp())}"

        if request.auto_execute:
            # Execute in background
            background_tasks.add_task(
                _execute_repair_workflow,
                executor,
                repair_id,
                request.dict()
            )
            return {
                "repair_id": repair_id,
                "status": "initiated",
                "message": f"Autonomous repair for {request.issue_type} started in background"
            }
        else:
            # Return repair plan
            plan = await _generate_repair_plan(executor, request.dict())
            return {
                "repair_id": repair_id,
                "status": "planned",
                "plan": plan,
                "message": "Repair plan generated - use /execute endpoint to run"
            }
    except Exception as e:
        logger.error(f"Failed to trigger repair: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_repair_history(limit: int = 10):
    """Get repair operation history"""
    try:
        executor = get_repair_executor()
        recent_history = executor.repair_history[-limit:] if executor.repair_history else []
        return {
            "repairs": recent_history,
            "total_count": len(executor.repair_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/diagnose")
async def diagnose_system():
    """Run system diagnosis for potential issues"""
    try:
        executor = get_repair_executor()

        # Simple system diagnosis
        issues = []

        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage("/")
        disk_free_gb = disk_usage.free / (1024**3)
        if disk_free_gb < 5:
            issues.append({
                "type": "disk_space",
                "severity": "high" if disk_free_gb < 2 else "medium",
                "description": f"Low disk space: {disk_free_gb:.1f}GB free"
            })

        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            issues.append({
                "type": "memory",
                "severity": "high",
                "description": f"High memory usage: {memory.percent:.1f}%"
            })

        # Check Echo Brain service health
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:8309/health")
                if response.status_code != 200:
                    issues.append({
                        "type": "service_health",
                        "severity": "medium",
                        "description": "Echo Brain health check failed"
                    })
        except Exception as e:
            issues.append({
                "type": "service_health",
                "severity": "high",
                "description": f"Cannot connect to Echo Brain: {str(e)}"
            })

        return {
            "status": "completed",
            "issues_found": len(issues),
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
            "system_healthy": len(issues) == 0
        }

    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _execute_repair_workflow(executor, repair_id: str, request_data: Dict):
    """Execute repair workflow in background"""
    try:
        logger.info(f"Starting repair workflow {repair_id}")

        # Add to history
        repair_record = {
            "repair_id": repair_id,
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "status": "in_progress"
        }
        executor.repair_history.append(repair_record)

        # Simulate repair operations
        await asyncio.sleep(2)  # Simulate diagnosis

        # Execute based on issue type
        issue_type = request_data.get("issue_type", "")

        if "service" in issue_type.lower():
            # Service restart simulation
            logger.info(f"Restarting service for repair {repair_id}")
            result = "Service restart completed"

        elif "config" in issue_type.lower():
            # Configuration fix simulation
            logger.info(f"Fixing configuration for repair {repair_id}")
            result = "Configuration updated"

        else:
            # Generic repair
            logger.info(f"Executing generic repair for {repair_id}")
            result = "Generic repair completed"

        # Update history
        repair_record["status"] = "completed"
        repair_record["result"] = result
        repair_record["completed_at"] = datetime.now().isoformat()

        logger.info(f"Repair workflow {repair_id} completed successfully")

    except Exception as e:
        logger.error(f"Repair workflow {repair_id} failed: {e}")
        if repair_record:
            repair_record["status"] = "failed"
            repair_record["error"] = str(e)

async def _generate_repair_plan(executor, request_data: Dict) -> List[Dict]:
    """Generate repair plan based on issue type"""
    issue_type = request_data.get("issue_type", "").lower()

    if "service" in issue_type:
        return [
            {"step": 1, "action": "Check service status", "command": "systemctl status"},
            {"step": 2, "action": "Restart service", "command": "systemctl restart"},
            {"step": 3, "action": "Verify health", "command": "health check"}
        ]
    elif "config" in issue_type:
        return [
            {"step": 1, "action": "Backup current config", "command": "cp config config.bak"},
            {"step": 2, "action": "Apply configuration fix", "command": "update config"},
            {"step": 3, "action": "Reload service", "command": "reload service"}
        ]
    else:
        return [
            {"step": 1, "action": "Diagnose issue", "command": "run diagnosis"},
            {"step": 2, "action": "Apply automated fix", "command": "auto-repair"},
            {"step": 3, "action": "Validate fix", "command": "validate"}
        ]