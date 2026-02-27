"""
Autonomous Repair API
Provides endpoints for triggering and monitoring autonomous repairs.
Wired to the existing SelfRepair system in src/autonomous/self_repair.py.
"""
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/repair", tags=["autonomous-repair"])

class RepairRequest(BaseModel):
    """Request for autonomous repair"""
    issue_type: str
    description: str
    severity: str = "medium"
    auto_execute: bool = False

class RepairStatus(BaseModel):
    """Status of a repair operation"""
    repair_id: str
    status: str
    progress: int
    message: str


def _get_repair():
    """Get the SelfRepair singleton"""
    try:
        from src.autonomous.self_repair import repair
        return repair
    except ImportError as e:
        logger.error(f"Failed to import SelfRepair: {e}")
        raise HTTPException(status_code=503, detail="Repair system not available")


@router.get("/status")
async def get_repair_system_status():
    """Get autonomous repair system status"""
    try:
        sr = _get_repair()
        from src.autonomous.self_repair import REPAIR_PLAYBOOK
        return {
            "status": "available",
            "auto_repair_enabled": sr.auto_repair_enabled,
            "max_risk_level": sr.max_risk_level,
            "history_count": len(sr.repair_history),
            "last_repair": sr.repair_history[-1] if sr.repair_history else None,
            "playbook_count": len(REPAIR_PLAYBOOK),
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/trigger")
async def trigger_repair(request: RepairRequest, background_tasks: BackgroundTasks):
    """Trigger autonomous repair operation"""
    try:
        sr = _get_repair()
        issue = {
            "type": request.issue_type,
            "description": request.description,
            "severity": request.severity,
        }
        if request.auto_execute:
            result = await sr.attempt_repair(issue)
            return {
                "status": "executed",
                "result": result,
            }
        else:
            action = sr.get_repair_action(issue)
            return {
                "status": "planned",
                "plan": action,
                "message": "Repair plan generated - set auto_execute=true to run",
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger repair: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_repair_history(limit: int = 10):
    """Get repair operation history"""
    try:
        sr = _get_repair()
        recent = sr.repair_history[-limit:] if sr.repair_history else []
        return {
            "repairs": recent,
            "total_count": len(sr.repair_history),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diagnose")
async def diagnose_system():
    """Run system diagnosis for potential issues"""
    try:
        from src.autonomous.self_diagnosis import diagnosis
        result = await diagnosis.run_full_diagnosis()
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="Diagnosis system not available")
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
