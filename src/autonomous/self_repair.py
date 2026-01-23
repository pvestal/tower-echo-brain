"""
Echo Brain Self-Repair
Takes diagnosis results and attempts automated fixes.
"""
import asyncio
import subprocess
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RepairAction:
    """Represents a repair action that can be taken."""
    def __init__(self, name: str, command: str, verify_command: str, risk_level: str = "low"):
        self.name = name
        self.command = command
        self.verify_command = verify_command
        self.risk_level = risk_level  # low, medium, high

REPAIR_PLAYBOOK = {
    "echo_brain_DOWN": RepairAction(
        name="restart_echo_brain",
        command="sudo systemctl restart tower-echo-brain",
        verify_command="curl -s http://localhost:8309/health",
        risk_level="medium"
    ),
    "anime_production_DOWN": RepairAction(
        name="restart_anime_production",
        command="sudo systemctl restart tower-anime-production",
        verify_command="curl -s http://localhost:8328/health",
        risk_level="medium"
    ),
    "ollama_DOWN": RepairAction(
        name="restart_ollama",
        command="sudo systemctl restart ollama",
        verify_command="curl -s http://localhost:11434/api/tags",
        risk_level="low"
    ),
    "comfyui_DOWN": RepairAction(
        name="restart_comfyui",
        command="sudo systemctl restart comfyui",
        verify_command="curl -s http://localhost:8188/system_stats",
        risk_level="low"
    ),
    "gpu_memory_high": RepairAction(
        name="clear_gpu_cache",
        command="curl -X DELETE http://localhost:11434/api/generate",  # Unload models
        verify_command="nvidia-smi --query-gpu=memory.used --format=csv,noheader",
        risk_level="low"
    ),
    "database_DOWN": RepairAction(
        name="restart_postgresql",
        command="sudo systemctl restart postgresql",
        verify_command="pg_isready -h localhost",
        risk_level="high"
    ),
    "qdrant_DOWN": RepairAction(
        name="restart_qdrant",
        command="sudo systemctl restart qdrant",
        verify_command="curl -s http://localhost:6333/collections",
        risk_level="medium"
    ),
}

class SelfRepair:
    def __init__(self, auto_repair_enabled: bool = False, max_risk_level: str = "medium"):
        self.auto_repair_enabled = auto_repair_enabled
        self.max_risk_level = max_risk_level
        self.repair_history = []
        self.risk_levels = {"low": 1, "medium": 2, "high": 3}

    def get_repair_action(self, issue: Dict) -> Optional[RepairAction]:
        """Find appropriate repair action for an issue."""
        component = issue.get("component", "")
        message = issue.get("message", "")

        # Build lookup key based on component and status
        if "DOWN" in message:
            key = f"{component}_DOWN"
            if key in REPAIR_PLAYBOOK:
                return REPAIR_PLAYBOOK[key]

        # Check for high resource usage
        if "high" in message.lower():
            if "gpu" in component.lower():
                return REPAIR_PLAYBOOK.get("gpu_memory_high")

        return None

    async def attempt_repair(self, issue: Dict, force: bool = False) -> Dict:
        """Attempt to repair an issue."""
        action = self.get_repair_action(issue)

        if not action:
            return {
                "success": False,
                "reason": "no_repair_action_found",
                "issue": issue
            }

        # Check risk level
        if self.risk_levels.get(action.risk_level, 0) > self.risk_levels.get(self.max_risk_level, 2):
            if not force:
                return {
                    "success": False,
                    "reason": f"risk_too_high_{action.risk_level}",
                    "action": action.name,
                    "requires_approval": True
                }

        # Execute repair
        logger.info(f"Attempting repair: {action.name}")
        start = datetime.now()

        try:
            result = subprocess.run(
                action.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Wait for service to start
            await asyncio.sleep(3)

            # Verify repair worked
            verify_result = subprocess.run(
                action.verify_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )

            success = verify_result.returncode == 0

            repair_record = {
                "timestamp": datetime.now().isoformat(),
                "action": action.name,
                "issue": issue,
                "success": success,
                "duration_seconds": (datetime.now() - start).total_seconds(),
                "output": result.stdout[:500] if result.stdout else None,
                "error": result.stderr[:500] if result.stderr else None
            }

            self.repair_history.append(repair_record)

            return repair_record

        except Exception as e:
            return {
                "success": False,
                "action": action.name,
                "error": str(e)
            }

    async def auto_repair_from_diagnosis(self, diagnosis_result: Dict) -> Dict:
        """Automatically repair all issues from a diagnosis."""
        if not self.auto_repair_enabled:
            return {
                "auto_repair": "disabled",
                "issues_found": len(diagnosis_result.get("issues", [])),
                "message": "Enable auto_repair to fix automatically"
            }

        repairs = []
        for issue in diagnosis_result.get("issues", []):
            if issue.get("severity") == "critical":
                repair = await self.attempt_repair(issue)
                repairs.append(repair)

        return {
            "repairs_attempted": len(repairs),
            "successful": sum(1 for r in repairs if r.get("success")),
            "details": repairs
        }

    def enable_auto_repair(self, enabled: bool = True):
        """Enable or disable auto-repair."""
        self.auto_repair_enabled = enabled
        logger.info(f"Auto-repair {'enabled' if enabled else 'disabled'}")

    def set_max_risk_level(self, level: str):
        """Set maximum risk level for auto-repairs."""
        if level in self.risk_levels:
            self.max_risk_level = level
            logger.info(f"Max risk level set to: {level}")

# Singleton
repair = SelfRepair(auto_repair_enabled=False)  # Start disabled for safety