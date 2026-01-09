#!/usr/bin/env python3
"""
Echo Brain Self-Awareness System
Real-time monitoring and diagnosis of Echo's own capabilities
"""

import asyncio
import sqlite3
import subprocess
import json
import os
import requests
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EchoSelfAwareness:
    """Echo's self-awareness and diagnostic system"""

    def __init__(self):
        self.base_path = Path("/opt/tower-echo-brain")
        self.services = {
            "vision": {"llava_model": "llava:7b", "photos_scanned": 0},
            "frontend": {"vue_interface": "/echo-brain/", "static_files": "/static/dist/"},
            "comfyui": {"endpoint": "http://localhost:8188", "workflows": []},
            "autonomous": {"tasks_active": 0, "background_worker": False},
            "memory": {"database_size": "0MB", "conversations": 0}
        }

    async def run_comprehensive_diagnosis(self):
        """Run complete self-diagnosis of all capabilities"""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "capabilities": {}
        }

        # Check vision capabilities
        diagnosis["capabilities"]["vision"] = await self._check_vision_system()

        # Check frontend interfaces
        diagnosis["capabilities"]["frontend"] = await self._check_frontend_system()

        # Check ComfyUI integration
        diagnosis["capabilities"]["comfyui"] = await self._check_comfyui_system()

        # Check autonomous systems
        diagnosis["capabilities"]["autonomous"] = await self._check_autonomous_system()

        # Check memory and learning
        diagnosis["capabilities"]["memory"] = await self._check_memory_system()

        # Determine overall health
        failed_systems = [k for k, v in diagnosis["capabilities"].items() if not v.get("healthy", False)]
        if failed_systems:
            diagnosis["overall_status"] = "degraded"
            diagnosis["failed_systems"] = failed_systems

        return diagnosis

    async def _check_vision_system(self):
        """Check LLaVA vision capabilities"""
        try:
            # Check if LLaVA model is available
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            llava_available = "llava:7b" in result.stdout

            # Check photo database
            photos_db = "/opt/tower-echo-brain/photos.db"
            photo_count = 0
            if os.path.exists(photos_db):
                conn = sqlite3.connect(photos_db)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM photos")
                photo_count = cursor.fetchone()[0]
                conn.close()

            # Count total photos available
            total_photos = 0
            google_photos_path = "/mnt/10TB2/Google_Takeout_2025/Takeout/Google Photos/"
            if os.path.exists(google_photos_path):
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    total_photos += len(list(Path(google_photos_path).rglob(ext)))

            return {
                "healthy": llava_available and photo_count > 0,
                "llava_model": "available" if llava_available else "missing",
                "photos_scanned": photo_count,
                "total_photos": total_photos,
                "scan_progress": f"{photo_count}/{total_photos}" if total_photos > 0 else "unknown",
                "vision_modules": ["EchoVision", "VisionQualityChecker"] if llava_available else []
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_frontend_system(self):
        """Check Vue.js and web interfaces"""
        try:
            vue_dist = self.base_path / "static" / "dist" / "index.html"
            vue_source = self.base_path / "frontend" / "src" / "App.vue"

            # Check if built files exist
            built_interface = vue_dist.exists()
            source_interface = vue_source.exists()

            # Check if interface is served on Echo's port
            interface_live = False
            try:
                response = requests.get("http://127.0.0.1:8309/echo-brain/", timeout=3)
                interface_live = response.status_code == 200
            except Exception as e:
                logger.warning(f"Frontend check failed: {e}")
                interface_live = False

            return {
                "healthy": built_interface and interface_live,
                "vue_built": built_interface,
                "vue_source": source_interface,
                "interface_live": interface_live,
                "endpoints": ["/echo-brain/", "/static/dist/"],
                "features": ["Chat", "Voice", "Metrics", "Financial"] if built_interface else []
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_comfyui_system(self):
        """Check ComfyUI integration and workflows"""
        try:
            # Check ComfyUI API
            comfyui_available = False
            try:
                response = requests.get("http://localhost:8188/system_stats", timeout=5)
                comfyui_available = response.status_code == 200
            except:
                pass

            # Check workflow files
            workflows = []
            workflow_dirs = [
                "/mnt/1TB-storage/ComfyUI/user/default/workflows/",
                "/mnt/1TB-storage/ComfyUI/custom_nodes/"
            ]

            for workflow_dir in workflow_dirs:
                if os.path.exists(workflow_dir):
                    workflows.extend([f.name for f in Path(workflow_dir).glob("*.json")])

            # Check recent outputs
            output_dir = "/mnt/1TB-storage/ComfyUI/output/"
            recent_outputs = []
            if os.path.exists(output_dir):
                output_files = sorted(Path(output_dir).glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
                recent_outputs = [f.name for f in output_files[:5]]

            return {
                "healthy": comfyui_available,
                "api_available": comfyui_available,
                "workflows_found": len(workflows),
                "recent_outputs": recent_outputs,
                "capabilities": ["AnimateDiff", "SDXL", "Video Generation"] if comfyui_available else []
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_autonomous_system(self):
        """Check autonomous task system and background workers"""
        try:
            # Check if background worker is running (part of main Echo process)
            result = subprocess.run(["pgrep", "-f", "tower-echo-brain.*src.main"], capture_output=True)
            worker_running = result.returncode == 0

            # Check task queue status - look for Redis connection
            task_status = {"redis_available": False}
            try:
                # Check if Redis is running (task queue backend)
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0)
                r.ping()
                task_status = {"redis_available": True, "task_queue_ready": True}
            except:
                task_status = {"redis_available": False, "task_queue_ready": False}

            # Check for autonomous behaviors
            autonomous_files = [
                "src/tasks/autonomous_behaviors.py",
                "src/tasks/background_worker.py",
                "src/tasks/task_queue.py"
            ]

            behaviors_available = all((self.base_path / f).exists() for f in autonomous_files)

            return {
                "healthy": behaviors_available and (worker_running or task_status),
                "background_worker": worker_running,
                "task_queue": task_status,
                "autonomous_modules": autonomous_files if behaviors_available else [],
                "capabilities": ["Infrastructure Repair", "Code Refactoring", "Proactive Monitoring"]
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_memory_system(self):
        """Check memory, database, and learning systems"""
        try:
            # Check main database
            db_size = "0MB"
            conversation_count = 0

            try:
                response = requests.get("http://127.0.0.1:8309/api/echo/db/stats", timeout=3)
                if response.status_code == 200:
                    db_stats = response.json()
                    db_size = db_stats.get("echo_brain", "0MB")
            except Exception as e:
                logger.warning(f"Failed to get db stats: {e}")
                pass

            # Check conversation history
            try:
                response = requests.get("http://127.0.0.1:8309/api/echo/status", timeout=3)
                if response.status_code == 200:
                    status = response.json()
                    conversation_count = len(status.get("recent_messages", []))
            except Exception as e:
                logger.warning(f"Failed to get status: {e}")
                pass

            # Check learning components
            learning_files = [
                "src/tasks/persona_trainer.py",
                "echo_media_scanner.py",
                "anime_memory_integration.py"
            ]

            learning_available = sum(1 for f in learning_files if (self.base_path / f).exists())

            return {
                "healthy": db_size not in ["0MB", ""] and conversation_count > 0,
                "database_size": db_size,
                "recent_conversations": conversation_count,
                "learning_modules": f"{learning_available}/{len(learning_files)}",
                "capabilities": ["Persona Training", "Media Learning", "Anime Memory"] if learning_available > 0 else []
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def create_improvement_plan(self, diagnosis):
        """Create improvement plan based on diagnosis"""
        improvements = []

        for system, status in diagnosis["capabilities"].items():
            if not status.get("healthy", False):
                if system == "vision":
                    improvements.append("Install or update LLaVA model: ollama pull llava:7b")
                    improvements.append("Resume photo scanning: python3 echo_media_scanner.py")
                elif system == "frontend":
                    improvements.append("Rebuild Vue interface: cd frontend && npm run build")
                    improvements.append("Restart Echo Brain service")
                elif system == "comfyui":
                    improvements.append("Check ComfyUI service status")
                    improvements.append("Verify NVIDIA GPU availability")
                elif system == "autonomous":
                    improvements.append("Restart background worker")
                    improvements.append("Check task queue Redis connection")
                elif system == "memory":
                    improvements.append("Check database connections")
                    improvements.append("Initialize conversation persistence")

        return improvements

# Create API endpoint for self-diagnosis
async def get_self_diagnosis():
    """API endpoint for Echo's self-diagnosis"""
    awareness = EchoSelfAwareness()
    diagnosis = await awareness.run_comprehensive_diagnosis()

    if diagnosis["overall_status"] != "healthy":
        improvements = await awareness.create_improvement_plan(diagnosis)
        diagnosis["improvement_plan"] = improvements

    return diagnosis

if __name__ == "__main__":
    async def test_awareness():
        awareness = EchoSelfAwareness()
        diagnosis = await awareness.run_comprehensive_diagnosis()
        print(json.dumps(diagnosis, indent=2))

    asyncio.run(test_awareness())