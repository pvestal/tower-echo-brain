"""
Echo Brain Self-Diagnosis
Runs checks and reports problems without human intervention.
"""
import asyncio
import httpx
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

CHECKS = {
    "services": [
        ("echo_brain", "http://localhost:8309/health"),
        ("anime_production", "http://localhost:8328/health"),
        ("ollama", "http://localhost:11434/api/tags"),
        ("comfyui", "http://localhost:8188/system_stats"),
        ("qdrant", "http://localhost:6333/collections"),
    ],
    "thresholds": {
        "gpu_memory_percent": 90,
        "cpu_percent": 85,
        "disk_percent": 90,
        "response_time_ms": 5000,
    }
}

class SelfDiagnosis:
    def __init__(self):
        self.last_check = None
        self.issues = []
        self.history = []

    async def check_service(self, name: str, url: str) -> Dict:
        """Check if a service is responding."""
        start = datetime.now()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                elapsed = (datetime.now() - start).total_seconds() * 1000
                return {
                    "service": name,
                    "status": "healthy" if response.status_code == 200 else "degraded",
                    "response_time_ms": elapsed,
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "service": name,
                "status": "DOWN",
                "error": str(e)
            }

    async def check_resources(self) -> Dict:
        """Check system resources."""
        gpu_info = {"status": "unknown"}
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                gpu_info = {
                    "memory_used_mb": int(parts[0]),
                    "memory_total_mb": int(parts[1]),
                    "utilization_percent": int(parts[2]),
                    "memory_percent": round(int(parts[0]) / int(parts[1]) * 100, 1)
                }
        except:
            pass

        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "gpu": gpu_info
        }

    async def check_database(self) -> Dict:
        """Check PostgreSQL connectivity and basic queries."""
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host="localhost",
                database="echo_brain",
                user="patrick",
                password="RP78eIrW7cI2jYvL5akt1yurE"
            )
            # Test query
            result = await conn.fetchval("SELECT COUNT(*) FROM task_results")
            await conn.close()
            return {"status": "healthy", "task_results_count": result}
        except Exception as e:
            return {"status": "DOWN", "error": str(e)}

    async def run_full_diagnosis(self) -> Dict:
        """Run all diagnostic checks."""
        self.last_check = datetime.now()
        self.issues = []

        results = {
            "timestamp": self.last_check.isoformat(),
            "services": {},
            "resources": {},
            "database": {},
            "issues": []
        }

        # Check services
        for name, url in CHECKS["services"]:
            check = await self.check_service(name, url)
            results["services"][name] = check
            if check.get("status") != "healthy":
                self.issues.append({
                    "severity": "critical" if check.get("status") == "DOWN" else "warning",
                    "component": name,
                    "message": f"Service {name} is {check.get('status')}: {check.get('error', '')}"
                })

        # Check resources
        resources = await self.check_resources()
        results["resources"] = resources

        thresholds = CHECKS["thresholds"]
        if resources["cpu_percent"] > thresholds["cpu_percent"]:
            self.issues.append({
                "severity": "warning",
                "component": "cpu",
                "message": f"CPU usage high: {resources['cpu_percent']}%"
            })
        if resources.get("gpu", {}).get("memory_percent", 0) > thresholds["gpu_memory_percent"]:
            self.issues.append({
                "severity": "warning",
                "component": "gpu",
                "message": f"GPU memory high: {resources['gpu']['memory_percent']}%"
            })

        # Check database
        results["database"] = await self.check_database()
        if results["database"].get("status") != "healthy":
            self.issues.append({
                "severity": "critical",
                "component": "database",
                "message": f"Database issue: {results['database'].get('error')}"
            })

        results["issues"] = self.issues
        results["health_score"] = self._calculate_health_score()

        # Store in history
        self.history.append(results)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return results

    def _calculate_health_score(self) -> int:
        """Calculate overall health score 0-100."""
        score = 100
        for issue in self.issues:
            if issue["severity"] == "critical":
                score -= 25
            elif issue["severity"] == "warning":
                score -= 10
        return max(0, score)

# Singleton instance
diagnosis = SelfDiagnosis()