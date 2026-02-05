"""
Unified Health Service for Echo Brain
Consolidates all health checks into a single service
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import psutil
import httpx
import logging

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    name: str
    status: str  # "healthy", "degraded", "down"
    latency_ms: float
    last_check: datetime
    details: Dict
    error: Optional[str] = None

@dataclass
class SystemHealth:
    overall_status: str
    uptime_seconds: float
    services: List[ServiceHealth]
    resources: Dict
    endpoints: Dict
    timestamp: datetime

class HealthService:
    def __init__(self):
        self.checks = {}
        self.start_time = datetime.now()

    async def check_postgres(self) -> ServiceHealth:
        """Check PostgreSQL health and statistics"""
        start = datetime.now()
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host="localhost",
                database="echo_brain",
                user="patrick",
                password=os.getenv("DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
            )

            # Get multiple stats in one query
            row = await conn.fetchrow("""
                SELECT
                    (SELECT count(*) FROM conversations) as conversations,
                    (SELECT count(*) FROM claude_conversations) as claude_conversations,
                    (SELECT count(*) FROM facts) as facts,
                    (SELECT pg_database_size('echo_brain')) as db_size,
                    (SELECT count(*) FROM echo_conversations) as echo_conversations
            """)
            await conn.close()

            latency = (datetime.now() - start).total_seconds() * 1000

            return ServiceHealth(
                name="postgres",
                status="healthy",
                latency_ms=round(latency, 2),
                last_check=datetime.now(),
                details={
                    "conversations": row["conversations"] if row["conversations"] else 0,
                    "claude_conversations": row["claude_conversations"] if row["claude_conversations"] else 0,
                    "echo_conversations": row["echo_conversations"] if row["echo_conversations"] else 0,
                    "facts": row["facts"] if row["facts"] else 0,
                    "db_size_mb": round(row["db_size"] / 1024 / 1024, 2) if row["db_size"] else 0
                }
            )
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return ServiceHealth(
                name="postgres",
                status="down",
                latency_ms=0,
                last_check=datetime.now(),
                details={},
                error=str(e)[:200]
            )

    async def check_ollama(self) -> ServiceHealth:
        """Check Ollama service health and loaded models"""
        start = datetime.now()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get available models
                resp = await client.get("http://localhost:11434/api/tags")
                data = resp.json()

                # Get running models
                ps_resp = await client.get("http://localhost:11434/api/ps")
                running = ps_resp.json()

            latency = (datetime.now() - start).total_seconds() * 1000

            # Calculate GPU usage from running models
            gpu_vram_mb = 0
            loaded_models = []
            for model in running.get("models", []):
                loaded_models.append(model.get("name", "unknown"))
                # Size is in bytes, convert to MB
                if "size_vram" in model:
                    gpu_vram_mb += model["size_vram"] / (1024 * 1024)

            return ServiceHealth(
                name="ollama",
                status="healthy",
                latency_ms=round(latency, 2),
                last_check=datetime.now(),
                details={
                    "models_available": len(data.get("models", [])),
                    "models_loaded": loaded_models,
                    "gpu_vram_mb": round(gpu_vram_mb, 2)
                }
            )
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return ServiceHealth(
                name="ollama",
                status="down",
                latency_ms=0,
                last_check=datetime.now(),
                details={},
                error=str(e)[:200]
            )

    async def check_qdrant(self) -> ServiceHealth:
        """Check Qdrant vector database health"""
        start = datetime.now()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get collection info
                resp = await client.get("http://localhost:6333/collections/echo_memory")
                data = resp.json()

                # Get cluster info for status
                cluster_resp = await client.get("http://localhost:6333/cluster")
                cluster_data = cluster_resp.json()

            latency = (datetime.now() - start).total_seconds() * 1000

            result = data.get("result", {})
            return ServiceHealth(
                name="qdrant",
                status="healthy",
                latency_ms=round(latency, 2),
                last_check=datetime.now(),
                details={
                    "vectors_count": result.get("vectors_count", 0),
                    "points_count": result.get("points_count", 0),
                    "indexed_vectors": result.get("indexed_vectors_count", 0),
                    "segments": result.get("segments_count", 0),
                    "status": result.get("status", "unknown"),
                    "cluster_status": cluster_data.get("result", {}).get("status", "unknown")
                }
            )
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return ServiceHealth(
                name="qdrant",
                status="down",
                latency_ms=0,
                last_check=datetime.now(),
                details={},
                error=str(e)[:200]
            )

    async def check_mcp(self) -> ServiceHealth:
        """Check MCP server functionality"""
        start = datetime.now()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Test MCP health
                health_resp = await client.get("http://localhost:8309/mcp/health")

                # Test actual MCP functionality
                mcp_resp = await client.post(
                    "http://localhost:8309/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "params": {},
                        "id": 1
                    }
                )

            latency = (datetime.now() - start).total_seconds() * 1000

            tools = mcp_resp.json().get("tools", [])
            return ServiceHealth(
                name="mcp",
                status="healthy",
                latency_ms=round(latency, 2),
                last_check=datetime.now(),
                details={
                    "version": health_resp.json().get("version", "unknown"),
                    "tools_available": len(tools),
                    "tools": [t.get("name") for t in tools]
                }
            )
        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            return ServiceHealth(
                name="mcp",
                status="down",
                latency_ms=0,
                last_check=datetime.now(),
                details={},
                error=str(e)[:200]
            )

    async def check_comfyui(self) -> ServiceHealth:
        """Check ComfyUI service health"""
        start = datetime.now()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check system stats
                resp = await client.get("http://localhost:8188/system_stats")
                data = resp.json()

            latency = (datetime.now() - start).total_seconds() * 1000

            devices = data.get("devices", [])
            gpu_info = {}
            for device in devices:
                if device.get("type") == "cuda":
                    gpu_info = {
                        "name": device.get("name", "unknown"),
                        "vram_total_mb": round(device.get("vram_total", 0) / (1024*1024), 2),
                        "vram_free_mb": round(device.get("vram_free", 0) / (1024*1024), 2)
                    }
                    break

            return ServiceHealth(
                name="comfyui",
                status="healthy",
                latency_ms=round(latency, 2),
                last_check=datetime.now(),
                details={
                    "gpu": gpu_info,
                    "python_version": data.get("python_version", "unknown"),
                    "pytorch_version": data.get("pytorch_version", "unknown")
                }
            )
        except Exception as e:
            # ComfyUI being down is not critical
            return ServiceHealth(
                name="comfyui",
                status="down",
                latency_ms=0,
                last_check=datetime.now(),
                details={},
                error=str(e)[:200]
            )

    async def check_all(self) -> SystemHealth:
        """Run all health checks and compile system health"""
        # Run all checks in parallel
        services = await asyncio.gather(
            self.check_postgres(),
            self.check_ollama(),
            self.check_qdrant(),
            self.check_mcp(),
            self.check_comfyui(),
            return_exceptions=True  # Don't fail if one check fails
        )

        # Filter out any exceptions and convert to failed services
        healthy_services = []
        for service in services:
            if isinstance(service, Exception):
                logger.error(f"Health check failed with exception: {service}")
            elif isinstance(service, ServiceHealth):
                healthy_services.append(service)

        # Determine overall status
        statuses = [s.status for s in healthy_services]
        if all(s == "healthy" for s in statuses):
            overall = "healthy"
        elif any(s == "down" for s in statuses[:3]):  # First 3 are critical
            overall = "critical"
        else:
            overall = "degraded"

        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()

        return SystemHealth(
            overall_status=overall,
            uptime_seconds=uptime_seconds,
            services=healthy_services,
            resources=await self._get_resource_stats(),
            endpoints={
                "total": 48,
                "by_router": {
                    "system": 14,
                    "memory": 4,
                    "intelligence": 6,
                    "reasoning": 5,
                    "echo": 5,
                    "moltbook": 7,
                    "search": 2,
                    "self_test": 2,
                    "mcp": 2,
                    "conversations": 3
                }
            },
            timestamp=datetime.now()
        )

    async def _get_resource_stats(self) -> Dict:
        """Get system resource statistics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Network stats
            net_io = psutil.net_io_counters()

            stats = {
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "network_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                "network_recv_mb": round(net_io.bytes_recv / (1024**2), 2)
            }

            # Try to get GPU stats
            gpu_stats = await self._get_gpu_stats()
            if gpu_stats:
                stats["gpu"] = gpu_stats

            return stats
        except Exception as e:
            logger.error(f"Failed to get resource stats: {e}")
            return {}

    async def _get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU statistics using nvidia-smi or rocm-smi"""
        try:
            import subprocess
            import asyncio

            # Try NVIDIA first
            try:
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL
                )
                stdout, _ = await proc.communicate()

                if proc.returncode == 0 and stdout:
                    parts = stdout.decode().strip().split(", ")
                    if len(parts) >= 5:
                        return {
                            "type": "nvidia",
                            "name": parts[4],
                            "utilization_percent": int(parts[0]) if parts[0].isdigit() else 0,
                            "memory_used_mb": int(parts[1]) if parts[1].isdigit() else 0,
                            "memory_total_mb": int(parts[2]) if parts[2].isdigit() else 0,
                            "temperature_c": int(parts[3]) if parts[3].isdigit() else 0
                        }
            except:
                pass

            # Try AMD ROCm
            try:
                proc = await asyncio.create_subprocess_exec(
                    "rocm-smi",
                    "--showmeminfo", "vram",
                    "--showtemp",
                    "--showuse",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL
                )
                stdout, _ = await proc.communicate()

                if proc.returncode == 0 and stdout:
                    # Parse rocm-smi output (format varies)
                    output = stdout.decode()
                    # This would need proper parsing based on rocm-smi output format
                    return {
                        "type": "amd",
                        "name": "AMD GPU",
                        "utilization_percent": 0,
                        "memory_used_mb": 0,
                        "memory_total_mb": 0,
                        "temperature_c": 0
                    }
            except:
                pass

            return None
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            return None

# Global instance
health_service = HealthService()