#!/usr/bin/env python3
"""
Echo Brain Self-Diagnostics Module
Provides comprehensive self-diagnosis capabilities for Echo Brain
"""

import os
import sys
import json
import psutil
import asyncio
import asyncpg
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE'
}

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"


class EchoDiagnostics:
    """Echo Brain self-diagnostics system"""

    def __init__(self):
        self.start_time = datetime.now()
        self.diagnostics_cache = {}
        self.last_check = None

    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and health"""
        try:
            conn = await asyncpg.connect(**DB_CONFIG)

            # Check connection
            version = await conn.fetchval("SELECT version()")

            # Check key tables
            tables_check = await conn.fetch("""
                SELECT tablename, n_live_tup as row_count
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
                ORDER BY n_live_tup DESC
                LIMIT 10
            """)

            # Check anime schema
            anime_check = await conn.fetchval("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'anime'
            """)

            await conn.close()

            return {
                "status": "healthy",
                "database": "echo_brain",
                "connected": True,
                "version": version.split(',')[0] if version else "Unknown",
                "tables": len(tables_check),
                "total_rows": sum(t['row_count'] for t in tables_check),
                "anime_schema_tables": anime_check,
                "top_tables": [
                    {"name": t['tablename'], "rows": t['row_count']}
                    for t in tables_check[:5]
                ]
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "database": "echo_brain",
                "connected": False,
                "error": str(e)
            }

    async def check_vector_store_health(self) -> Dict[str, Any]:
        """Check Qdrant vector database health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get collections info
                response = await client.get(f"{QDRANT_URL}/collections")

                if response.status_code == 200:
                    data = response.json()
                    collections = data.get('result', {}).get('collections', [])

                    # Get detailed info for echo_memory
                    echo_memory = await client.get(f"{QDRANT_URL}/collections/echo_memory")

                    if echo_memory.status_code == 200:
                        memory_data = echo_memory.json()['result']
                        vector_count = memory_data.get('points_count', 0)
                        dimensions = memory_data.get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
                    else:
                        vector_count = 0
                        dimensions = 0

                    return {
                        "status": "healthy",
                        "service": "Qdrant",
                        "connected": True,
                        "collections": len(collections),
                        "collection_names": [c['name'] for c in collections],
                        "echo_memory_vectors": vector_count,
                        "vector_dimensions": dimensions,
                        "url": QDRANT_URL
                    }
                else:
                    return {
                        "status": "degraded",
                        "service": "Qdrant",
                        "connected": True,
                        "error": f"HTTP {response.status_code}"
                    }

        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "Qdrant",
                "connected": False,
                "error": str(e)
            }

    async def check_ollama_health(self) -> Dict[str, Any]:
        """Check Ollama service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{OLLAMA_URL}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    models = data.get('models', [])

                    return {
                        "status": "healthy",
                        "service": "Ollama",
                        "connected": True,
                        "models_available": len(models),
                        "model_names": [m['name'] for m in models[:5]],
                        "url": OLLAMA_URL
                    }
                else:
                    return {
                        "status": "degraded",
                        "service": "Ollama",
                        "connected": True,
                        "error": f"HTTP {response.status_code}"
                    }

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "Ollama",
                "connected": False,
                "error": str(e)
            }

    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage('/')

            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()

            return {
                "status": "healthy",
                "cpu": {
                    "usage_percent": cpu_percent,
                    "cores": cpu_count,
                    "status": "normal" if cpu_percent < 80 else "high"
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory.percent,
                    "status": "normal" if memory.percent < 80 else "high"
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": disk.percent,
                    "status": "normal" if disk.percent < 80 else "high"
                },
                "process": {
                    "memory_mb": round(process_memory.rss / (1024**2), 2),
                    "uptime_hours": round((datetime.now() - self.start_time).total_seconds() / 3600, 2)
                }
            }

        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {
                "status": "unknown",
                "error": str(e)
            }

    async def check_module_health(self) -> Dict[str, Any]:
        """Check health of Echo Brain modules"""
        modules = {}

        # Check anime module
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    "http://localhost:8309/api/echo/anime/scene/plan",
                    json={"session_id": "health_check", "scene_description": "test"}
                )
                modules["anime"] = {
                    "status": "operational" if response.status_code in [200, 422] else "degraded",
                    "endpoint": "/api/echo/anime/*",
                    "response_code": response.status_code
                }
        except Exception as e:
            modules["anime"] = {"status": "offline", "error": str(e)}

        # Check main Echo API
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:8309/health")
                if response.status_code == 200:
                    health_data = response.json()
                    modules["core"] = {
                        "status": "operational",
                        "agents": health_data.get("agents", []),
                        "service": health_data.get("service", "unknown")
                    }
                else:
                    modules["core"] = {"status": "degraded", "response_code": response.status_code}
        except Exception as e:
            modules["core"] = {"status": "offline", "error": str(e)}

        return modules

    async def perform_comprehensive_diagnosis(self) -> Dict[str, Any]:
        """Perform comprehensive system diagnosis"""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "system_name": "Echo Brain",
            "version": "2.0",
            "diagnosis_type": "comprehensive",
            "components": {}
        }

        # Run all health checks in parallel
        tasks = [
            ("database", self.check_database_health()),
            ("vector_store", self.check_vector_store_health()),
            ("ollama", self.check_ollama_health()),
            ("system_resources", self.check_system_resources()),
            ("modules", self.check_module_health())
        ]

        for name, task in tasks:
            try:
                diagnosis["components"][name] = await task
            except Exception as e:
                diagnosis["components"][name] = {
                    "status": "error",
                    "error": str(e)
                }

        # Calculate overall health
        statuses = []
        for component in diagnosis["components"].values():
            if isinstance(component, dict) and "status" in component:
                statuses.append(component["status"])

        if all(s == "healthy" or s == "operational" for s in statuses):
            diagnosis["overall_status"] = "healthy"
            diagnosis["summary"] = "All systems operational"
        elif any(s == "unhealthy" or s == "offline" for s in statuses):
            diagnosis["overall_status"] = "unhealthy"
            diagnosis["summary"] = "Critical issues detected"
        else:
            diagnosis["overall_status"] = "degraded"
            diagnosis["summary"] = "Some components experiencing issues"

        # Add recommendations
        diagnosis["recommendations"] = self.generate_recommendations(diagnosis)

        # Cache the diagnosis
        self.diagnostics_cache = diagnosis
        self.last_check = datetime.now()

        return diagnosis

    def generate_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnosis"""
        recommendations = []

        # Check database
        if diagnosis["components"].get("database", {}).get("status") == "unhealthy":
            recommendations.append("Database connection failed - check PostgreSQL service and credentials")

        # Check vector store
        vector_status = diagnosis["components"].get("vector_store", {})
        if vector_status.get("status") == "unhealthy":
            recommendations.append("Qdrant vector database offline - restart the service")
        elif vector_status.get("echo_memory_vectors", 0) < 1000:
            recommendations.append("Low vector count - consider running memory ingestion")

        # Check Ollama
        if diagnosis["components"].get("ollama", {}).get("status") == "unhealthy":
            recommendations.append("Ollama service offline - AI features limited")

        # Check resources
        resources = diagnosis["components"].get("system_resources", {})
        if resources.get("memory", {}).get("status") == "high":
            recommendations.append("High memory usage - consider restarting services")
        if resources.get("disk", {}).get("status") == "high":
            recommendations.append("High disk usage - clean up old logs and temporary files")

        if not recommendations:
            recommendations.append("All systems operating normally - no action needed")

        return recommendations

    async def format_telegram_diagnosis(self, diagnosis: Dict[str, Any]) -> str:
        """Format diagnosis for Telegram response"""
        lines = []
        lines.append("🔍 **Echo Brain Self-Diagnosis Report**")
        lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Overall status
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌",
            "unknown": "❓"
        }

        overall = diagnosis.get("overall_status", "unknown")
        lines.append(f"{status_emoji[overall]} **Overall Status:** {overall.upper()}")
        lines.append(f"📝 {diagnosis.get('summary', 'No summary available')}")
        lines.append("")

        # Component status
        lines.append("**Component Status:**")

        for name, component in diagnosis.get("components", {}).items():
            if isinstance(component, dict):
                status = component.get("status", "unknown")
                emoji = status_emoji.get(status, "❓")

                if name == "database":
                    lines.append(f"{emoji} Database: {status}")
                    if status == "healthy":
                        lines.append(f"  • Tables: {component.get('tables', 0)}")
                        lines.append(f"  • Total rows: {component.get('total_rows', 0):,}")

                elif name == "vector_store":
                    lines.append(f"{emoji} Vector Store: {status}")
                    if status == "healthy":
                        lines.append(f"  • Collections: {component.get('collections', 0)}")
                        lines.append(f"  • Echo memories: {component.get('echo_memory_vectors', 0):,}")

                elif name == "ollama":
                    lines.append(f"{emoji} AI Models: {status}")
                    if status == "healthy":
                        lines.append(f"  • Models: {component.get('models_available', 0)}")

                elif name == "system_resources":
                    lines.append(f"{emoji} System Resources:")
                    if "cpu" in component:
                        lines.append(f"  • CPU: {component['cpu']['usage_percent']}%")
                    if "memory" in component:
                        lines.append(f"  • Memory: {component['memory']['percent_used']:.1f}%")
                    if "disk" in component:
                        lines.append(f"  • Disk: {component['disk']['percent_used']:.1f}%")

                elif name == "modules":
                    lines.append(f"{emoji} Modules:")
                    for module_name, module_info in component.items():
                        if isinstance(module_info, dict):
                            m_status = module_info.get("status", "unknown")
                            m_emoji = "✅" if m_status == "operational" else "❌"
                            lines.append(f"  {m_emoji} {module_name}: {m_status}")

        lines.append("")

        # Recommendations
        recommendations = diagnosis.get("recommendations", [])
        if recommendations:
            lines.append("**Recommendations:**")
            for rec in recommendations[:3]:  # Limit to 3 for Telegram
                lines.append(f"• {rec}")

        return "\n".join(lines)


# Singleton instance
echo_diagnostics = EchoDiagnostics()


# API Endpoints
@router.get("/health")
async def get_diagnostics_health():
    """Quick health check endpoint"""
    return {
        "status": "operational",
        "service": "Echo Brain Diagnostics",
        "last_check": echo_diagnostics.last_check.isoformat() if echo_diagnostics.last_check else None
    }


@router.get("/full")
async def perform_full_diagnosis():
    """Perform comprehensive system diagnosis"""
    diagnosis = await echo_diagnostics.perform_comprehensive_diagnosis()
    return diagnosis


@router.get("/telegram")
async def get_telegram_diagnosis():
    """Get diagnosis formatted for Telegram"""
    diagnosis = await echo_diagnostics.perform_comprehensive_diagnosis()
    formatted = await echo_diagnostics.format_telegram_diagnosis(diagnosis)
    return {
        "diagnosis": diagnosis,
        "telegram_format": formatted
    }


@router.get("/component/{component_name}")
async def check_component(component_name: str):
    """Check health of specific component"""
    if component_name == "database":
        return await echo_diagnostics.check_database_health()
    elif component_name == "vector":
        return await echo_diagnostics.check_vector_store_health()
    elif component_name == "ollama":
        return await echo_diagnostics.check_ollama_health()
    elif component_name == "system":
        return await echo_diagnostics.check_system_resources()
    elif component_name == "modules":
        return await echo_diagnostics.check_module_health()
    else:
        raise HTTPException(status_code=404, detail=f"Component {component_name} not found")


# Chat integration function for Telegram bot
async def handle_diagnosis_request(query: str) -> str:
    """Handle diagnosis request from chat interface"""
    query_lower = query.lower()

    # Check if this is a diagnosis request
    diagnosis_keywords = [
        "diagnos", "self-check", "health", "status",
        "check yourself", "system check", "are you ok",
        "what's wrong", "analyze yourself"
    ]

    if any(keyword in query_lower for keyword in diagnosis_keywords):
        # Perform diagnosis
        diagnosis = await echo_diagnostics.perform_comprehensive_diagnosis()

        # Format for chat response
        return await echo_diagnostics.format_telegram_diagnosis(diagnosis)

    return None  # Not a diagnosis request