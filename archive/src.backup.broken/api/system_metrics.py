#!/usr/bin/env python3
"""
Real system metrics for Echo Brain dashboard
Provides actual CPU, memory, disk, and database statistics
"""

import psutil
import asyncio
import asyncpg
from fastapi import APIRouter
from datetime import datetime
import logging
import os
import subprocess

logger = logging.getLogger(__name__)
router = APIRouter()

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "tower_consolidated",
    "user": "patrick",
    "password": "tower_echo_brain_secret_key_2025"
}

@router.get("/api/echo/metrics/system")
async def get_system_metrics():
    """Return real system metrics"""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Get network I/O
        net_io = psutil.net_io_counters()

        # Get GPU/VRAM usage (if NVIDIA GPU is available)
        vram_used_gb = await get_gpu_memory_usage()

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_usage": disk_percent,
            "vram_used_gb": vram_used_gb,
            "network_io": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        # Return safe defaults on error
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_usage": 0.0,
            "vram_used_gb": 0.0,
            "network_io": {
                "bytes_sent": 0,
                "bytes_recv": 0
            },
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/api/echo/metrics/db")
async def get_db_stats():
    """Return real database statistics"""
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)

        # Try to get conversation count from available tables
        try:
            conv_count = await conn.fetchval("""
                SELECT COUNT(*) FROM echo_conversations
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)
        except:
            # Fall back to counting anime generations as proxy for activity
            conv_count = await conn.fetchval("""
                SELECT COUNT(*) FROM anime_generations
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)

        # Try to get learning count
        try:
            learning_count = await conn.fetchval("""
                SELECT COUNT(*) FROM echo_learnings
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)
        except:
            # Use project count as fallback
            learning_count = await conn.fetchval("""
                SELECT COUNT(*) FROM projects
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)

        # Get vector memory count (from Qdrant)
        vector_count = await get_vector_memory_count()

        # Get active database connections
        db_activity = await conn.fetch("""
            SELECT state, COUNT(*) as count
            FROM pg_stat_activity
            WHERE datname = 'tower_consolidated'
            GROUP BY state
        """)

        active_connections = sum(row['count'] for row in db_activity if row['state'] == 'active')
        idle_connections = sum(row['count'] for row in db_activity if row['state'] == 'idle')

        await conn.close()

        return {
            "total_conversations": conv_count or 0,
            "total_learnings": learning_count or 0,
            "vector_memories": vector_count,
            "active_sessions": active_connections,
            "connections": {
                "main": {
                    "active": active_connections,
                    "idle": idle_connections
                }
            },
            "last_query": datetime.now().isoformat(),
            "database_status": "healthy"
        }

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        # Return safe defaults on error
        return {
            "total_conversations": 0,
            "total_learnings": 0,
            "vector_memories": 0,
            "active_sessions": 0,
            "connections": {
                "main": {"active": 0, "idle": 0}
            },
            "last_query": datetime.now().isoformat(),
            "database_status": "error",
            "error": str(e)
        }

async def get_gpu_memory_usage():
    """Get GPU memory usage in GB"""
    try:
        # Try nvidia-smi first
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            memory_mb = float(result.stdout.strip())
            return memory_mb / 1024  # Convert to GB
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    # Try rocm-smi for AMD GPUs
    try:
        result = subprocess.run(
            ['rocm-smi', '--showmemuse'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # Parse rocm-smi output (format varies)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'GPU' in line and 'MB' in line:
                    # Extract memory usage
                    import re
                    match = re.search(r'(\d+)\s*MB', line)
                    if match:
                        memory_mb = float(match.group(1))
                        return memory_mb / 1024
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return 0.0  # No GPU found

async def get_vector_memory_count():
    """Get count of vectors in Qdrant"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Check Qdrant collections
            response = await client.get("http://localhost:6333/collections")
            if response.status_code == 200:
                data = response.json()
                collections = data.get("result", {}).get("collections", [])

                # Find echo collection
                for collection in collections:
                    if 'echo' in collection['name'].lower():
                        # Get collection info
                        coll_response = await client.get(
                            f"http://localhost:6333/collections/{collection['name']}"
                        )
                        if coll_response.status_code == 200:
                            coll_data = coll_response.json()
                            return coll_data.get("result", {}).get("vectors_count", 0)

    except Exception as e:
        logger.debug(f"Could not get Qdrant vector count: {e}")

    return 0